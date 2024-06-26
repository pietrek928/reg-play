from os import environ, path
from random import randrange
from sys import argv
from typing import Any
from model import Values, ValuesRec

import numpy as np
from pydantic import BaseModel
from torch import Tensor

from adapt import adapt_rc_dab_control, compute_rc_dab_sym_params, score_controller, fill_with_grad, adapt_rc_dab_reg
from dab import DABRCOptimModel, prepare_test_cases, prepare_out_params, DABRCModel
from grad import compute_grad
from obj import DynSystem, Block, SystemBlock
from utils import default_device, get_at_pos, get_shapes, merge_values
from vis import plot_controller_sym, plot_time_graphs


DATASET_PATH = path.join(environ.get('DATASET_PATH', '.'), 'reg-test')


class SolderState(BaseModel):
    T1: Any = 0.
    T2: Any = 0
    Tx: Any = 0.


class SolderDrive(BaseModel):
    supply: Any = 0.
    T0: Any = 0.


class SolderSensor(BaseModel):
    T1: Any = 0.


class SolderParams(BaseModel):
    ku1: Any = 0.

    k10: Any = 0.
    k12: Any = 0.

    k20: Any = 0.
    k21: Any = 0.
    k2x: Any = 0.

    kx0: Any = 0.  # estim ?
    kx2: Any = 0.  # estim ?

    @staticmethod
    def setup(
            C1=1., C2=1., Cx=1.,
            ku1=1.,
            k12=1., k10=1.,
            k20=1., k2x=1.,
            kx0=1.
    ) -> 'SolderParams':
        return SolderParams(
            ku1=ku1 / C1,
            k10=k10 / C1, k12=k12 / C1,
            k20=k20 / C2, k21=k12 / C2, k2x=k2x / C2,
            kx0=kx0 / Cx, kx2=k2x / Cx
        )


class SolderSystem(DynSystem):
    def __init__(self):
        super().__init__(SolderState, SolderDrive, SolderSensor, SolderParams)

    @staticmethod
    def _dx(x: SolderState, u: SolderDrive, p: SolderParams):
        return SolderState(
            T1=x.T1 * p.k12 + u.T0 * p.k10 - x.T1 * (p.k12 + p.k10) + u.supply * p.ku1,
            T2=x.T1 * p.k21 + x.Tx * p.k2x + u.T0 * p.k20 - x.T2 * (p.k21 + p.k2x + p.k20),
            Tx=x.T2 * p.kx2 + u.T0 * p.kx0 - x.Tx * (p.kx2 + p.kx0),
        )

    @staticmethod
    def _y(x: SolderState, p: SolderParams):
        return SolderSensor(T1=x.T1)


class SolderControllerState(BaseModel):
    pass


class SolderControllerParams(BaseModel):
    k: Any = 0.
    ki: Any = 0.


class SolderController(Block):
    def __init__(self, params: SolderControllerParams):
        super().__init__()
        self.params = params
        self._e = 0.
        self._i = 0.

    def process(self, *inputs, dt):
        current: SolderSensor
        setup: float
        current, setup = inputs
        self._e = setup - current.T1
        self._i += self._e * dt

    @property
    def state(self):
        return dict(
            e=self._e,
            # i=self._i,
        )

    @property
    def output(self):
        supply = self._e * self.params.k + self._i * self.params.ki
        if supply < 0.:
            # supply = set_value(supply, 0.)  # clamp without gradient loss ;p
            supply = 0.
        if supply > 100.:
            # supply = set_value(supply, 100.)  # clamp without gradient loss ;p
            supply = 100.
        return SolderDrive(
            supply=supply,
            T0=25.
        )


# a = GradVar(2)
# b = GradVar(3)
# c = a + a * b
# d = b + a * b
#
# e = c + d
#
# compute_grad(c)
#
# print(a.compute_grad())
# print(b.compute_grad())


def plot_sym():
    plot_controller_sym(
        SystemBlock(SolderSystem(), SolderParams.setup(
            C1=2., C2=2., Cx=.02,
            k10=.2, k20=.2, kx0=.1,
            k12=10.,
            k2x=.4
        )),
        SolderController(SolderControllerParams(k=11.300826300591783, ki=2.6559545295704656)),
        lambda t: (120. if t < 50. else 40.),
        100., .01
    )


def score_func(state: SolderState, target: float):
    e = state.Tx - target
    if e < 0.:
        return e * e
    else:
        return e * e * 10.


def optimize():
    k = 5.77974002996507
    ki = 10.194518292221606
    lv = 0.
    for i in range(100):
        controller_params = fill_with_grad(SolderControllerParams(k=k, ki=ki))
        vg = score_controller(
            SystemBlock(SolderSystem(), SolderParams.setup(
                C1=2., C2=2., Cx=.02,
                k10=.2, k20=.2, kx0=.1,
                k12=10.,
                k2x=.4
            )),
            SolderController(controller_params),
            lambda t: (120. if t < 50. else 40.),
            score_func,
            100., .01
        )
        compute_grad(vg)
        grad_array = np.array([controller_params.k.compute_grad(), controller_params.ki.compute_grad()])
        print(vg.value - lv)
        lv = vg.value
        print(grad_array / vg.value)
        grad_array /= vg.value
        grad_array *= 10.
        k -= grad_array[0]
        ki -= grad_array[1]
        print(f'k={k}, ki={ki}')


# plot_sym()
# r = CSVNumbersReader('/home/pietrek/Downloads/data.csv')
# r.read(1024)  # skip rows
# data = transform_sim_data(
#     r.read(100000000000)
# )
# samples = tuple(sample_many(data, 2000, 52, ('VS', 'd', 'I')))
# # plot_dataset(samples[100])
# dataset = prepare_dataset(samples)
#
# adapt_model(DABLowRef, dataset, l1_loss_normalized(dataset), .15, device=default_device())


def dab_rc_loss_func(outputs: Values, inputs: ValuesRec):
    from torch import linspace

    # device = outputs['VOUT'].device
    # steps_count = outputs['VOUT'].shape[0]
    # loss_weights = linspace(.1, 1., steps_count, device=device) ** 1.7
    values = (
            (outputs['VOUT'] - inputs['VOUT_set'])  # * loss_weights
    ).abs() + outputs['iout'].abs() * .01 + outputs['fi_v'].abs() * .01

    # values[~values.isfinite() | (values > 1e3)] = 0.

    # smaller gradient for clamped values
    # values = values.clamp(max=1e3)
    # return (clamped + (values - clamped) * .001).mean()
    return values


def dab_rc_control_loss_func(outputs: Values, inputs: ValuesRec):
    from torch import linspace

    # device = outputs['VOUT'].device
    # steps_count = outputs['VOUT'].shape[0]
    # loss_weights = linspace(.1, 1., steps_count, device=device)
    P_out = outputs['VOUT'] * outputs['iout']
    P_in = inputs['VIN'] * outputs['iin']
    values = (
            (outputs['VOUT'] - inputs['VOUT_set']) # * loss_weights
    ).abs() + P_out.abs() * .001 + P_in.abs() * .001 + inputs['fi_reg'].abs() * .01

    # values[~values.isfinite() | (values > 1e3)] = 0.

    # smaller gradient for clamped values
    # values = values.clamp(max=1e3)
    # return (clamped + (values - clamped) * .001).mean()
    return values


def prepare_train_control():
    import torch

    n = 1024 * 3
    case_count = 3006
    model_input = prepare_test_cases(n, case_count) | prepare_out_params(n, case_count)
    controls = adapt_rc_dab_control(DABRCOptimModel(), model_input, dab_rc_control_loss_func, ('fi_reg', ), device=default_device())
    rest_params = compute_rc_dab_sym_params(DABRCOptimModel(), merge_values(model_input, controls), device=default_device())

    all_params = merge_values(model_input, controls, rest_params)
    torch.save(all_params, path.join(DATASET_PATH, 'controls.pt'))


def show_optimized_control():
    import torch

    data = torch.load(path.join(DATASET_PATH, 'controls.pt'), map_location='cpu')
    print(f'loss={float(dab_rc_control_loss_func(data, data).mean())}')
    for _ in range(16):
        n = randrange(0, 3006)
        # dims are time, case, ...
        plot_time_graphs(get_at_pos(data, n, dim=1), ('VIN', 'VOUT_set', 'VOUT', 'R', 'C', 'iout', 'fi_reg'))


def optimize_test_dab():
    import torch
    device = default_device()
    data = torch.load(path.join(DATASET_PATH, 'controls.pt'), map_location=device)
    model = DABRCModel()

    # for k, v in model_input.items():
    #     print(k, tuple(v[it].mean() for it in range(50)))
    data['fi_guide'] = data['fi_reg']
    try:
        adapt_rc_dab_reg(model, data, dab_rc_loss_func, ('fi_reg', ), 150, 150., device=default_device())
        # adapt_rc_dab_control(DABRCOptimModel(), model_input, dab_rc_control_loss_func, ('fi_reg', ), device=default_device())
    finally:
        print('Saving model...')
        torch.save(model.reg_model.state_dict(), path.join(DATASET_PATH, 'reg-model.pt'))


def sim_test_dab_reg():
    import torch
    with torch.no_grad():
        device = default_device()
        data = torch.load(path.join(DATASET_PATH, 'controls.pt'), map_location=device)
        data.pop('fi_reg', None)

        model = DABRCModel()
        model.eval()
        checkpoint = torch.load(path.join(DATASET_PATH, 'reg-model.pt'), map_location='cpu')
        model.reg_model.load_state_dict(checkpoint)
        model.to(device)

        computed_params = compute_rc_dab_sym_params(model, data, device=device)
        torch.save(merge_values(data, computed_params), path.join(DATASET_PATH, 'model-sim.pt'))


if __name__ == '__main__':
    func_name = argv[1]
    globals()[func_name](*argv[2:])
