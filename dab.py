from typing import Tuple, Dict, Type

from torch import Tensor, cat
from torch.nn import Sequential, Linear, Module, BatchNorm1d, SELU, AlphaDropout

from model import OutputValue, InputValue, Model, Values, TorchModel, ParamValue, StateValue
from named_tensor import NamedTensor
from utils import compute_dt


class ResidualSequential(Sequential):
    @property
    def creates_layer(self):
        return any(
            m.creates_layer for m in self
        )

    def forward(self, input: Tensor) -> Tensor:
        for block in self:
            input = input + block(input)
        return input


class LinearBlock(Module):
    def __init__(
            self, in_features, out_features,
            activation_cls: Type[Module] = SELU,
            normalize=True,
    ):
        super().__init__()
        layers = []
        layers.append(Linear(in_features, out_features))
        if normalize:
            layers.append(BatchNorm1d(out_features))
        layers.append(activation_cls(inplace=True))
        self.net = Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.net.forward(input)


def assemble_inputs(
        history_size: int,
        inputs: Tuple[Values, ...], input_keys: Tuple[str, ...],
        outputs: Tuple[Values, ...], output_keys: Tuple[str, ...],
) -> Tensor:
    items = []
    for it in range(history_size - 1):
        for k in input_keys:
            items.append(inputs[-it - 1][k])
        for k in output_keys:
            items.append(outputs[-it][k])
    for k in input_keys:
        items.append(inputs[-1][k])
    return cat(items, dim=-1)


class DABLowRef(Model):
    history_size = 2

    class Model(Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            n = 256

            self.model = Sequential(
                LinearBlock(10, 64),
                AlphaDropout(0.1),
                LinearBlock(64, n),
                ResidualSequential(
                    LinearBlock(n, n),
                    LinearBlock(n, n),
                    LinearBlock(n, n),
                    LinearBlock(n, n),
                ),
                LinearBlock(n, 64),
                Linear(64, 5),

                # LinearBlock(17, 256),
                # ResidualSequential(
                #     LinearBlock(256, 256),
                # ),
                # Linear(256, 5),

                # BatchNorm1d(5),
                # SELU(inplace=True),
                # Linear(5, 5),
            )

        def forward(self, x: Tensor):
            return self.model(x)

    # Inputs
    # V = InputValue(shape=(2,), descr='Both sides voltage')
    VS = InputValue(descr='Secondary side voltage')
    # f = InputValue(descr='Switching frequency[Hz]')
    d = InputValue(descr='Switching phase shift')
    dt = InputValue(descr='Time step')

    # Outputs
    PD = OutputValue(shape=(2,), descr='Both sides diode avg power loss')
    I = OutputValue(shape=(2,), descr='Both sides avg current')

    # State
    # state = StateValue(shape=(1,), descr='DAB internal state')

    # Torch models
    model = TorchModel(model=Model, descr='Main model computing derivatives for integration')

    @classmethod
    def compute_step(
            cls, params: Values, torch_models: Dict[str, Module],
            inputs: Tuple[Values, ...], outputs: Tuple[Values, ...]
    ) -> Tuple[Values, Values]:  # new_state, outputs
        model_out = torch_models['model'](
            # cat((state['state'], inputs['VS'], inputs['d'], inputs['dt']), dim=-1)
            assemble_inputs(
                cls.history_size,
                inputs, ('VS', 'd', 'dt'),
                outputs, ('PD', 'I'),
            )
        )
        return dict(
            # TODO: improve integration
            # state=state['state'] + model_out[..., 4:5] * inputs['dt']
            # state=model_out[..., 4:5]
        ), dict(
            PD=model_out[..., 0:2],
            I=model_out[..., 2:4],
        )


def transform_sim_data(sim_data: NamedTensor) -> NamedTensor:
    return NamedTensor.from_params(
        dt=compute_dt(sim_data[..., 'Time']),
        VS=sim_data[..., 'Vm1:Measured voltage'],
        d=sim_data[..., 'Triangular Wave'],

        PD=cat((
            (sim_data[..., 'D3:Diode voltage'] * sim_data[..., 'D3:Diode current']).abs()
            + (sim_data[..., 'D4:Diode voltage'] * sim_data[..., 'D4:Diode current']).abs(),
            (sim_data[..., 'D1:Diode voltage'] * sim_data[..., 'D1:Diode current']).abs()
            + (sim_data[..., 'D7:Diode voltage'] * sim_data[..., 'D7:Diode current']).abs(),
        ), dim=-1),
        I=cat((
            sim_data[..., 'Am1:Measured current'], sim_data[..., 'Am3:Measured current'],
        ), dim=-1),
    )


class DABLowSimple(Model):
    history_size = 0

    # Parameters
    Leq = ParamValue(descr='Equivalent inductance[H]')
    nt = ParamValue(descr='Transformer turns ratio')

    # Inputs
    VIN = InputValue(descr='Input voltage[V]')
    VOUT = InputValue(descr='Output voltage[V]')
    f = InputValue(descr='Switching frequency[Hz]')
    fi = InputValue(descr='Switching phase shift[Rad]')

    # Outputs
    IIN = OutputValue(descr='Input current[A]')
    IOUT = OutputValue(descr='Output current[A]')

    @classmethod
    def compute_step(
            cls, params: Values, torch_models: Dict[str, Module],
            inputs: Tuple[Values, ...], outputs: Tuple[Values, ...]
    ) -> Tuple[Values, Values]:  # new_state, outputs
        in_vals = inputs[-1]

        k = in_vals['fi'] * (1. - 2. * in_vals['fi'].abs()) / (in_vals['f'] * params['Leq'])
        return dict(), dict(
            IIN=in_vals['VOUT'] * k,
            IOUT=in_vals['VIN'] * params['nt'] * k,
        )


class TestDABReg(Model):
    history_size = 2

    class Model(Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            n = 256

            self.model = Sequential(
                LinearBlock(30, 64),
                AlphaDropout(0.1),
                LinearBlock(64, n),
                ResidualSequential(
                    LinearBlock(n, n),
                    LinearBlock(n, n),
                    LinearBlock(n, n),
                    LinearBlock(n, n),
                ),
                LinearBlock(n, 64),
                Linear(64, 6),
            )

        def forward(self, x: Tensor):
            return self.model(x)

    # Inputs
    VIN = InputValue(descr='Input voltage[V]')
    e = InputValue(descr='Difference from set value')
    f = InputValue(descr='Switching frequency[Hz]')

    # Outputs
    fi = OutputValue(shape=(1,), descr='Switching phase shift[Rad]')

    # State
    fi_I = OutputValue(shape=(1,), descr='Switching phase shift[Rad] - integrated')
    state = StateValue(shape=(4,), descr='DAB regulator internal state')

    # Torch models
    model = TorchModel(model=Model, descr='Main model computing regulator')

    @classmethod
    def compute_step(
            cls, params: Values, torch_models: Dict[str, Module],
            inputs: Tuple[Values, ...], outputs: Tuple[Values, ...]
    ) -> Tuple[Values, Values]:  # new_state, outputs
        model_out = torch_models['model'](
            assemble_inputs(
                cls.history_size,
                inputs, ('VIN', 'e', 'f', 'fi_I', 'state'),
                outputs, ('fi',),
            )
        )

        fi_v = model_out[..., 0:1]
        fi_I = inputs[-1]['fi_I'] + model_out[..., 1:2]
        return dict(
            fi_I=fi_I,
            state=model_out[..., 2:],
        ), dict(
            fi=(fi_v + fi_I).tanh() * .5,
        )


def u_step_sin_case(n, f, uinrms, fin, uout, t_step):
    import numpy as np
    UIN = (uinrms * np.sqrt(2)) * np.sin((2 * np.pi * fin / f) * np.arange(n))

    UOUT = np.zeros(n)
    nt = int(t_step * f)
    it = nt
    while it < n:
        UOUT[it:it + nt] = uout
        it += 2 * nt

    return dict(UIN=UIN, UOUT=UOUT, f=np.ones(n) * f)


def u_const_dc_step_case(n, f, uin, uout, t_step):
    import numpy as np
    UIN = np.zeros(n)
    nt = int(t_step * f)
    it = nt
    while it < n:
        UIN[it:it + nt] = uin
        it += 2 * nt

    UOUT = np.ones(n) * uout

    return dict(UIN=UIN, UOUT=UOUT, f=np.ones(n) * f)


def u_sin_sin_case(n, f, uinrms, fin, uoutrms, fout):
    import numpy as np
    UIN = (uinrms * np.sqrt(2)) * np.sin((2 * np.pi * fin / f) * np.arange(n))
    UOUT = (uoutrms * np.sqrt(2)) * np.sin((2 * np.pi * fout / f) * np.arange(n))

    return dict(UIN=UIN, UOUT=UOUT, f=np.ones(n) * f)


def union_cases(cases):
    import numpy as np
    return dict(
        UIN=np.concatenate([c['UIN'] for c in cases], axis=-1)[..., np.newaxis],
        UOUT=np.concatenate([c['UOUT'] for c in cases], axis=-1)[..., np.newaxis],
        f=np.concatenate([c['f'] for c in cases], axis=-1)[..., np.newaxis],
    )


def prepare_test_cases(n):
    import numpy as np

    f_min = 3e3
    f_max = 1e4
    uin_min = 100
    uin_max = 300
    uout_min = 30
    uout_max = 80
    fin_min = 50
    fin_max = 100
    fout_min = 50
    fout_max = 100
    t_step_min = .1
    t_step_max = .5

    cases = []
    for f in np.linspace(f_min, f_max + 1e-6, 1000):
        for _ in range(100):
            uin = np.random.uniform(uin_min, uin_max)
            uout = np.random.uniform(uout_min, uout_max)
            fin = np.random.uniform(fin_min, fin_max)
            fout = np.random.uniform(fout_min, fout_max)
            t_step = np.random.uniform(t_step_min, t_step_max)

            cases.append(u_step_sin_case(n, f, uin, fin, uout, t_step))
            cases.append(u_const_dc_step_case(n, f, uin, uout, t_step))
            cases.append(u_sin_sin_case(n, f, uin, fin, uout, fout))

    return union_cases(cases)