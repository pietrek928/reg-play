from typing import Tuple, Dict, Type

from torch import Tensor, cat, exp
from torch.nn import Sequential, Linear, Module, BatchNorm1d, SELU, GELU, ELU, ReLU, Sigmoid, AlphaDropout, LSTM

from model import OutputValue, InputValue, Values, TorchModel, StateValue, SymBlock, ValuesRec, ValuesDescr
from named_tensor import NamedTensor
from utils import compute_dt, stack_values_np


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
            activation_cls: Type[Module] = ReLU,
            normalize=True,
    ):
        super().__init__()
        layers = []
        layers.append(Linear(in_features, out_features))
        if normalize:
            layers.append(BatchNorm1d(out_features))
        try:
            layers.append(activation_cls(inplace=True))
        except TypeError:
            layers.append(activation_cls())
        self.net = Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.net.forward(input)


def assemble_inputs(
        inputs: Values, keys: Tuple[str, ...],
) -> Tensor:
    try:
        return cat([
            inputs[k] for k in keys
        ], dim=-1)
    except RuntimeError as e:
        shapes = {k: inputs[k].shape for k in keys}
        raise RuntimeError(f'Failed to assemble inputs with shapes: {shapes}') from e


class DABLowRef(SymBlock):
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


class DABLowSimple(SymBlock):
    # Parameters
    Leq = InputValue(descr='Equivalent inductance[H]')
    nt = InputValue(descr='Transformer turns ratio')

    # Inputs
    VIN = InputValue(descr='Input voltage[V]')
    VOUT = InputValue(descr='Output voltage[V]')
    f = InputValue(descr='Switching frequency[Hz]')
    fi = InputValue(descr='Switching phase shift[Rad]')

    # Outputs
    IIN = OutputValue(descr='Input current[A]')
    IOUT = OutputValue(descr='Output current[A]')

    def compute_step(
            self, inputs: Values
    ) -> Tuple[Values, Values]:  # new_state, outputs
        fi = inputs['fi']
        k = fi * (1. - 2. * fi.abs()) * inputs['nt'] / (inputs['f'] * inputs['Leq'])
        return dict(), dict(
            IIN=inputs['VOUT'] * k,
            IOUT=inputs['VIN'].abs() * k,
        )


class TestDABReg(SymBlock):
    # Inputs
    VIN = InputValue(descr='Input voltage[V]')
    e = InputValue(descr='Difference from set value')
    f = InputValue(descr='Switching frequency[Hz]')

    # Outputs
    fi = OutputValue(shape=(1,), descr='Switching phase shift[Rad]')

    # State
    e_I = StateValue(shape=(1,), descr='Difference from set value - integrated')
    # state = StateValue(shape=(16,), descr='DAB regulator internal state')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        lstm_layers = 2
        n_lstm = 64
        n = 64

        self.lstm_state_1 = StateValue(shape=(lstm_layers, n_lstm), descr='DAB regulator internal state')
        self.lstm_state_2 = StateValue(shape=(lstm_layers, n_lstm), descr='DAB regulator internal state')

        self.lstm = LSTM(5, n_lstm, num_layers=lstm_layers, batch_first=True)
        self.model = Sequential(
            LinearBlock(n_lstm, n),
            # AlphaDropout(0.04),
            ResidualSequential(
                LinearBlock(n, n),
                LinearBlock(n, n),
                LinearBlock(n, n),
                LinearBlock(n, n),
                LinearBlock(n, n),
                LinearBlock(n, n),
            ),
            # AlphaDropout(0.04),
            LinearBlock(n, n),
            Linear(n, 2),
        )

    def compute_step(
            self, inputs: ValuesRec
    ) -> Tuple[ValuesRec, ValuesRec]:  # new_state, outputs
        e = inputs['VOUT_set'] - inputs['VOUT']
        e_I = (inputs['e_I'] + e / inputs['f']).tanh()
        # e_I = inputs['e_I']

        lstm_in = assemble_inputs(
            inputs | dict(
                e_I=e_I,
            ), ('VIN', 'VOUT', 'VOUT_set', 'e_I', 'f')
        )
        lstm_out, (lstm_state_1, lstm_state_2) = self.lstm(lstm_in.unsqueeze(-2), (  # lstm needs additional dim
            inputs['lstm_state_1'].transpose(0, 1).contiguous(),  # batch as second dim
            inputs['lstm_state_2'].transpose(0, 1).contiguous()
        ))
        model_out = self.model.forward(lstm_out.squeeze(-2))

        fi_v = model_out[..., 0:1]
        return dict(
            e_I=e_I,
            lstm_state_1=lstm_state_1.transpose(0, 1),
            lstm_state_2=lstm_state_2.transpose(0, 1),
        ), dict(
            fi_v=fi_v,
            fi=fi_v.tanh() * .5,
        )


class DABRCModel(SymBlock):
    R = InputValue(descr='Load resistance[Ohm]')
    C = InputValue(descr='Load capacitance[F]')
    VOUT_set = InputValue(descr='target output voltage value[V]')

    VOUT = StateValue(descr='Output capacitor voltage[V]')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dab_model = DABLowSimple()
        self.reg_model = TestDABReg()

    def get_inputs(self) -> ValuesDescr:
        return super().get_inputs() | dict(
            Leq=self.dab_model.Leq,
            nt=self.dab_model.nt,
            VIN=self.dab_model.VIN,
            f=self.dab_model.f,
        )

    def compute_step(
            self, inputs: ValuesRec
    ) -> Tuple[ValuesRec, ValuesRec]:  # new_state, outputs
        reg_state, reg_outputs = self.reg_model.compute_step(
            inputs | inputs['reg_model'] | dict(
                VOUT_set=inputs['VOUT_set'],
                VOUT=inputs['VOUT'],
            )
        )

        dab_state, dab_outputs = self.dab_model.compute_step(
            inputs | dict(
                fi=reg_outputs['fi'],
                Leq=25e-6, nt=.12,
            )
        )

        vout = inputs['VOUT']
        iout = dab_outputs['IOUT']
        
        R = inputs['R']
        a = exp(-1. / (R * inputs['C'] * inputs['f']))
        vout = vout * a + (1. - a) * iout * R

        return dict(
            reg_model=reg_state
        ) | dict(
            VOUT=vout,
        ), reg_outputs | dict(
            VOUT=vout,
            iout=iout,
        )


def u_step_sin_case(n, f, vinrms, fin, vout, t_step):
    import numpy as np
    VIN = (vinrms * np.sqrt(2)) * np.sin((2 * np.pi * fin / f) * np.arange(n))

    VOUT_set = np.zeros(n)
    nt = int(t_step * f)
    it = nt
    while it < n:
        VOUT_set[it:it + nt] = vout
        it += 2 * nt

    return dict(VIN=VIN, VOUT_set=VOUT_set, f=np.ones(n) * f)


def u_const_dc_step_case(n, f, vin, vout, t_step):
    import numpy as np
    VIN = np.zeros(n)
    nt = int(t_step * f)
    it = nt
    while it < n:
        VIN[it:it + nt] = vin
        it += 2 * nt

    VOUT_set = np.ones(n) * vout

    return dict(VIN=VIN, VOUT_set=VOUT_set, f=np.ones(n) * f)


def u_sin_sin_case(n, f, uinrms, fin, uoutrms, fout):
    import numpy as np
    VIN = (uinrms * np.sqrt(2)) * np.sin((2 * np.pi * fin / f) * np.arange(n))
    VOUT_set = (uoutrms * np.sqrt(2)) * np.sin((2 * np.pi * fout / f) * np.arange(n))

    return dict(VIN=VIN, VOUT_set=VOUT_set, f=np.ones(n) * f)


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
    for f in np.linspace(f_min, f_max + 1e-6, 100):
        for _ in range(5):
            uin = np.random.uniform(uin_min, uin_max)
            uout = np.random.uniform(uout_min, uout_max)
            fin = np.random.uniform(fin_min, fin_max)
            fout = np.random.uniform(fout_min, fout_max)
            t_step = np.random.uniform(t_step_min, t_step_max)

            cases.append(u_step_sin_case(n, f, uin, fin, uout, t_step))
            cases.append(u_const_dc_step_case(n, f, uin, uout, t_step))
            cases.append(u_sin_sin_case(n, f, uin, fin, uout, fout))

    return stack_values_np(cases, axis=-2, append_dim=True)


def make_random_steps(n, tmin, tmax):
    import numpy as np

    v = np.zeros(n, dtype=np.float32)
    it = 0
    while it < n:
        t = int(np.random.uniform(tmin, tmax))
        v[it:it + t] = 1
        it += t

        it += int(np.random.uniform(tmin, tmax))

    return v


# prepare output resistance and capacitance
def prepare_out_params(n):
    import numpy as np

    steps_min_period = 1000
    steps_max_period = 10000
    r_min = .1
    r_max = 1
    c_min = 1e-4
    c_max = 1e-3

    cases = []
    for _ in range(1500):
        cases.append(dict(
            R=make_random_steps(n, steps_min_period, steps_max_period) * np.random.uniform(0, r_max - r_min) + r_min,
            C=make_random_steps(n, steps_min_period, steps_max_period) * np.random.uniform(0, c_max - c_min) + c_min,
        ))
    return stack_values_np(cases, axis=-2, append_dim=True)
