from typing import Tuple, Dict

from torch import Tensor, cat, stack
from torch.nn import Sequential, Linear, SELU, Module

from model import OutputValue, InputValue, Model, Values, StateValue, TorchModel
from named_tensor import NamedTensor
from utils import compute_dt


class DABLowRef(Model):
    class Model(Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.model = Sequential(
                Linear(7, 16),
                SELU(inplace=True),
                Linear(16, 32),
                SELU(inplace=True),
                Linear(32, 8),
            )

        def forward(self, x: Tensor):
            return self.model(x)

    # Inputs
    # V = InputValue(shape=(2,), descr='Both sides voltage')
    VS = InputValue(descr='Secondary side voltage')
    # f = InputValue(desct='Switching frequency[Hz]')
    d = InputValue(descr='Switching phase shift')
    dt = InputValue(descr='Time step')

    # Outputs
    PD = OutputValue(shape=(2,), descr='Both sides diode avg power loss')
    I = OutputValue(shape=(2,), descr='Both sides avg current')

    # State
    state = StateValue(shape=(4,), descr='DAB internal state')

    # Torch models
    model = TorchModel(model=Model, descr='Main model computing derivatives for integration')

    @classmethod
    def compute_step(
            cls, params: Values, torch_models: Dict[str, Module],
            state: Values, inputs: Values
    ) -> Tuple[Values, Values]:  # new_state, outputs
        model_out = torch_models['models'](
            cat((state['state'], inputs['VS'], inputs['d'], inputs['dt']), dim=-1)
        )
        return dict(
            # TODO: improve integration
            state=state['state'] + model_out[..., :4] * inputs['dt']
        ), dict(
            PD=model_out[..., 4:6],
            I=model_out[..., 6:8],
        )


def transform_sim_data(sim_data: NamedTensor) -> NamedTensor:
    return NamedTensor.from_params(
        dt=compute_dt(sim_data[..., 'Time']).unsqueeze(-1),
        VS=sim_data[..., 'Vm1:Measured voltage'].unsqueeze(-1),
        d=sim_data[..., 'Triangular Wave'].unsqueeze(-1),

        PD=stack((
            (sim_data[..., 'D3:Diode voltage'] * sim_data[..., 'D3:Diode current']).abs()
            + (sim_data[..., 'D4:Diode voltage'] * sim_data[..., 'D4:Diode current']).abs(),
            (sim_data[..., 'D1:Diode voltage'] * sim_data[..., 'D1:Diode current']).abs()
            + (sim_data[..., 'D7:Diode voltage'] * sim_data[..., 'D7:Diode current']).abs(),
        ), dim=-1),
        I=stack((
            sim_data[..., 'Am1:Measured current'], sim_data[..., 'Am3:Measured current'],
        ), dim=-1),
    )
