from typing import Tuple, Dict

from torch import Module, Tensor, cat
from torch.nn import Sequential, Linear, SELU

from model import OutputValue, InputValue, Model, Values, StateValue, TorchModel


class DABLowRef(Model):
    class Model(Module):
        def __init__(self):
            self.model = Sequential(
                Linear(8, 16),
                SELU(inplace=True),
                Linear(16, 32),
                SELU(inplace=True),
                Linear(32, 8),
            )

        def forward(self, x: Tensor):
            return self.model(x)

    # Outputs
    VD = OutputValue(shape=(2,), descr='Both sides diode avg voltage')
    I = OutputValue(shape=(2,), descr='Both sides avg current')

    # Inputs
    V = InputValue(shape=(2,), descr='Both sides voltage')
    f = InputValue(desct='Switching frequency[Hz]')
    d = InputValue(descr='Switching phase shift')

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
            cat((state['state'], inputs['V'], inputs['f'].unsqueeze(-1), inputs['d'].unsqueeze(-1)), dim=-1)
        )
        return dict(
            # TODO: improve integration
            state=state['state'] + model_out[..., :4] * inputs['dt'].unsqueeze(-1)
        ), dict(
            VD=model_out[..., 4:6],
            I=model_out[..., 6:8],
        )
