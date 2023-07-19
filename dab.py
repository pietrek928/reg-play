from typing import Tuple, Dict, Type

from torch import Tensor, cat
from torch.nn import Sequential, Linear, Module, BatchNorm1d, SELU, ReLU, AlphaDropout

from model import OutputValue, InputValue, Model, Values, TorchModel
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
