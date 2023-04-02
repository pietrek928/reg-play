from typing import Tuple, Dict, Type, Any

from pydantic import BaseModel
from torch import float32, dtype, zeros, Tensor
from torch.nn import Parameter, Module


class ValueDescr(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    # Empty means scalar
    shape: Tuple[int, ...] = (1,)
    type: dtype = float32
    descr: str = ''


class TorchModel(BaseModel):
    model: Type[Module]
    params: Dict[str, Any] = {}
    descr: str = ''


TorchModelsDescr = Dict[str, TorchModel]
Values = Dict[str, Tensor]
ValuesDescr = Dict[str, ValueDescr]


class InputValue(ValueDescr):
    pass


class ParamValue(ValueDescr):
    pass


class StateValue(ValueDescr):
    pass


class OutputValue(ValueDescr):
    pass


# TODO: add submodels ?
class Model:
    @classmethod
    def get_inputs(cls) -> ValuesDescr:
        return {
            k: v for k, v in vars(cls).items()
            if isinstance(v, InputValue)
        }

    @classmethod
    def get_outputs(cls) -> ValuesDescr:
        return {
            k: v for k, v in vars(cls).items()
            if isinstance(v, OutputValue)
        }

    @classmethod
    def get_params(cls) -> ValuesDescr:
        return {
            k: v for k, v in vars(cls).items()
            if isinstance(v, ParamValue)
        }

    @classmethod
    def get_state(cls) -> ValuesDescr:
        return {
            k: v for k, v in vars(cls).items()
            if isinstance(v, StateValue)
        }

    @classmethod
    def get_torch_models(cls) -> TorchModelsDescr:
        return {
            k: v for k, v in vars(cls).items()
            if isinstance(v, TorchModel)
        }

    @classmethod
    def compute_step(
            cls, params: Values, torch_models: Dict[str, Module],
            state: Values, inputs: Values
    ) -> Tuple[Values, Values]:  # new_state, outputs
        raise NotImplementedError('compute_step not implemented')


def init_zero_params(
        values: ValuesDescr,
        base_shape: Tuple[int, ...] = (),
        device=None,
) -> Values:
    tensors = {}
    for k, v in dict(values).items():
        shape = base_shape + v.shape
        if not shape:
            shape = (1,)
        tensors[k] = Parameter(
            data=zeros(*shape, dtype=v.type, device=device)
        )
    return tensors


def init_torch_models(
        models_descr: TorchModelsDescr, device=None
) -> Dict[str, Module]:
    models = {}
    for k, m in models_descr.items():
        mm = m.model(**m.params)
        if device is not None:
            mm.to(device)
        models[k] = mm
    return models


def get_parameters(*items):
    for o in items:
        if isinstance(o, Module):
            yield from o.parameters()
        elif isinstance(o, dict):
            yield from o.values()


def validate_values(values_descr: ValuesDescr, values: Values):
    prefix = None

    for k, v in values.items():
        descr = values_descr[k]
        if v.shape[-len(descr.shape):] != descr.shape:
            raise ValueError(
                f'Argument {k}({descr.descr}): '
                f'Invalid value shape {v.shape} for required shape {descr.shape}'
            )

        current_prefix = v.shape[:-len(descr.shape)]
        if prefix is None:
            prefix = current_prefix
        elif prefix != current_prefix:
            raise ValueError(f'Inconsistent shape prefix {prefix} != {current_prefix}')

    return prefix
