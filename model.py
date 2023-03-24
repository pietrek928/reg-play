from typing import Tuple, Dict

from pydantic import BaseModel
from torch import float32, dtype, zeros, Tensor
from torch.nn import Parameter


class ValueDescr(BaseModel):
    # Empty means scalar
    shape: Tuple[int, ...] = ()
    type: dtype = float32
    descr: str = ''


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
class Model(BaseModel):
    @classmethod
    def get_inputs(cls):
        return {
            k: v for k, v in dir(cls)
            if isinstance(v, InputValue)
        }

    @classmethod
    def get_outputs(cls):
        return {
            k: v for k, v in dir(cls)
            if isinstance(v, OutputValue)
        }

    @classmethod
    def get_params(cls):
        return {
            k: v for k, v in dir(cls)
            if isinstance(v, ParamValue)
        }

    @classmethod
    def get_state(cls):
        return {
            k: v for k, v in dir(cls)
            if isinstance(v, StateValue)
        }

    @classmethod
    def compute_step(
            self, params: Values, state: Values, inputs: Values
    ) -> Tuple[Values, Values]:
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
