from typing import Tuple, Dict, Type, Any, Union

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
ValuesRec = Dict[str, Union[Tensor, 'ValuesRec']]
ValuesDescr = Dict[str, ValueDescr]
ValuesDescrRec = Dict[str, Union[ValueDescr, 'ValuesDescrRec']]
ParamsRec = Dict[str, Union[Parameter, 'ParamsRec']]


class InputValue(ValueDescr):
    pass


class StateValue(ValueDescr):
    pass


class OutputValue(ValueDescr):
    pass


class SymBlock(Module):
    def extract_attrs(self, parent_cls):
        return {
            k: getattr(self, k) for k in dir(self)
            if not k.startswith('_') and isinstance(getattr(self, k), parent_cls)
        }

    def get_inputs(self) -> ValuesDescr:
        return self.extract_attrs(InputValue)

    def get_outputs(self) -> ValuesDescr:
        return self.extract_attrs(OutputValue)

    def get_state(self) -> ValuesDescrRec:
        states = self.extract_attrs(StateValue)
        for name, submodel in self.extract_attrs(SymBlock).items():
            if isinstance(submodel, SymBlock):
                substate = submodel.get_state()
                if substate:
                    states[name] = submodel.get_state()
        return states

    def compute_step(
            self, inputs: ValuesRec
    ) -> Tuple[ValuesRec, ValuesRec]:  # new_state, outputs
        raise NotImplementedError('compute_step not implemented')


def init_zero_values(
        values: ValuesDescrRec,
        base_shape: Tuple[int, ...] = (),
        device=None,
) -> Values:
    tensors = {}
    for k, v in dict(values).items():
        if isinstance(v, dict):
            tensors[k] = init_zero_values(v, base_shape, device)
        else:
            shape = v.shape
            if not shape:
                shape = (1,)
            shape = base_shape + shape
            tensors[k] = zeros(*shape, dtype=v.type, device=device, requires_grad=False)
    return tensors


def init_zero_params(
        values: ValuesDescrRec,
        base_shape: Tuple[int, ...] = (),
        device=None,
) -> ValuesRec:
    tensors = {}
    for k, v in dict(values).items():
        if isinstance(v, dict):
            tensors[k] = init_zero_params(v, base_shape, device)
        else:
            shape = v.shape
            if not shape:
                shape = (1,)
            shape = base_shape + shape
            tensors[k] = Parameter(
                data=zeros(*shape, dtype=v.type, device=device)
            )
    return tensors


def init_torch_models(
        models_descr: TorchModelsDescr, device=None, train=False,
) -> Dict[str, Module]:
    models = {}
    for k, m in models_descr.items():
        mm = m.model(**m.params)
        mm.train(train)
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
        if k in values_descr:
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


def scale_values_grad(values: Values, scale: float):
    for v in values.values():
        v.grad *= scale
