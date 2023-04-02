from typing import Tuple

from torch import Tensor, ones, cat, float32


class NamedTensor:
    def __init__(self, tensor: Tensor, axis_descr):
        self.tensor = tensor
        self.axis_descr = self._prepare_axis_mapping(axis_descr)

    @classmethod
    def from_params(cls, **params):
        it = 0
        axis_descr = {}
        values = []
        for k, v in params.items():
            if v.shape[-1] == 1:
                axis_descr[k] = it
                it += 1
            else:
                axis_descr[k] = tuple(range(it, it + v.shape[-1]))
                it += v.shape[-1]

            values.append(v)
        return cls(
            tensor=cat(values, dim=-1),
            axis_descr=axis_descr
        )

    def values(self, dtype=float32, device=None):
        return {
            k: self[..., k].to(dtype=dtype, device=device)
            for k in self.axis_descr
        }

    @staticmethod
    def _prepare_axis_mapping(axis_descr):
        if isinstance(axis_descr, (list, tuple)):
            return {
                n: it for it, n in enumerate(axis_descr)
            }
        elif isinstance(axis_descr, dict):
            return axis_descr
        else:
            raise ValueError(f'Unknown axis_descr type {type(axis_descr)}')

    def _convert_axis(self, axis):
        if axis is None:
            return None

        if isinstance(axis, int):
            return axis

        if isinstance(axis, (list, tuple)):
            return tuple(self._convert_axis(a) for a in axis)

        if isinstance(axis, str):
            d = self.axis_descr.get(axis)
            if d is None:
                raise ValueError(f'Unknown axis name {axis}')

            if isinstance(d, int):
                return d
            elif isinstance(d, tuple):
                start, end = d
                return slice(start, end + 1)
            else:
                raise ValueError(f'Unknown axis_descr type {type(d)}')

        if axis is Ellipsis:
            return axis

        if isinstance(axis, slice):
            return slice(
                self._convert_axis(axis.start),
                self._convert_axis(axis.stop),
                self._convert_axis(axis.step),
            )

        raise ValueError(f'Unknown axis type {type(axis)}')

    def __getitem__(self, idx):
        idx = self._convert_axis(idx)
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        if isinstance(idx, tuple) and isinstance(idx[-1], int):
            idx = (*idx[:-1], slice(idx[-1], idx[-1] + 1))
        return self.tensor.__getitem__(idx)

    def sample_sums(self, start: int, end: int, step: int, mean_axis: Tuple[str, ...] = ()):
        end = (end - start) // step * step + start

        sampled = self.tensor[start:end, ...].view(-1, step, *self.tensor.shape[1:])
        sampled = sampled.sum(dim=1)

        scale_tensor = ones(*self.tensor.shape[1:], dtype=sampled.dtype, device=sampled.device)
        for axis in mean_axis:
            scale_tensor[self._convert_axis(axis)] = 1. / step

        return NamedTensor(
            sampled * scale_tensor,
            self.axis_descr,
        )
