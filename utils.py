from typing import Tuple

from torch import tensor, stack, float32

from named_tensor import NamedTensor


def compute_dt(time: tensor):
    dt = time - time.roll(shifts=-1, dims=-1)
    dt[..., -1] = dt[..., :-1].mean(dim=-1)
    return dt


def prepare_dataset(items: Tuple[NamedTensor]):
    joined = stack([i.tensor for i in items], dim=0).to(dtype=float32)
    return NamedTensor(joined, axis_descr=items[0].axis_descr)


def sample_many(
        dataset: NamedTensor,
        count: int, step: int,
        mean_axis: Tuple[str, ...] = ()
):
    n = dataset.tensor.shape[0]
    l = n // (count // 2)
    for start in range(0, n - l, l // 2):
        yield dataset.sample_sums(start, start + l, step, mean_axis=mean_axis)
