from functools import partial
from itertools import chain
from typing import Tuple, Dict, Any, List

import numpy as np
from torch import tensor, stack, float32, save, load, Tensor, from_numpy
from torch.nn.functional import smooth_l1_loss

from model import Values, ValuesRec
from named_tensor import NamedTensor


def save_models_state(fname, models: Dict):
    state = {
        k: v.state_dict()
        for k, v in models.items()
    }
    save(state, fname)


def load_models_state(fname, models: Dict):
    state = load(fname)
    for n, m in models.items():
        if n in state:
            m.load_state_dict(state[n])


def values_to(
        values: Dict[str, Any], dtype=float32, device=None
) -> Values:
    return {
        k: (
            v if isinstance(v, Tensor) else from_numpy(v)
        ).to(dtype=dtype, device=device)
        for k, v in values.items()
    }


def detach_values(values: ValuesRec):
    r = {}
    for k, v in values.items():
        if isinstance(v, dict):
            r[k] = detach_values(v)
        else:
            r[k] = v.detach()
    return r


def stack_values(values: List[ValuesRec], axis: int = 0):
    r = {}
    for k in values[0].keys():
        vs = [v[k] for v in values]
        if isinstance(vs[0], dict):
            r[k] = stack_values(vs, axis=axis)
        else:
            r[k] = stack(vs, dim=axis)
    return r


def merge_values(*values: ValuesRec):
    r = {}
    all_keys = set(
        chain.from_iterable(v.keys() for v in values)
    )
    for k in all_keys:
        key_values = [v[k] for v in values if v.get(k) is not None]
        if len(key_values) == 0:
            continue

        if all(isinstance(v, dict) for v in key_values):
            r[k] = merge_values(*key_values)
        elif all(not isinstance(v, dict) for v in key_values):
            r[k] = key_values[-1]
        else:
            raise ValueError(f'Cannot merge values for key {k}')
    return r


def get_range(values: ValuesRec, start: int, end: int):
    r = {}
    for k, v in values.items():
        if isinstance(v, dict):
            r[k] = get_range(v, start, end)
        else:
            r[k] = v[start:end]
    return r


def get_at_pos(values: ValuesRec, pos: int):
    r = {}
    for k, v in values.items():
        if isinstance(v, dict):
            r[k] = get_at_pos(v, pos)
        else:
            r[k] = v[pos]
    return r


def set_range(values: ValuesRec, start: int, new_values: Values):
    for k, nv in new_values.items():
        v = values[k]
        if isinstance(nv, dict):
            set_range(v, start, nv)
        else:
            v[start:start + nv.shape[0]] = nv


def stack_values_np(
        values: List[Dict[str, np.ndarray]],
        axis: int = 0, append_dim=False
):
    return {
        k: np.stack([
            v[k] if not append_dim else v[k][..., np.newaxis]
            for v in values
        ], axis=axis)
        for k in values[0]
    }


def default_device():
    from torch import cuda
    if cuda.device_count():
        return 'cuda:0'
    return 'cpu'


def compute_dt(time: tensor):
    dt = time - time.roll(shifts=-1, dims=0)
    dt[-1, ...] = dt[:-1, ...].mean(dim=0)
    return dt


def prepare_dataset(items: Tuple[NamedTensor]):
    joined = stack([i.tensor for i in items], dim=1).to(dtype=float32)
    return NamedTensor(joined, axis_descr=items[0].axis_descr)


def plot_dataset(dataset: NamedTensor):
    import matplotlib.pyplot as plt
    for k, v in dataset.values().items():
        plt.plot(v, label=k)
    plt.legend()
    plt.show()


def sample_many(
        dataset: NamedTensor,
        count: int, step: int,
        mean_axis: Tuple[str, ...] = ()
):
    n = dataset.tensor.shape[0]
    l = n // (count // 2)
    for start in range(0, n - l, l // 2):
        yield dataset.sample_sums(start, start + l, step, mean_axis=mean_axis)


def compute_means(dataset: NamedTensor):
    return {
        k: float(v.mean())
        for k, v in dataset.values().items()
    }


def l1_loss_func(means, outputs, gt):
    return sum(
        (outputs[k] - gt[k]).abs().mean() * (1. / abs(means[k]))
        for k in outputs
    )


def l1_loss_normalized(dataset: NamedTensor):
    means = compute_means(dataset)
    return partial(l1_loss_func, means)


def l2_loss_func(means, outputs, gt):
    return sum(
        smooth_l1_loss(outputs[k] * (1. / abs(means[k])), gt[k] * (1. / abs(means[k])))
        for k in outputs
    )


def l2_loss_normalized(dataset: NamedTensor):
    means = compute_means(dataset)
    return partial(l2_loss_func, means)
