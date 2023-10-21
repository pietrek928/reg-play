from functools import partial
from typing import Tuple, Dict, Any, List

import numpy as np
from torch import tensor, stack, float32, save, load, Tensor, from_numpy
from torch.nn.functional import smooth_l1_loss

from model import Values
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


def stack_values(values: List[Values], axis: int = 0):
    return {
        k: stack([v[k] for v in values], dim=axis)
        for k in values[0]
    }


def stack_values_np(values: List[Dict[str, np.ndarray]], axis: int = 0):
    return {
        k: np.stack([v[k] for v in values], axis=axis)
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
