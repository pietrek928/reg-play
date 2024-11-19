from collections import defaultdict
from typing import Dict, Tuple
import torch


class LFModelSimple:
    def __init__(self, params):
        self.params = params

    def compute_dx(self, data):
        p = self.params

        u1 = data['u1']
        u2 = data['u2']
        v = data['v']
        a = data['a']
        w = data['w']
        v1 = v + w*p[:, 0]
        v2 = v - w*p[:, 0]

        am1 = p[:, 1]*u1 + p[:, 2]*v1
        am1 = torch.where(am1 > 0, (am1 - p[:, 4]).max(0), (am1 + p[:, 4]).min(0))

        am2 = p[:, 1]*u2 + p[:, 2]*v2
        am2 = torch.where(am2 > 0, (am2 - p[:, 4]).max(0), (am2 + p[:, 4]).min(0))

        vp = am1 + am2
        wp = p[:, 3] * (am1 - am2)
        return {
            'v': vp,
            'w': wp,
            'a': w,
            'x': v * a.cos(),
            'y': v * a.sin(),
        }


# transform angles tensor to drot
def compute_drot(angles):
    drot = angles[..., 1:] - angles[..., :-1]
    drot = torch.where(drot < -torch.pi, drot + 2*torch.pi, drot)
    drot = torch.where(drot > torch.pi, drot - 2*torch.pi, drot)
    return drot


def accum_drot(drot):
    rot = torch.cumsum(drot, dim=-1)
    init_angle = torch.zeros_like(rot[:, :1])
    return torch.cat([init_angle, rot], dim=-1)


# dims: (sample, time)
def score_sim_fit(
        step_func, score_func, state: Dict[torch.Tensor],
        samples: Dict[torch.Tensor], out_keys: Tuple[str, ...]
):
    scores = []
    n = tuple(samples.values()).shape[1]
    for it in range(n):
        in_data = {
            k: v[:, it]
            for k, v in samples.items()
            if k not in out_keys
        }
        out_dx = step_func(in_data)
        state = {
            k: v + out_dx[k]
            for k, v in state.keys()
        }
        out_gt = {
            k: v[:, it]
            for k, v in samples.items()
            if k in out_keys
        }
        scores.append(score_func(state, out_gt))


# dims: (sample, time)
def run_model_sim(
        step_func, state: Dict[torch.Tensor],
        samples: Dict[torch.Tensor], out_keys: Tuple[str, ...]
):
    states = {
        k: [v] for k, v in state.items()
    }
    n = tuple(samples.values()).shape[1]
    for it in range(n):
        in_data = {
            k: v[:, it]
            for k, v in samples.items()
            if k not in out_keys
        }
        out_dx = step_func(in_data)
        state = {
            k: v + out_dx[k]
            for k, v in state.keys()
        }
        for k, v in state.items():
            states[k].append(v)
    return {
        k: torch.stack(vs, dim=-1)
        for k, vs in states.items()
    }
