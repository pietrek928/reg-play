from collections import defaultdict
from gc import collect
from typing import Dict, Tuple
import torch


class LFModelSimple:
    def __init__(self, params):
        self.params = params

    def compute_dx(self, data):
        p = self.params

        u1 = data['u_l']
        u2 = data['u_r']
        v = data['v']
        a = data['a']
        w = data['w']
        v1 = v + w*p[:, 0]
        v2 = v - w*p[:, 0]

        am1 = p[:, 3]*(u1 - p[:, 1]*v1)
        am1 = torch.where(am1 > 0, (am1 - p[:, 4]).clamp(min=0), (am1 + p[:, 4]).clamp(max=0))

        am2 = p[:, 3]*(u2 - p[:, 2]*v2)
        am2 = torch.where(am2 > 0, (am2 - p[:, 4]).clamp(min=0), (am2 + p[:, 4]).clamp(max=0))

        dv = am1 + am2
        dw = p[:, 5] * (am1 - am2)
        return {
            'v': dv,
            'w': dw,
            'a': w,
            'x': v * a.cos(),
            'y': v * a.sin(),
            'dl': v1,
            'dr': v2
        }


# transform angles tensor to drot
def compute_drot(angles):
    drot = angles[..., 1:] - angles[..., :-1]
    drot = torch.where(drot < -torch.pi, drot + 2*torch.pi, drot)
    drot = torch.where(drot > torch.pi, drot - 2*torch.pi, drot)
    return drot


def accum_drot(drot):
    rot = torch.cumsum(drot, dim=-1)
    init_angle = torch.zeros_like(rot[..., :1])
    return torch.cat([init_angle, rot], dim=-1)


# dims: (sample, time)
def score_sim_fit(
        step_func, score_func, state: Dict[str, torch.Tensor],
        samples: Dict[str, torch.Tensor], out_keys: Tuple[str, ...]
):
    state = {
        k: v.clone() for k, v in state.items()
    }
    scores = []
    n = tuple(samples.values())[0].shape[-1]
    for it in range(n):
        if it % 100 == 0:
            collect()

        in_data = {
            k: v[:, it]
            for k, v in samples.items()
            if k not in out_keys
        }
        out_dx = step_func(in_data | state)
        # state = {
        #     k: v + out_dx[k] * in_data['dt']
        #     for k, v in state.items()
        # }
        for k in state.keys():
            state[k] += out_dx[k] * in_data['dt']
        out_gt = {
            k: v[:, it]
            for k, v in samples.items()
            if k in out_keys
        }
        scores.append(score_func(state, out_gt))
    scores = torch.stack(scores, dim=-1)
    return scores.mean(dim=-1)


# dims: (sample, time)
def run_model_sim(
        step_func, state: Dict[str, torch.Tensor],
        samples: Dict[str, torch.Tensor], out_keys: Tuple[str, ...]
):
    states = {
        k: [] for k in state.keys()
    }
    n = tuple(samples.values())[0].shape[-1]
    for it in range(n):
        in_data = {
            k: v[:, it]
            for k, v in samples.items()
            if k not in out_keys
        }
        dx = step_func(in_data | state)
        state = {
            k: v + dx[k] * in_data['dt']
            for k, v in state.items()
        }
        for k, v in state.items():
            states[k].append(v)
    return {
        k: torch.stack(vs, dim=-1)
        for k, vs in states.items()
    }


def prepare_lf_data(fname, device='cpu'):
    import pandas as pd

    df = pd.read_excel(fname)
    dt = torch.from_numpy(df['dt'].to_numpy()).to(
        dtype=torch.float64, device=device
    ).unsqueeze(0)
    t = torch.cumsum(dt, -1)

    angle_l = torch.from_numpy(df['angle_l'].to_numpy()).to(
        dtype=torch.float64, device=device
    ).unsqueeze(0)
    drot_l = compute_drot(angle_l)
    rot_l = accum_drot(drot_l)

    angle_r = torch.from_numpy(df['angle_r'].to_numpy()).to(
        dtype=torch.float64, device=device
    ).unsqueeze(0)
    drot_r = -compute_drot(angle_r)
    rot_r = accum_drot(drot_r)

    return dict(
        dt=dt,
        t=t,
        # drot_l=drot_l,
        rot_l=rot_l,
        dl=rot_l * (.02 * torch.pi),
        u_l=torch.from_numpy(df['motor_l_u'].to_numpy()).to(
            dtype=torch.float64, device=device
        ).unsqueeze(0),
        # drot_r=drot_r,
        rot_r=rot_r,
        dr=rot_r * (.02 * torch.pi),
        u_r=torch.from_numpy(df['motor_r_u'].to_numpy()).to(
            dtype=torch.float64, device=device
        ).unsqueeze(0),
    )


def plot_lf_case(case_num, data, data_sim=None):
    import matplotlib.pyplot as plt

    t = data['t'][case_num]

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Plot wheel left rotations
    axs[0, 0].plot(t, data['dl'][case_num], label='Wheel left distance', color='b')
    axs[0, 0].set_title('Wheel left distance')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Distance')
    axs[0, 0].legend()

    # Plot wheel right rotation
    axs[0, 1].plot(t, data['dr'][case_num], label='Wheel right distance', color='g')
    axs[0, 1].set_title('Wheel right distance')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Distance')
    axs[0, 1].legend()

    # Plot voltage 1
    axs[1, 0].plot(t, data['u_l'][case_num], label='Voltage left', color='r')
    axs[1, 0].set_title('Voltage 1')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Voltage (V)')
    axs[1, 0].legend()

    # Plot voltage 2
    axs[1, 1].plot(t, data['u_r'][case_num], label='Voltage right', color='m')
    axs[1, 1].set_title('Voltage 2')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Voltage (V)')
    axs[1, 1].legend()

    if data_sim is not None:
        axs[0, 0].plot(t, data_sim['dl'][case_num], label='Simulated left distance', color='orange', linestyle='--')
        axs[0, 0].legend()
        axs[0, 1].plot(t, data_sim['dr'][case_num], label='Simulated right distance', color='orange', linestyle='--')
        axs[0, 1].legend()

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def score_lf_fit(state_gt, state_sim):
    return (
        (state_gt['dl'] - state_sim['dl']) ** 2.
        + (state_gt['dr'] - state_sim['dr']) ** 2.
    )


def validate_lf_params(params):
    return torch.all(params > 0, dim=-1)


def score_points(params):
    model = LFModelSimple(params)
    z = torch.zeros_like(params[..., 0])
    sim_scores = score_sim_fit(
        model.compute_dx, score_lf_fit, dict(v=z, w=z, a=z, x=z, y=z, dl=z, dr=z),
        data, ('x', 'y', 'a', 'dl', 'dr')
    )
    return sim_scores + params.abs().mean(dim=-1) * .01


# TODO: better interpolation
def add_randoms(pts, a):
    v = pts.std(dim=0) + pts.abs().mean(dim=0) * .25
    r = torch.rand_like(pts) - 0.5
    return pts + r * (v * a)


def update_momentum(m, a, pts_old, scores_old, pts_new, scores_new):
    score_diff = scores_new - scores_old
    pts_diff = pts_new - pts_old
    return m + pts_diff * score_diff.unsqueeze(-1) * a


# TODO: better selection
# first value is score
def select_min_indexes(scores, n):
    # remove invalid values first
    indexes = torch.arange(scores.shape[0], dtype=torch.int32, device=scores.device)
    finite_mask = torch.isfinite(scores)
    scores = scores[finite_mask]
    indexes = indexes[finite_mask]

    pts_order = scores.sort().indices
    indexes = indexes[pts_order]
    selected = indexes[:n//2]
    rest = indexes[n//2:]
    return torch.cat([selected, rest[torch.randperm(rest.shape[0], device=scores.device)[:n//2]]], dim=0)


def search_step(old_pts, old_scores, old_moments, n, a, b):
    new_pts = torch.cat([
        add_randoms(old_pts, a),
        old_pts + old_moments * b,
        add_randoms(old_pts, a*.3),
        old_pts + old_moments * b * .3,
        add_randoms(old_pts, a*.1),
        old_pts + old_moments * b * .1,
        add_randoms(old_pts, a*.03),
        old_pts + old_moments * b * .03,
        add_randoms(old_pts, a*.01),
        old_pts + old_moments * b * .01,
        add_randoms(old_pts, a*.003),
        old_pts + old_moments * b * .003,
        add_randoms(old_pts, a*.001),
        old_pts + old_moments * b * .001,
    ], dim=0)
    prev_pts = torch.cat([old_pts] * 14, dim=0)
    prev_scores = torch.cat([old_scores] * 14, dim=0)
    prev_moments = torch.cat([old_moments] * 14, dim=0)

    valid_indexes = validate_lf_params(new_pts)
    new_pts = new_pts[valid_indexes]
    prev_pts = prev_pts[valid_indexes]
    prev_scores = prev_scores[valid_indexes]
    prev_moments = prev_moments[valid_indexes]

    new_scores = score_points(new_pts)
    new_moments = update_momentum(prev_moments, .2, prev_pts, prev_scores, new_pts, new_scores)

    all_pts = torch.cat([old_pts, new_pts], dim=0)
    all_scores = torch.cat([old_scores, new_scores], dim=0)
    all_moments = torch.cat([old_moments, new_moments], dim=0)

    selected_indexes = select_min_indexes(all_scores, n)
    return all_pts[selected_indexes], all_scores[selected_indexes], all_moments[selected_indexes]


def vis_params(data, params):
    params = torch.tensor([params], dtype=torch.float64)
    model = LFModelSimple(params)
    z = torch.zeros((1, ), dtype=torch.float64)
    sim_data = run_model_sim(
        model.compute_dx, dict(v=z, w=z, a=z, x=z, y=z, dl=z, dr=z),
        data, ('x', 'y', 'a', 'dl', 'dr')
    )
    # print(sim_data)
    plot_lf_case(0, data, sim_data)
    # sim_score = score_sim_fit(
    #     model.compute_dx, score_lf_fit, dict(v=z, w=z, a=z, x=z, y=z, dl=z, dr=z),
    #     data, ('x', 'y', 'a', 'dl', 'dr')
    # )
    # print(sim_score)

data = prepare_lf_data('/home/pietrek/Downloads/mcal.xlsx')
vis_params(data, [2.165925225621571, 0.0516630785650092, 0.041066706577213015, 112.50283206725963, 0.7414677216321874, 0.506700264484151])


# device='cuda:0'
# # device='cpu'
# data = prepare_lf_data('/home/pietrek/Downloads/mcal.xlsx', device=device)

# params = torch.tensor([
#     [0.9386564921990348, 0.0510397941187321, 0.04097628708440863, 149.38405005788226, 9.72265956724609, 0.2052234886348298],
#     # [.5, -.1, 100, 50, 80, .1, 1],
#     [0, 0, 0, 0, 0, 0]
# ], dtype=torch.float64, device=device) + .01
# scores = torch.tensor([1e9, 1e9], dtype=torch.float64, device=device)
# moments = torch.zeros_like(params)

# a = 1
# b = 5
# for it in range(100):
#     params, scores, moments = search_step(params, scores, moments, 5000, a, b)
#     # a *= .97
#     # print(params)
#     print(it, a, float(scores[0]), tuple(map(float, params[0])))
# print(params.shape)
# print(tuple(map(float, params[0])))
