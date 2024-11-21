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
    init_angle = torch.zeros_like(rot[..., :1])
    return torch.cat([init_angle, rot], dim=-1)


# dims: (sample, time)
def score_sim_fit(
        step_func, score_func, state: Dict[str, torch.Tensor],
        samples: Dict[str, torch.Tensor], out_keys: Tuple[str, ...]
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
        step_func, state: Dict[str, torch.Tensor],
        samples: Dict[str, torch.Tensor], out_keys: Tuple[str, ...]
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


def prepare_lf_data(fname):
    import pandas as pd

    df = pd.read_excel(fname)
    dt = torch.from_numpy(df['dt'].to_numpy()).to(torch.float64)
    t = torch.cumsum(dt, 0)

    angle_l = torch.from_numpy(df['angle_l'].to_numpy()).to(torch.float64)
    drot_l = compute_drot(angle_l)
    rot_l = accum_drot(drot_l)

    angle_r = torch.from_numpy(df['angle_r'].to_numpy()).to(torch.float64)
    drot_r = -compute_drot(angle_r)
    rot_r = accum_drot(drot_r)

    return dict(
        dt=dt,
        t=t,
        drot_l=drot_l,
        rot_l=rot_l,
        u_l=torch.from_numpy(df['motor_l_u'].to_numpy()).to(torch.float64),
        drot_r=drot_r,
        rot_r=rot_r,
        u_r=torch.from_numpy(df['motor_r_u'].to_numpy()).to(torch.float64),
    )


def plot_case(data, data_sim=None):
    import matplotlib.pyplot as plt

    t = data['t']

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Plot wheel left rotations
    axs[0, 0].plot(t, data['rot_l'], label='Wheel left rotation', color='b')
    axs[0, 0].set_title('Wheel left rotation')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Rotation')
    axs[0, 0].legend()

    # Plot wheel right rotation
    axs[0, 1].plot(t, data['rot_r'], label='Wheel right rotation', color='g')
    axs[0, 1].set_title('Wheel right rotation')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Rotation')
    axs[0, 1].legend()

    # Plot voltage 1
    axs[1, 0].plot(t, data['u_l'], label='Voltage left', color='r')
    axs[1, 0].set_title('Voltage 1')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Voltage (V)')
    axs[1, 0].legend()

    # Plot voltage 2
    axs[1, 1].plot(t, data['u_r'], label='Voltage right', color='m')
    axs[1, 1].set_title('Voltage 2')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Voltage (V)')
    axs[1, 1].legend()

    if data_sim is not None:
        axs[0, 0].plot(t, data_sim['rot_l'], label='Simulated left angle', color='orange', linestyle='--')
        axs[0, 0].legend()
        axs[0, 1].plot(t, data_sim['rot_r'], label='Simulated right angle', color='orange', linestyle='--')
        axs[0, 1].legend()

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()


data = prepare_lf_data('/home/pietrek/Downloads/mcal.xlsx')
plot_case(data, data)
