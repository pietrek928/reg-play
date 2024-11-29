from gc import collect
from sys import argv
from typing import Dict, Tuple
from click import command, option
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
        am2 = torch.where(am2 > 0, (am2 - p[:, 5]).clamp(min=0), (am2 + p[:, 5]).clamp(max=0))

        dv = am1 + am2
        dw = p[:, 6] * (am1 - am2)
        return {
            'v': dv,
            'w': dw,
            'a': w,
            'x': v * a.cos(),
            'y': v * a.sin(),
            'dl': v1,
            'dr': v2
        }


def _unflatten_matrix(data: torch.Tensor, x, y):
    s = data.shape
    return data.view(*s[:-1], x, y)


class LFReg:
    def __init__(self, params):
        self.params = params

    def compute_step(self, data):
        A = _unflatten_matrix(self.params[..., :16], 4, 4)
        B = _unflatten_matrix(self.params[..., 16:32], 4, 4)
        C = _unflatten_matrix(self.params[..., 32:40], 2, 4)
        D = _unflatten_matrix(self.params[..., 40:48], 2, 4)

        state = data['state']
        dt = data['dt']
        drot_l = data['drot_l']
        drot_r = data['drot_r']
        line_pos = data['line_pos']
        line_pos_last = data['line_pos_last']

        u = torch.stack([
            drot_l,
            # drot_l / dt,
            drot_r,
            # drot_r / dt,
            line_pos,
            (line_pos - line_pos_last) / dt,
        ], dim=-1)

        state = state + (A * state + B * u) * dt
        y = C * state + D * u

        return {
            'u_l': y[..., 0],
            'u_r': y[..., 1],
        }, {
            'line_pos_last': line_pos,
            'state': state,
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


def segment_distance(S, P):
    # Unpack segment endpoints
    A, B = S[:-1, :], S[1:, :]

    # Vector from segment start to end
    seg_vec = B - A

    # Vector from segment start to points
    point_vec = P.unsqueeze(1) - A.unsqueeze(0)  # Shape (N, M, 2)

    # Project point_vec onto seg_vec
    seg_len_squared = (seg_vec ** 2).sum(dim=1)  # Shape (M,)
    t = (point_vec * seg_vec.unsqueeze(0)).sum(dim=2) / seg_len_squared.unsqueeze(0)  # Shape (N, M)

    # Clamp t to the range [0, 1]
    t = torch.clamp(t, 0, 1)

    # Find the closest point on the segment to the point
    closest_point = A.unsqueeze(0) + t.unsqueeze(2) * seg_vec.unsqueeze(0)  # Shape (N, M, 2)

    # Compute the distance from the point to the closest point on the segment
    distance = torch.norm(P.unsqueeze(1) - closest_point, dim=2).amin(dim=1)  # Shape (N, M)

    return distance


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
def score_reg_fit(
        step_func, reg_func, score_func,
        state: Dict[str, torch.Tensor], reg_state: Dict[str, torch.Tensor],
        samples: Dict[str, torch.Tensor], out_keys: Tuple[str, ...]
):
    state = {
        k: v.clone() for k, v in state.items()
    }
    reg_state = {
        k: v.clone() for k, v in reg_state.items()
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
        reg_out, reg_state = reg_func(in_data | reg_state)
        out_dx = step_func(in_data | reg_out | state)
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
    drot_l = compute_drot(angle_l) * .1  # gear is 10x
    rot_l = accum_drot(drot_l)

    angle_r = torch.from_numpy(df['angle_r'].to_numpy()).to(
        dtype=torch.float64, device=device
    ).unsqueeze(0)
    drot_r = -compute_drot(angle_r) * .1  # gear is 10x
    rot_r = accum_drot(drot_r)

    # Wheel R=2cm, wheels distance d=15cm
    d_l = drot_l * .02
    d_r = drot_r * .02
    ddist = torch.cat((torch.zeros_like(d_l[..., :1]), (d_l + d_r) * .5), dim=-1)
    drot = (d_l - d_r) * (1 / .15)
    rot = accum_drot(drot)
    # print(tuple(map(float, drot_l[0])))
    x = torch.cumsum(ddist * rot.cos(), -1)
    y = torch.cumsum(ddist * rot.sin(), -1)

    return dict(
        dt=dt,
        t=t,
        # drot_l=drot_l,
        rot_l=rot_l,
        dl=accum_drot(d_l),
        u_l=torch.from_numpy(df['motor_l_u'].to_numpy()).to(
            dtype=torch.float64, device=device
        ).unsqueeze(0),
        # drot_r=drot_r,
        rot_r=rot_r,
        dr=accum_drot(d_r),
        u_r=torch.from_numpy(df['motor_r_u'].to_numpy()).to(
            dtype=torch.float64, device=device
        ).unsqueeze(0),
        x=x,
        y=y
    )


def plt_on_key(event):
    import matplotlib.pyplot as plt
    if event.key == 'escape' or event.key == ' ':
        plt.close('all')


def plot_lf_case(case_num, data, data_sim=None):
    import matplotlib.pyplot as plt

    t = data['t'][case_num]

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.canvas.mpl_connect('key_press_event', plt_on_key)

    # Plot wheel left rotations
    axs[0, 0].plot(t, data['dl'][case_num], label='Wheel left distance', color='b')
    axs[0, 0].set_title('Wheel left distance')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Distance')

    # Plot wheel right rotation
    axs[0, 1].plot(t, data['dr'][case_num], label='Wheel right distance', color='g')
    axs[0, 1].set_title('Wheel right distance')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Distance')

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
        axs[0, 1].plot(t, data_sim['dr'][case_num], label='Simulated right distance', color='orange', linestyle='--')

    axs[0, 0].legend()
    axs[0, 1].legend()

    plt.tight_layout()

    # Plot robot trace
    fig = plt.figure(figsize=(7, 7))
    fig.canvas.mpl_connect('key_press_event', plt_on_key)
    plt.plot(data['x'][case_num], data['y'][case_num], label='Robot trace', color='b')
    plt.title('Robot Trace')
    plt.xlabel('X position')
    plt.ylabel('Y position')

    if data_sim:
        plt.plot(data_sim['x'][case_num], data_sim['y'][case_num], label='Simulated trace', color='orange', linestyle='--')

    plt.legend()
    plt.axis('equal')  # This ensures the aspect ratio is 1:1

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def score_lf_fit(state_gt, state_sim):
    return (
        ((state_gt['dl'] - state_sim['dl']) ** 2.
        + (state_gt['dr'] - state_sim['dr']) ** 2.) * .001
        + (state_gt['x'] - state_sim['x']) ** 2.
        + (state_gt['y'] - state_sim['y']) ** 2.
    )


def validate_lf_params(params):
    return torch.all(params > 0, dim=-1)


def score_points(data, params):
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
    pts_diff = (pts_new - pts_old) / pts_old.abs()
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


def search_step(data, old_pts, old_scores, old_moments, n, a, b):
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

    new_scores = score_points(data, new_pts)
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


@command()
def test_model():
    data = prepare_lf_data('/home/pietrek/Downloads/mcal.xlsx')
    vis_params(data, [0.06690421842070352, 1.6685777799435322, 1.340981065985289, 5.751083839594803, 0.014004082325513676, 0.0026276411826163344, 5.140782116967004])
    # plot_lf_case(0, data)


@command()
@option('--device', default='cpu')
def fit_model(device):
    data = prepare_lf_data('/home/pietrek/Downloads/mcal.xlsx', device=device)

    params = torch.tensor([
        [0.06690421842070352, 1.6685777799435322, 1.340981065985289, 5.751083839594803, 0.014004082325513676, 0.0026276411826163344, 5.140782116967004],
        # [.5, .1, 100, 50, 80, .1],
        [0, 0, 0, 0, 0, 0, 0]
    ], dtype=torch.float64, device=device) + .01
    scores = torch.tensor([1e9, 1e9], dtype=torch.float64, device=device)
    moments = torch.zeros_like(params)

    a = 1
    b = 5
    for it in range(100):
        params, scores, moments = search_step(data, params, scores, moments, 5000, a, b)
        # a *= .97
        # print(params)
        print(it, a, float(scores[0]), tuple(map(float, params[0])))
    print(params.shape)
    print(tuple(map(float, params[0])))


if __name__ == '__main__':
    func_name = argv[1]
    argv[:] = argv[0:1] + argv[2:]
    globals()[func_name]()
