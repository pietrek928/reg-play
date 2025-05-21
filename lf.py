from gc import collect
from sys import argv
from typing import Dict, Tuple
from click import command, option
from functools import partial

import torch
from torch.nn import functional as torch_func


def stack_tensor_dicts(dicts, dim=0):
    return {
        k: torch.stack([d[k] for d in dicts], dim=dim)
        for k in {k for d in dicts for k in d.keys()}
    }


def clone_tensor_dict(d):
    return {
        k: v.clone()
        for k, v in d.items()
    }


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

def compute_pid(e, eprev, i, dt, kp, ki, kd, ilimit=None):
    i += e * dt
    if ilimit is not None:
        i = i.clamp(-ilimit, ilimit)
    d = (e - eprev) / dt
    return kp * e + ki * i + kd * d, i


class LFRegPID:
    def __init__(self, params):
        self.params = params

    def compute_step(self, data):
        kp_turn = self.params[..., 0]
        ki_turn = self.params[..., 1]
        kd_turn = self.params[..., 2]
        ilimit_turn = self.params[..., 3]
        kadd_speed = self.params[..., 4]
        kline_speed = self.params[..., 5]
        limit_speed = self.params[..., 6]
        kp_mot = self.params[..., 7]
        ki_mot = self.params[..., 8]
        kd_mot = self.params[..., 9]
        ilimit_mot = self.params[..., 10]

        state = data['state']
        dt = data['dt']
        drot_l = data['drot_l']
        drot_r = data['drot_r']
        line_sensor = data['line_sensor']
        # print(line_sensor)

        line_sensor_last = state[..., 0]
        iturn = state[..., 1]
        ispeed = state[..., 2]
        ileft = state[..., 3]
        e_l_prev = state[..., 4]
        iright = state[..., 5]
        e_r_prev = state[..., 6]

        # print('--------------', line_sensor, line_sensor_last)
        cturn, iturn = compute_pid(line_sensor, line_sensor_last, iturn, dt, kp_turn, ki_turn, kd_turn, ilimit_turn)
        ispeed += (kadd_speed - line_sensor.abs() * kline_speed) * dt
        ispeed = ispeed.clamp(torch.zeros_like(limit_speed), limit_speed)

        vl = ispeed * (1 - cturn)
        vl_m = drot_l / dt
        e_l = vl - vl_m
        uleft, ileft = compute_pid(e_l, e_l_prev, ileft, dt, kp_mot, ki_mot, kd_mot, ilimit_mot)
        vr = ispeed * (1 + cturn)
        vr_m = drot_r / dt
        e_r = vr - vr_m
        uright, iright = compute_pid(e_r, e_r_prev, iright, dt, kp_mot, ki_mot, kd_mot, ilimit_mot)

        # print('!!!!!!!!!!!!!!', uleft, uright)

        state = torch.stack([
            line_sensor,
            iturn,
            ispeed,
            ileft,
            e_l,
            iright,
            e_r,
        ], dim=-1)

        return {
            'u_l': uleft,
            'u_r': uright,
        }, {
            'state': state,
        }


class LFRegTest:
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
        line_sensor = data['line_sensor']
        line_sensor_last = data['line_sensor_last']

        u = torch.stack([
            drot_l,
            # drot_l / dt,
            drot_r,
            # drot_r / dt,
            line_sensor,
            (line_sensor - line_sensor_last) / dt,
        ], dim=-1)

        state = state + (A * state + B * u) * dt
        y = C * state + D * u

        return {
            'u_l': y[..., 0],
            'u_r': y[..., 1],
        }, {
            'line_sensor_last': line_sensor,
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


def linestring_distance(S, P):
    """
    Distances from points to line strings
    S: Tensor of shape [N..., M, 2] where N is the number of lines and M is lines poitn count
    P: Tensor of shape [N..., 2] where N is the number of points
    """

    # Unpack segment endpoints
    A, B = S[..., :-1, :], S[..., 1:, :]

    # Vector from segment start to end
    seg_vec = B - A

    # Vector from segment start to points
    point_vec = P.unsqueeze(-2) - A  # Shape (N, M-1, 2)

    # compute cross product
    seg_len_squared = seg_vec.square().sum(dim=-1)  # Shape (N, M-1)
    t = (point_vec * seg_vec).sum(dim=-1) / seg_len_squared  # Shape (N, M-1)
 
    # Clamp t to the range [0, 1]
    t = t.clamp(0, 1).unsqueeze(-1)

    # Compute the distance from the point to the closest point on the segment
    distance = torch.norm(point_vec - t * seg_vec, dim=-1).amin(dim=-1)  # Shape (N, )

    return distance

def _compare_num_list(l1, l2, e=1e-6):
    l1 = tuple(float(v) for v in l1)
    l2 = tuple(float(v) for v in l2)
    assert len(l1) == len(l2), 'Length is not equal'
    for it, (v1, v2) in enumerate(zip(l1, l2)):
        assert abs(v1 - v2) < e, f'item {it}: {v1} != {v2}'


def test_linestring_distance():
    from shapely.geometry import LineString, Point

    l = ((0, 0), (1, 2))
    pts = ((0, 0), (1, 2), (1, 1), (0.5, 0.5), (0, 2), (2, 0), (2, 2))

    l_tensor = torch.tensor((l, ), dtype=torch.float64)
    pts_tensor = torch.tensor(pts, dtype=torch.float64)

    l_shapely = LineString(l)
    distances_shapely = [l_shapely.distance(Point(pt)) for pt in pts]

    _compare_num_list(linestring_distance(l_tensor, pts_tensor), distances_shapely)


def linestring_distance_along(S, P, D):
    """
    Sign of distance determines if its on left or right
    S: sections Tensor of shape [N..., M, 2] where N is the number of lines and M is lines poitn count
    P: points Tensor of shape [N..., 2] where N is the number of points
    D: directions Tensor of shape [N..., 2] where N is the number of points
    """

    # Unpack sections into endpoints
    P = P.unsqueeze(-2)
    D = torch_func.normalize(D.unsqueeze(-2), dim=-1)

    A, B = S[..., :-1, :], S[..., 1:, :]

    # Calculate direction vectors of the sections
    AB = B - A

    # Calculate the denominator for the intersection formula
    denominator = AB[..., 0] * D[..., 1] - AB[..., 1] * D[..., 0]

    # Check where denominator is not zero (i.e., lines are not parallel)
    non_parallel_mask = denominator.abs() > 1e-9

    # Calculate t and u for the intersection point formulas
    AP = P - A
    t = (AP[..., 0] * D[..., 1] - AP[..., 1] * D[..., 0]) / denominator

    # Calculate vector from point to intersection
    vector_to_intersection = t.unsqueeze(-1) * AB - AP

    # Calculate dot product to determine sign of distance
    dot_product = (vector_to_intersection * D).sum(dim=-1)

    # Calculate distances only for valid intersections
    valid_intersections = (0 <= t) & (t <= 1) & non_parallel_mask

    distances = vector_to_intersection.norm(dim=-1)

    # Assign negative distance if the dot product is negative
    distances[dot_product < 0] *= -1  # ????

    # Compute signed distances for valid intersections
    distances[~valid_intersections] = 1e9

    min_indices = torch.min(distances.abs(), dim=-1).indices
    return torch.gather(distances, -1, min_indices.unsqueeze(-1)).squeeze(-1)


def linestring_distance_along_shapely(linestring, pt, v):
    from shapely import LineString, Point, shortest_line

    x, y = map(float, pt.coords[0])
    pt_line = LineString(((x - v[0] * 10000, y - v[1] * 10000), (x + v[0] * 10000, y + v[1] * 10000)))
    intersection = pt_line.intersection(linestring)
    if intersection.is_empty:
        return 1e9
    intersection = shortest_line(intersection, pt).coords[0]
    distance = Point(intersection).distance(pt)
    ix, iy = intersection
    dx, dy = ix - x, iy - y
    if dx * v[0] + dy * v[1] < 0:
        distance *= -1
    return distance


def test_linestring_distance_along():
    from shapely.geometry import LineString, Point

    l = ((0, 0), (1, 2), (1, 0))
    pts = ((0, 0), (1, 2), (1, 1), (1, 1), (0.5, 0.5), (0, 2), (2, 0), (2, 2))
    d = ((1, 0), (0, 1), (-1, 0), (1, 0), (0, -1), (1, 0), (0, 1), (-1, 0))

    l_tensor = torch.tensor((l, ), dtype=torch.float64)
    pts_tensor = torch.tensor(pts, dtype=torch.float64)
    d_tensor = torch.tensor(d, dtype=torch.float64)
    distances = linestring_distance_along(l_tensor, pts_tensor, d_tensor)

    l_shapely = LineString(l)
    distance_shapely = [
        linestring_distance_along_shapely(l_shapely, Point(pt), v)
        for pt, v in zip(pts, d)
    ]

    print(tuple(float(d) for d in distances))
    print(tuple(distance_shapely))


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
    step_func, reg_func, score_func, dt,
    state: Dict[str, torch.Tensor]
):
    state = {
        k: v.clone() for k, v in state.items()
    }
    scores = []
    n = dt.shape[0]
    for it in range(n):
        if it % 100 == 0:
            collect()
        dt_val = dt[it]

        state['dt'] = dt_val
        reg_out, reg_state = reg_func(state)
        state |= reg_state | reg_out
        state |= step_func(state | reg_out)
        scores.append(score_func(state))
    scores = torch.stack(scores, dim=-1)
    return scores.mean(dim=-1), state


def sim_reg_fit(
        step_func, reg_func, dt,
        state: Dict[str, torch.Tensor]
):
    state = {
        k: v.clone() for k, v in state.items()
    }

    states = []
    n = dt.shape[0]
    for it in range(n):
        if it % 100 == 0:
            collect()
        dt_val = dt[it]

        state['dt'] = dt_val
        reg_out, reg_state = reg_func(state)
        state |= reg_state
        state |= step_func(state | reg_out)
        states.append(clone_tensor_dict(state))
    return stack_tensor_dicts(states, dim=-1)


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


def lf_line_track_step(state, lf_model, params):
    model_dx = lf_model.compute_dx(state)
    dt = state['dt']
    for k, v in model_dx.items():
        state[k] += v * dt

    lf_len = params['lf_len']
    r_wheel = params['r_wheel']
    S = params['segments']
    P = torch.stack([state['x'], state['y']], dim=-1)
    a = state['a']
    line_dist = linestring_distance(S, P)

    max_d = .06

    dx, dy = lf_len * a.cos(), lf_len * a.sin()
    P = torch.stack([state['x'] + dx, state['y'] + dy], dim=-1)
    D = torch.stack([dy, -dx], dim=-1)
    line_sensor = linestring_distance_along(S, P, D)
    # print('...............', P, line_sensor)

    # update only values in visible distance
    line_sensor = torch.where(line_sensor.abs() <= max_d, line_sensor, state['line_sensor'])
    # print(line_sensor)

    return {
        'line_dist': line_dist,
        'line_sensor': line_sensor,
        'drot_l': model_dx['dl'] * dt / r_wheel,
        'drot_r': model_dx['dr'] * dt / r_wheel,
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


def plot_lf_reg_sim(case_num, states):
    import matplotlib.pyplot as plt

    t = torch.cumsum(states['dt'], dim=-1).cpu()

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    fig.canvas.mpl_connect('key_press_event', plt_on_key)

    # plot trace by x and y
    print(states.keys())
    print(states['v'])
    print(states['line_dist'])
    axs[0].plot(states['x'][case_num].cpu(), states['y'][case_num].cpu(), label='Robot trace', color='b')
    axs[0].legend()
    # axs[0].equal()  # This ensures the aspect ratio is 1:1

    # Plot line_dist and line_sensor
    axs[1].plot(t, states['line_dist'][case_num].cpu(), label='Line distance', color='r')
    axs[1].plot(t, states['line_sensor'][case_num].cpu(), label='Line sensor', color='g')
    axs[1].legend()

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    # plt.show()
    plt.savefig(f'reg_test_{case_num}.jpg')
    

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


def score_line_trace(state_sim):
    return (
        state_sim['line_dist'] ** 2.
        + state_sim['line_sensor'] ** 2. * .01
        + state_sim['u_l'] ** 2. * .001
        + state_sim['u_r'] ** 2. * .001
    )


def lf_score_result(state):
    return (state['x'].square() + state['y'].square()).sqrt()


def score_reg_points(reg_params, lf_model, sim_params):
    reg = LFRegPID(reg_params)
    z = torch.zeros_like(reg_params[..., 0])
    n = 5000
    sim_scores, end_state = score_reg_fit(
        partial(lf_line_track_step, lf_model=lf_model, params=sim_params), reg.compute_step, score_line_trace,
        torch.ones_like(reg_params[:1, 0]).repeat(n) * 0.01,
        dict(
            v=z, w=z, a=z + 3.14/64, x=z+.01, y=z-.01, dl=z, dr=z, drot_l=z, drot_r=z, line_sensor=z,
            state=z.unsqueeze(-1).expand(-1, 7)
        ),
    )
    return sim_scores - lf_score_result(end_state) * .1 + reg_params.abs().mean(dim=-1) * .01


def sim_reg_for_points(reg_params, lf_model, sim_params):
    segments = sim_params['segments']
    ncases = reg_params.shape[0]
    nsegments = segments.shape[0]
    reg_params = reg_params.repeat(nsegments, 1)
    segments = segments.repeat(ncases, 1, 1)
    sim_params = sim_params | {'segments': segments}

    reg = LFRegPID(reg_params)
    z = torch.zeros_like(reg_params[..., 0])
    n = 5000
    return sim_reg_fit(
        partial(lf_line_track_step, lf_model=lf_model, params=sim_params), reg.compute_step,
        torch.ones_like(reg_params[:1, 0]).repeat(n) * 0.01,
        dict(
            v=z, w=z, a=z + 3.14/64, x=z+.01, y=z-.01, dl=z, dr=z, drot_l=z, drot_r=z, line_sensor=z,
            state=z.unsqueeze(-1).expand(-1, 7)
        ),
    )


def score_lf_points(params, lf_model, sim_params):
    segments = sim_params['segments']
    ncases = params.shape[0]
    nsegments = segments.shape[0]
    params = params.repeat_interleave(nsegments, dim=0)
    segments = segments.repeat(ncases, 1, 1)

    scores = score_reg_points(params, lf_model, sim_params | {'segments': segments})
    return scores.view(ncases, nsegments).mean(dim=-1)


def validate_reg_params(params):
    # return torch.all(params > 0, dim=-1)
    return torch.all(torch.isfinite(params) & (params > 0), dim=-1)


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


def search_step(
        score_pts, validate_pts,
        old_pts, old_scores, old_moments,
        hyper_params, n_pts
):
    a = hyper_params['a']
    b = hyper_params['b']
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

    valid_indexes = validate_pts(new_pts)
    new_pts = new_pts[valid_indexes]
    prev_pts = prev_pts[valid_indexes]
    prev_scores = prev_scores[valid_indexes]
    prev_moments = prev_moments[valid_indexes]

    new_scores = score_pts(new_pts)
    new_moments = update_momentum(prev_moments, .2, prev_pts, prev_scores, new_pts, new_scores)

    all_pts = torch.cat([old_pts, new_pts], dim=0)
    all_scores = torch.cat([old_scores, new_scores], dim=0)
    all_moments = torch.cat([old_moments, new_moments], dim=0)

    selected_indexes = select_min_indexes(all_scores, n_pts)
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


def vis_reg_params(lf_params, reg_params, sim_params):
    lf_model = LFModelSimple(lf_params)
    states = sim_reg_for_points(reg_params, lf_model, sim_params)
    for it in range(states['x'].shape[0]):
        plot_lf_reg_sim(it, states)


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


@command()
@option('--device', default='cpu')
def test_reg(device):
    with torch.inference_mode():
        lf_params = torch.tensor([
            [0.06690421842070352, 1.6685777799435322, 1.340981065985289, 5.751083839594803, 0.014004082325513676, 0.0026276411826163344, 5.140782116967004],
        ], dtype=torch.float64, device=device)
        reg_params = torch.tensor([
            [1.7705734654693264, 0.5041871633283294, 0.13775492738923034, 0.026215076141577785, 2.3202036414377134, 0.11290563489064577, 106.31518134382023, 0.09357056535224194, 0.0022088191456681827, 1.8032512958061575e-06, 0.10597822390838066],
            # [0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=torch.float64, device=device)
        lines = torch.tensor([
            [[0, 0], [2, .05], [20, 10]],
            [[0, 0], [1.5, -.05], [10, 20]],
            [[0, 0], [1.2, .01], [20, 20]],
            [[0, 0], [1.5, .1], [10, 20]],
            [[0, 0], [.25, -.02], [5, 20]],
        ], dtype=torch.float64, device=device)
        sim_params = dict(
            a=1, b=5, segments=lines, lf_len=0.15, r_wheel=0.01
        )

        vis_reg_params(lf_params, reg_params, sim_params)


@command()
@option('--device', default='cpu')
def fit_reg(device):
    with torch.inference_mode():
        lf_params = torch.tensor([
            [0.06690421842070352, 1.6685777799435322, 1.340981065985289, 5.751083839594803, 0.014004082325513676, 0.0026276411826163344, 5.140782116967004],
        ], dtype=torch.float64, device=device)
        lf_model = LFModelSimple(lf_params)

        lines = torch.tensor([
            [[0, 0], [2, .05], [20, 10]],
            [[0, 0], [1.5, -.05], [10, 20]],
            [[0, 0], [1.2, .01], [20, 20]],
            [[0, 0], [1.5, .1], [10, 20]],
            [[0, 0], [.25, -.02], [5, 20]],
        ], dtype=torch.float64, device=device)
        reg_params = torch.tensor([
            # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],
            [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1],
            [2.4412428180862094, 0.6429266446072572, 0.14861268954212808, 0.024600151551036786, 2.8192529322178843, 0.08238251172463415, 82.7216835657122, 0.09279805742766256, 0.002092433882416961, 5.362541601932549e-05, 0.11936847566964214],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=torch.float64, device=device)
        moments = torch.zeros_like(reg_params)
        scores = torch.tensor([1e9, 1e9, 1e9], dtype=torch.float64, device=device)

        sim_params = dict(
            a=1, b=5, segments=lines, lf_len=0.15, r_wheel=0.01
        )

        for it in range(100):
            reg_params, scores, moments = search_step(
                partial(score_lf_points, lf_model=lf_model, sim_params=sim_params), validate_reg_params,
                reg_params, scores, moments,
                sim_params, 64
            )
            # a *= .97
            # print(params)
            print(it, sim_params['a'], float(scores[0]), tuple(map(float, reg_params[0])))
        print(lf_params.shape)
        print(tuple(map(float, reg_params[0])))


if __name__ == '__main__':
    func_name = argv[1]
    argv[:] = argv[0:1] + argv[2:]
    globals()[func_name]()
