from random import randint, uniform
from typing import Callable, Dict, Any

from torch.optim import Adamax, SGD, Adam, AdamW, ASGD
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from grad import sum_vals, GradVar
from model import ValuesRec, validate_values, init_zero_params, init_torch_models, get_parameters, SymBlock, init_zero_values
from named_tensor import NamedTensor
from obj import Block, Tm
from utils import detach_values, get_at_pos, get_range, set_range, values_to, stack_values, merge_values


def set_optimizer_params(optimizer, new_params):
    for param_group in optimizer.param_groups:
        param_group.update(new_params)


def get_step_params(lr, step):
    lr /= step
    return dict(
        lr=lr,
        weight_decay=lr * 1e-3,
    )


def score_controller(block: Block, controller: Block, fin: Callable, fscore: Callable, t: float, dt: float):
    n = int(t / dt)
    scores = []
    for i in range(n):
        controller_out = controller.output
        block_out = block.output

        tt = t * i / n
        ctrl = fin(tt)
        scores.append(fscore(block.state, ctrl))
        controller.process(block_out, ctrl, dt=dt)
        block.process(controller_out, dt=dt)
    return sum_vals(scores)


def fill_with_grad(m: Tm) -> Tm:
    keys = tuple(m.schema()['properties'].keys())
    return type(m).parse_obj({
        k: GradVar(getattr(m, k))
        for k in keys
    })


def adapt_model(
        model: SymBlock, dataset: NamedTensor, loss_func,
        target_loss: float, device=None
):
    dataset = dataset.values(device=device)

    inputs_descr = model.get_inputs()
    outputs_descr = model.get_outputs()

    dataset_shape_prefix = validate_values(inputs_descr | outputs_descr, dataset)
    if len(dataset_shape_prefix) != 2:
        raise ValueError(f'Invalid dataset shape prefix {dataset_shape_prefix}')

    steps_count = dataset_shape_prefix[0]

    # Values to be adjusted by optimization
    params = init_zero_params(model.get_params(), device=device)
    torch_models = init_torch_models(model.get_torch_models(), device=device, train=True)
    # start_states = init_zero_params(model.get_state(), base_shape=(dataset_shape_prefix[1],), device=device)

    model_lr = 1e-3
    # start_states_lr = 4e-3

    # AdamW ?
    # Adamax +
    optimizer_params = Adamax(
        get_parameters(params, *torch_models.values()),
        **get_step_params(model_lr)
    )
    # optimizer_start_states = RMSprop(
    #     get_parameters(start_states),
    #     **get_step_params(start_states_lr)
    # )

    history_size = model.history_size

    # step = 0
    try:
        while True:  # Training loop
            optimizer_params.zero_grad()
            # optimizer_start_states.zero_grad()
            # state = start_states
            loss = None

            dataset_steps = tuple(
                {
                    k: v[step] for k, v in dataset.items()
                } for step in range(history_size)
            )

            for step in range(history_size - 1, steps_count):  # Model simulation loop
                dataset_steps = dataset_steps[1:] + ({
                                                         k: v[step] for k, v in dataset.items()
                                                     },)
                state, outputs = model.compute_step(params, torch_models, dataset_steps, dataset_steps[:-1])
                # state = detach_values(state)  # ???????????
                # scale_values_grad(state, 1e-1)
                if loss is None:
                    loss = loss_func(outputs, dataset_steps[-1])
                else:
                    loss += loss_func(outputs, dataset_steps[-1])
                # if step:
                #     loss += loss_step
                #     if not (float(loss) < target_loss * 5.):
                #         break
                # else:
                #     loss = loss_step

            # loss += (1. - sum(
            #     v.abs().mean(dim=0).sum() for v in start_states.values()
            # )).abs() * .004
            # loss += sum(
            #     (v - v.roll(1, dims=0)).abs().mean(dim=0).sum() for v in start_states.values()
            # ).abs() * .004

            loss.backward()
            optimizer_params.step()
            # optimizer_start_states.step()

            loss = float(loss)
            # print(f'{step + 1}/{steps_count} loss={loss}')
            print(f'loss={loss}')

            set_optimizer_params(optimizer_params, get_step_params(model_lr * loss))
            # set_optimizer_params(optimizer_start_states, get_step_params(start_states_lr * loss / (step + 1) ** 2.3))

            # if loss < target_loss and step + 1 == steps_count:
            if loss < target_loss:
                break

    except KeyboardInterrupt:
        pass

    finally:
        return tuple(get_parameters(params, *torch_models))


def run_dab_rc_sim(
        model: SymBlock, input_data: ValuesRec, model_state: ValuesRec
):
    output_history = []
    state_history = []

    steps_count = input_data[next(iter(input_data.keys()))].shape[0]
    for step in range(steps_count):
        model_inputs = get_at_pos(input_data, step)

        model_state, outputs = model.compute_step(
            merge_values(model_inputs, model_state)
        )
        state_history.append(model_state)
        output_history.append(outputs)

    return stack_values(state_history), stack_values(output_history)


def adapt_rc_dab_reg(
        model: SymBlock, dataset: Dict[str, Any], init_state, loss_func,
        target_steps_count, target_loss: float, device=None
):
    model.to(device)
    model.train()
    dataset = values_to(dataset, device=device)
    init_state = values_to(init_state, device=device)

    inputs_descr = model.get_inputs()
    outputs_descr = model.get_outputs()

    dataset_shape_prefix = validate_values(inputs_descr | outputs_descr, dataset)
    if len(dataset_shape_prefix) != 2:
        raise ValueError(f'Invalid dataset shape prefix {dataset_shape_prefix}')
    
    model_states = init_zero_values(
        model.get_state(), base_shape=dataset_shape_prefix, device=device
    )
    model_states = merge_values(model_states, init_state)

    steps_count = 2

    model_lr = 1e-5

    # AdamW ?
    # Adamax +
    optimizer_params = ASGD(model.parameters(), **get_step_params(model_lr, steps_count))

    start_pos = 0

    last_max_loss = 0.
    last_mean_loss = 0.
    max_stuck_count = 0
    while True:  # Training loop
        optimizer_params.zero_grad()

        start = 1024
        # start = randint(0, dataset_shape_prefix[0] - steps_count - 1)
        # start = start_pos
        end = start + steps_count
        model_inputs = get_range(dataset, start, end)

        new_states, outputs = run_dab_rc_sim(
            # FIXME: why detach needed ???
            model, model_inputs, detach_values(get_at_pos(model_states, start))
        )

        loss = loss_func(outputs, model_inputs)

        loss_mean = loss.mean()
        loss_max = loss.max()

        (loss_max * uniform(.01, .05) + loss_mean).backward()
        loss_mean = float(loss_mean)
        loss_max = float(loss_max)
        if abs(loss_max - last_max_loss) < 1e-6:
            max_stuck_count += 1
        else:
            max_stuck_count = 0
        bad_loss = not (
            abs(loss_mean) < 1e4
            and abs(loss_max) < 1e7
            and max_stuck_count < 15
        )
        last_max_loss = loss_max
        last_mean_loss = loss_mean

        # clip_grad_norm_(model.parameters(), clip_value=1.0)
        # clip_grad_value_(model.parameters(), clip_value=1.0)
        if not bad_loss or steps_count < 5:
            # set_range(model_states, start+1, detach_values(new_states))
            optimizer_params.step()
            set_optimizer_params(optimizer_params, get_step_params(model_lr, steps_count))

        print(f'{steps_count + 1}/{dataset_shape_prefix[0]} stert_pos={start} loss_mean={loss_mean} loss_max={loss_max} bad_loss={bad_loss}')

        # start_pos += steps_count
        # if start_pos >= dataset_shape_prefix[0] - steps_count - 1:
        #     start_pos = 0

        if loss_mean < target_loss and not bad_loss:
            if steps_count < target_steps_count:
                # steps_count += randint(0, 10) // 8
                steps_count += 1
            # else:
            #     break
        elif bad_loss:
            if steps_count > 2:
                steps_count -= randint(1, 2)
            elif steps_count > 1:
                steps_count -= 1
            # else:
                # break
