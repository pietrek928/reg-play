from typing import Callable, Dict, Any

from torch.optim import Adamax

from grad import sum_vals, GradVar
from model import validate_values, init_zero_params, init_torch_models, get_parameters, SymBlock, init_zero_values
from named_tensor import NamedTensor
from obj import Block, Tm
from utils import values_to, stack_values, merge_values


def set_optimizer_params(optimizer, new_params):
    for param_group in optimizer.param_groups:
        param_group.update(new_params)


def get_step_params(lr):
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
        model: SymBlock, input_data, steps_count, shape_prefix, device
):
    # inputs_descr = dab_model.get_inputs() | reg_model.get_inputs()
    # outputs_descr = dab_model.get_outputs() | reg_model.get_outputs()

    # input_dataset_shape_prefix = validate_values(inputs_descr, input_data)
    # output_dataset_shape_prefix = validate_values(outputs_descr, start_outputs)
    # if len(dataset_shape_prefix) != 2:
    #     raise ValueError(f'Invalid dataset shape prefix {dataset_shape_prefix}')
    output_history = []

    model_state = init_zero_values(model.get_state(), base_shape=shape_prefix[1:], device=device)

    for step in range(steps_count):
        model_inputs = {
            k: v[step] for k, v in input_data.items()
        }

        model_state, outputs = model.compute_step(
            merge_values(model_inputs, model_state)
        )
        output_history.append(outputs)

    return stack_values(output_history)


def adapt_rc_dab_reg(
        model: SymBlock, dataset: Dict[str, Any], loss_func,
        target_loss: float, device=None
):
    model.to(device)
    dataset = values_to(dataset, device=device)

    inputs_descr = model.get_inputs()
    outputs_descr = model.get_outputs()

    dataset_shape_prefix = validate_values(inputs_descr | outputs_descr, dataset)
    if len(dataset_shape_prefix) != 2:
        raise ValueError(f'Invalid dataset shape prefix {dataset_shape_prefix}')

    target_steps_count = dataset_shape_prefix[0]
    steps_count = 1

    model_lr = 1e-3

    # AdamW ?
    # Adamax +
    optimizer_params = Adamax(model.parameters())

    while True:  # Training loop
        optimizer_params.zero_grad()

        outputs = run_dab_rc_sim(model, dataset, steps_count, dataset_shape_prefix, device)

        loss = loss_func(outputs, dataset, steps_count)

        loss.backward()
        optimizer_params.step()

        loss = float(loss)
        print(f'{steps_count + 1}/{target_steps_count} loss={loss}')

        set_optimizer_params(optimizer_params, get_step_params(model_lr * loss))

        if loss < target_loss:
            if steps_count < target_steps_count:
                steps_count += 1
            else:
                break
