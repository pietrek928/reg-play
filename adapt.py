from typing import Callable, Type

from torch.optim import RMSprop, Adam

from grad import sum_vals, GradVar
from model import Model, validate_values, init_zero_params, init_torch_models, get_parameters
from named_tensor import NamedTensor
from obj import Block, Tm


def set_optimizer_params(optimizer, new_params):
    for param_group in optimizer.param_groups:
        param_group.update(new_params)


def get_step_params(lr):
    return dict(
        lr=lr,
        weight_decay=lr * 5e-3,
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
        model: Type[Model], dataset: NamedTensor, loss_func,
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
    torch_models = init_torch_models(model.get_torch_models(), device=device)
    start_states = init_zero_params(model.get_state(), base_shape=(dataset_shape_prefix[1],), device=device)

    model_lr = 2e-2
    start_states_lr = 2e-3

    optimizer_params = Adam(
        get_parameters(params, *torch_models.values()),
        **get_step_params(model_lr)
    )
    optimizer_start_states = RMSprop(
        get_parameters(start_states),
        **get_step_params(start_states_lr)
    )

    step = 0
    while True:  # Training loop
        optimizer_params.zero_grad()
        optimizer_start_states.zero_grad()
        state = start_states
        loss = None

        for step in range(steps_count):  # Model simulation loop
            dataset_step = {
                k: v[step] for k, v in dataset.items()
            }
            state, outputs = model.compute_step(params, torch_models, state, dataset_step)
            # state = detach_values(state)  # ???????????
            # scale_values_grad(state, 1e-1)
            loss_step = loss_func(outputs, dataset_step)
            if step:
                loss += loss_step
                if not (float(loss) < target_loss):
                    break
            else:
                loss = loss_step

        # loss += (1. - sum(
        #     v.abs().mean(dim=0).sum() for v in start_states.values()
        # )).abs() * .004
        # loss += sum(
        #     (v - v.roll(1, dims=0)).abs().mean(dim=0).sum() for v in start_states.values()
        # ).abs() * .004

        loss.backward()
        loss = float(loss)
        optimizer_params.step()
        optimizer_start_states.step()
        print(f'{step + 1}/{steps_count} loss={loss}')
        lr_scale = loss ** .5  # / (step + 1)

        set_optimizer_params(optimizer_params, get_step_params(model_lr * lr_scale))
        set_optimizer_params(optimizer_start_states, get_step_params(start_states_lr * lr_scale))

        if loss < target_loss and step + 1 == steps_count:
            break

    return tuple(get_parameters(params, *torch_models))
