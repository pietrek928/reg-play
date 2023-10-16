from typing import Callable, Type

from torch.optim import Adamax

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


def compute_dab_rc_step(dab_model: Model, reg_model: Model, torch_models, inputs, outputs):
    vout = outputs[-1]['VOUT']
    dab_state, dab_outputs = dab_model.compute_step(
        {}, torch_models,
        ({
             'VIN': inputs[-1]['VIN'],
             'VOUT': vout,
             'f': inputs[-1]['f'],
             'fi': inputs[-1]['fi'],
         },),
        outputs
    )

    iin = dab_outputs['iin']
    iout = dab_outputs['iout']
    e = vout - inputs[-1]['VIN_set']

    reg_state, reg_outputs = reg_model.compute_step(
        {}, torch_models, inputs, outputs
    )

    ic = iout - vout / inputs[-1]['R']
    vout += ic / inputs[-1]['C'] * inputs[-1]['dt']

    return reg_state, reg_outputs


def run_model_sim(model: Model, torch_models, input_data, steps_count):
    history_size = model.history_size

    inputs_descr = model.get_inputs()
    outputs_descr = model.get_outputs()

    input_dataset_shape_prefix = validate_values(inputs_descr, input_data)
    # output_dataset_shape_prefix = validate_values(outputs_descr, start_outputs)
    # if len(dataset_shape_prefix) != 2:
    #     raise ValueError(f'Invalid dataset shape prefix {dataset_shape_prefix}')

    model_inputs = tuple(
        {
            k: v[step] for k, v in input_data.items()
        } for step in range(history_size)
    )

    model_states = tuple(
        {
            init_zero_params(model.get_state())
        } for _ in range(history_size)
    )
    model_outputs = tuple(
        {
            init_zero_params(outputs_descr)
        } for _ in range(history_size - 1)
    )

    for step in range(steps_count):
        model_inputs = model_inputs[1:] + (
            {
                k: v[step] for k, v in input_data.items()
            },
        )

        state, outputs = model.compute_step({}, torch_models, model_inputs, model_outputs)

        model_states = model_states[1:] + (state,)
        model_outputs = model_outputs[1:] + (outputs,)
