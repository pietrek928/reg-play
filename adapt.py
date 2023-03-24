from typing import Callable

from torch import Tensor, tensor
from torch.optim import Adam

from grad import sum_vals, GradVar
from model import Model, Values, validate_values, init_zero_params
from obj import Block, Tm


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


def fit_model(model: Model, dataset: Values, loss_func, loss_limit: float):
    inputs_descr = model.get_inputs()
    outputs_descr = model.get_outputs()

    dataset_shape_prefix = validate_values(inputs_descr | outputs_descr, dataset)
    if len(dataset_shape_prefix) != 2:
        raise ValueError(f'Invalid dataset shape prefix {dataset_shape_prefix}')

    steps_count = dataset_shape_prefix[0]

    # Values to be adjusted by optimization
    params = init_zero_params(model.get_params())
    start_states = init_zero_params(model.get_state(), base_shape=dataset_shape_prefix)

    optimizer = Adam((params, start_states), lr=1e-6, weight_decay=1e-9)

    step = 0
    loss: Tensor = tensor(1e9)
    while float(loss) > 1e-3:  # Training loop

        optimizer.zero_grad()
        state = start_states
        loss: Tensor = tensor(0)

        for step in range(steps_count):  # Model simulation loop
            dataset_step = {
                k: v[step] for k, v in dataset.items()
            }
            state, outputs = model.compute_step(params, state, dataset_step)
            loss_step = loss_func(outputs, dataset)
            if step:
                if loss_step > loss_limit:
                    break
                loss += loss_step
            else:
                loss = loss_step

        loss.backward()
        optimizer.step()
        print(float(loss), step)

    return params
