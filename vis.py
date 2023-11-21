from collections import defaultdict
from typing import Callable, Optional, Tuple

import numpy as np
from model import ValuesRec

from obj import Block


def plot_controller_sym(block: Block, controller: Block, fin: Callable, t: float, dt: float):
    n = int(t / dt)
    graphs = defaultdict(list)
    for i in range(n):
        controller_out = controller.output
        block_out = block.output
        state = {**dict(block.state), **dict(controller.state)}
        for k, v in state.items():
            graphs[k].append(v)

        tt = t * i / n
        ctrl = fin(tt)
        controller.process(block_out, ctrl, dt=dt)
        block.process(controller_out, dt=dt)

    state = {**dict(block.state), **dict(controller.state)}
    for k, v in state.items():
        graphs[k].append(v)

    import matplotlib.pyplot as plt

    taxis = np.arange(n + 1) * t / n
    for k in sorted(graphs):
        plt.plot(taxis, graphs[k], label=k)

    plt.legend()
    plt.show()


def plot_time_graphs(data: ValuesRec, draw_keys: Tuple[str, ...], time_key: Optional[str] = None):
    from matplotlib import pyplot as plt

    if time_key is not None:
        time = data[time_key]
    else:
        first_data = data[draw_keys[0]]
        time = np.arange(first_data.shape[0])

    fig, axs = plt.subplots(len(draw_keys), sharex=True, figsize=(10, 6))

    for ax, key in zip(axs, draw_keys):
        ax.plot(time, data[key], label=key)
        ax.set_ylabel(key)
        ax.legend(loc='upper left')
        ax.grid(True)

    plt.tight_layout()
    # plt.show()
    
    # close graph on key press
    plt.waitforbuttonpress()
    plt.close(fig)
