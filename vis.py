from collections import defaultdict
from typing import Callable

import numpy as np

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
