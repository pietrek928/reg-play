import numpy as np


class SimpleIntegrator:
    def __init__(self, coeff, start_state):
        self._coeff = coeff
        self._state = start_state

    def do_step(self, dx):
        self._state = np.concatenate((self._state[1:, :], dx), axis=0)
        return self._state * self._coeff
