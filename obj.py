from typing import Type, TypeVar

from pydantic import BaseModel


class Block:
    def process(self, *inputs, dt):
        return

    @property
    def state(self):
        return {}

    @property
    def output(self):
        return ()


class DynSystem(Block):
    def __init__(
            self,
            x_class: Type[BaseModel],
            u_class: Type[BaseModel],
            y_class: Type[BaseModel],
            params_class: Type[BaseModel]
    ):
        self.x_class = x_class
        self.u_class = u_class
        self.y_class = y_class
        self.params_class = params_class

    def dx(self, x, u, p):
        assert isinstance(x, self.x_class)
        assert isinstance(u, self.u_class)
        assert isinstance(p, self.params_class)
        r = self._dx(x, u, p)
        return self.x_class.parse_obj(r)

    def y(self, x, p):
        assert isinstance(x, self.x_class)
        assert isinstance(p, self.params_class)
        r = self._y(x, p)
        return self.y_class.parse_obj(r)

    @staticmethod
    def _dx(x, u, params):
        raise NotImplemented('dx unimplemented')

    @staticmethod
    def _y(x, params):
        return x


Tm = TypeVar('Tm', bound=BaseModel)


def sum_models(m1: Tm, m2: Tm, dt: float) -> Tm:
    d1 = m1.dict()
    d2 = m2.dict()
    for k, v in d2.items():
        d1[k] += v * dt
    return type(m1).parse_obj(d1)


class SystemBlock(Block):
    def __init__(self, system: DynSystem, params: BaseModel):
        super().__init__()
        self.system = system
        self.params = params
        self._state = system.x_class()

    def process(self, *inputs, dt):
        u, = inputs
        dx = self.system.dx(self._state, u, self.params)
        self._state = sum_models(self._state, dx, dt)

    @property
    def state(self):
        return self._state

    @property
    def output(self):
        return self.system.y(self.state, self.params)
