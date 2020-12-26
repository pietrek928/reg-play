from collections import defaultdict
from typing import Tuple


def sum_vals(vals):
    vals = list(vals)

    if not vals:
        return 0.

    while len(vals) > 1:
        sums = []
        if len(vals) % 2:
            sums.append(vals[-1])
        for i in range(len(vals) // 2):
            sums.append(vals[i * 2] + vals[i * 2 + 1])
        vals = sums
    return vals[0]


def to_value(v):
    if isinstance(v, GradValue):
        return v
    return GradConst(v)


def get_value(v):
    if isinstance(v, GradValue):
        return v.value
    return v


def set_value(v, nv):
    if isinstance(v, GradValue):
        v.value = nv
        return v
    return nv


class GradValue:
    accept_grad = False

    def __init__(self, v):
        self._v = v

    @property
    def value(self):
        return self._v

    @value.setter
    def value(self, v):
        self._v = v

    def append_grad(self, grad):
        pass

    def compute_grad(self):
        return 0.

    def prop_grad(self):
        pass

    def clear_grad(self):
        pass

    def get_deps(self) -> Tuple['GradValue', ...]:
        return ()

    def __add__(self, v):
        return GradAdd(self, to_value(v))

    def __radd__(self, v):
        return GradAdd(to_value(v), self)

    def __sub__(self, v):
        return GradSub(self, to_value(v))

    def __rsub__(self, v):
        return GradSub(to_value(v), self)

    def __mul__(self, v):
        return GradMul(self, to_value(v))

    def __rmul__(self, v):
        return GradMul(to_value(v), self)

    def __gt__(self, v):
        return self.value > get_value(v)

    def __lt__(self, v):
        return self.value < get_value(v)


class GradConst(GradValue):
    pass


class GradVar(GradValue):
    accept_grad = True

    def __init__(self, v):
        super().__init__(v)
        self._grads = []

    def append_grad(self, grad):
        self._grads.append(grad)

    def clear_grad(self):
        self._grads = []

    def compute_grad(self):
        return sum_vals(self._grads)


class GradOp(GradValue):
    accept_grad = True

    def __init__(self, *args):
        super().__init__(self.compute(*args))
        self._deps = tuple(args)
        self._grads = []

    def get_deps(self) -> Tuple['GradValue', ...]:
        return self._deps

    def compute(self, *args):
        raise NotImplemented('No compute method')

    def append_grad(self, grad):
        self._grads.append(grad)

    def clear_grad(self):
        self._grads = []

    def compute_grad(self):
        return sum_vals(self._grads)


class GradAdd(GradOp):
    def compute(self, a, b):
        return a.value + b.value

    def prop_grad(self):
        base_grad = self.compute_grad()
        a, b = self.get_deps()
        if a.accept_grad:
            a.append_grad(base_grad)
        if b.accept_grad:
            b.append_grad(base_grad)


class GradSub(GradOp):
    def compute(self, a, b):
        return a.value - b.value

    def prop_grad(self):
        base_grad = self.compute_grad()
        a, b = self.get_deps()
        if a.accept_grad:
            a.append_grad(base_grad)
        if b.accept_grad:
            b.append_grad(-base_grad)


class GradMul(GradOp):
    def compute(self, a, b):
        return a.value * b.value

    def prop_grad(self):
        base_grad = self.compute_grad()
        a, b = self.get_deps()
        if a.accept_grad:
            a.append_grad(base_grad * b.value)
        if b.accept_grad:
            b.append_grad(base_grad * a.value)


def compute_grad(start, start_grad=1.):
    st = [start]
    deps_fwd = defaultdict(int)
    visited = set()
    while st:
        a = st.pop()
        if a not in visited:
            a.clear_grad()
            visited.add(a)
            for d in a.get_deps():
                deps_fwd[d] += 1
                st.append(d)

    start.append_grad(start_grad)
    st = [start]
    while st:
        a = st.pop()
        a.prop_grad()
        for d in a.get_deps():
            deps_fwd[d] -= 1
            if not deps_fwd[d]:
                st.append(d)
