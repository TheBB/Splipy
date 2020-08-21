import numpy as np


class ControlPointOperation(object):

    def __mul__(self, other):
        if isinstance(other, ControlPointOperation):
            return Compose(self, other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, ControlPointOperation):
            return Compose(other, self)
        return NotImplemented


class Compose(ControlPointOperation):

    def __init__(self, first, second):
        self.first = first
        self.second = second

    def __call__(self, cps):
        return self.second(self.first(cps))


class Identity(ControlPointOperation):

    def __call__(self, cps):
        return cps


class Index(ControlPointOperation):

    def __init__(self, index):
        self.index = index

    def __call__(self, cps):
        return cps[self.index]


def Reverse(axis):
    index = (slice(None, None, None),) * axis + (slice(None, None, -1),)
    return Index(index)


class Rationalize(ControlPointOperation):

    def __call__(self, cps):
        return np.insert(cps, cps.shape[-1], np.ones(cps.shape[:-1]), cps.ndim - 1)


class Roll(ControlPointOperation):

    def __init__(self, shift, axis):
        self.shift = shift
        self.axis = axis

    def __call__(self, cps):
        return np.roll(cps, self.shift, self.axis)


class TensorDot(ControlPointOperation):

    def __init__(self, matrix, axes):
        self.matrix = matrix
        self.axes = axes

    def __call__(self, cps):
        return np.tensordot(self.matrix, cps, axes=self.axes)


class Transpose(ControlPointOperation):

    def __init__(self, permutation):
        self.permutation = permutation

    def __call__(self, cps):
        return cps.transpose(self.permutation)


def Swap(dir1, dir2, pardim):
    permutation = list(range(pardim + 1))
    permutation[dir1] = dir2
    permutation[dir2] = dir1
    return Transpose(tuple(permutation))
