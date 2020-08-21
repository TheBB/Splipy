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
