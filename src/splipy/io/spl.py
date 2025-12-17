from __future__ import annotations

from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from types import TracebackType
from typing import Self, TextIO

import numpy as np

from splipy.basis import BSplineBasis
from splipy.curve import Curve
from splipy.splineobject import SplineObject
from splipy.surface import Surface
from splipy.volume import Volume

from .master import MasterIO


class SPL(MasterIO):
    fstream: TextIO
    filename: str
    trimming_curves: list[Curve]

    def __init__(self, filename: str) -> None:
        if not filename.endswith(".spl"):
            filename += ".spl"
        self.filename = filename
        self.trimming_curves = []

    def __enter__(self) -> Self:
        self.fstream = Path(self.filename).open()
        return self

    def lines(self) -> Iterator[str]:
        for line in self.fstream:
            yield line.split("#", maxsplit=1)[0].strip()

    def read(self) -> list[SplineObject]:
        lines = self.lines()

        version = next(lines).split()
        assert version[0] == "C"
        assert version[3] == "0"  # No support for rational SPL yet
        pardim = int(version[1])
        physdim = int(version[2])

        orders = [int(k) for k in islice(lines, pardim)]
        ncoeffs = [int(k) for k in islice(lines, pardim)]
        totcoeffs = int(np.prod(ncoeffs))
        nknots = [a + b for a, b in zip(orders, ncoeffs)]

        next(lines)  # Skip spline accuracy

        knots = [[float(k) for k in islice(lines, nkts)] for nkts in nknots]
        bases = [BSplineBasis(p, kts, -1) for p, kts in zip(orders, knots)]

        cpts = np.array([float(k) for k in islice(lines, totcoeffs * physdim)])
        cpts = cpts.reshape(physdim, *(ncoeffs[::-1])).transpose()

        obj = SplineObject.construct_subclass(bases, cpts, rational=False, raw=True)
        return [obj]

    def __exit__(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        traceback: TracebackType,
    ) -> None:
        self.fstream.close()
