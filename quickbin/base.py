from numbers import Integral, Number
import signal
from typing import Literal, Optional, Sequence, Union

import numpy as np

from quickbin.definitions import OPS
from quickbin._quickbin_core import genhist

INTERRUPTS_RECEIVED = []


OpName = Literal["count", "sum", "mean", "std", "median", "min", "max"]
BINERR = "n_bins must be either an integer or a sequence of two integers."


def bin2d(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    val_arr: np.ndarray,
    op: Union[OpName, Sequence[OpName]],
    n_bins: Union[Integral, Sequence[int]],
    bbounds: Optional[
        tuple[tuple[Number, Number], tuple[Number, Number]]
    ] = None
):
    arrs = [x_arr, y_arr, val_arr]
    for i, arr in enumerate(arrs):
        if arr is None:
            if i != 2:
                raise TypeError("x and y arrays may not be none")
            elif op != 'count':
                raise TypeError("val array may only be none for count op")
            else:
                arrs[i] = np.array([])
        elif arr.dtype != np.float64:
            arrs[i] = arr.astype(np.float64)
    if not isinstance(n_bins, Sequence):
        n_bins = [n_bins, n_bins]
    elif len(n_bins) != 2:
        raise ValueError(BINERR)
    if not all(map(lambda n: isinstance(n, Integral), n_bins)):
        raise TypeError(BINERR)
    n_bins = tuple(map(int, n_bins))
    if min(n_bins) <= 0:
        raise ValueError("Must have a strictly positive number of bins.")
    xbounds, ybounds = (x_arr.min(), x_arr.max()), (y_arr.min(), y_arr.max())
    if bbounds is None:
        ranges = (
            xbounds[0],
            xbounds[1] + np.finfo('f8').resolution * 5,
            ybounds[0],
            ybounds[1] + np.finfo('f8').resolution * 5
        )
    elif len(bbounds) != 2:
        raise ValueError(
            "bbounds must be a sequence like [[xmin, xmax], [ymin, ymax]]"
        )
    else:
        for (rmin, rmax), (amin, amax) in zip(bbounds, (xbounds, ybounds)):
            if (rmin > amin) or (rmax < amax):
                raise ValueError("x and y values must fall within bbounds")
        ranges = (bbounds[0][0], bbounds[0][1], bbounds[1][0], bbounds[1][1])
    ranges = tuple(map(float, ranges))
    ops = {op} if isinstance(op, str) else set(op)
    if not set(ops).issubset(OPS.keys()):
        raise ValueError(
            f"Unknown operation(s) {ops.difference(OPS.keys())}. "
            f"Valid operations are {', '.join(OPS.keys())}."
        )

    def make_interrupter():
        # TODO: politely replace default keyboardinterrupt
        def interrupter(signalnum, frame):
            INTERRUPTS_RECEIVED.append(signal.SIGINT)
            raise KeyboardInterrupt

        return interrupter
    oparg = sum(OPS[o].value for o in map(str.lower, ops))
    try:
        signal.signal(signal.SIGINT, make_interrupter())
        res = genhist(*arrs, *ranges, *n_bins, oparg)
        if isinstance(res, np.ndarray):
            return res.reshape(n_bins)
        elif len(res) == 1:
            return tuple(res.values())[0].reshape(n_bins)
        return {k: v.reshape(n_bins) for k, v in res.items()}
    except Exception as ex:
        if len(INTERRUPTS_RECEIVED) > 0:
            raise KeyboardInterrupt
        raise ex

