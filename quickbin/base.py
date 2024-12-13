from enum import Flag, auto
from numbers import Integral, Number, Real
from types import new_class
from typing import Callable, Literal, Optional, Sequence, Union

import numpy as np

from quickbin.quickbin_core import (
    _binned_count,
    _binned_countvals,
    _binned_sum,
    _binned_median,
    _binned_minmax,
    _binned_std,
)

class Ops(Flag):
    count = auto()
    sum = auto()
    mean = auto()
    std = auto()
    median = auto()
    min = auto()
    max = auto()

OPS = { op.name: op for op in Ops }
OpName = Literal[tuple(OPS.keys())]


BINERR = "n_bins must be either an integer or a sequence of two integers."

# signature of C binning functions from Python's perspective
# TODO: this is overly generic. Write out all actual signatures explicitly
Binfunc = Callable[
    [np.ndarray, ..., Real, Real, Real, Real, Integral, Integral, ...], None
]


def binned_unary(
    arrs: Union[
          tuple[np.ndarray, np.ndarray],
          tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    ranges: tuple[Real, Real, Real, Real],
    n_bins: tuple[Integral, Integral],
    binfunc: Binfunc,
    dtype: np.dtype
) -> np.ndarray:
    """
    Handler for C binning functions that populate only one array:
    count, sum, median.
    """
    result = np.zeros(n_bins[0] * n_bins[1], dtype=dtype)
    binfunc(*arrs, result, *ranges, *n_bins)
    return result.reshape(n_bins)


def binned_countvals(
    arrs: tuple[np.ndarray, np.ndarray, np.ndarray],
    ranges: tuple[Real, Real, Real, Real],
    n_bins: tuple[Integral, Integral],
    oparg: Ops
) -> dict[str, np.ndarray]:
    """Handler for C binned_countvals()."""
    countarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    sumarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    if oparg & Ops.mean:
        meanarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    else:
        meanarr = None
    _binned_countvals(*arrs, countarr, sumarr, meanarr, *ranges, *n_bins)
    output = {}
    for op, arr in zip(
        (Ops.count, Ops.sum, Ops.mean),
        (countarr, sumarr, meanarr)
    ):
        if oparg & op:
            output[op.name] = arr.reshape(n_bins)
    if len(output) == 1:
        return tuple(output.values())[0]
    return output


# TODO, maybe: Perhaps a bit redundant with binned_countvals().
def binned_std(
    arrs: tuple[np.ndarray, np.ndarray, np.ndarray],
    ranges: tuple[Real, Real, Real, Real],
    n_bins: tuple[Integral, Integral],
    oparg: Ops
) -> dict[str, np.ndarray]:
    """Handler for C binned_std()."""
    countarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    sumarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    stdarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    if oparg & Ops.mean:
        meanarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    else:
        meanarr = None
    _binned_std(*arrs, countarr, sumarr, stdarr, meanarr, *ranges, *n_bins)
    output = {}
    for op, arr in zip(
        (Ops.count, Ops.sum, Ops.mean, Ops.std),
        (countarr, sumarr, meanarr, stdarr)
    ):
        if oparg & op:
            output[op.name] = arr.reshape(n_bins)
    if len(output) == 1:
        return tuple(output.values())[0]
    return output


def binned_minmax(
    arrs: tuple[np.ndarray, np.ndarray, np.ndarray],
    ranges: tuple[Real, Real, Real, Real],
    n_bins: tuple[Integral, Integral],
) -> dict[str, np.ndarray]:
    """Handler for C binned_minmax()."""
    minarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    maxarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    _binned_minmax(*arrs, minarr, maxarr, *ranges, *n_bins)
    return {"min": minarr.reshape(n_bins), "max": maxarr.reshape(n_bins)}


def binned_min(
    arrs: tuple[np.ndarray, np.ndarray, np.ndarray],
    ranges: tuple[Real, Real, Real, Real],
    n_bins: tuple[Integral, Integral],
) -> np.ndarray:
    minarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    _binned_minmax(*arrs, minarr, None, *ranges, *n_bins)
    return minarr.reshape(n_bins)


def binned_max(
    arrs: tuple[np.ndarray, np.ndarray, np.ndarray],
    ranges: tuple[Real, Real, Real, Real],
    n_bins: tuple[Integral, Integral],
) -> np.ndarray:
    maxarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    _binned_minmax(*arrs, None, maxarr, *ranges, *n_bins)
    return maxarr.reshape(n_bins)


def bin2d(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    val_arr: Optional[np.ndarray],
    op: Union[Ops, OpName, Sequence[OpName]],
    n_bins: Union[Integral, Sequence[int]],
    bbounds: Optional[
        tuple[tuple[Number, Number], tuple[Number, Number]]
    ] = None
) -> Union[dict[str, np.ndarray], np.ndarray]:
    arrs = [x_arr, y_arr, val_arr]
    for i, arr in enumerate(arrs):
        if arr is None:
            if i != 2:
                raise TypeError("x and y arrays may not be none")
            elif op != 'count':
                raise TypeError("val array may only be none for 'count'")
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

    if isinstance(op, Ops):
        oparg = op
    elif isinstance(op, str):
        oparg = OPS.get(op.lower())
        if oparg is None:
            raise ValueError(
                f"Unknown operation {op.lower()}. "
                f"Valid operations are {', '.join(OPS.keys())}."
            )
    else:
        oparg = Ops(0)
        unknown = set()
        for o in op:
            oo = OPS.get(o.lower())
            if oo is None:
                unknown.add(o.lower())
            else:
                oparg = oparg | oo
        if unknown:
            s = "s" if len(unknown) > 1 else ""
            raise ValueError(
                f"Unknown operation{s} {', '.join(sorted(unknown))}. "
                f"Valid operations are {', '.join(OPS.keys())}."
            )

    if not oparg:
        raise ValueError("must request at least one operation")

    if oparg & Ops.median:
        if oparg != Ops.median:
            raise ValueError("median can only be computed alone.")
        return binned_unary(arrs, ranges, n_bins, _binned_median, np.float64)
    if oparg == Ops.min | Ops.max:
        if oparg & ~(Ops.min | Ops.max):
            raise ValueError("min/max can only be computed alongside min/max")
    if oparg == Ops.min | Ops.max:
        return binned_minmax(arrs, ranges, n_bins)
    if oparg == Ops.min:
        return binned_min(arrs, ranges, n_bins)
    if oparg == Ops.max:
        return binned_max(arrs, ranges, n_bins)
    if oparg == Ops.count:
        return binned_unary(arrs[:2], ranges, n_bins, _binned_count, np.int64)
    if oparg == Ops.sum:
        return binned_unary(arrs, ranges, n_bins, _binned_sum, np.float64)
    if oparg & Ops.std:
        return binned_std(arrs, ranges, n_bins, oparg)
    if oparg & ~(Ops.count | Ops.sum | Ops.mean):
        raise ValueError("Failure in binning operation selection.")
    return binned_countvals(arrs, ranges, n_bins, oparg)
