from enum import Flag
from functools import partial, reduce
from numbers import Integral, Real
from operator import add
from types import MappingProxyType, new_class
from typing import Callable, Literal, Optional, Sequence, Union, TypeAlias

import numpy as np

from quickbin.quickbin_core import (
    _binned_count,
    _binned_countvals,
    _binned_sum,
    _binned_median,
    _binned_min,
    _binned_max,
    _binned_minmax,
    _binned_std,
    OPS
)

OpName = Literal[tuple(OPS.keys())]
"""Names of valid binning operations."""

Ops = new_class("Ops", bases=(Flag,), exec_body=lambda ns: ns.update(OPS))
"""Enum mapping operation names to flags."""

Opword: TypeAlias = int
"""Represents 'legal' additive combination of flag values in Ops / OPS."""

BINERR = "n_bins must be either an integer or a sequence of two integers."

# TODO: this is overly generic. Write out all actual signatures explicitly
Binfunc = Callable[
    [np.ndarray, ..., Real, Real, Real, Real, Integral, Integral, ...], None
]
"""Signature of C binning functions from Python's perspective."""


def binned_unary(
    binfunc: Binfunc,
    arrs: Union[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    ranges: tuple[float, float, float, float],
    n_bins: tuple[int, int],
    dtype: np.dtype
) -> np.ndarray:
    """
    Handler for C binning functions that only ever populate one array:
    count, sum, median, min, and max.
    """
    result = np.zeros(n_bins[0] * n_bins[1], dtype=dtype)
    binfunc(*arrs, result, *ranges, *n_bins)
    return result.reshape(n_bins)


def binned_countvals(
    arrs: tuple[np.ndarray, np.ndarray, np.ndarray],
    ranges: tuple[float, float, float, float],
    n_bins: tuple[int, int],
    opword: Opword
) -> Union[np.ndarray, dict[str, np.ndarray]]:
    """Handler for C binned_countvals()."""
    count = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    sum_ = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    if opword & OPS["mean"]:
        mean_ = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    else:
        mean_ = np.array([])  # should never be touched by C in this case
    _binned_countvals(*arrs, count, sum_, mean_, *ranges, *n_bins, opword)
    if opword == OPS["mean"]:
        return mean_.reshape(n_bins)
    output = {}
    for name, arr in zip(("count", "sum", "mean"), (count, sum_, mean_)):
        if opword & OPS[name]:
            output[name] = arr.reshape(n_bins)
    return output


# TODO, maybe: Perhaps a bit redundant with binned_countvals().
def binned_std(
    arrs: tuple[np.ndarray, np.ndarray, np.ndarray],
    ranges: tuple[float, float, float, float],
    n_bins: tuple[int, int],
    opword: int
) -> np.ndarray | dict[str, np.ndarray]:
    """
    Handler for C binned_std().

    Warning:
        In normal usage, should only be called by bin2d(), which performs a
        variety of input sanitization tasks. Not doing do may cause undesired
        results.
    """
    count = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    sum_ = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    std = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    if opword & OPS["mean"]:
        mean_ = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    else:
        mean_ = np.array([])  # should never be touched by C in this case
    _binned_std(
        *arrs, count, sum_, mean_, std, *ranges, *n_bins, opword
    )
    if opword == OPS["std"]:
        return std.reshape(n_bins)
    output = {}
    for name, arr in zip(
        ("count", "sum", "mean", "std"), (count, sum_, mean_, std)
    ):
        if opword & OPS[name]:
            output[name] = arr.reshape(n_bins)
    if len(output) == 1:
        return tuple(output.values())[0]
    return output


def binned_minmax(
    arrs: tuple[np.ndarray, np.ndarray, np.ndarray],
    ranges: tuple[float, float, float],
    n_bins: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Handler for C binned_minmax()."""
    minarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    maxarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    _binned_minmax(*arrs, minarr, maxarr, *ranges, *n_bins)
    return {"min": minarr.reshape(n_bins), "max": maxarr.reshape(n_bins)}


def _set_up_xyval(
    op: Opword,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    val_arr: Optional[np.ndarray]
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray]
):
    arrs = [x_arr, y_arr, val_arr]
    for i, arr in enumerate(arrs):
        if arr is None and i != 2:
            raise TypeError("x and y arrays may not be none")
        elif arr is None and op != OPS["count"]:
            raise TypeError("val array may only be none for 'count'")
        elif arr.dtype != np.float64:
            arrs[i] = arr.astype(np.float64)
    return arrs[:2] if op == OPS["count"] else tuple(arrs)


FIXED_OP_BINFUNCS = MappingProxyType(
    {
        OPS["count"]: partial(binned_unary, _binned_count, dtype=np.int64),
        OPS["sum"]: partial(binned_unary, _binned_sum, dtype=np.float64),
        OPS["min"]: partial(binned_unary, _binned_min, dtype=np.float64),
        OPS["max"]: partial(binned_unary, _binned_max, dtype=np.float64),
        OPS["median"]: partial(binned_unary, _binned_median, dtype=np.float64),
        OPS["min"] & OPS["max"]: binned_minmax
    }
)
"""
Mapping from opwords to binning handler functions that always produce the 
same number and type of output arrays.
"""


def check_opword(opword: Opword):
    """Check validity of an Opword."""
    if not (128 > opword > 0):
        raise ValueError("opword out of range.")
    if opword & (opword - 1) == 0:
        return
    if opword & OPS["median"]:
        raise ValueError("median can only be computed alone.")
    if opword == OPS["min"] & OPS["max"]:
        return
    if opword & (OPS["min"] | OPS["max"]):
        raise ValueError("min/max can only be computed alongside min/max.")
    if opword & OPS["std"]:
        return
    if opword & ~(OPS["count"] | OPS["sum"] | OPS["mean"]):
        raise ValueError("Unusually invalid opword")


def opspec2opword(opspec: OpName | Sequence[OpName] | Integral) -> Opword:
    """
    Check an op specification as passed to bin2d and, if valid, return the
    corresponding Opword.
    """
    if isinstance(opspec, Integral):
        opword = int(opspec)
    else:
        opspec = (opspec,) if isinstance(opspec, str) else opspec
        if not set(opspec).issubset(OPS.keys()):
            raise KeyError(
                f"invalid opname(s): {set(opspec).difference(OPS.keys())}."
                f"Known ops are {', '.join(OPS.keys())}."
            )
        opword = reduce(add, (OPS[s] for s in opspec))
    check_opword(opword)
    return opword


def _set_up_bins(
    n_bins: Union[Sequence[Integral], Integral]
) -> tuple[int, int]:
    """
    Helper function for bin2d(). Formats bin-shape specification correctly.
    Should not be called directly.
    """
    if not isinstance(n_bins, Sequence):
        n_bins = (n_bins, n_bins)
    elif len(n_bins) != 2:
        raise ValueError(BINERR)
    if not all(map(lambda n: isinstance(n, Integral), n_bins)):
        raise TypeError(BINERR)
    if min(n_bins) <= 0:
        raise ValueError("Must have a strictly positive number of bins.")
    return int(n_bins[0]), int(n_bins[1])


def opword2binfunc(opword: Opword) -> Callable:
    """
    Given a valid opword, return a corresponding binning handler function.
    """
    if opword in FIXED_OP_BINFUNCS.keys():
        return FIXED_OP_BINFUNCS[opword]
    if opword & OPS["std"]:
        return partial(binned_std, opword=opword)
    return partial(binned_countvals, opword=opword)


def _set_up_bounds(
    bbounds: Optional[tuple[tuple[Real, Real], tuple[Real, Real]]],
    x_arr: np.ndarray,
    y_arr: np.ndarray
) -> tuple[float, float, float, float]:
    """
    Helper function for bin2d(). Checks and formats binning bounds
    specifications. Should not be called directly.
    """
    xbounds, ybounds = (x_arr.min(), x_arr.max()), (y_arr.min(), y_arr.max())
    if bbounds is None:
        ranges = (
            float(xbounds[0]),
            float(xbounds[1] + np.finfo('f8').resolution * 5),
            float(ybounds[0]),
            float(ybounds[1] + np.finfo('f8').resolution * 5)
        )
    elif len(bbounds) != 2:
        raise ValueError(
            "bbounds must be a sequence like [[xmin, xmax], [ymin, ymax]]"
        )
    else:
        for (rmin, rmax), (amin, amax) in zip(bbounds, (xbounds, ybounds)):
            if (rmin > amin) or (rmax < amax):
                raise ValueError("x and y values must fall within bbounds")
        ranges = tuple(
            float(f) for f in
            (bbounds[0][0], bbounds[0][1], bbounds[1][0], bbounds[1][1])
        )
    return ranges


def bin2d(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    val_arr: Optional[np.ndarray],
    opspec: Union[OpName, Sequence[OpName], Integral],
    n_bins: Union[Integral, Sequence[Integral]],
    bbounds: Optional[tuple[tuple[Real, Real], tuple[Real, Real]]] = None
) -> Union[dict[str, np.ndarray], np.ndarray]:
    """
    2D generalized histogram function.

    Available statistics are count (classical histogram), sum, mean, median,
    std (standard deviation), min, and max. Min and max may be computed
    simultaneously for efficiency, as may any combination of count, sum, mean,
    and std.

    Args:
        x_arr: 1-D array of x-coordinate values.
        y_arr: 1-D array of y-coordinate values. Must have the same size as
            x_arr.
        val_arr: 1-D Array of values. For a solo "count" operation, this may
            be None (and will be ignored in any case). If present, it must
            have the same length as x_arr and y_arr.
        opspec: Specification for statistical operation to perform. This can
            be a single string (e.g. `"count"`) or a sequence of strings
            (e.g. `("sum", "count")`). It may also be a flag word formed from
            an additive combination of the values of `quickbin.base.OPS`,
            expressed as an integer (e.g. `OPS["sum"]` is 2 and `OPS["count"]`
            is 1, so `opspec=3` is equivalent to `opspec=("sum", "count")`.
        n_bins: Number of bins for output array(s). May either be an integer,
            which specifies square arrays of shape `(n_bins, n_bins)`, or a
            sequence of two integers, which specifies arrays of shape
            `(n_bins[0], n_bins[1])`.
        bbounds: Optional restricted bounds specification, like
            `[[xmin, xmax], [ymin, ymax]]`. If not given, uses the min/max
            values of `x_arr` and `y_arr`.

    Returns:
        If `opspec` specifies a single statistic (e.g. `opspec="count"`),
        returns a single `ndarray`. If `opspec` specifies more than one
        statistic (e.g. `opspec=("min", "max")`), returns a `dict` like
        `statistic_name: ndarray for that statistic`.

    Note:
        Can be used in most cases as a drop-in replacement for
        `scipy.stats.binned_statistic_2d()`.
    """
    opword = opspec2opword(opspec)
    arrs = _set_up_xyval(opword, x_arr, y_arr, val_arr)
    n_bins = _set_up_bins(n_bins)
    ranges = _set_up_bounds(bbounds, x_arr, y_arr)
    # TODO: return dict w/flagword keys if flagword passed for opspec
    return opword2binfunc(opword)(arrs, ranges, n_bins)
