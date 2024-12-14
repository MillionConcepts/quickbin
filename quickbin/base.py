from enum import auto, IntFlag
from functools import partial
from numbers import Integral, Real
from types import MappingProxyType
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

class Ops(IntFlag):
    """Enum mapping operation names to flags."""
    count = auto()
    sum = auto()
    mean = auto()
    std = auto()
    median = auto()
    min = auto()
    max = auto()

OPS = MappingProxyType({ op.name: op for op in Ops })
"""Mapping expansion of `Ops`."""

OpName = Literal[tuple(OPS.keys())]
"""Names of valid binning operations."""


BINERR = "n_bins must be either an integer or a sequence of two integers."

# signature of C binning functions from Python's perspective
# TODO: this is overly generic. Write out all actual signatures explicitly
Binfunc = Callable[
    [np.ndarray, ..., Real, Real, Real, Real, Integral, Integral, ...], None
]


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
    count, sum, median.
    """
    result = np.zeros(n_bins[0] * n_bins[1], dtype=dtype)
    binfunc(*arrs, result, *ranges, *n_bins)
    return result.reshape(n_bins)


def binned_countvals(
    arrs: tuple[np.ndarray, np.ndarray, np.ndarray],
    ranges: tuple[float, float, float, float],
    n_bins: tuple[int, int],
    ops: Ops
) -> dict[str, np.ndarray]:
    """Handler for C binned_countvals()."""
    countarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    sumarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    if ops & Ops.mean:
        meanarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    else:
        meanarr = None
    _binned_countvals(*arrs, countarr, sumarr, meanarr, *ranges, *n_bins)
    output = {}
    for op, arr in zip(
        (Ops.count, Ops.sum, Ops.mean),
        (countarr, sumarr, meanarr)
    ):
        if ops & op:
            output[op.name] = arr.reshape(n_bins)
    if len(output) == 1:
        return tuple(output.values())[0].reshape(n_bins)
    return output


# TODO, maybe: Perhaps a bit redundant with binned_countvals().
def binned_std(
    arrs: tuple[np.ndarray, np.ndarray, np.ndarray],
    ranges: tuple[float, float, float, float],
    n_bins: tuple[int, int],
    ops: Ops
) -> np.ndarray | dict[str, np.ndarray]:
    """
    Handler for C binned_std().

    Warning:
        In normal usage, should only be called by bin2d(), which performs a
        variety of input sanitization tasks. Not doing do may cause undesired
        results.
    """
    countarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    sumarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    stdarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    if ops & Ops.mean:
        meanarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    else:
        meanarr = None
    _binned_std(*arrs, countarr, sumarr, stdarr, meanarr, *ranges, *n_bins)
    if ops == Ops.std:
        return stdarr.reshape(n_bins)
    output = {}
    for op, arr in zip(
        (Ops.count, Ops.sum, Ops.mean, Ops.std),
        (countarr, sumarr, meanarr, stdarr)
    ):
        if ops & op:
            output[op.name] = arr.reshape(n_bins)
    return output


def binned_minmax(
    arrs: tuple[np.ndarray, np.ndarray, np.ndarray],
    ranges: tuple[float, float, float, float],
    n_bins: tuple[int, int],
    ops: Ops
) -> dict[str, np.ndarray]:
    """Handler for C binned_minmax()."""
    minarr, maxarr = None, None
    if ops & Ops.min:
        minarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    if ops & Ops.max:
        maxarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    _binned_minmax(*arrs, minarr, maxarr, *ranges, *n_bins)
    if ops == Ops.min & Ops.max:
        return {"min": minarr.reshape(n_bins), "max": maxarr.reshape(n_bins)}
    return next(
        filter(lambda arr: arr is not None, (minarr, maxarr))
    ).reshape(n_bins)



def _maybe_error(
    do_raise: bool = True, exc_class: type[Exception] | None = ValueError
) -> Callable[[str], bool]:
    """
    If do_raise is True, make a function that raises an Exception of type
    exc_class when passed a string; otherwise, make a function that returns
    False when passed a string.
    """
    if do_raise is True and exc_class is None:
        raise TypeError("Must give exc_class if raising Exceptions")

    def error_maybe(msg: str):
        if do_raise is False:
            return False
        raise exc_class(msg)

    return error_maybe


def check_ops(ops: Ops, raise_exc: bool = True) -> bool:
    """
    Check validity of an Ops object: does it specify a legal combination of
    operations? If so, return True. If it does not and `raise_exc` is True,
    raise a useful ValueError; otherwise return False.
    """
    if ops.bit_count() == 1 or ops == (Ops.min & Ops.max):
        return True
    razor = _maybe_error(raise_exc)
    if ops == 0:
        return razor("No operations specified.")
    if ops & ops.median:
        return razor("median can only be computed alone.")
    if ops & (Ops.min | Ops.max):
        razor("min/max can only be computed alongside min/max.")
    if ops & Ops.std:
        return True
    if ops & ~(Ops.count | Ops.sum | Ops.mean):
        razor("Operation invalid in an unusual way: is this not an Ops object?")
    return True


def opspec2ops(ops: OpName | Sequence[OpName] | Integral | Ops) -> Ops:
    """
    Construct an Ops object from an operation specification as passed to bin2d.
    If it is valid, return a corresponding Ops object; otherwise, raise a
    ValueError.
    """
    if isinstance(ops, Integral):
        if not 0 < int(ops) <= sum(Ops):
            raise ValueError(
                f"{ops} out of bounds; must be between 1 and {sum(Ops)}."
            )
        ops = Ops(int(ops))
    elif not isinstance(ops, Ops):
        ops = (ops,) if isinstance(ops, str) else ops
        unknown = set(ops).difference(OPS.keys())
        if len(unknown) > 0:
            s = "s" if len(unknown) > 1 else ""
            raise KeyError(
                f"invalid opname{s}: {unknown}. "
                f"Known ops are {', '.join(OPS.keys())}."
            )
        ops = Ops(sum(OPS[s] for s in ops))
    check_ops(ops)
    return ops

OPWORD_BINFUNC_MAP = MappingProxyType(
    {
        Ops.count: partial(binned_unary, _binned_count, dtype=np.int64),
        Ops.sum: partial(binned_unary, _binned_sum, dtype=np.float64),
        Ops.min: partial(binned_minmax, ops=Ops.min),
        Ops.max: partial(binned_minmax, ops=Ops.max),
        Ops.median: partial(binned_unary, _binned_median, dtype=np.float64),
        Ops.min | Ops.max: partial(binned_minmax, ops=Ops.min | Ops.max)
    }
)
"""
Mapping from some valid opwords to binning handler functions. Does not include 
the many possible permutations of count, sum, mean, and std (see `ops2binfunc`).
"""


def ops2binfunc(ops: Ops) -> Callable:
    """
    Given a valid opword, return a corresponding binning handler function.
    """
    if ops in OPWORD_BINFUNC_MAP.keys():
        return OPWORD_BINFUNC_MAP[ops]
    if ops & Ops.std:
        return partial(binned_std, ops=ops)
    return partial(binned_countvals, ops=ops)



def _set_up_bins(
    n_bins: Union[Sequence[Integral], Integral]
) -> tuple[int, int]:
    """
    Helper function for bin2d(). Formats bin-shape specification correctly.
    Pointless to call this directly.
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


def _set_up_bounds(
    bbounds: Optional[tuple[tuple[Real, Real], tuple[Real, Real]]],
    x_arr: np.ndarray,
    y_arr: np.ndarray
) -> tuple[float, float, float, float]:
    """
    Helper function for bin2d(). Checks and formats binning region bounds
    specifications. Pointless to call this directly.
    """
    xbounds, ybounds = (x_arr.min(), x_arr.max()), (y_arr.min(), y_arr.max())
    # TODO: push this off to C.
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
        # TODO: also push this off to C.
        for (rmin, rmax), (amin, amax) in zip(bbounds, (xbounds, ybounds)):
            if (rmin > amin) or (rmax < amax):
                raise ValueError("x and y values must fall within bbounds")
        ranges = tuple(
            float(f) for f in
            (bbounds[0][0], bbounds[0][1], bbounds[1][0], bbounds[1][1])
        )
    return ranges


def _set_up_xyval(
    op: Ops,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    val_arr: Optional[np.ndarray]
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray]
):
    """
    Helper function for bin2d(). Checks and regularizes array types and
    presence. Pointless to call this directly.
    """
    arrs = [x_arr, y_arr, val_arr]
    for i, arr in enumerate(arrs):
        if arr is None and i != 2:
            raise TypeError("x and y arrays may not be none")
        elif arr is None and op != OPS["count"]:
            raise TypeError("val array may only be none for 'count'")
        elif arr.dtype != np.float64:
            arrs[i] = arr.astype(np.float64)
    return tuple(arrs[:2]) if op == OPS["count"] else tuple(arrs)


def bin2d(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    val_arr: Optional[np.ndarray],
    ops: Union[OpName, Sequence[OpName], Integral, Ops],
    n_bins: Union[Integral, Sequence[Integral]],
    bbounds: Optional[tuple[tuple[Real, Real], tuple[Real, Real]]] = None
) -> dict[str, np.ndarray] | np.ndarray:
    """
    2D generalized histogram function.

    Available statistics are count (classical histogram), sum, mean, median,
    std (standard deviation), min, and max. Min and max may be computed
    simultaneously for efficiency, as may any combination of count, sum, mean,
    and std.

    Args:
        x_arr: 1-D ndarray of x-coordinate values.
        y_arr: 1-D ndarray of y-coordinate values. Must have the same size as
            x_arr.
        val_arr: 1-D ndarray of values. For a solo "count" operation, this may
            be None (and will be ignored in any case). If present, it must
            have the same length as x_arr and y_arr.
        ops: Specification for statistical operation to perform.
            Legal formats are:
                1. a single string (e.g. `"count"`)
                2. a sequence of strings (e.g. `("sum", "count")`).
                3. An instance of `quickbin.base.Ops` (e.g. `Ops.sum
                    | Ops.count`)
                4. An integer "flag word" formed from an additive combination
                    of the values of `Ops`, expressed as an integer (e.g.
                    `Ops.sum` is 2 and `Ops.count` is 1, so `ops=3` is
                    equivalent to `ops=("sum", "count")` and
                    `ops=Ops.sum | Ops.count`.
        n_bins: Number of bins for output array(s). May either be an integer,
            which specifies square arrays of shape `(n_bins, n_bins)`, or a
            sequence of two integers, which specifies arrays of shape
            `(n_bins[0], n_bins[1])`.
        bbounds: Optional restricted bounds specification, like
            `[[xmin, xmax], [ymin, ymax]]`. If not given, uses the min/max
            values of `x_arr` and `y_arr`.

    Returns:
        If `ops` specifies a single statistic (e.g. `ops="count"`),
        returns a single `ndarray`. If `opspec` specifies more than one
        statistic (e.g. `opspec=("min", "max")`), returns a `dict` like
        `"statistic_name": ndarray for that statistic`.

    Note:
        Can be used in most cases as a drop-in replacement for
        `scipy.stats.binned_statistic_2d()`.
    """
    ops = opspec2ops(ops)
    arrs = _set_up_xyval(ops, x_arr, y_arr, val_arr)
    n_bins = _set_up_bins(n_bins)
    ranges = _set_up_bounds(bbounds, x_arr, y_arr)
    # TODO: return dict w/Ops or int keys if Ops / int passed for opspec
    return ops2binfunc(ops)(arrs, ranges, n_bins)

