"""
This module contains `bin2d()` and subroutines. `bin2d()` is the on-label
entry point for most of `quickbin`'s functionality.
"""
from numbers import Integral, Real
from typing import Collection, Optional, Sequence, Union

import numpy as np

from quickbin._binning_handlers import ops2binfunc
from quickbin.definitions import OpName, Ops, OPS, BINERR, opspec2ops


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
) -> tuple[float, float, float, float]:
    """
    Helper function for bin2d(). Formats binning region bound specifications.
    Pointless to call this directly.

    Note:
        The C code has responsibility for actual bounds checks. This is so that
        we don't have to calculate the min/max of x_arr and y_arr twice, which
        can be expensive on large arrays.

        If the user doesn't specify bounds, we set them to NaN here, which cues
        the C code to assign them based on x/y array min/max values.
    """
    if bbounds is None:
        ranges = (float('nan'),) * 4
    elif len(bbounds) != 2:
        raise ValueError(
            "bbounds must be a sequence like [[xmin, xmax], [ymin, ymax]]"
        )
    else:
        ranges = tuple(map(float, (*bbounds[0], *bbounds[1])))
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
        elif arr is not None and arr.dtype != np.float64:
            arrs[i] = arr.astype(np.float64)
    return tuple(arrs[:2]) if op == OPS["count"] else tuple(arrs)


def bin2d(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    val_arr: Optional[np.ndarray],
    ops: Union[OpName, Collection[OpName], Integral, Ops],
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
    ranges = _set_up_bounds(bbounds)
    # TODO: return dict w/Ops or int keys if Ops / int passed for opspec
    return ops2binfunc(ops)(arrs, ranges, n_bins)
