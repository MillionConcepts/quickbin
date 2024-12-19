"""
This module contains `bin2d()` and subroutines. `bin2d()` is the on-label
entry point for most of `quickbin`'s functionality.
"""
from numbers import Integral, Real
from typing import Collection, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from quickbin._binning_handlers import ops2binfunc
from quickbin.definitions import BINERR, OpName, Ops, opspec2ops


def _set_up_bins(
    n_bins: Union[tuple[Integral, Integral], Integral]
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
        we don't have to calculate the min/max of i_arr and j_arr twice, which
        can be expensive on large arrays.

        If the user doesn't specify bounds, we set them to NaN here, which cues
        the C code to assign them based on i/j array min/max values.
    """
    if bbounds is None:
        return (float('nan'),) * 4
    elif len(bbounds) != 2:
        raise ValueError(
            "bbounds must be a sequence like [[imin, imax], [jmin, jmax]]"
        )
    return (
        float(bbounds[0][0]),
        float(bbounds[0][1]),
        float(bbounds[1][0]),
        float(bbounds[1][1])
    )


def _set_up_ijval(
    op: Ops,
    i_arr: NDArray[np.integer | np.floating],
    j_arr: NDArray[np.integer | np.floating],
    val_arr: Optional[NDArray[np.integer | np.floating]]
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    | tuple[NDArray[np.float64], NDArray[np.float64]]
):
    """
    Helper function for bin2d(). Checks and regularizes array types and
    presence. Pointless to call this directly.
    """
    if not (isinstance(i_arr, np.ndarray) and isinstance(j_arr, np.ndarray)):
        raise TypeError("i and j coordinate arguments must be ndarrays")
    if i_arr.dtype != np.float64:
        i_arr = i_arr.astype(np.float64)
    if j_arr.dtype != np.float64:
        j_arr = j_arr.astype(np.float64)
    if op == Ops.count:
        return i_arr, j_arr
    if not isinstance(val_arr, np.ndarray):
        raise TypeError("value argument may only be None for 'count' op")
    if val_arr.dtype != np.float64:
        val_arr = val_arr.astype(np.float64)
    return i_arr, j_arr, val_arr


def bin2d(
    i_arr: NDArray[np.integer | np.floating],
    j_arr: NDArray[np.integer | np.floating],
    val_arr: Optional[NDArray[np.integer | np.floating]],
    ops: Union[OpName, Collection[OpName], Integral, Ops],
    n_bins: Union[Integral, tuple[Integral, Integral]],
    bbounds: Optional[tuple[tuple[Real, Real], tuple[Real, Real]]] = None
) -> dict[str, np.ndarray] | np.ndarray:
    """
    2D generalized histogram function.

    Available statistics are count (classical histogram), sum, mean, median,
    std (standard deviation), min, and max. Min and max may be computed
    simultaneously for efficiency, as may any combination of count, sum, mean,
    and std.

    Note:
        This can be used as a nearly drop-in replacement for
        `scipy.stats.binned_statistic_2d()`. Major differences are:
            1. It can be used to compute multiple statistics at once (see below)
            2. It does not return a `BinnedStatistic2dResult` object, but
               rather an `np.ndarray` (if computing one statistic) or a
               `dict[str, np.ndarray]` (if computing multiple statistics).
            3. It does not explicitly compute and return bin edges. If needed,
               you can calculate them as a simple regular partition of the
               space. (They should be identical to `binned_statistic_2d()`'s
               down to floating-point error.)

    Note:
        Similar functions that interface with numpy (including
        `binned_statistic_2d()`) sometimes call their first two arguments "x"
        and "y", but map them to the first and second axes respectively.
        Because numpy uses (i, j) / (y, x) / row-major axis order,
        this maps arguments named "x" to the y axis and arguments named "y"
        to the x axis. We find that using "x" and "y" in this way creates
        confusion. We have given "i" and "j" identifiers to the arguments
        in order to emphasize that they map to the first and second axes. The
        behavior is identical.

    Args:
        i_arr: 1-D ndarray of coordinates to bin along the first axis.
        j_arr: 1-D ndarray of coordinates to bin along the second axis.
        val_arr: 1-D ndarray of values. For a solo "count" operation, this may
            be None (and will be ignored in any case). If present, it must
            have the same length as i_arr and j_arr.
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
            `[[imin, imax], [jmin, jmax]]`. If not given, uses the min/max
            values of `i_arr` and `j_arr`.

    Returns:
        If `ops` specifies a single statistic (e.g. `ops="count"`),
        returns a single `ndarray`. If `opspec` specifies more than one
        statistic (e.g. `opspec=("min", "max")`), returns a `dict` like
        `"statistic_name": ndarray for that statistic`.
    """
    ops = opspec2ops(ops)
    arrs = _set_up_ijval(ops, i_arr, j_arr, val_arr)
    n_bins = _set_up_bins(n_bins)
    ranges = _set_up_bounds(bbounds)
    # TODO: return dict w/Ops or int keys if Ops / int passed for opspec
    return ops2binfunc(ops)(arrs, ranges, n_bins)
