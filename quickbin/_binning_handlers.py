"""
Handlers for C-side binning functions.

Caution:
    In normal usage, the functions in this module should only be called by
    `quickbin.base.bin2d()`. Skipping the setup steps it performs may
    produce undesired results.
"""
from functools import partial
from types import MappingProxyType
from typing import Callable, Union

import numpy as np

from quickbin.definitions import Binfunc, Ops
from quickbin.quickbin_core import (
    _binned_count,
    _binned_countvals,
    _binned_sum,
    _binned_median,
    _binned_minmax,
    _binned_std,
)


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
    Given a valid opword, return a corresponding binning handler function,
    partially evaluated with appropriate arguments for your convenience.
    """
    if ops in OPWORD_BINFUNC_MAP.keys():
        return OPWORD_BINFUNC_MAP[ops]
    if ops & Ops.std:
        return partial(binned_std, ops=ops)
    return partial(binned_countvals, ops=ops)
