"""
Handlers for C-layer binning functions.

Caution:
    In normal usage, the functions in this module should only be called by
    `quickbin.base.bin2d()`. Skipping the setup steps it performs may
    produce undesired results.
"""
from functools import partial
from types import MappingProxyType
from typing import Callable, Union, Optional

import numpy as np
from numpy.typing import NDArray

from quickbin.definitions import I8, F8, Ops
from quickbin._quickbin_core import (
    binned_count,
    binned_countvals,
    binned_sum,
    binned_median,
    binned_minmax,
    binned_std,
)


def binned_unary_handler(
    binfunc: Callable[..., None],
    arrs: Union[
        tuple[NDArray[np.float64], NDArray[np.float64]],
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    ],
    ranges: tuple[float, float, float, float],
    n_bins: tuple[int, int],
    dtype: np.dtype[np.int64 | np.float64]
) -> NDArray[np.float64 | np.int64]:
    """
    Handler for C binning functions that only ever populate one array:
    count, sum, median.
    """
    constructor = np.empty if binfunc == binned_median else np.zeros
    result = constructor(n_bins[0] * n_bins[1], dtype=dtype)
    binfunc(*arrs, result, *ranges, *n_bins)
    return result.reshape(n_bins)


def binned_countvals_handler(
    arrs: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ranges: tuple[float, float, float, float],
    n_bins: tuple[int, int],
    ops: Ops
) ->  NDArray[np.float64] | dict[str, NDArray[np.float64 | np.int64]]:
    """Handler for C binned_countvals()."""
    countarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    sumarr = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    if ops & Ops.mean:
        meanarr = np.empty(n_bins[0] * n_bins[1], dtype='f8')
    else:
        meanarr = None
    binned_countvals(*arrs, countarr, sumarr, meanarr, *ranges, *n_bins)
    if ops == Ops.mean:
        if meanarr is None:
            raise TypeError("Something went wrong in array construction.")
        arrout: NDArray[np.float64] = meanarr.reshape(n_bins)
        return arrout
    output: dict[str, NDArray[np.int64] | NDArray[np.float64]] = {}
    for op, arr in zip(
        (Ops.count, Ops.sum, Ops.mean), (countarr, sumarr, meanarr)
    ):
        if ops & op == 0:
            continue
        if arr is None:
            raise TypeError("Something went wrong in array construction.")
        output[op.name] = arr.reshape(n_bins)
        if op == Ops.count:
            output["count"] = output["count"].astype(np.int64)
    return output


# TODO, maybe: Perhaps a bit redundant with binned_countvals().
def binned_std_handler(
    arrs: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ranges: tuple[float, float, float, float],
    n_bins: tuple[int, int],
    ops: Ops
) -> NDArray[np.float64] | dict[str, NDArray[np.float64 | np.int64]]:
    """
    Handler for C binned_std().

    Warning:
        In normal usage, should only be called by bin2d(), which performs a
        variety of input sanitization tasks. Not doing do may cause undesired
        results.
    """
    countarr: NDArray[np.float64] = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    sumarr: NDArray[np.float64] = np.zeros(n_bins[0] * n_bins[1], dtype='f8')
    stdarr: NDArray[np.float64] = np.empty(n_bins[0] * n_bins[1], dtype='f8')
    if ops & Ops.mean:
        meanarr = np.empty(n_bins[0] * n_bins[1], dtype='f8')
    else:
        meanarr = None
    binned_std(*arrs, countarr, sumarr, stdarr, meanarr, *ranges, *n_bins)
    if ops == Ops.std:
        outarr: NDArray[np.float64] = stdarr.reshape(n_bins)
        return outarr
    output: dict[str, NDArray[np.float64 | np.int64]] = {}
    if ops & Ops.count:
        countout: NDArray[np.int64] = countarr.reshape(n_bins).astype("int64")
        output["count"] = countout
    for op, arr in zip((Ops.sum, Ops.mean, Ops.std), (sumarr, meanarr, stdarr)):
        if ops & op == 0:
            continue
        if arr is None:
            raise TypeError("Something went wrong in array construction.")
        arrout: NDArray[np.float64] = arr.reshape(n_bins)
        output[op.name] = arrout
    return output

from typing import TypeAlias

HandlerFunc: TypeAlias = Callable[
    [
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        tuple[float, float, float, float],
        tuple[int, int],
        Ops,
        Optional[np.dtype[np.int64 | np.float64]]
    ],
    dict[str, NDArray[np.float64 | np.int64]] | NDArray[np.float64 | np.int64]
]


def binned_minmax_handler(
    arrs: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ranges: tuple[float, float, float, float],
    n_bins: tuple[int, int],
    ops: Ops
) -> dict[str, NDArray[np.float64]] | NDArray[np.float64]:
    """Handler for C binned_minmax()."""
    minarr, maxarr = None, None
    if ops & Ops.min:
        minarr = np.empty(n_bins[0] * n_bins[1], dtype='f8')
    if ops & Ops.max:
        maxarr = np.empty(n_bins[0] * n_bins[1], dtype='f8')
    binned_minmax(*arrs, minarr, maxarr, *ranges, *n_bins)
    if ops == Ops.min | Ops.max:
        if maxarr is None or minarr is None:
            raise TypeError("something went wrong in array construction.")
        return {"min": minarr.reshape(n_bins), "max": maxarr.reshape(n_bins)}
    if ops == Ops.min:
        if minarr is None:
            raise TypeError("something went wrong in array construction.")
        return minarr.reshape(n_bins)
    if maxarr is None:
        raise TypeError("something went wrong in array construction.")
    return maxarr.reshape(n_bins)

from typing import Mapping, TypeAlias

HandlerPartial: TypeAlias = (
    partial[
        NDArray[np.float64]
        | dict[str, NDArray[np.int64 | np.float64]]
    ]
    | partial[NDArray[np.float64] |  dict[str, NDArray[np.float64]]]
    | partial[NDArray[np.int64 | np.float64]]
)


OPWORD_BINFUNC_MAP: Mapping[Ops, HandlerPartial] = MappingProxyType(
    {
        Ops.count: partial(binned_unary_handler, binned_count, dtype=I8),
        Ops.sum: partial(binned_unary_handler, binned_sum, dtype=F8),
        Ops.min: partial(binned_minmax_handler, ops=Ops.min),
        Ops.max: partial(binned_minmax_handler, ops=Ops.max),
        Ops.median: partial(binned_unary_handler, binned_median, dtype=F8),
        Ops.min | Ops.max: partial(binned_minmax_handler, ops=Ops.min | Ops.max)
    }
)
"""
Mapping from some valid opwords to binning handler functions. Does not include 
the many possible permutations of count, sum, mean, and std (see `ops2binfunc`).
"""


def ops2binfunc(ops: Ops) -> HandlerPartial:
    """
    Given a valid opword return a corresponding binning handler function,
    partially evaluated with appropriate arguments for your convenience.
    Preferably, this should be prece be called first
    """
    if ops in OPWORD_BINFUNC_MAP.keys():
        return OPWORD_BINFUNC_MAP[ops]
    if ops & Ops.std:
        return partial(binned_std_handler, ops=ops)
    return partial(binned_countvals_handler, ops=ops)
