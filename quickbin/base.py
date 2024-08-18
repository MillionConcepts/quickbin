from numbers import Integral, Number
from typing import Literal, Optional, Sequence, Union

import numpy as np

from quickbin._quickbin_core import genhist

# TODO: find a way to share an object between this and the C module
OpName = Literal["count", "sum", "mean", "std", "median", "min", "max"]
OPS = {
    'count': 0,
    'sum': 1,
    'mean': 2,
    'std': 3,
    'median': 4,
    'min': 5,
    'max': 6
}

BINERR = "n_bins must be either an integer or a sequence of two integers."


def bin2d(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    val_arr: np.ndarray,
    op: OpName,
    n_bins: Union[Integral, Sequence[int]],
    bbounds: Optional[
        tuple[tuple[Number, Number], tuple[Number, Number]]
    ] = None
):
    arrs = [x_arr, y_arr, val_arr]
    for i, arr in enumerate(arrs):
        if arr.dtype != np.float64:
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
    if (op := op.lower()) not in OPS.keys():
        raise ValueError(
            f"Unknown operation {op}. "
            f"Valid operations are {', '.join(OPS.keys())}."
        )
    return genhist(*arrs, *ranges, *n_bins, OPS[op]).reshape(n_bins)
