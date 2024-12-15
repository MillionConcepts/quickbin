from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def binned_count(
    xarr: "NDArray[np.float64]",
    yarr: "NDArray[np.float64]",
    count_result: "NDArray[np.int64]",
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    nx: int,
    ny: int
) -> None: ...


def binned_sum(
    xarr: "NDArray[np.float64]",
    yarr: "NDArray[np.float64]",
    varr: "NDArray[np.float64]",
    sum_result: "NDArray[np.float64]",
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    nx: int,
    ny: int
) -> None: ...


def binned_median(
    xarr: "NDArray[np.float64]",
    yarr: "NDArray[np.float64]",
    varr: "NDArray[np.float64]",
    sum_result: "NDArray[np.float64]",
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    nx: int,
    ny: int
) -> None: ...


def binned_countvals(
    xarr: "NDArray[np.float64]",
    yarr: "NDArray[np.float64]",
    varr: "NDArray[np.float64]",
    count_result: "NDArray[np.float64]",
    sum_result: "NDArray[np.float64]",
    mean_result: "Optional[NDArray[np.float64]]",
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    nx: int,
    ny: int
) -> None: ...


def binned_std(
    xarr: "NDArray[np.float64]",
    yarr: "NDArray[np.float64]",
    varr: "NDArray[np.float64]",
    count_result: "NDArray[np.float64]",
    sum_result: "NDArray[np.float64]",
    mean_result: "Optional[NDArray[np.float64]]",
    std_result: "NDArray[np.float64]",
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    nx: int,
    ny: int
) -> None: ...


def binned_minmax(
    xarr: "NDArray[np.float64]",
    yarr: "NDArray[np.float64]",
    varr: "NDArray[np.float64]",
    min_result: "Optional[NDArray[np.float64]]",
    max_result: "Optional[NDArray[np.float64]]",
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    nx: int,
    ny: int
) -> None: ...
