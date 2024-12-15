import gc
from itertools import product
import time
from typing import Collection, Literal

import psutil
import numpy as np
from scipy.stats import binned_statistic_2d

from quickbin.benchmarks.memlogger import Memwatcher
from quickbin import bin2d
from quickbin.definitions import check_ops, opspec2ops, Ops

RNG = np.random.default_rng()
PROC = psutil.Process()


def benchmark(
    size,
    n_bins,
    ops,
    *,
    n_iter: int = 1,
    spatial_correlation: float = 0,
    verbose: bool = False,
    checkmem: bool = False,
    which: Collection[Literal["scipy", "quickbin"]] = ("scipy", "quickbin")
):
    if len(which) == 0:
        return {}
    maybeprint = print if verbose is True else lambda *_, **__: None
    ops = opspec2ops(ops)
    check_ops(ops)
    xarr = RNG.random(size) * 100
    yarr = RNG.random(size) * 100
    if ops != Ops.count:
        varr = RNG.poisson(100, size)
        if spatial_correlation != 0:
            varr += np.clip(abs(spatial_correlation), 0, 1) * xarr
            varr += np.clip(abs(spatial_correlation), 0, 1) * yarr
    else:
        varr = None
    rec, qtimes, qmems, stimes, smems = {}, [], [], [], []
    args = (xarr, yarr, varr, ops, n_bins)
    memwatch = Memwatcher(PROC.pid, fake=checkmem is False)
    if "quickbin" in which:
        maybeprint('q: ', end='')
        for n in range(n_iter):
            maybeprint(f'{n + 1}/{n_iter}...', end='')
            with memwatch:
                start = time.time()
                bin2d(*args)
                qtimes.append(time.time() - start)
            qmems.append(memwatch.last)
            gc.collect()
            maybeprint(f'({round(qtimes[-1], 1)}s)', end=' ')
        rec['qtime'] = float(np.mean(qtimes))
        rec['qtime_ptp'] = float(np.ptp(qtimes))
        if checkmem is True:
            rec['qmem'] = float(np.mean(qmems))
            rec['qmem_ptp'] = float(np.ptp(qmems))
        maybeprint(f"    qtime: {round(rec['qtime'], 3)}s")
        if checkmem is True:
            maybeprint(
                f"    qmem: {round(rec['qmem'] / 10 ** 6, 2)} MB\n"
            )
    if "scipy" in which:
        stimes, smems = [], []
        maybeprint('s: ', end='')
        for n in range(n_iter):
            maybeprint(f'{n + 1}/{n_iter}...', end='')
            with memwatch:
                start = time.time()
                for opname in ops.name.split("|"):
                    binned_statistic_2d(xarr, yarr, varr, opname, n_bins)
                stimes.append(time.time() - start)
            smems.append(memwatch.last)
            gc.collect()
            maybeprint(f'({round(stimes[-1], 1)}s)', end=' ')
        rec['stime'] = float(np.mean(stimes))
        rec['stime_ptp'] = float(np.ptp(stimes))
        if checkmem is True:
            rec['smem'] = float(np.mean(smems))
            rec['smem_ptp'] = float(np.ptp(smems))
        maybeprint(f"    stime: {round(rec['stime'], 3)}s")
        if checkmem is True:
            maybeprint(
                f"    smem: {round(rec['smem'] / 10 ** 6, 2)} MB\n"
            )
    return rec


DEFAULT_BINCOUNTS = np.hstack([
    10 ** np.arange(1, 4),
    10 ** np.arange(1, 4) * 2,
])
DEFAULT_BINCOUNTS.sort()
DEFAULT_BINCOUNTS.flags.writeable = False

DEFAULT_SIZES = np.hstack([
   10 ** np.arange(3, 9),
   10 ** np.arange(3, 9) * 2,
   # 10 ** np.arange(2, 9) * 3,
   # 10 ** np.arange(3, 8) * 7,
])
DEFAULT_SIZES.sort()
DEFAULT_SIZES.flags.writeable = False


def run_benchmarks(
    ops,
    *,
    sizes: Collection[int] = DEFAULT_SIZES,
    bincounts: Collection[int] = DEFAULT_BINCOUNTS,
    n_iter: int = 1,
    verbose: bool = False,
    which: Collection[Literal["quickbin", "scipy"]] = ("quickbin", "scipy"),
    checkmem: bool = False
):
    maybeprint = print if verbose is True else lambda *_, **__: None
    cases, benches = [], []
    for size, n_bins in product(sizes, bincounts):
        cases.append({'size': size, 'n_bins': n_bins})
    benchkwargs = {
        'n_iter': n_iter,
        'ops': ops,
        'verbose': verbose,
        'which': which,
        'checkmem': checkmem
    }
    try:
        for i, case in enumerate(cases):
            maybeprint(
                f"----cnt: {case['cnt']} size: {case['sz']} "
                f"({i + 1} / {len(cases)})----"
            )
            benches.append(benchmark(**(case | benchkwargs)))
    finally:
        for b in benches:
            b['cnt'] = int(b['cnt'])
            b['sz'] = int(b['sz'])
    return benches
