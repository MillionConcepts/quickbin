import gc
from itertools import product
import time

import numpy as np
import psutil
import pytest

from quickbin import bin2d
from quickbin.base import OPS

N_TILE_OPTIONS = np.arange(50, 250, 50)
TILESIZE_OPTIONS = np.arange(50, 10050, 1000)

RNG = np.random.default_rng()


# TODO: more

def _simpletest(n_tiles, op, tilesize):
    # TODO: these tiles don't need to be identical
    tile = (RNG.random(tilesize) - 0.5) * RNG.integers(1, 10) ** 10
    tiled = np.tile(tile, n_tiles)
    axchunk = np.arange(0, n_tiles, dtype='f8')
    ax = np.repeat(axchunk, tilesize)
    res = bin2d(ax, ax, tiled, op, n_tiles)
    flipres = bin2d(ax, np.flip(ax), tiled, op, n_tiles)
    if op in ("median", "mean", "std", "sum", "min", "max"):
        compfunc = getattr(np, op)
    elif op == "count":
        compfunc = lambda k: k.size
    else:
        raise ValueError(f"unknown test operation {op}")
    try:
        assert (np.diag(res) == compfunc(tile)).all()
        assert (np.diag(np.flip(flipres, axis=1)) == compfunc(tile)).all()
    except AssertionError:
        return False
    return True


# TODO: replace / supplement this stuff with hypothesize
@pytest.mark.parametrize("op", OPS.keys())
def test_op_simple(op):
    results = []
    for tilesize, n_tiles in product(TILESIZE_OPTIONS, N_TILE_OPTIONS):
        results.append(_simpletest(n_tiles, op, tilesize))
    failures = tuple(filter(lambda r: None, results))
    if len(failures) > 0:
        raise ValueError(f"{len(failures)} failed value comparisons")


# The following may be too crude / non-portable an idea to work at all, and is
# not working currently for at least the reason that there are memory usage
# jumps and drops of unknown and confusing etiology when operations are
# executed in particular orders (e.g. running a sum 10
# times followed by a count causes the process RSS to jump; running a  count
# 10 times followed by a sum does not; running a count 10 times, then a sum
# 10 times, then a count does not; running a sum 10 times and then a median
# does not). I suspect this is related to some weird Python / numpy import
# plumbing crap but there are plenty of other possible explanations.


# don't, of course, set these too small
# TMLC_INPUT_SIZE = int(4e6)
# TMLC_BINSIZE = int(3e3)
# TMLC_ITERATIONS = 15


# @pytest.mark.parametrize("op", ["count", "sum"])
# def test_memory_leak_crudely():
#     if (
#         (3 * TMLC_INPUT_SIZE + TMLC_BINSIZE ** 2) * np.finfo('f8').bits * 8
#     ) < 100:
#         raise ValueError("Set the TMLC size variables in this module higher")
#     proc = psutil.Process()
#     # time.sleep(1)
#     basemem = proc.memory_info().rss
#     print(basemem)
#     for i in range(TMLC_ITERATIONS):
#         xarr = RNG.random(TMLC_INPUT_SIZE)
#         yarr = RNG.random(TMLC_INPUT_SIZE)
#         varr = RNG.random(TMLC_INPUT_SIZE)
#         _result = bin2d(xarr, yarr, varr, op, TMLC_BINSIZE)
#         # del _result
#         # import time
#         # time.sleep(1)
#         del xarr, yarr, varr, _result
#         # gc.collect()
#         # this slop of 60 MB is intended to give a little grace for stuff
#         # still allocated to the process for some reason, Python objects not
#         # controlled by us, etc. -- if we're actually leaking, we should get
#         # past this at some point
#         print((proc.memory_info().rss - basemem) / 1e6)
#         if proc.memory_info().rss - basemem > 6e7:
#             time.sleep(2)
#             if proc.memory_info().rss - basemem > 6e7:
#                 raise RuntimeError(f"possible memory leak on iteration {i}")
#         print((proc.memory_info().rss - basemem) / 1e6)
#         # # print(refcounts())
#         print(proc.memory_info().rss)


# def test_memory_leak_crudely_2():
#     op = "sum"
#     if (
#         (3 * TMLC_INPUT_SIZE + TMLC_BINSIZE ** 2) * np.finfo('f8').bits * 8
#     ) < 100:
#         raise ValueError("Set the TMLC size variables in this module higher")
#     proc = psutil.Process()
#     # time.sleep(1)
#     basemem = proc.memory_info().rss
#     print(basemem)
#     ops = iter(("sum", "count", "sum", "count2", "count", "sum"))
#     for op in ops:
#         xarr = RNG.random(TMLC_INPUT_SIZE)
#         yarr = RNG.random(TMLC_INPUT_SIZE)
#         print(op)
#         if op == "count2":
#             varr = None
#             op = "count"
#         else:
#             varr = RNG.random(TMLC_INPUT_SIZE)
#         _result = bin2d(xarr, yarr, varr, op, TMLC_BINSIZE)
#         # del _result
#         # import time
#         # time.sleep(1)
#         del xarr, yarr, varr, _result
#         # gc.collect()
#         # this slop of 60 MB is intended to give a little grace for stuff
#         # still allocated to the process for some reason, Python objects not
#         # controlled by us, etc. -- if we're actually leaking, we should get
#         # past this at some point
#         if proc.memory_info().rss - basemem > 6e7:
#             time.sleep(2)
#             if proc.memory_info().rss - basemem > 6e7:
#                 raise RuntimeError
#         print((proc.memory_info().rss - basemem) / 1e6)