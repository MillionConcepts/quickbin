"""
Simple validation tests of bin2d() outputs at various array sizes for all valid
op combinations.
"""
from itertools import chain, combinations, product

import numpy as np
import pytest

from quickbin import bin2d, Ops
from quickbin.definitions import check_ops

N_TILE_OPTIONS = np.arange(50, 250, 50)
TILESIZE_OPTIONS = np.arange(50, 10050, 2000)

RNG = np.random.default_rng()
COUNTSUM_OPS = (Ops.count, Ops.sum, Ops.mean, Ops.std)
VALID_COMBOS = tuple(
    map(sum, chain(*[combinations(COUNTSUM_OPS, i) for i in range(2, 5)]))
) + (Ops.min | Ops.max,)


def _check_against_tiles(res, iix, jix, tiles, op):
    check_ops(op)
    if op == Ops.count:
        stack = np.hstack(np.full(len(tiles), len(tiles[0])))
    else:
        stack = np.hstack([getattr(np, op.name)(t) for t in tiles])
    return np.allclose(res[iix, jix], stack)


def _make_test_tiles(n_tiles, tilesize, op):
    if op != Ops.count:
        tiles = [
            (RNG.random(tilesize) - 0.5) * RNG.integers(1, 10) ** 10
            for _ in range(n_tiles)
        ]
    else:
        # NOTE: this is a goofy placeholder to not pass extra arguments to
        # _check_against_tiles
        tiles = [[None for _ in range(tilesize)] for _ in range(n_tiles)]
    iix, jix = np.arange(n_tiles), np.arange(0, n_tiles)
    return iix, jix, tiles


def _simpletest(n_tiles, op, tilesize):
    iix, jix, tiles = _make_test_tiles(n_tiles, tilesize, op)
    # TODO, maybe: non-repeating coords. it becomes slow to check against naive
    #  numpy operations, though, which is sort of the point here.
    np.random.shuffle(iix)
    np.random.shuffle(jix)
    iarr = np.repeat(iix, tilesize)
    jarr = np.repeat(jix, tilesize)
    varr = np.hstack(tiles) if op != Ops.count else None
    res = bin2d(iarr, jarr, varr, op, n_tiles)
    return bool(_check_against_tiles(res, iix, jix, tiles, op))


# TODO: replace / supplement this stuff with hypothesize

@pytest.mark.parametrize("op", Ops)
def test_op_simple(op):
    results = [
        _simpletest(n_tiles, op, tilesize)
        for tilesize, n_tiles in product(TILESIZE_OPTIONS, N_TILE_OPTIONS)
    ]
    if len(failed := tuple(filter(lambda r: r is False, results))) > 0:
        raise ValueError(f"{len(failed)} failed value comps for {op.name}")


@pytest.mark.parametrize("ops", VALID_COMBOS)
def test_op_combo(ops):
    n_failed, ops = 0, Ops(ops)
    for tilesize, n_tiles in product(TILESIZE_OPTIONS, N_TILE_OPTIONS):
        iix, jix, tiles = _make_test_tiles(n_tiles, tilesize, ops)
        res = bin2d(
            np.repeat(iix, tilesize),
            np.repeat(jix, tilesize),
            np.hstack(tiles),
            ops,
            n_tiles
        )
        for op in filter(lambda op: ops & op, list(Ops)):
            if _check_against_tiles(res[op.name], iix, jix, tiles, op) is False:
                n_failed += 1
    if n_failed > 0:
        raise ValueError(f"{n_failed} failed value comps for {ops.name}")
