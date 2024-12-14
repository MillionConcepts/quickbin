"""Tests of various things that should fail."""
import numpy as np

from quickbin import bin2d, Ops
from quickbin._binning_handlers import binned_minmax, binned_std, binned_unary
from quickbin.quickbin_core import _binned_count, _binned_sum


def test_bad_bounds():
    xarr = np.arange(0, 10)
    yarr = np.arange(0, 10)
    try:
        bin2d(xarr, yarr, None, Ops.count, 5, ((1, 2), (1, 2)))
        raise ValueError("Those bounds should have been rejected")
    except ValueError as ve:
        assert str(ve).startswith("binned_count: specified bounds")


def _check_op_fails(ops: Ops, exc_msg: str):
    try:
        bin2d(None, None, None, ops, None, None)
        raise ValueError(exc_msg)
    except ValueError:
        pass


def test_bad_ops():
    _check_op_fails(
        Ops.count | Ops.median, "Should not be able to compute count and median"
    )
    _check_op_fails(
        Ops.min | Ops.std, "should not be able to compute min and std"
    )


def _check_call_fails(
    binfunc, args, exc_msg: str, expected_exctype: type[Exception]
):
    try:
        binfunc(*args)
        raise ValueError(exc_msg)
    except expected_exctype:
        return
    except Exception as ex:
        raise ValueError(
            f"Wrong exception type (expected {expected_exctype}, got {type(ex)}"
        )


def test_bad_arrays():
    xarr = np.arange(100, dtype=np.float64)
    yarr = np.arange(100, dtype=np.float64)
    varr = np.ones(100, dtype=np.float64)
    _check_call_fails(
        binned_minmax,
        ((xarr, yarr, varr[:2]), (np.nan,) * 4, (5, 5), Ops.min),
        "Mismatched array lengths should fail",
        TypeError
    )
    _check_call_fails(
        binned_std,
        (
            (xarr.astype('u8'), yarr, varr),
            (np.nan,) * 4,
            (5, 5),
            Ops.std | Ops.mean
        ),
        "unsigned int xarr should fail",
        TypeError
    )
    _check_call_fails(
        binned_unary,
        (
            _binned_sum,
            (xarr, yarr, varr.astype('f2')),
            (np.nan,) * 4,
            (5, 5),
            np.float64
        ),
        "varr with itemsize 2 should fail",
        TypeError
    )
    _check_call_fails(
        binned_unary,
        (
            _binned_count,
            (xarr, yarr, varr),
            (np.nan,) * 4,
            (5, 5),
            np.float64
        ),
        "_binned_count should fail with this many args",
        TypeError
    )
