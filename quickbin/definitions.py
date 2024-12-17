"""Operation definitions, types, and validity rules."""

from enum import auto, IntFlag
from numbers import Integral
from types import MappingProxyType
from typing import Callable, Collection, Literal

import numpy as np

F8 = np.dtype('f8')
I8 = np.dtype('i8')


class Ops(IntFlag):
    """Enum mapping operation names to flags."""
    count = auto()
    sum = auto()
    mean = auto()
    std = auto()
    median = auto()
    min = auto()
    max = auto()


OPS = MappingProxyType({ op.name: op for op in Ops })
"""Mapping expansion of `Ops`."""

OpName = Literal["count", "sum", "mean", "std", "median", "min", "max"]
"""Names of valid binning operations."""

BINERR = "n_bins must be either an integer or a sequence of two integers."


def _maybe_error(
    do_raise: bool = True, exc_class: type[Exception] = ValueError
) -> Callable[[str], bool]:
    """
    If do_raise is True, make a function that raises an Exception of type
    exc_class when passed a string; otherwise, make a function that returns
    False when passed a string.
    """

    def error_maybe(msg: str) -> bool:
        if do_raise is False:
            return False
        raise exc_class(msg)

    return error_maybe


def check_ops(ops: Ops, raise_exc: bool = True) -> bool:
    """
    Check validity of an Ops object: does it specify a legal combination of
    operations? If so, return True. If it does not and `raise_exc` is True,
    raise a useful ValueError; otherwise return False.
    """
    if ops.bit_count() == 1 or ops == (Ops.min | Ops.max):
        return True
    razor = _maybe_error(raise_exc)
    if ops == 0:
        return razor("No operations specified.")
    if ops & ops.median:
        return razor("median can only be computed alone.")
    if ops & (Ops.min | Ops.max):
        razor("min/max can only be computed alongside min/max.")
    if ops & Ops.std:
        return True
    if ops & ~(Ops.count | Ops.sum | Ops.mean):
        razor("Operation invalid in an unusual way: is this not an Ops object?")
    return True


def opspec2ops(ops: OpName | Collection[OpName] | Integral | Ops) -> Ops:
    """
    Construct an Ops object from an operation specification as passed to bin2d.
    If it is valid, return a corresponding Ops object; otherwise, raise a
    ValueError.
    """
    if isinstance(ops, Integral):
        if not 0 < int(ops) <= sum(Ops):
            raise ValueError(
                f"{ops} out of bounds; must be between 1 and {sum(Ops)}."
            )
        ops = Ops(int(ops))
    elif isinstance(ops, str):
        ops = (ops,)
    if isinstance(ops, Collection) and not isinstance(ops, Ops):
        if not all(map(lambda s: isinstance(s, str), ops)):
            raise TypeError(
                "A set/list/tuple-like ops specification can contain only "
                "strings."
            )
        if len(ops) > len(Ops):
            raise ValueError(f"Too many operations specified!")
        unknown = set(ops).difference(OPS.keys())
        if len(unknown) > 0:
            s = "s" if len(unknown) > 1 else ""
            # mypy can't see the 'name' field generated by auto()
            raise KeyError(
                f"invalid opname{s}: {unknown}. "
                f"Known ops are {', '.join(OPS.keys())}."
            )
        ops = Ops(sum(OPS[s] for s in ops))
    elif not isinstance(ops, Ops):
        raise TypeError(f"Invalid ops specification: got {type(ops)}")
    check_ops(ops)
    return ops
