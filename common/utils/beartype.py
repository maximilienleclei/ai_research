"""Beartype validators for type-annotated runtime checks.

Provides validator factories for use with typing.Annotated:
- not_empty(): String must be non-empty
- equal(val): Value must equal val
- one_of(*vals): Value must be in vals
- ge(val), gt(val), le(val), lt(val): Numeric comparisons

Usage:
    from typing import Annotated as An
    from common.utils.beartype import ge, one_of

    @dataclass
    class Config:
        count: An[int, ge(0)]
        mode: An[str, one_of("train", "eval")]
"""

from beartype.vale import Is
from beartype.vale._core._valecore import BeartypeValidator


def not_empty() -> BeartypeValidator:
    """Create validator that checks string is non-empty."""
    def _not_empty(x: object) -> bool:
        return isinstance(x, str) and len(x) > 0

    return Is[lambda x: _not_empty(x)]


def equal(element: object) -> BeartypeValidator:
    """Create validator that checks value equals element."""
    def _equal(x: object, element: object) -> bool:
        return x == element

    return Is[lambda x: _equal(x, element)]


def one_of(*elements: object) -> BeartypeValidator:
    """Create validator that checks value is in elements."""
    def _one_of(x: object, elements: tuple[object, ...]) -> bool:
        return x in elements

    return Is[lambda x: _one_of(x, elements)]


def ge(val: float) -> BeartypeValidator:
    """Create validator that checks value >= val."""
    def _ge(x: object, val: float) -> bool:
        return isinstance(x, int | float) and x >= val

    return Is[lambda x: _ge(x, val)]


def gt(val: float) -> BeartypeValidator:
    """Create validator that checks value > val."""
    def _gt(x: object, val: float) -> bool:
        return isinstance(x, int | float) and x > val

    return Is[lambda x: _gt(x, val)]


def le(val: float) -> BeartypeValidator:
    """Create validator that checks value <= val."""
    def _le(x: object, val: float) -> bool:
        return isinstance(x, int | float) and x <= val

    return Is[lambda x: _le(x, val)]


def lt(val: float) -> BeartypeValidator:
    """Create validator that checks value < val."""
    def _lt(x: object, val: float) -> bool:
        return isinstance(x, int | float) and x < val

    return Is[lambda x: _lt(x, val)]
