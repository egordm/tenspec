"""Scalar formats, their families, and what a declaration requires of them."""

import functools
from collections.abc import Mapping
from enum import StrEnum
from typing import ClassVar, TypeVar

from pydantic import Field
from tenspec.errors import AnnotationError
from tenspec.types.base import Frozen


class DTypeFamily(StrEnum):
    """A set of scalar formats that one declaration accepts."""

    FLOATING = "floating"
    INTEGER = "integer"
    UNSIGNED = "unsigned"
    COMPLEX = "complex"
    BOOLEAN = "boolean"


class DTypeId(StrEnum):
    """The logical scalar format, independent of the array library.

    NumPy float32 and Torch float32 are one DTypeId, so a shared variable accepts both.
    """

    BOOL = "bool"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"

    @property
    def family(self) -> DTypeFamily:
        """The family this scalar format belongs to."""
        return DTYPE_FAMILIES[self]


DTYPE_FAMILIES: Mapping[DTypeId, DTypeFamily] = {
    DTypeId.BOOL: DTypeFamily.BOOLEAN,
    DTypeId.INT8: DTypeFamily.INTEGER,
    DTypeId.INT16: DTypeFamily.INTEGER,
    DTypeId.INT32: DTypeFamily.INTEGER,
    DTypeId.INT64: DTypeFamily.INTEGER,
    DTypeId.UINT8: DTypeFamily.UNSIGNED,
    DTypeId.UINT16: DTypeFamily.UNSIGNED,
    DTypeId.UINT32: DTypeFamily.UNSIGNED,
    DTypeId.UINT64: DTypeFamily.UNSIGNED,
    DTypeId.FLOAT16: DTypeFamily.FLOATING,
    DTypeId.BFLOAT16: DTypeFamily.FLOATING,
    DTypeId.FLOAT32: DTypeFamily.FLOATING,
    DTypeId.FLOAT64: DTypeFamily.FLOATING,
    DTypeId.COMPLEX64: DTypeFamily.COMPLEX,
    DTypeId.COMPLEX128: DTypeFamily.COMPLEX,
}
"""The family of every scalar format. Each format belongs to exactly one family."""


class DType:
    """The bound of a shared dtype variable. Every scalar format satisfies it."""

    permitted: ClassVar[DTypeId | DTypeFamily | None] = None


class Floating(DType):
    """The bound of a shared dtype variable that must stay in the floating family."""

    permitted: ClassVar[DTypeId | DTypeFamily | None] = DTypeFamily.FLOATING


def permitted_formats(permitted: DTypeId | DTypeFamily | None) -> frozenset[DTypeId]:
    """Every scalar format one constraint accepts. None accepts them all.

    Returns:
        The accepted scalar formats.
    """
    if permitted is None:
        return frozenset(DTypeId)
    if isinstance(permitted, DTypeFamily):
        return frozenset(dtype for dtype in DTypeId if dtype.family is permitted)
    return frozenset({permitted})


class DTypeRequirement(Frozen):
    """What a declaration requires of one array's scalar format."""

    permitted: DTypeId | DTypeFamily = Field(
        description="the one accepted format, or the family the format must belong to"
    )
    variable: TypeVar | None = Field(
        default=None,
        description="the shared variable that every operand carrying it must agree on",
    )

    @functools.cached_property
    def permitted_formats(self) -> frozenset[DTypeId]:
        """Every scalar format this whole requirement accepts.

        A shared variable narrows the alias. Float with a variable bound to a float64-only
        subset permits float64 alone, so the bound is a requirement and not a hint. Both
        the runtime check and operation applicability read this one set.

        Returns:
            The accepted scalar formats.

        Raises:
            AnnotationError: when the variable is bound to something other than DType.
        """
        alias = permitted_formats(self.permitted)
        if self.variable is None:
            return alias
        bound = self.variable.__bound__
        if not (isinstance(bound, type) and issubclass(bound, DType)):
            raise AnnotationError(
                f"the shared variable {self.variable.__name__} must be bound to DType, "
                f"or to one of its subsets"
            )
        return alias & permitted_formats(bound.permitted)

    def accepts(self, dtype: DTypeId) -> bool:
        """True when `dtype` satisfies the permitted format, the family, and the bound.

        DTypeRequirement(permitted=DTypeFamily.FLOATING).accepts(DTypeId.FLOAT32) is True.

        Returns:
            True when the format satisfies this requirement.
        """
        return dtype in self.permitted_formats
