"""The immutable requirement that one declaration places on one array."""

from pydantic import Field
from tenspec.types import operations
from tenspec.types.base import Frozen
from tenspec.types.dtypes import DTypeRequirement
from tenspec.types.properties import ArrayProperty, DeviceRequirement
from tenspec.types.shapes import ShapeSpec


class TensorType(Frozen):
    """Everything one declaration requires of one array.

    The requirement holds no array, no backend, and no binding.
    """

    shape: ShapeSpec = Field(description="the axes the array must have")
    dtype: DTypeRequirement = Field(description="the scalar format the array must have")
    device: DeviceRequirement | None = Field(
        default=None, description="where the array must sit, when a declaration says so"
    )
    transforms: tuple[operations.TransformClass, ...] = Field(
        default=(),
        description="the explicit transforms the array passes through, in declared order",
    )
    properties: tuple[type[ArrayProperty], ...] = Field(
        default=(), description="the independent value properties the array must have"
    )
