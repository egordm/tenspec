"""What one requirement accepts in the operation fields it holds."""

from typing import Any

import pytest

from pydantic import ValidationError
from tenspec.types import operations
from tenspec.types.dtypes import DTypeFamily, DTypeRequirement
from tenspec.types.shapes import parse_shape
from tenspec.types.tensor import TensorType


class Unchanged(operations.ArrayTransform[Any]):
    """A transform that returns what it receives."""

    @staticmethod
    def transform(values: Any) -> Any:
        return values


class NotATransform:
    """A caller's class that derives from no operation interface."""


@pytest.mark.parametrize(
    "declared",
    [NotATransform, Unchanged()],
    ids=["a class of no interface", "an instance rather than a class"],
)
def test_transform_refused_when_the_field_value_is_malformed(declared: Any) -> None:
    # Act and Assert  The ancestry rule reads a method resolution order, so a value that
    # has none is refused before it is asked for one.
    with pytest.raises(ValidationError, match="is not a transform"):
        TensorType(
            shape=parse_shape("rows"),
            dtype=DTypeRequirement(permitted=DTypeFamily.FLOATING),
            transforms=(declared,),
        )
