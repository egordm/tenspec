"""A caller who writes `from __future__ import annotations` reaches the same checks."""

from __future__ import annotations

from typing import Literal as Shape

import numpy as np
import pytest

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tenspec import TensorContracts, checked
from tenspec.numpy import Float


class Base(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Pair(TensorContracts, Base):
    left: Float[Shape["rows"]]
    right: Float[Shape["rows"]]


@checked
def widen(values: Float[Shape["rows"]]) -> Float[Shape["rows"]]:
    """Return twice the rows the declaration binds, which its own return refuses.

    Returns:
        The concatenated array, which never meets the declared rows.
    """
    return np.concatenate([values, values])


def test_return_rejected_when_extent_differs() -> None:
    # Act and Assert
    with pytest.raises(ValidationError, match="rows=2"):
        widen(np.ones(2))


def test_fields_accepted_when_shapes_match() -> None:
    # Act
    model = Pair(left=np.ones(2), right=np.ones(2))

    # Assert
    assert model.left.shape == (2,)


def test_fields_rejected_when_shapes_differ() -> None:
    # Act and Assert
    with pytest.raises(ValidationError, match="rows=2"):
        Pair(left=np.ones(2), right=np.ones(5))


def test_local_alias_accepted_when_shapes_match() -> None:
    # Arrange
    Block = Float[Shape["rows"]]  # noqa: N806  a local type alias keeps type case.

    class Aliased(TensorContracts, Base):
        left: Block
        right: Block

    # Act
    model = Aliased(left=np.ones(2), right=np.ones(2))

    # Assert
    assert model.right.shape == (2,)


def test_local_alias_rejected_when_shapes_differ() -> None:
    # Arrange
    Block = Float[Shape["rows"]]  # noqa: N806  a local type alias keeps type case.

    class Aliased(TensorContracts, Base):
        left: Block
        right: Block

    # Act and Assert
    with pytest.raises(ValidationError, match="rows=2"):
        Aliased(left=np.ones(2), right=np.ones(5))


def test_forward_alias_accepted_when_shapes_match() -> None:
    # Arrange
    class Deferred(TensorContracts, Base):
        left: Later
        right: Later

    Later = Float[Shape["rows"]]  # noqa: N806  a local type alias keeps type case.
    Deferred.model_rebuild(_types_namespace={"Later": Later})

    # Act
    model = Deferred(left=np.ones(2), right=np.ones(2))

    # Assert
    assert model.left.shape == (2,)


def test_forward_alias_rejected_when_shapes_differ() -> None:
    # Arrange
    class Deferred(TensorContracts, Base):
        left: Later
        right: Later

    Later = Float[Shape["rows"]]  # noqa: N806  a local type alias keeps type case.
    Deferred.model_rebuild(_types_namespace={"Later": Later})

    # Act and Assert
    with pytest.raises(ValidationError, match="rows=2"):
        Deferred(left=np.ones(2), right=np.ones(5))


def test_field_description_preserved_when_model_prepared() -> None:
    # Act
    class Sized(TensorContracts, Base):
        values: Float[Shape["rows"]] = Field(description="the block this model holds")

    # Assert
    assert Sized.model_fields["values"].description == "the block this model holds"
