"""The standalone route: its cache, and the heading a refusal carries."""

from typing import Annotated, Any, TypeVar
from typing import Literal as Shape

import numpy as np
import pytest

from pydantic import ValidationError
from tenspec import Finite, Floating, validate
from tenspec.errors import AnnotationError, TensorMismatch
from tenspec.numpy import CopyReadOnly, Float
from tenspec.pydantic.standalone import hashable, prepared_adapter

type Weights = Float[Shape["rows cols"]]
"""A caller's own alias, which names the whole declaration."""

RANK_MISMATCH = "expected rank 2, but the array has rank 1 (3,)"
"""What a rank-one array reports against a declaration of two axes."""


def test_cache_hit_when_declaration_reused() -> None:
    # Arrange
    prepared_adapter.cache_clear()
    validate(np.ones((2, 3)), Float[Shape["rows cols"]])

    # Act
    validate(np.ones((4, 5)), Float[Shape["rows cols"]])

    # Assert
    assert prepared_adapter.cache_info().hits == 1


@pytest.mark.parametrize("extent", [(2, 3), (40, 50)], ids=["a small array", "a larger array"])
def test_fresh_extents_when_schema_reused(
    extent: tuple[int, int],
) -> None:
    # Arrange
    validate(np.ones((7, 11)), Float[Shape["rows cols"]])
    values = np.ones(extent)

    # Act
    accepted = validate(values, Float[Shape["rows cols"]])

    # Assert
    assert accepted.shape == extent


def test_mismatch_rejected_when_schema_reused() -> None:
    # Arrange
    pair = tuple[Float[Shape["rows"]], Float[Shape["rows"]]]
    validate((np.ones(3), np.ones(3)), pair)

    # Act and Assert
    with pytest.raises(ValidationError, match="rows=3"):
        validate((np.ones(3), np.ones(4)), pair)


def test_changed_values_checked_when_schema_reused() -> None:
    # Arrange
    values = np.ones(3)
    validate(values, Float[Shape["rows"], Finite])
    values[1] = np.inf

    # Act and Assert
    with pytest.raises(ValidationError, match="Finite"):
        validate(values, Float[Shape["rows"], Finite])


def test_separate_schemas_when_variables_share_name() -> None:
    # Arrange
    alias: Any = Float
    variable: Any = TypeVar

    one = alias[Shape["rows"], variable("Scalar", bound=Floating)]
    other = alias[Shape["rows"], variable("Scalar", bound=Floating)]
    first = prepared_adapter(one)

    # Act
    second = prepared_adapter(other)

    # Assert
    assert second is not first


def test_cache_bypassed_when_metadata_unhashable() -> None:
    # Arrange
    declaration = Annotated[Float[Shape["rows"]], ["a note the caller owns"]]
    prepared_adapter.cache_clear()
    assert not hashable(declaration)

    # Act
    accepted = validate(np.ones(3), declaration)

    # Assert
    assert accepted.shape == (3,)
    assert prepared_adapter.cache_info().currsize == 0


def test_constraints_checked_when_metadata_unhashable() -> None:
    # Arrange
    declaration = Annotated[Float[Shape["rows"], Finite], ["a note the caller owns"]]

    # Act and Assert
    with pytest.raises(ValidationError, match="Finite"):
        validate(np.array([np.nan]), declaration)


@pytest.mark.parametrize(
    ("declaration", "title"),
    [
        pytest.param(Float[Shape["rows cols"]], "Float", id="a subscripted Tenspec alias"),
        pytest.param(Weights, "Weights", id="a caller's own alias"),
    ],
)
def test_title_names_the_declaration_when_it_carries_a_name(declaration: Any, title: str) -> None:
    # Act and Assert
    with pytest.raises(ValidationError) as refusal:
        validate(np.ones(3), declaration)

    # Assert
    assert refusal.value.title == title
    assert RANK_MISMATCH in str(refusal.value)


def test_title_falls_back_when_declaration_is_a_tuple() -> None:
    # Arrange
    pair = tuple[Float[Shape["rows"]], Float[Shape["rows"]]]

    # Act and Assert
    with pytest.raises(ValidationError) as refusal:
        validate((np.ones(3), np.ones(4)), pair)

    # Assert
    assert refusal.value.title == "Tenspec"
    entry = refusal.value.errors()[0]
    assert entry["loc"] == (1,)
    original = entry["ctx"]["error"]
    assert isinstance(original, TensorMismatch)
    assert (original.requirement, original.observed) == ("rows=3", "rows=4")
    assert original.bindings == {"rows": 3}


def test_title_applied_when_annotation_unhashable() -> None:
    # Arrange
    declaration = Annotated[Float[Shape["rows cols"]], ["a note the caller owns"]]
    prepared_adapter.cache_clear()

    # Act and Assert
    with pytest.raises(ValidationError) as refusal:
        validate(np.ones(3), declaration)

    # Assert
    assert refusal.value.title == "Tenspec"
    assert RANK_MISMATCH in str(refusal.value)
    assert prepared_adapter.cache_info().currsize == 0


def test_title_unchanged_when_declaration_has_no_tensor() -> None:
    # Act and Assert
    with pytest.raises(ValidationError) as refusal:
        validate("text", int)

    # Assert
    assert refusal.value.title == "int"


def test_refuses_property_outside_the_alias() -> None:
    # Arrange
    declaration = Annotated[Float[Shape["rows"]], Finite]

    # Act & Assert
    with pytest.raises(AnnotationError, match="Finite is a Tenspec property"):
        validate(np.array([np.inf]), declaration)


def test_refuses_transform_outside_the_alias() -> None:
    # Arrange
    declaration = Annotated[Float[Shape["rows"]], CopyReadOnly]

    # Act & Assert
    with pytest.raises(AnnotationError, match="CopyReadOnly is a Tenspec transform"):
        validate(np.ones(2), declaration)
