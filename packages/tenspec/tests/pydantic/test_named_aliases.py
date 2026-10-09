"""What a caller's own alias keeps when it names a whole declaration."""

from typing import Annotated
from typing import Literal as Shape

import numpy as np
import pytest
from numpy.typing import NDArray

from pydantic import AfterValidator, BaseModel, ConfigDict, ValidationError
from tenspec import AnnotationError, Finite, NonEmpty, TensorContracts, checked, validate
from tenspec.numpy import Float64
from tenspec.pydantic.annotations import prepare_annotation

type Scores = Float64[Shape["layers heads"], Finite, NonEmpty]
type StoredScores = Scores
type Row = Float64[Shape["rows"]]
type MaybeRow = Row | None
type RowPair = tuple[Row, Row]
type OtherRow = Float64[Shape["cols"]]
type Label = str
type Chain = tuple[Row, Chain]
type Nest = tuple[int, Nest]

WRONG_RANK = np.ones((2, 2))
"""A default of rank two, which a declaration of one named axis refuses."""


class Base(BaseModel):
    """The caller's own base, with its own configuration."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class Block(TensorContracts, Base):
    """Two fields that name the same alias, so they share its axis."""

    left: Row
    right: Row


@checked
def heads(scores: Scores) -> int:
    """Read the head count from a declaration the caller named.

    Returns:
        The head count.
    """
    return scores.shape[1]


@checked
def widths(values: Row = WRONG_RANK) -> int:
    """Take a named declaration whose default holds the wrong rank.

    Returns:
        The row count.
    """
    return values.shape[0]


def test_argument_accepted_when_alias_matches() -> None:
    # Arrange
    scores = np.ones((2, 3))

    # Act
    count = heads(scores)

    # Assert
    assert count == 3


def test_argument_refused_when_alias_rank_differs() -> None:
    # Arrange
    scores = np.ones(3)

    # Act and Assert
    with pytest.raises(ValidationError, match="expected rank 2"):
        heads(scores)


def test_default_refused_when_alias_declares_parameter() -> None:
    # Act and Assert
    with pytest.raises(ValidationError, match="expected rank 1"):
        widths()


def test_fields_related_when_alias_names_shared_axis() -> None:
    # Act and Assert
    with pytest.raises(ValidationError, match="rows=2"):
        Block(left=np.ones(2), right=np.ones(5))


def test_value_refused_when_alias_names_declaration() -> None:
    # Act and Assert
    with pytest.raises(ValidationError, match="expected rank 2"):
        validate(np.ones((4, 5, 6)), Scores)


def test_value_accepted_when_alias_names_another_alias() -> None:
    # Arrange
    scores = np.ones((2, 3))

    # Act
    accepted = validate(scores, StoredScores)

    # Assert
    assert accepted is scores


def test_none_accepted_when_alias_wraps_optional_declaration() -> None:
    # Act
    accepted = validate(None, MaybeRow)

    # Assert
    assert accepted is None


def test_items_related_when_alias_wraps_tuple() -> None:
    # Act and Assert
    with pytest.raises(ValidationError, match="rows=2"):
        validate((np.ones(2), np.ones(5)), RowPair)


def test_declaration_checked_before_caller_validator_when_alias_wrapped() -> None:
    # Arrange
    seen: list[NDArray[np.float64]] = []

    def record(given: NDArray[np.float64]) -> NDArray[np.float64]:
        seen.append(given)
        return given

    type Wrapped = Annotated[Row, AfterValidator(record)]

    # Act and Assert
    with pytest.raises(ValidationError, match="float64"):
        validate(np.ones(2, dtype=np.float32), Wrapped)

    # Assert
    assert seen == []


def test_caller_validator_runs_when_alias_accepts_array() -> None:
    # Arrange
    seen: list[NDArray[np.float64]] = []

    def record(given: NDArray[np.float64]) -> NDArray[np.float64]:
        seen.append(given)
        return given

    type Wrapped = Annotated[Row, AfterValidator(record)]
    values = np.ones(2)

    # Act
    accepted = validate(values, Wrapped)

    # Assert
    assert len(seen) == 1
    assert seen[0] is values
    assert accepted is values


def test_union_accepted_when_aliases_name_alternatives() -> None:
    # Arrange
    values = np.ones(2)

    # Act
    result = validate(values, Row | OtherRow)

    # Assert
    assert result is values


def test_alias_preserved_when_no_tensor_declared() -> None:
    # Act
    prepared = prepare_annotation(Label)

    # Assert
    assert prepared is Label


def test_alias_preserved_when_ordinary_alias_contains_itself() -> None:
    # Act
    prepared = prepare_annotation(Nest)

    # Assert
    assert prepared is Nest


def test_alias_refused_when_tensor_alias_contains_itself() -> None:
    # Act and Assert
    with pytest.raises(AnnotationError, match="contains itself"):
        validate((np.ones(2), ()), Chain)
