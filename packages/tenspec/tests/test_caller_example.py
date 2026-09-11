"""One caller, written three ways: a checked call, a model, and a standalone check.

This is the readable example of the first release. Everything a caller must know appears
in the declarations, and every test here uses the surface a caller uses.
"""

from typing import Any
from typing import Literal as Shape

import numpy as np
import pytest
import torch
from numpy.testing import assert_array_equal
from numpy.typing import NDArray

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tenspec import (
    AnnotationError,
    Finite,
    Floating,
    NonEmpty,
    TensorContracts,
    checked,
    validate,
)
from tenspec.numpy import ArrayProperty, CopyReadOnly, Float, Float64, Int
from tenspec.torch import Float as TorchFloat
from tenspec.types.dtypes import DType, DTypeId


class Float64Only(DType):
    """The bound of a shared variable that permits float64 alone."""

    permitted = DTypeId.FLOAT64


RELATED_OPERANDS = tuple[
    TorchFloat[Shape["heads keys"]], TorchFloat[Shape["kv_heads keys head_dim"]]
]
"""Two operands of one attention step, which must agree on the key axis."""


@checked
def weigh[Scalar: Floating](
    values: Float[Shape["rows cols"], Scalar],
    weights: Float[Shape["cols"], Scalar],
) -> Float[Shape["rows cols"], Scalar]:
    """Weigh every column of `values`, and keep the shape and the scalar format.

    Returns:
        The weighed array, on the same axes and in the same scalar format.
    """
    return values * weights


class NonPositive(ArrayProperty[np.floating[Any]]):
    """The caller's own check: no entry of the array is above zero.

    The scalar domain says the method reads floating values. A declaration must
    guarantee that domain, and `Finite` runs before this check, so it reads no NaN.
    """

    @staticmethod
    def validate(values: NDArray[np.floating[Any]]) -> None:
        if bool((values > 0).any()):
            raise ValueError("no entry may be positive")


type LogProbs = Float64[Shape["masks tokens"], NonEmpty, Finite, NonPositive]
"""Log probabilities: never empty, always finite, and never above zero."""


type StoredLogProbs = Float64[Shape["masks tokens"], CopyReadOnly, NonEmpty, Finite, NonPositive]
"""The same log probabilities, stored as a read-only copy the caller cannot write."""


@checked
def logit_targets(logprobs: LogProbs) -> NDArray[np.float64]:
    """Read log probabilities the caller's own check has already accepted."""
    return logprobs


@checked
def scale64[Exact: Float64Only](values: Float[Shape["rows"], Exact]) -> NDArray[np.floating[Any]]:
    """Take a floating array that the shared variable narrows to float64."""
    return values


@checked
def store_logprobs(logprobs: LogProbs) -> StoredLogProbs:
    """Accept log probabilities, and return them as the stored read-only copy."""
    return logprobs


class Measurements(BaseModel):
    """The caller's own base. Tenspec adds no base and changes no setting here."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class HeadMeasurements(TensorContracts, Measurements):
    """Two blocks that must share their axes, stated once per field."""

    source: Float64[Shape["steps heads"], Finite] = Field(
        description="Source measurements per step and head."
    )
    background: Float64[Shape["steps heads"], Finite] = Field(
        description="Background measurements on the same axes."
    )


class StoredMeasurements(TensorContracts, Measurements):
    """One field whose declaration stores a sealed copy of what it received."""

    logprobs: StoredLogProbs = Field(description="Log probabilities, stored read-only.")


def test_shape_and_dtype_preserved_when_call_valid() -> None:
    # Arrange
    values = np.ones((2, 3), dtype=np.float32)
    weights = np.full((3,), 2.0, dtype=np.float32)

    # Act
    weighed = weigh(values, weights)

    # Assert
    assert weighed.shape == (2, 3)
    assert weighed.dtype == np.float32


def test_call_rejected_when_shared_axis_differs() -> None:
    # Arrange
    values = np.ones((2, 3), dtype=np.float32)

    # Act and Assert
    with pytest.raises(ValidationError, match="cols=3"):
        weigh(values, np.ones((4,), dtype=np.float32))


def test_call_rejected_when_shared_dtype_differs() -> None:
    # Arrange
    values = np.ones((2, 3), dtype=np.float32)

    # Act and Assert
    with pytest.raises(ValidationError, match="float32"):
        weigh(values, np.ones((3,), dtype=np.float64))


def test_return_rejected_when_extent_differs() -> None:
    # Arrange
    @checked
    def widen(values: Float[Shape["rows"]]) -> Float[Shape["rows"]]:
        return np.concatenate([values, values])

    # Act and Assert
    with pytest.raises(ValidationError, match="rows=2"):
        widen(np.ones(2, dtype=np.float32))


def test_array_identity_preserved_when_call_checked() -> None:
    # Arrange
    values = np.ones((2, 3), dtype=np.float32)
    seen: list[NDArray[np.float32]] = []

    @checked
    def record(operand: Float[Shape["rows cols"]]) -> Float[Shape["rows cols"]]:
        seen.append(operand)
        return operand

    # Act
    returned = record(values)

    # Assert
    assert len(seen) == 1
    assert seen[0] is values
    assert returned is values


@pytest.mark.parametrize("rows", [2, 9], ids=["two rows", "nine rows"])
def test_fresh_extents_when_callable_reused(rows: int) -> None:
    # Arrange
    weights = np.ones((3,), dtype=np.float32)
    weigh(np.ones((5, 3), dtype=np.float32), weights)
    values = np.ones((rows, 3), dtype=np.float32)

    # Act
    weighed = weigh(values, weights)

    # Assert
    assert weighed.shape == (rows, 3)


def test_field_identity_preserved_when_shapes_match() -> None:
    # Arrange
    source = np.zeros((2, 4))
    background = np.ones((2, 4))

    # Act
    measurements = HeadMeasurements(source=source, background=background)

    # Assert
    assert measurements.source is source
    assert measurements.background is background


def test_model_rejected_when_shared_axis_differs() -> None:
    # Act and Assert
    with pytest.raises(ValidationError, match="steps=2"):
        HeadMeasurements(source=np.zeros((2, 4)), background=np.zeros((5, 4)))


def test_model_rejected_when_field_nonfinite() -> None:
    # Act and Assert
    with pytest.raises(ValidationError, match="Finite"):
        HeadMeasurements(source=np.full((2, 4), np.nan), background=np.zeros((2, 4)))


def test_declarations_retained_when_model_prepared() -> None:
    # Arrange
    declared = Float64[Shape["steps heads"], Finite]

    # Act
    class Blocks(TensorContracts, Measurements):
        source: Float64[Shape["steps heads"], Finite]
        background: Float64[Shape["steps heads"], Finite]

    # Assert
    assert Blocks.__tensor_declarations__["source"] == declared
    assert Blocks.__tensor_declarations__["background"] == declared


def test_operand_identity_preserved_when_shapes_match() -> None:
    # Arrange
    attention = torch.ones(4, 7)
    values = torch.ones(2, 7, 8)

    # Act
    accepted = validate((attention, values), RELATED_OPERANDS)

    # Assert
    assert accepted[0] is attention
    assert accepted[1] is values


def test_operands_rejected_when_shared_axis_differs() -> None:
    # Act and Assert
    with pytest.raises(ValidationError, match="keys=7"):
        validate((torch.ones(4, 7), torch.ones(2, 5, 8)), RELATED_OPERANDS)


def test_nan_accepted_when_finite_not_required() -> None:
    # Arrange
    values: NDArray[np.float64] = np.array([1.0, np.nan])

    # Act
    accepted = validate(values, Float64[Shape["time"]])

    # Assert
    assert accepted is values


def test_call_accepted_when_caller_check_holds() -> None:
    # Arrange  The first entry is log 1, which is zero. The rule refuses an entry above
    # zero, so zero itself must pass.
    logprobs = np.log(np.array([[1.0, 0.25], [0.75, 0.5]]))

    # Act
    targets = logit_targets(logprobs)

    # Assert
    assert targets is logprobs


def test_call_rejected_when_caller_check_refuses() -> None:
    # Arrange
    positive = np.array([[0.5, -1.0], [-2.0, -0.5]])

    # Act and Assert
    with pytest.raises(ValidationError, match="no entry may be positive"):
        logit_targets(positive)


def test_caller_check_reached_when_declaration_is_standalone() -> None:
    # Arrange
    logprobs = np.log(np.array([[0.5, 0.25]]))

    # Act
    accepted = validate(logprobs, LogProbs)

    # Assert
    assert accepted is logprobs


def test_declaration_rejected_when_caller_check_needs_a_wider_scalar() -> None:
    # Act and Assert  NonPositive reads floating values, so an integer alias cannot run it.
    with pytest.raises(AnnotationError, match="does not guarantee the scalar domain"):
        validate(np.ones(2, dtype=np.int64), Int[Shape["rows"], NonPositive])


def test_call_rejected_when_narrowed_variable_bound_differs() -> None:
    # Act and Assert  The Float alias permits float32, and the bound of Exact does not.
    with pytest.raises(ValidationError, match="dtype floating that Exact narrows to float64"):
        scale64(np.ones(2, dtype=np.float32))


def test_call_accepted_when_narrowed_variable_bound_matches() -> None:
    # Arrange
    values = np.ones(2, dtype=np.float64)

    # Act
    accepted = scale64(values)

    # Assert
    assert accepted is values


def test_stored_copy_returned_when_declaration_names_a_transform() -> None:
    # Arrange
    source = np.log(np.array([[0.5, 0.25], [0.75, 0.5]]))

    # Act
    stored = validate(source, StoredLogProbs)

    # Assert  The caller keeps its own writeable array, and gets a sealed copy back.
    assert_array_equal(stored, source)
    assert not np.shares_memory(source, stored)
    assert not stored.flags.writeable
    assert source.flags.writeable


def test_caller_check_still_refuses_after_the_copy() -> None:
    # Arrange
    positive = np.array([[0.5, -1.0]])

    # Act and Assert  The copy runs first, and NonPositive then reads its values.
    with pytest.raises(ValidationError, match="no entry may be positive"):
        validate(positive, StoredLogProbs)

    # Assert
    assert positive.flags.writeable


def test_field_sealed_when_its_declaration_names_a_transform() -> None:
    # Arrange
    source = np.log(np.array([[0.5, 0.25]]))

    # Act
    measurements = StoredMeasurements(logprobs=source)

    # Assert  The model stores the copy, and the caller keeps its own writeable array.
    assert_array_equal(measurements.logprobs, source)
    assert not measurements.logprobs.flags.writeable
    assert source.flags.writeable


def test_return_sealed_when_the_return_declaration_names_a_transform() -> None:
    # Arrange
    source = np.log(np.array([[0.5, 0.25]]))

    # Act
    stored = store_logprobs(source)

    # Assert  The return is its own boundary, so the transform runs there too.
    assert_array_equal(stored, source)
    assert not stored.flags.writeable
    assert source.flags.writeable
