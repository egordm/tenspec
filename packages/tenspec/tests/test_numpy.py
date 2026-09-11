"""The NumPy facts a Torch caller never meets."""

from typing import Any, TypeVar
from typing import Literal as Shape

import numpy as np
import pytest
from numpy._typing import _64Bit
from numpy.testing import assert_array_equal
from numpy.typing import NDArray

from pydantic import ValidationError
from tenspec import Contiguous, Materialized, NonEmpty, validate
from tenspec.errors import AnnotationError, TensorMismatch
from tenspec.numpy import (
    NUMPY_ARRAYS,
    ArrayProperty,
    CopyReadOnly,
    Float,
    ReadOnly,
    scalar_formats,
)
from tenspec.numpy import NonEmpty as NumpyNonEmpty
from tenspec.torch import CPU
from tenspec.torch import Float as TorchFloat
from tenspec.types.dtypes import DTypeId


def test_array_accepted_when_properties_match() -> None:
    # Arrange
    values = np.ones((2, 3))

    # Act
    accepted = validate(values, Float[Shape["rows cols"], NonEmpty, Contiguous, Materialized])

    # Assert
    assert accepted is values


def test_array_rejected_when_readonly_required_but_writeable() -> None:
    # Act and Assert
    with pytest.raises(ValidationError, match="ReadOnly"):
        validate(np.ones(3), Float[Shape["rows"], ReadOnly])


def test_array_unchanged_when_validation_passes() -> None:
    # Arrange
    values = np.asfortranarray(np.ones((2, 3), dtype=np.float32))
    values.flags.writeable = False

    # Act
    accepted = validate(values, Float[Shape["rows cols"], ReadOnly])

    # Assert
    assert accepted is values
    assert accepted.dtype == np.float32
    assert accepted.flags.f_contiguous
    assert not accepted.flags.writeable


def test_declaration_rejected_when_property_belongs_to_numpy_alone() -> None:
    # Act and Assert  ReadOnly is a NumPy property, so a Torch declaration cannot run it.
    with pytest.raises(AnnotationError, match="not a Torch operation"):
        validate(np.ones(3), TorchFloat[Shape["rows"], ReadOnly])


def test_declaration_rejected_when_device_specified() -> None:
    # Act and Assert
    with pytest.raises(AnnotationError, match="places no array"):
        validate(np.ones(3), Float[Shape["rows"], CPU])


class Precision64(ArrayProperty[np.floating[_64Bit]]):
    """A property whose family carries a precision this release does not read."""

    @staticmethod
    def validate(values: NDArray[np.floating[_64Bit]]) -> None:
        """Accept nothing, because resolution refuses this class before any value."""


Width = TypeVar("Width")
"""A precision the declaration never resolves."""


class AnyFloating(ArrayProperty[np.floating[Any]]):
    """A property on the accepted family form."""

    @staticmethod
    def validate(values: NDArray[np.floating[Any]]) -> None:
        """Accept every floating array."""


@pytest.mark.parametrize(
    "scalar",
    [
        np.floating[_64Bit],
        np.floating[Width],  # ty: ignore[invalid-type-arguments]  the case under test.
        np.complexfloating[Any, _64Bit],
    ],
    ids=["a fixed precision", "an unresolved precision", "one restricted of two"],
)
def test_scalar_formats_refused_when_family_carries_a_precision(scalar: Any) -> None:
    # Act and Assert  Widening any of these would hand float32 to a float64 check.
    with pytest.raises(AnnotationError, match="names a precision"):
        scalar_formats(scalar)


def test_property_refused_when_its_declared_family_carries_a_precision() -> None:
    # Act and Assert
    with pytest.raises(AnnotationError, match="names a precision"):
        NUMPY_ARRAYS.operation_formats(Precision64)


def test_family_with_any_covers_every_floating_format() -> None:
    # Act
    accepted = NUMPY_ARRAYS.operation_formats(AnyFloating)

    # Assert
    assert accepted == {DTypeId.FLOAT16, DTypeId.FLOAT32, DTypeId.FLOAT64}


def test_complex_family_covered_when_both_precisions_are_any() -> None:
    # Act  np.complexfloating takes two precision parameters, not one.
    accepted = scalar_formats(np.complexfloating[Any, Any])

    # Assert
    assert accepted == {DTypeId.COMPLEX64, DTypeId.COMPLEX128}


def test_copy_owns_separate_storage_and_leaves_input_unchanged() -> None:
    # Arrange
    source = np.ones((2, 2))

    # Act  test_caller_example covers the public declaration, so this calls the transform.
    stored = CopyReadOnly.transform(source)

    # Assert
    assert_array_equal(stored, source)
    assert not np.shares_memory(source, stored)
    assert not stored.flags.writeable
    assert source.flags.writeable


@pytest.mark.parametrize(
    ("extent", "observed"),
    [
        ((0, 3), "shape (0, 3), with no elements at axis 0"),
        ((2, 0, 0), "shape (2, 0, 0), with no elements at axis 1, axis 2"),
    ],
    ids=["one empty axis", "two empty axes"],
)
def test_refusal_names_the_empty_axes_when_array_holds_no_elements(
    extent: tuple[int, ...], observed: str
) -> None:
    # Arrange
    values = np.zeros(extent)

    # Act and Assert
    with pytest.raises(TensorMismatch) as refusal:
        NumpyNonEmpty.validate(values)

    # Assert
    assert refusal.value.requirement == "NonEmpty"
    assert refusal.value.observed == observed
