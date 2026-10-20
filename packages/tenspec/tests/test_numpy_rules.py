"""NumPy value rules through checked function boundaries."""

from collections.abc import Callable
from typing import Any
from typing import Literal as Shape

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from numpy.typing import NDArray

from pydantic import ValidationError
from tenspec import Finite, checked
from tenspec.errors import TensorMismatch
from tenspec.numpy import (
    Bool,
    CanonicalBool,
    CopyReadOnly,
    ExactNDArray,
    Float,
    NonNegative,
    NonPositive,
    ReadOnly,
)


@checked
def exact_array(values: Float[Shape["*axes"], ExactNDArray]) -> NDArray[np.floating[Any]]:
    return values


@checked
def non_positive(values: Float[Shape["*axes"], NonPositive]) -> NDArray[np.floating[Any]]:
    return values


@checked
def non_negative(values: Float[Shape["*axes"], NonNegative]) -> NDArray[np.floating[Any]]:
    return values


@checked
def finite_non_positive(
    values: Float[Shape["*axes"], NonPositive, Finite],
) -> NDArray[np.floating[Any]]:
    return values


@checked
def finite_non_negative(
    values: Float[Shape["*axes"], NonNegative, Finite],
) -> NDArray[np.floating[Any]]:
    return values


@checked
def canonical_bool(values: Bool[Shape["*axes"], CanonicalBool]) -> NDArray[np.bool_]:
    return values


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_exact_array_preserved_when_native_class_supplied(
    dtype: type[np.floating[Any]],
) -> None:
    # Arrange
    values = np.ones((2, 3), dtype=dtype)

    # Act
    accepted = exact_array(values)

    # Assert
    assert accepted is values


@pytest.mark.parametrize("array_class", [np.ma.MaskedArray, np.matrix])
def test_exact_array_refused_when_subclass_supplied(array_class: type[np.ndarray]) -> None:
    # Arrange
    values = np.ones((2, 3)).view(array_class)

    # Act & Assert
    with pytest.raises(ValidationError, match="ExactNDArray") as refusal:
        exact_array(values)

    # Assert
    original = refusal.value.errors()[0]["ctx"]["error"]
    assert isinstance(original, TensorMismatch)
    assert original.requirement == "ExactNDArray"
    assert original.observed == array_class.__name__
    assert type(values) is array_class


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize(
    ("accept", "sign"),
    [(non_positive, 1), (non_negative, -1)],
    ids=["NonPositive", "NonNegative"],
)
@pytest.mark.parametrize(
    "entries", [[-np.inf, -1.0, -0.0, 0.0], [], -0.0], ids=["entries", "empty", "scalar"]
)
def test_sign_array_preserved_when_no_entry_has_forbidden_sign(
    dtype: type[np.floating[Any]],
    accept: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
    sign: int,
    entries: list[float] | float,
) -> None:
    # Arrange
    values = np.array(entries, dtype=dtype)
    values *= sign

    # Act
    accepted = accept(values)

    # Assert
    assert accepted is values


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize(
    ("accept", "sign", "requirement", "observed"),
    [
        (non_positive, 1, "NonPositive", "an entry is positive"),
        (non_negative, -1, "NonNegative", "an entry is negative"),
    ],
    ids=["NonPositive", "NonNegative"],
)
def test_sign_array_refused_when_an_entry_has_forbidden_sign(
    dtype: type[np.floating[Any]],
    accept: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
    sign: int,
    requirement: str,
    observed: str,
) -> None:
    # Arrange
    values = np.array([0.0, np.finfo(dtype).smallest_subnormal], dtype=dtype) * sign
    unchanged = values.copy()

    # Act & Assert
    with pytest.raises(ValidationError, match=observed) as refusal:
        accept(values)

    # Assert
    original = refusal.value.errors()[0]["ctx"]["error"]
    assert isinstance(original, TensorMismatch)
    assert original.requirement == requirement
    assert original.observed == observed
    assert_array_equal(values, unchanged)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("accept", [non_positive, non_negative])
def test_sign_array_preserved_when_entry_is_nan(
    dtype: type[np.floating[Any]],
    accept: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
) -> None:
    # Arrange
    values = np.array([np.nan], dtype=dtype)

    # Act
    accepted = accept(values)

    # Assert
    assert accepted is values
    assert bool(np.isnan(accepted).all())


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("accept", [finite_non_positive, finite_non_negative])
def test_nan_refused_when_sign_rule_composes_with_finite(
    dtype: type[np.floating[Any]],
    accept: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
) -> None:
    # Arrange
    values = np.array([np.nan], dtype=dtype)

    # Act & Assert
    with pytest.raises(ValidationError, match="Finite") as refusal:
        accept(values)

    # Assert
    original = refusal.value.errors()[0]["ctx"]["error"]
    assert isinstance(original, TensorMismatch)
    assert original.requirement == "Finite"


@pytest.mark.parametrize(("raw", "expected"), [([0, 1, 1], [0, 1, 1]), ([0, 2, 255], [0, 1, 1])])
@pytest.mark.parametrize("writeable", [False, True])
def test_boolean_bytes_canonicalized_when_checked(
    raw: list[int], expected: list[int], *, writeable: bool
) -> None:
    # Arrange
    values = np.array(raw, dtype=np.uint8).view(np.bool_)
    values.setflags(write=writeable)

    # Act
    accepted = canonical_bool(values)

    # Assert
    assert_array_equal(accepted.view(np.uint8), expected)
    assert_array_equal(values.view(np.uint8), raw)
    assert_array_equal(accepted, values)
    assert not np.shares_memory(accepted, values)
    assert accepted.shape == values.shape
    assert accepted.dtype == values.dtype
    assert type(accepted) is type(values)
    assert values.flags.writeable == writeable


def test_boolean_copy_readonly_when_canonicalization_precedes_storage_transform() -> None:
    # Arrange
    values = np.array([0, 2], dtype=np.uint8).view(np.bool_)

    @checked
    def store(
        supplied: Bool[Shape["rows"], CanonicalBool, CopyReadOnly, ReadOnly],
    ) -> NDArray[np.bool_]:
        return supplied

    # Act
    stored = store(values)

    # Assert
    assert_array_equal(stored.view(np.uint8), [0, 1])
    assert_array_equal(values.view(np.uint8), [0, 2])
    assert not stored.flags.writeable
    assert values.flags.writeable
    assert not np.shares_memory(stored, values)


def test_boolean_transform_refused_when_input_is_integer_array() -> None:
    # Arrange
    values: Any = np.array([0, 2], dtype=np.uint8)

    # Act & Assert
    with pytest.raises(ValidationError, match="bool") as refusal:
        canonical_bool(values)

    # Assert
    assert isinstance(refusal.value.errors()[0]["ctx"]["error"], TensorMismatch)
    assert_array_equal(values, [0, 2])
