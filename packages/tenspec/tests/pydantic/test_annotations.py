"""What preparation accepts, and how an unsupported declaration fails."""

from abc import ABCMeta
from typing import Annotated, TypeVar, get_args
from typing import Literal as Shape

import numpy as np
import pytest
from numpy.typing import NDArray

from pydantic import AfterValidator, BeforeValidator, TypeAdapter, ValidationError
from tenspec import AnnotationError, Finite, Floating, NonEmpty, checked, validate
from tenspec.errors import TransformContractError
from tenspec.numpy import (
    ArrayProperty,
    ArrayTransform,
    CopyReadOnly,
    Float,
    Float32,
    Float64,
    Int,
)
from tenspec.pydantic.annotations import prepare_annotation

Scalar = TypeVar("Scalar", bound=Floating)
Other = TypeVar("Other", bound=Floating)
Unbounded = TypeVar("Unbounded")


@checked
def batch_size(
    values: Float32[Shape["batch time"]],
    mask: Float32[Shape["batch time"]] | None = None,
) -> int:
    """Read the batch extent from a required or optional array.

    Returns:
        The batch extent shared by the supplied arrays.
    """
    return values.shape[0] if mask is None else mask.shape[0]


def test_schema_rejected_when_declaration_unprepared() -> None:
    # Act and Assert
    with pytest.raises(AnnotationError, match="no entry point prepared"):
        TypeAdapter(Float[Shape["rows"]], config={"arbitrary_types_allowed": True})


def test_annotation_preserved_when_no_tensor_declared() -> None:
    # Act
    prepared = prepare_annotation(int)

    # Assert
    assert prepared is int


def test_declaration_unchanged_when_prepared() -> None:
    # Arrange
    declaration = Float[Shape["rows"], Finite]
    written = get_args(declaration)

    # Act
    prepared = prepare_annotation(declaration)

    # Assert
    assert prepared is not declaration
    assert get_args(declaration) == written


def test_declaration_rejected_when_shape_missing() -> None:
    # Act and Assert
    with pytest.raises(AnnotationError, match="needs a shape"):
        validate(np.ones(3), Float)


def test_declaration_rejected_when_first_argument_not_shape() -> None:
    # Act and Assert
    with pytest.raises(AnnotationError, match="must be Shape"):
        validate(np.ones(3), Float[Finite])


def test_declaration_rejected_when_dtype_variables_conflict() -> None:
    # Act and Assert
    with pytest.raises(AnnotationError, match="at most one dtype variable"):
        validate(np.ones(3), Float[Shape["rows"], Scalar, Other])


def test_declaration_rejected_when_dtype_bound_excludes_alias() -> None:
    # Act and Assert
    with pytest.raises(AnnotationError, match="no scalar format"):
        validate(np.ones(2, dtype=np.int32), Int[Shape["rows"], Scalar])


def test_declaration_rejected_when_variable_unbounded() -> None:
    # Act and Assert
    with pytest.raises(AnnotationError, match="must be bound to DType"):
        validate(np.ones(2), Float[Shape["rows"], Unbounded])


def test_decorator_rejected_when_function_async() -> None:
    # Act and Assert
    with pytest.raises(AnnotationError, match="async"):

        @checked
        async def measure(values: Float[Shape["rows"]]) -> None: ...


def test_none_default_used_when_optional_array_omitted() -> None:
    # Arrange
    values = np.ones((2, 3), dtype=np.float32)

    # Act
    result = batch_size(values)

    # Assert
    assert result == 2


def test_optional_array_accepted_when_shapes_match() -> None:
    # Arrange
    values = np.ones((2, 3), dtype=np.float32)

    # Act
    result = batch_size(values, np.zeros((2, 3), dtype=np.float32))

    # Assert
    assert result == 2


def test_optional_array_rejected_when_shapes_differ() -> None:
    # Arrange
    values = np.ones((2, 3), dtype=np.float32)

    # Act and Assert
    with pytest.raises(ValidationError, match="batch=2"):
        batch_size(values, np.zeros((5, 3), dtype=np.float32))


def test_union_accepted_when_second_tensor_alternative_matches() -> None:
    # Arrange
    values = np.ones(2)

    # Act
    result = validate(
        (values, "second"),
        tuple[Float[Shape["rows"]], Shape["first"]] | tuple[Float[Shape["cols"]], Shape["second"]],
    )

    # Assert
    assert result[0] is values
    assert result[1] == "second"


def test_union_accepted_when_wrapper_prepares_array() -> None:
    # Act
    result = validate(
        [1.0, 2.0],
        Annotated[Float[Shape["rows"]] | Float[Shape["cols"], Finite], BeforeValidator(np.asarray)],
    )

    # Assert
    np.testing.assert_array_equal(result, [1.0, 2.0])


def test_union_rejected_when_tensor_mixed_with_ordinary_type() -> None:
    # Act and Assert
    with pytest.raises(AnnotationError, match="ordinary non-tensor"):
        validate(np.ones(2), Float[Shape["rows"]] | int)


def test_union_accepted_when_no_tensor_declared() -> None:
    # Act
    accepted = validate(3, int | str)

    # Assert
    assert accepted == 3


def test_conversion_precedes_validation_when_before_validator_used() -> None:
    # Arrange
    prepared = Annotated[
        Float32[Shape["rows"]], BeforeValidator(lambda given: np.asarray(given, np.float32))
    ]

    # Act
    accepted = validate([1.0, 2.0, 3.0], prepared)

    # Assert
    assert accepted.dtype == np.float32
    assert accepted.shape == (3,)


def test_array_preserved_when_after_validator_observes() -> None:
    # Arrange
    seen: list[NDArray[np.float32]] = []

    def record(given: NDArray[np.float32]) -> NDArray[np.float32]:
        seen.append(given)
        return given

    prepared = Annotated[Float32[Shape["rows"]], AfterValidator(record)]
    values = np.ones(2, dtype=np.float32)

    # Act
    accepted = validate(values, prepared)

    # Assert
    assert len(seen) == 1
    assert seen[0] is values
    assert accepted is values


def test_declaration_rejected_when_transform_follows_a_property() -> None:
    # Act and Assert  Preparation splits the two sequences, so the order must be right.
    with pytest.raises(AnnotationError, match="before its properties"):
        validate(np.ones(2), Float64[Shape["rows"], NonEmpty, CopyReadOnly])


class Unhashable(ABCMeta):
    """A metaclass that leaves every class it makes without a hash."""

    __hash__ = None


def test_declaration_accepted_when_its_property_class_carries_no_hash() -> None:
    # Arrange  Sorting the subscript must not need a hash either, or the uncached
    # resolution route would be unreachable from a public declaration.
    class NoHash(ArrayProperty[np.generic], metaclass=Unhashable):
        """A caller's property whose class carries no hash."""

        @staticmethod
        def validate(values: NDArray[np.generic]) -> None:
            """Accept every array."""

    values = np.ones(2)

    # Act
    accepted = validate(values, Float64[Shape["rows"], NoHash])

    # Assert
    assert accepted is values


def test_declaration_accepted_when_its_transform_class_carries_no_hash() -> None:
    # Arrange  The requirement holds transforms in a type-of-abstract-class field, whose
    # own subclass check hashes the candidate. Only the public route reaches that field.
    class NoHashCopy(ArrayTransform[np.generic], metaclass=Unhashable):
        """A caller's transform whose class carries no hash."""

        @staticmethod
        def transform(values: NDArray[np.generic]) -> NDArray[np.generic]:
            """Return a copy of the values.

            Returns:
                The copy.
            """
            return values.copy()

    values = np.ones(2)

    # Act
    accepted = validate(values, Float64[Shape["rows"], NoHashCopy])

    # Assert
    assert accepted is not values
    assert np.array_equal(accepted, values)


class Reshape(ArrayTransform[np.generic]):
    """A transform that breaks its contract by changing the shape of its result."""

    @staticmethod
    def transform(values: NDArray[np.generic]) -> NDArray[np.generic]:
        return values.reshape(-1)


def test_outer_validators_keep_their_order_when_a_transform_runs() -> None:
    # Arrange
    seen: list[NDArray[np.float64]] = []

    def record(given: NDArray[np.float64]) -> NDArray[np.float64]:
        seen.append(given)
        return given

    prepared = Annotated[
        Float64[Shape["rows"], CopyReadOnly, Finite],
        BeforeValidator(np.asarray),
        AfterValidator(record),
    ]

    # Act
    accepted = validate([1.0, 2.0], prepared)

    # Assert  The list converts first, Tenspec copies and checks, and the copy reaches
    # the outer validator. Tenspec reorders neither validator.
    assert len(seen) == 1
    assert not seen[0].flags.writeable
    assert accepted is seen[0]


def test_transform_contract_error_raised_when_a_transform_breaks_it() -> None:
    # Act and Assert  A broken structural guarantee is not a value mismatch, so it keeps
    # its own type through the Pydantic boundary rather than arrive as a ValidationError.
    with pytest.raises(TransformContractError, match="shape"):
        validate(np.ones((2, 2)), Float64[Shape["rows cols"], Reshape])
