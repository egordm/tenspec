"""The atomic contract of one array check."""

from dataclasses import FrozenInstanceError
from typing import Any, TypeVar

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from tenspec.errors import (
    AnnotationError,
    BindingError,
    BindingReason,
    TensorMismatch,
    TransformContractError,
)
from tenspec.numpy import NUMPY_ARRAYS, CopyReadOnly, NumpyArrays, ReadOnly
from tenspec.numpy import ArrayTransform as NumpyTransform
from tenspec.runtime.arrays import ArrayBackend
from tenspec.runtime.bindings import Bindings
from tenspec.runtime.validation import ArrayValidator, validate_array
from tenspec.torch import TORCH_ARRAYS
from tenspec.torch import ArrayTransform as TorchTransform
from tenspec.types import operations
from tenspec.types.dtypes import DType, DTypeFamily, DTypeId, DTypeRequirement
from tenspec.types.properties import ArrayProperty, DeviceRequirement, Finite
from tenspec.types.shapes import parse_shape
from tenspec.types.tensor import TensorType


@pytest.fixture
def bindings() -> Bindings:
    """One boundary with nothing committed yet.

    Returns:
        The empty bindings.
    """
    return Bindings()


def tensor_requirement(
    shape: str,
    permitted: DTypeId | DTypeFamily = DTypeFamily.FLOATING,
    properties: tuple[type[ArrayProperty], ...] = (),
    device: DeviceRequirement | None = None,
    transforms: tuple[type[operations.ArrayTransform], ...] = (),
) -> TensorType:
    """One requirement, carrying only the parts a scenario names.

    Returns:
        The requirement.
    """
    return TensorType(
        shape=parse_shape(shape),
        dtype=DTypeRequirement(permitted=permitted),
        properties=properties,
        device=device,
        transforms=transforms,
    )


def test_array_identity_preserved_when_shape_matches(bindings: Bindings) -> None:
    # Arrange
    values = np.ones((2, 3))

    # Act
    accepted = validate_array(
        values,
        tensor_type=tensor_requirement("rows cols"),
        backend=NUMPY_ARRAYS,
        bindings=bindings,
    )

    # Assert
    assert accepted is values
    assert bindings.dimensions == {"rows": 2, "cols": 3}


@pytest.mark.parametrize(
    ("values", "wanted", "message"),
    [
        (
            np.ones((2, 3), dtype=np.int64),
            tensor_requirement("rows cols", DTypeId.FLOAT64),
            "dtype float64",
        ),
        (np.array([1.0, np.inf]), tensor_requirement("rows", properties=(Finite,)), "Finite"),
        (np.ones((2, 3)), tensor_requirement("size size"), "size=2"),
        (np.ones((2, 4)), tensor_requirement("2 3"), "an axis of 3"),
    ],
    ids=["dtype", "property", "repeated name", "fixed extent"],
)
def test_bindings_unchanged_when_array_rejected(
    values: Any, wanted: TensorType, message: str, bindings: Bindings
) -> None:
    # Act and Assert
    with pytest.raises(TensorMismatch, match=message):
        validate_array(values, tensor_type=wanted, backend=NUMPY_ARRAYS, bindings=bindings)

    assert bindings.dimensions == {}


def test_prior_bindings_preserved_when_operand_rejected(bindings: Bindings) -> None:
    # Arrange
    bindings.bind_dimension("rows", 2)

    # Act and Assert
    with pytest.raises(TensorMismatch):
        validate_array(
            np.ones((5, 3)),
            tensor_type=tensor_requirement("batch rows"),
            backend=NUMPY_ARRAYS,
            bindings=bindings,
        )

    assert bindings.dimensions == {"rows": 2}


def test_group_bound_when_rank_sufficient(bindings: Bindings) -> None:
    # Act
    validate_array(
        np.ones((2, 3, 4)),
        tensor_type=tensor_requirement("*batch time"),
        backend=NUMPY_ARRAYS,
        bindings=bindings,
    )

    # Assert
    assert bindings.dimensions == {"batch": (2, 3), "time": 4}


def test_operand_rejected_when_group_differs(bindings: Bindings) -> None:
    # Arrange
    bindings.bind_dimension("batch", (2, 3))

    # Act and Assert
    with pytest.raises(TensorMismatch, match=r"batch=\(2, 3\)"):
        validate_array(
            np.ones((5, 4)),
            tensor_type=tensor_requirement("*batch time"),
            backend=NUMPY_ARRAYS,
            bindings=bindings,
        )


def test_axis_unbound_when_broadcast_extent_one(
    bindings: Bindings,
) -> None:
    # Act
    validate_array(
        np.ones((1, 3)),
        tensor_type=tensor_requirement("#rows cols"),
        backend=NUMPY_ARRAYS,
        bindings=bindings,
    )

    # Assert
    assert bindings.dimensions == {"cols": 3}


def test_expression_accepted_when_extent_matches(
    bindings: Bindings,
) -> None:
    # Arrange
    bindings.bind_dimension("width", 3)
    values = np.ones(6)

    # Act
    accepted = validate_array(
        values,
        tensor_type=tensor_requirement("2*width"),
        backend=NUMPY_ARRAYS,
        bindings=bindings,
    )

    # Assert
    assert accepted is values


def test_expression_rejected_when_extent_differs(bindings: Bindings) -> None:
    # Arrange
    bindings.bind_dimension("width", 3)

    # Act and Assert
    with pytest.raises(TensorMismatch, match=r"2\*width=6"):
        validate_array(
            np.ones(5),
            tensor_type=tensor_requirement("2*width"),
            backend=NUMPY_ARRAYS,
            bindings=bindings,
        )


def test_expression_fails_when_name_unbound(
    bindings: Bindings,
) -> None:
    # Act and Assert
    with pytest.raises(BindingError) as refusal:
        validate_array(
            np.ones(6),
            tensor_type=tensor_requirement("2*width"),
            backend=NUMPY_ARRAYS,
            bindings=bindings,
        )

    assert refusal.value.reason is BindingReason.UNBOUND


@pytest.mark.parametrize(
    ("values", "wanted", "backend", "message"),
    [
        (
            torch.ones(3),
            tensor_requirement("rows", properties=(ReadOnly,)),
            TORCH_ARRAYS,
            "not a Torch operation",
        ),
        (
            np.ones(3),
            tensor_requirement("rows", device=DeviceRequirement(kind="cpu")),
            NUMPY_ARRAYS,
            "places no array",
        ),
    ],
    ids=["a property of another backend", "a device it never reports"],
)
def test_requirement_rejected_when_backend_unsupported(
    values: Any, wanted: TensorType, backend: ArrayBackend[Any], message: str, bindings: Bindings
) -> None:
    # Act and Assert
    with pytest.raises(AnnotationError, match=message):
        validate_array(values, tensor_type=wanted, backend=backend, bindings=bindings)

    assert bindings.dimensions == {}


class Float64Only(DType):
    """The bound of a shared variable that permits float64 alone."""

    permitted = DTypeId.FLOAT64


Impossible = TypeVar("Impossible", bound=Float64Only)


class ObservedArray(np.ndarray):
    """An array whose shape read fails, so a refusal that reads it is visible."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Fail, because nothing may read this array.

        Raises:
            AssertionError: whenever the check reads the array.
        """
        raise AssertionError("the check read the array before refusing the declaration")


def test_declaration_refused_before_the_array_is_read(bindings: Bindings) -> None:
    # Arrange  An integer alias whose variable bound permits float64 alone.
    impossible = TensorType(
        shape=parse_shape("rows"),
        dtype=DTypeRequirement(permitted=DTypeFamily.INTEGER, variable=Impossible),
    )

    # Act and Assert
    with pytest.raises(AnnotationError, match="permits no scalar format"):
        validate_array(
            np.ones(2).view(ObservedArray),
            tensor_type=impossible,
            backend=NUMPY_ARRAYS,
            bindings=bindings,
        )

    # Assert
    assert bindings.dimensions == {}


class Reshape(NumpyTransform[np.generic]):
    """A transform that returns another shape."""

    @staticmethod
    def transform(values: NDArray[np.generic]) -> NDArray[np.generic]:
        return values.reshape(-1)


class ByteSwap(NumpyTransform[np.generic]):
    """A transform that keeps the logical format and changes the byte order."""

    @staticmethod
    def transform(values: NDArray[np.generic]) -> NDArray[np.generic]:
        return values.astype(values.dtype.newbyteorder())


class Tripwire(NumpyTransform[np.generic]):
    """A transform that fails the test when the runtime lets it run."""

    @staticmethod
    def transform(values: NDArray[np.generic]) -> NDArray[np.generic]:  # noqa: ARG004  it must never run, so it reads nothing.
        raise AssertionError("a transform ran that the runtime should not have reached")


class CloneParameter(TorchTransform):
    """A transform whose clone turns a Parameter into a plain Tensor."""

    @staticmethod
    def transform(values: Tensor) -> Tensor:
        return values.clone()


class ToMeta(TorchTransform):
    """A transform that moves the tensor to the meta device."""

    @staticmethod
    def transform(values: Tensor) -> Tensor:
        return values.to("meta")


@pytest.mark.parametrize(
    ("values", "wanted", "backend", "fact"),
    [
        (
            np.ones((2, 2)),
            tensor_requirement("rows cols", transforms=(Reshape,)),
            NUMPY_ARRAYS,
            "shape",
        ),
        (
            np.ones(2),
            tensor_requirement("rows", transforms=(ByteSwap,)),
            NUMPY_ARRAYS,
            "native dtype",
        ),
        (
            nn.Parameter(torch.ones(2)),
            tensor_requirement("rows", transforms=(CloneParameter,)),
            TORCH_ARRAYS,
            "array class",
        ),
        (torch.ones(2), tensor_requirement("rows", transforms=(ToMeta,)), TORCH_ARRAYS, "device"),
    ],
    ids=["a changed shape", "a changed native dtype", "a changed class", "a changed device"],
)
def test_transform_refused_when_a_structural_fact_changed(
    values: Any, wanted: TensorType, backend: ArrayBackend[Any], fact: str, bindings: Bindings
) -> None:
    # Act and Assert
    with pytest.raises(TransformContractError, match=fact):
        validate_array(values, tensor_type=wanted, backend=backend, bindings=bindings)

    # Assert  A broken transform commits none of the facts its array proposed.
    assert bindings.dimensions == {}


def test_transform_not_reached_when_input_structure_wrong(bindings: Bindings) -> None:
    # Act and Assert  The rank check refuses first, so Tripwire never runs.
    with pytest.raises(TensorMismatch, match="rank 2"):
        validate_array(
            np.ones(3),
            tensor_type=tensor_requirement("rows cols", transforms=(Tripwire,)),
            backend=NUMPY_ARRAYS,
            bindings=bindings,
        )


def test_later_transform_not_reached_when_earlier_one_broke_its_contract(
    bindings: Bindings,
) -> None:
    # Act and Assert  Reshape fails, so Tripwire never receives its result.
    with pytest.raises(TransformContractError, match="shape"):
        validate_array(
            np.ones((2, 2)),
            tensor_type=tensor_requirement("rows cols", transforms=(Reshape, Tripwire)),
            backend=NUMPY_ARRAYS,
            bindings=bindings,
        )


def test_prior_bindings_preserved_when_transform_fails(bindings: Bindings) -> None:
    # Arrange
    bindings.bind_dimension("rows", 2)

    # Act and Assert
    with pytest.raises(TransformContractError):
        validate_array(
            np.ones((2, 2)),
            tensor_type=tensor_requirement("rows cols", transforms=(Reshape,)),
            backend=NUMPY_ARRAYS,
            bindings=bindings,
        )

    # Assert
    assert bindings.dimensions == {"rows": 2}


def test_bindings_uncommitted_when_property_refuses_after_a_copy(bindings: Bindings) -> None:
    # Arrange  The copy succeeds, and Finite then refuses the values it holds.
    values = np.array([[1.0, np.inf]])

    # Act and Assert
    with pytest.raises(TensorMismatch, match="Finite"):
        validate_array(
            values,
            tensor_type=tensor_requirement(
                "rows cols", transforms=(CopyReadOnly,), properties=(Finite,)
            ),
            backend=NUMPY_ARRAYS,
            bindings=bindings,
        )

    # Assert
    assert bindings.dimensions == {}
    assert values.flags.writeable


class InPlaceShape(TorchTransform):
    """A transform that changes its input in place and returns that same tensor.

    Torch supplies the in-place case, because NumPy 2.5 deprecates both of its own
    in-place shape changes. `apply_transform` is shared, so either backend proves it.
    """

    @staticmethod
    def transform(values: Tensor) -> Tensor:
        return values.unsqueeze_(0)


def test_transform_refused_when_it_changed_its_input_in_place(bindings: Bindings) -> None:
    # Arrange
    values = torch.ones(2, 2)

    # Act and Assert  The facts are read before the call, so both sides cannot move together.
    with pytest.raises(TransformContractError, match="shape"):
        validate_array(
            values,
            tensor_type=tensor_requirement("rows cols", transforms=(InPlaceShape,)),
            backend=TORCH_ARRAYS,
            bindings=bindings,
        )

    # Assert
    assert bindings.dimensions == {}


def test_later_value_still_transformed_when_resolution_is_warm(bindings: Bindings) -> None:
    # Arrange  The first array warms the resolution cache for this requirement.
    wanted = tensor_requirement("rows cols", transforms=(CopyReadOnly,), properties=(Finite,))
    first = validate_array(
        np.ones((2, 2)), tensor_type=wanted, backend=NUMPY_ARRAYS, bindings=Bindings()
    )
    later = np.full((2, 2), 2.0)

    # Act
    second = validate_array(later, tensor_type=wanted, backend=NUMPY_ARRAYS, bindings=bindings)

    # Assert  The cache holds classes alone, so this array got its own sealed copy of its
    # own values. A separate allocation alone would also pass on the earlier values.
    assert not second.flags.writeable
    assert np.array_equal(second, later)
    assert not np.array_equal(second, first)
    assert not np.shares_memory(second, first)
    assert not np.shares_memory(second, later)
    assert later.flags.writeable


def test_later_value_still_refused_when_resolution_is_warm(bindings: Bindings) -> None:
    # Arrange  The first array warms the resolution cache for this requirement.
    wanted = tensor_requirement("rows cols", transforms=(CopyReadOnly,), properties=(Finite,))
    validate_array(np.ones((2, 2)), tensor_type=wanted, backend=NUMPY_ARRAYS, bindings=Bindings())

    # Act and Assert
    with pytest.raises(TensorMismatch, match="Finite"):
        validate_array(
            np.array([[1.0, np.inf]]),
            tensor_type=wanted,
            backend=NUMPY_ARRAYS,
            bindings=bindings,
        )

    # Assert
    assert bindings.dimensions == {}


class CountingArrays(NumpyArrays):
    """A NumPy backend that counts resolution at the public seam, and still resolves."""

    def __init__(self) -> None:
        super().__init__()
        self.resolutions = 0

    def resolve_property(
        self, property_type: type[ArrayProperty], *, permitted_formats: frozenset[DTypeId]
    ) -> type[operations.ArrayProperty[Any]]:
        self.resolutions += 1
        return super().resolve_property(property_type, permitted_formats=permitted_formats)

    def resolve_transform(
        self,
        transform_type: type[operations.ArrayTransform[Any]],
        *,
        permitted_formats: frozenset[DTypeId],
    ) -> type[operations.ArrayTransform[Any]]:
        self.resolutions += 1
        return super().resolve_transform(transform_type, permitted_formats=permitted_formats)


def test_operations_resolved_once_when_validator_reused(bindings: Bindings) -> None:
    # Arrange
    backend = CountingArrays()
    wanted = tensor_requirement("rows cols", transforms=(CopyReadOnly,), properties=(Finite,))

    validator = ArrayValidator(tensor_type=wanted, backend=backend)
    at_construction = backend.resolutions
    validator.validate(np.ones((2, 2)), bindings=bindings)

    # Act
    validator.validate(np.full((2, 2), 2.0), bindings=Bindings())

    # Assert  Construction resolved the transform and the property, and no later check
    # resolved either one again.
    assert at_construction == 2
    assert backend.resolutions == at_construction


def test_later_value_checked_on_its_own_when_validator_reused() -> None:
    # Arrange
    validator = ArrayValidator(
        tensor_type=tensor_requirement("rows cols", properties=(Finite,)), backend=NUMPY_ARRAYS
    )
    shared = Bindings()
    validator.validate(np.ones((2, 3)), bindings=shared)

    # Act and Assert  The second array is refused on its own values.
    with pytest.raises(TensorMismatch, match="Finite"):
        validator.validate(np.array([[1.0, np.inf, 2.0], [1.0, 2.0, 3.0]]), bindings=shared)

    # Assert  The supplied bindings kept what the accepted array established.
    assert shared.dimensions == {"rows": 2, "cols": 3}


def test_later_bindings_independent_when_validator_reused() -> None:
    # Arrange
    validator = ArrayValidator(tensor_type=tensor_requirement("rows cols"), backend=NUMPY_ARRAYS)
    first = Bindings()
    validator.validate(np.ones((2, 3)), bindings=first)
    later = Bindings()

    # Act  A later boundary supplies its own bindings, so other extents are free.
    accepted = validator.validate(np.ones((5, 7)), bindings=later)

    # Assert
    assert accepted.shape == (5, 7)
    assert first.dimensions == {"rows": 2, "cols": 3}
    assert later.dimensions == {"rows": 5, "cols": 7}


def test_prepared_state_refuses_assignment() -> None:
    # Arrange
    validator = ArrayValidator(
        tensor_type=tensor_requirement("rows", properties=(Finite,)), backend=NUMPY_ARRAYS
    )

    # Act and Assert  A deliberate caller error: the runtime refusal is what this proves.
    with pytest.raises(FrozenInstanceError):
        validator.properties = ()  # ty: ignore[invalid-assignment]
