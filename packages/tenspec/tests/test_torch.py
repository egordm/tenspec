"""The Torch facts that a NumPy caller never meets."""

from typing import Literal as Shape
from typing import TypeVar

import pytest
import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensorMode

from pydantic import ValidationError
from tenspec import ConstraintEvaluationError, Finite, Materialized, validate
from tenspec.errors import AnnotationError, TensorMismatch
from tenspec.torch import CPU, CUDA, ArrayProperty, Float, Int, NonEmpty
from tenspec.types.properties import Device

Placement = TypeVar("Placement", bound=Device)
OnCuda = TypeVar("OnCuda", bound=CUDA)


class NonPositive(ArrayProperty):
    """A Torch caller's own check: no entry of the tensor is above zero."""

    @staticmethod
    def validate(values: Tensor) -> None:
        """Refuse a tensor holding a positive entry.

        Raises:
            ValueError: when any entry is above zero.
        """
        if bool((values > 0).any().item()):
            raise ValueError("no entry may be positive")


TWO_PLACEMENTS = tuple[Float[Shape["a"], Placement], Float[Shape["b"], Placement]]
"""Two operands that must sit on one placement, whatever that placement is."""


def test_finite_unavailable_when_tensor_meta() -> None:
    # Act and Assert
    with pytest.raises(ConstraintEvaluationError, match="metadata alone"):
        validate(torch.ones(3, device="meta"), Float[Shape["rows"], Finite])


def test_finite_unavailable_when_tensor_fake() -> None:
    # Arrange
    with FakeTensorMode():
        fake = torch.ones(3)

        # Act and Assert
        with pytest.raises(ConstraintEvaluationError, match="metadata alone"):
            validate(fake, Float[Shape["rows"], Finite])


def test_materialized_rejected_when_tensor_fake() -> None:
    # Arrange
    with FakeTensorMode():
        fake = torch.ones(3)

        # Act and Assert
        with pytest.raises(ValidationError, match="Materialized"):
            validate(fake, Float[Shape["rows"], Materialized])


def test_custom_property_refused_when_its_own_rule_fails() -> None:
    # Act and Assert  A Torch caller's own check reaches the tensor through one shared path.
    with pytest.raises(ValidationError, match="no entry may be positive"):
        validate(torch.ones(3), Float[Shape["rows"], NonPositive])


def test_custom_property_accepted_when_its_own_rule_holds() -> None:
    # Arrange
    values = -torch.ones(3)

    # Act
    accepted = validate(values, Float[Shape["rows"], NonPositive])

    # Assert
    assert accepted is values


def test_finite_accepted_when_tensor_integer() -> None:
    # Arrange
    values = torch.ones(3, dtype=torch.int64)

    # Act
    accepted = validate(values, Int[Shape["rows"], Finite])

    # Assert
    assert accepted is values


def test_device_accepted_when_cpu_required() -> None:
    # Arrange
    values = torch.ones(3)

    # Act
    accepted = validate(values, Float[Shape["rows"], CPU])

    # Assert
    assert accepted is values


def test_device_rejected_when_kind_differs() -> None:
    # Act and Assert
    with pytest.raises(ValidationError, match="device cuda"):
        validate(torch.ones(3), Float[Shape["rows"], CUDA])


def test_operands_accepted_when_devices_match() -> None:
    # Arrange
    left, right = torch.ones(2), torch.ones(3)

    # Act
    accepted = validate((left, right), TWO_PLACEMENTS)

    # Assert
    assert accepted[0] is left
    assert accepted[1] is right


def test_operand_rejected_when_devices_differ() -> None:
    # Arrange
    on_cpu, on_meta = torch.ones(2), torch.ones(3, device="meta")

    # Act and Assert
    with pytest.raises(ValidationError, match="device cpu for Placement"):
        validate((on_cpu, on_meta), TWO_PLACEMENTS)


def test_declaration_rejected_when_device_kind_unspecified() -> None:
    # Act and Assert
    with pytest.raises(AnnotationError, match="Name a kind"):
        validate(torch.ones(3), Float[Shape["rows"], Device])


def test_device_rejected_when_variable_bound_differs() -> None:
    # Act and Assert
    with pytest.raises(ValidationError, match="device cuda"):
        validate(torch.ones(3), Float[Shape["rows"], OnCuda])


def test_declaration_rejected_when_device_bounds_conflict() -> None:
    # Act and Assert
    with pytest.raises(AnnotationError, match="contradicts"):
        validate(torch.ones(3), Float[Shape["rows"], CPU, OnCuda])


@pytest.mark.parametrize(
    ("extent", "device", "observed"),
    [
        ((0, 3), "cpu", "shape (0, 3), with no elements at axis 0"),
        ((2, 0, 0), "cpu", "shape (2, 0, 0), with no elements at axis 1, axis 2"),
        ((0, 3), "meta", "shape (0, 3), with no elements at axis 0"),
    ],
    ids=["one empty axis", "two empty axes", "metadata alone"],
)
def test_refusal_names_the_empty_axes_when_tensor_holds_no_elements(
    extent: tuple[int, ...], device: str, observed: str
) -> None:
    # Arrange
    values = torch.zeros(extent, device=device)

    # Act and Assert
    with pytest.raises(TensorMismatch) as refusal:
        NonEmpty.validate(values)

    # Assert
    assert refusal.value.requirement == "NonEmpty"
    assert refusal.value.observed == observed
