"""Selection owns the commit of a tensor alternative's binding facts."""

from typing import Annotated, TypeVar
from typing import Literal as Shape

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch import Tensor

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from tenspec import (
    ConstraintEvaluationError,
    Finite,
    Floating,
    TensorContracts,
    axis_size,
    checked,
    validate,
)
from tenspec.numpy import Float
from tenspec.pydantic.annotations import prepare_annotation
from tenspec.runtime.bindings import validation_scope
from tenspec.torch import Device
from tenspec.torch import Float as TorchFloat

Scalar = TypeVar("Scalar", bound=Floating)
Placement = TypeVar("Placement", bound=Device)

type Either = Float[Shape["rows hidden"]] | TorchFloat[Shape["rows hidden"]]
type LateFailure = tuple[Float[Shape["discarded"]], Shape["left"]]
type Accepted = tuple[Float[Shape["rows"]], Shape["right"]]
type Overlapping = tuple[Float[Shape["first"]], float] | tuple[Float[Shape["second"]], int]


@checked
def measured(values: Either) -> Either:
    return values


class Block(TensorContracts, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    first: Either = Field(description="first operand")
    second: Either = Field(description="related operand")


@pytest.mark.parametrize("values", [np.ones((2, 3)), torch.ones((2, 3))], ids=["numpy", "torch"])
@pytest.mark.parametrize("route", ["call", "model", "standalone"])
def test_native_value_preserved_when_either_backend_accepted(
    values: NDArray[np.float64] | Tensor, route: str
) -> None:
    # Act
    if route == "call":
        result = measured(values)
    elif route == "model":
        result = Block(first=values, second=values).first
    else:
        result = validate(values, Either)

    # Assert
    assert result is values


def test_failed_arm_discarded_when_later_literal_refuses() -> None:
    # Arrange
    adapter = TypeAdapter(
        prepare_annotation(LateFailure | Accepted),
        config=ConfigDict(strict=True, arbitrary_types_allowed=True),
    )
    values = np.ones(3)

    # Act
    with validation_scope() as bindings:
        bindings.bind_dimension("rows", 3)
        result = adapter.validate_python((values, "right"))

    # Assert
    assert result[0] is values
    assert dict(bindings.dimensions) == {"rows": 3}


def test_only_selected_arm_commits_when_both_arms_accept() -> None:
    # Arrange
    adapter = TypeAdapter(
        prepare_annotation(
            tuple[Float[Shape["first"]], float] | tuple[Float[Shape["second"]], int]
        ),
        config=ConfigDict(strict=True, arbitrary_types_allowed=True),
    )

    # Act
    with validation_scope() as bindings:
        result = adapter.validate_python((np.ones(3), 1))

    # Assert
    assert type(result[1]) is int
    assert dict(bindings.dimensions) == {"second": 3}


@pytest.mark.parametrize("route", ["call", "model", "standalone"])
def test_caller_selection_preserved_when_left_to_right_requested(route: str) -> None:
    # Arrange
    @checked
    def configured(value: Annotated[Overlapping, Field(union_mode="left_to_right")]) -> float:
        return value[1]

    class Configured(TensorContracts, BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        value: Overlapping = Field(union_mode="left_to_right", description="caller selection")

    values = (np.ones(3), 1)

    # Act
    if route == "call":
        result = configured(values)
    elif route == "model":
        result = Configured(value=values).value[1]
    else:
        result = validate(values, Annotated[Overlapping, Field(union_mode="left_to_right")])[1]

    # Assert
    assert type(result) is float


def test_all_errors_preserved_when_no_arm_accepts() -> None:
    # Arrange
    adapter = TypeAdapter(
        prepare_annotation(LateFailure | Accepted),
        config=ConfigDict(strict=True, arbitrary_types_allowed=True),
    )

    # Act and Assert
    with validation_scope() as bindings:
        bindings.bind_dimension("existing", 7)
        with pytest.raises(ValidationError) as refused:
            adapter.validate_python((np.ones(3), "neither"))

    # Assert
    assert [entry["type"] for entry in refused.value.errors()] == ["literal_error"] * 2
    assert dict(bindings.dimensions) == {"existing": 7}


def test_nested_facts_discarded_when_enclosing_arm_fails() -> None:
    # Arrange
    adapter = TypeAdapter(
        prepare_annotation(
            tuple[Float[Shape["inner_a"]] | Float[Shape["inner_b"]], Shape["left"]] | Accepted
        ),
        config=ConfigDict(strict=True, arbitrary_types_allowed=True),
    )

    # Act
    with validation_scope() as bindings:
        adapter.validate_python((np.ones(3), "right"))

    # Assert
    assert dict(bindings.dimensions) == {"rows": 3}


def test_known_axis_enforced_when_alternative_follows_argument() -> None:
    # Arrange
    @checked
    def related(first: Float[Shape["rows hidden"]], second: Either) -> int:
        return axis_size("rows") + first.shape[1] + second.shape[1]

    # Act and Assert
    with pytest.raises(ValidationError, match="rows=2"):
        related(np.ones((2, 3)), torch.ones((5, 3)))


def test_discarded_arm_variables_absent_when_dtype_and_device_were_bound() -> None:
    # Arrange
    adapter = TypeAdapter(
        prepare_annotation(
            tuple[TorchFloat[Shape["discarded"], Scalar, Placement], Shape["left"]]
            | tuple[TorchFloat[Shape["rows"]], Shape["right"]]
        ),
        config=ConfigDict(strict=True, arbitrary_types_allowed=True),
    )

    # Act
    with validation_scope() as bindings:
        adapter.validate_python((torch.ones(3), "right"))

    # Assert
    assert dict(bindings.dimensions) == {"rows": 3}
    assert not bindings.dtypes
    assert not bindings.devices


@pytest.mark.parametrize("values", [None, np.ones((2, 3))], ids=["none", "array"])
def test_optional_value_preserved_when_union_includes_none(
    values: NDArray[np.float64] | None,
) -> None:
    # Act
    result = validate(values, Either | None)

    # Assert
    assert result is values


def test_inspection_error_propagates_when_an_arm_cannot_read_values() -> None:
    # Arrange
    values = torch.ones(3, device="meta")

    # Act and Assert
    with pytest.raises(ConstraintEvaluationError, match="metadata alone"):
        validate(values, TorchFloat[Shape["rows"], Finite] | TorchFloat[Shape["rows"]])
