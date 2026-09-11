"""What the model mixin adds, and what it leaves to the caller's own base."""

from typing import Annotated, Any
from typing import Literal as Shape

import numpy as np
import pytest
from pydantic_core import CoreSchema, core_schema

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    ValidationError,
    field_validator,
)
from tenspec import Finite, TensorContracts
from tenspec.errors import AnnotationError
from tenspec.numpy import CopyReadOnly, Float


class Base(BaseModel):
    """The caller's own base, with its own configuration."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class Pair(TensorContracts, Base):
    left: Float[Shape["rows"]]
    right: Float[Shape["rows"]] = Field(
        default=np.ones(3), description="the second block, on the same axis"
    )


class Checked(Base):
    """A caller base that carries its own field validator."""

    @field_validator("label", check_fields=False)
    @classmethod
    def label_is_short(cls, given: str) -> str:
        """Refuse a label of more than three characters.

        Returns:
            The label.
        """
        if len(given) > 3:
            raise ValueError("the label is too long")
        return given


class Sized(TensorContracts, Checked):
    label: str
    values: Float[Shape["rows"]]


class Hooked(Base):
    """A caller base whose own schema hook adds a check of the built model."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Wrap the caller's own model check around whatever schema the handler builds.

        Returns:
            The schema, with the caller's check after it.
        """

        def refuse_an_all_zero_block(model: Any) -> Any:
            if not model.values.any():
                raise ValueError("the caller hook refuses an all-zero block")
            return model

        return core_schema.no_info_after_validator_function(
            refuse_an_all_zero_block, handler(source)
        )


class Composed(TensorContracts, Hooked):
    values: Float[Shape["rows"]]


BUILD_SETTINGS = [{}, {"defer_build": False}, {"defer_build": True}]
"""Every build setting a caller's own base can carry into the mixin."""


def test_default_accepted_when_shapes_match() -> None:
    # Act
    model = Pair(left=np.ones(3))

    # Assert
    assert model.right.shape == (3,)


def test_default_rejected_when_shapes_differ() -> None:
    # Act and Assert
    with pytest.raises(ValidationError, match="rows=2"):
        Pair(left=np.ones(2))


def test_field_metadata_preserved_when_model_prepared() -> None:
    # Arrange
    description = "the second block, on the same axis"

    class CallerBase(BaseModel):
        model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # Act
    class Related(TensorContracts, CallerBase):
        left: Float[Shape["rows"]]
        right: Float[Shape["rows"]] = Field(default=np.ones(3), description=description)

    # Assert
    assert Related.model_fields["right"].description == description
    assert Related.model_config["frozen"] is True


@pytest.mark.parametrize("setting", BUILD_SETTINGS, ids=["absent", "False", "True"])
def test_build_setting_preserved_when_model_prepared(
    setting: dict[str, bool],
) -> None:
    # Arrange
    class CallerBase(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True, **setting)

    # Act
    class Related(TensorContracts, CallerBase):
        left: Float[Shape["rows"]]
        right: Float[Shape["rows"]] = Field(default_factory=lambda: np.ones(3))

    # Assert
    assert Related.model_config.get("defer_build", "absent") == setting.get("defer_build", "absent")


@pytest.mark.parametrize("setting", BUILD_SETTINGS, ids=["absent", "False", "True"])
def test_matching_fields_accepted_when_build_setting_varies(
    setting: dict[str, bool],
) -> None:
    # Arrange
    class CallerBase(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True, **setting)

    class Related(TensorContracts, CallerBase):
        left: Float[Shape["rows"]]
        right: Float[Shape["rows"]] = Field(default_factory=lambda: np.ones(3))

    # Act
    model = Related(left=np.ones(3))

    # Assert
    assert model.right.shape == (3,)


@pytest.mark.parametrize("setting", BUILD_SETTINGS, ids=["absent", "False", "True"])
def test_mismatched_fields_rejected_when_build_setting_varies(
    setting: dict[str, bool],
) -> None:
    # Arrange
    class CallerBase(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True, **setting)

    class Related(TensorContracts, CallerBase):
        left: Float[Shape["rows"]]
        right: Float[Shape["rows"]] = Field(default_factory=lambda: np.ones(3))

    # Act and Assert
    with pytest.raises(ValidationError, match="rows=2"):
        Related(left=np.ones(2))


def test_caller_hook_accepts_when_check_passes() -> None:
    # Act
    model = Composed(values=np.ones(2))

    # Assert
    assert model.values.shape == (2,)


def test_caller_hook_rejects_when_check_fails() -> None:
    # Act and Assert
    with pytest.raises(ValidationError, match="all-zero block"):
        Composed(values=np.zeros(2))


def test_inherited_validator_accepts_when_check_passes() -> None:
    # Act
    model = Sized(label="ok", values=np.ones(2))

    # Assert
    assert model.label == "ok"


def test_inherited_validator_rejects_when_check_fails() -> None:
    # Act and Assert
    with pytest.raises(ValidationError, match="too long"):
        Sized(label="far too long", values=np.ones(2))


def test_refuses_property_outside_the_alias() -> None:
    # Act & Assert
    with pytest.raises(AnnotationError, match="Finite is a Tenspec property"):

        class Scores(TensorContracts, Base):
            values: Annotated[Float[Shape["rows"]], Finite]


def test_refuses_transform_outside_the_alias() -> None:
    # Act & Assert
    with pytest.raises(AnnotationError, match="CopyReadOnly is a Tenspec transform"):

        class Stored(TensorContracts, Base):
            values: Annotated[Float[Shape["rows"]], CopyReadOnly]
