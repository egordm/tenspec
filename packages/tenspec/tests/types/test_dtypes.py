"""What a whole dtype requirement accepts, once its shared variable narrows it."""

from typing import TypeVar

from tenspec.types.dtypes import DType, DTypeFamily, DTypeId, DTypeRequirement


class Float64Only(DType):
    """The bound of a shared variable that permits float64 alone."""

    permitted = DTypeId.FLOAT64


Narrow = TypeVar("Narrow", bound=Float64Only)


def test_family_kept_whole_when_no_variable_narrows_it() -> None:
    # Arrange
    requirement = DTypeRequirement(permitted=DTypeFamily.FLOATING)

    # Act
    accepted = requirement.permitted_formats

    # Assert
    assert accepted == {DTypeId.FLOAT16, DTypeId.BFLOAT16, DTypeId.FLOAT32, DTypeId.FLOAT64}


def test_family_narrowed_when_variable_bound_permits_one_format() -> None:
    # Arrange
    requirement = DTypeRequirement(permitted=DTypeFamily.FLOATING, variable=Narrow)

    # Act
    accepted = requirement.permitted_formats

    # Assert
    assert accepted == {DTypeId.FLOAT64}
