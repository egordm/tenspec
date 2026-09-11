"""The errors Tenspec raises, and what each one reports."""

from collections.abc import Mapping
from enum import StrEnum


class AnnotationError(TypeError):
    """A declaration is invalid, unsupported, or never reached Tenspec's preparation."""


class ConstraintEvaluationError(RuntimeError):
    """A backend cannot inspect a declared property of this value.

    This never reports a measured mismatch. It reports that the check could not run.
    """


class BindingReason(StrEnum):
    """Why a dimension lookup found no extent."""

    NO_SCOPE = "no_scope"
    UNBOUND = "unbound"
    NOT_SCALAR = "not_scalar"


class BindingError(ValueError):
    """A required dimension was unavailable, rather than different.

    The reason says which lookup failed. The name is the dimension the caller asked for.
    """

    def __init__(self, reason: BindingReason, name: str | None = None) -> None:
        self.reason = reason
        self.name = name
        messages = {
            BindingReason.NO_SCOPE: "no Tenspec validation scope is open",
            BindingReason.UNBOUND: f"dimension {name!r} has no binding in this scope",
            BindingReason.NOT_SCALAR: f"dimension {name!r} holds a group of axes, not one extent",
        }
        super().__init__(messages[reason])


class TensorMismatch(ValueError):  # noqa: N818  the accepted contract fixes this name.
    """An array disagrees with its declared requirement.

    The requirement, the observed fact, and the relevant bindings stay readable on the
    error. The error holds no array.
    """

    def __init__(
        self,
        *,
        requirement: str,
        observed: str,
        bindings: Mapping[str, int | tuple[int, ...]] | None = None,
    ) -> None:
        self.requirement = requirement
        self.observed = observed
        self.bindings: Mapping[str, int | tuple[int, ...]] = dict(bindings or {})
        detail = f", with {self.bindings}" if self.bindings else ""
        super().__init__(f"expected {requirement}, but the array has {observed}{detail}")


def describe_empty(shape: tuple[int, ...]) -> str:
    """What a refusal reports about an array that holds no elements.

    It reads the shape alone. Axis indices start at 0.

    describe_empty((2, 0, 0)) returns "shape (2, 0, 0), with no elements at axis 1, axis 2".

    Returns:
        The observed text for a TensorMismatch.
    """
    empty = ", ".join(f"axis {index}" for index, extent in enumerate(shape) if extent == 0)
    return f"shape {shape}, with no elements at {empty}"


class TransformContractError(RuntimeError):
    """A transform changed a structural fact that it had to preserve.

    This reports a defect in the transform, not invalid caller input, so it is a
    RuntimeError and never a ValueError. It names the transform and the one fact that
    changed. Every field is text, so the error holds no array.
    """

    def __init__(self, *, transform: str, fact: str, before: str, after: str) -> None:
        self.transform = transform
        self.fact = fact
        super().__init__(f"{transform} changed the {fact} of its input, from {before} to {after}")
