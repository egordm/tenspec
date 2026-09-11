"""What a checked call does with a default, and what it leaves to the caller."""

import threading
from collections.abc import Callable
from typing import Annotated, Self
from typing import Literal as Shape

import numpy as np
import pytest
from numpy.typing import NDArray

from pydantic import ValidationError
from tenspec import Finite, checked
from tenspec.errors import AnnotationError
from tenspec.numpy import CopyReadOnly, Float, Float64

THREE_ROWS = np.ones(3)
"""A default the declaration must check, like any supplied argument."""


@checked
def pair(first: Float[Shape["rows"]], second: Float[Shape["rows"]] = THREE_ROWS) -> int:
    """Two operands on one axis, the second of them defaulted.

    Returns:
        The total extent of both operands.
    """
    return first.shape[0] + second.shape[0]


class Scale:
    """A caller's own class, whose fit and widen return the receiver's class."""

    def __init__(self, mean: float) -> None:
        self.mean = mean

    @classmethod
    @checked
    def fit(cls, values: Float[Shape["*batch time"]], *, floor: float = 1e-3) -> Self:
        return cls(float(values.mean()) + floor)

    @checked
    def widen(self, values: Float[Shape["rows"]]) -> Self:
        self.mean += float(values.sum())
        return self


class Sharper(Scale):
    """A subclass whose calls must return the subclass."""


class Broken(Scale):
    """A subclass whose fit returns the base class, which its Self return refuses."""

    @classmethod
    @checked
    def fit(cls, values: Float[Shape["*batch time"]], *, floor: float = 1e-3) -> Self:
        # A deliberate caller error: the runtime refusal is what this class proves.
        return Scale(float(values.mean()) + floor)  # ty: ignore[invalid-return-type]


def test_default_used_when_shapes_match() -> None:
    # Act
    result = pair(np.ones(3))

    # Assert
    assert result == 6


def test_default_rejected_when_shapes_differ() -> None:
    # Act and Assert
    with pytest.raises(ValidationError, match="rows=2"):
        pair(np.ones(2))


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [((), 4.0), ((3.0,), 6.0)],
    ids=["the default applies", "a supplied value wins"],
)
def test_ordinary_default_preserved_when_tensor_declared(
    supplied: tuple[float, ...], expected: float
) -> None:
    # Arrange
    @checked
    def scale(values: Float[Shape["rows"]], factor: float = 2.0) -> float:
        return float(values.sum()) * factor

    # Act
    result = scale(np.ones(2), *supplied)

    # Assert
    assert result == expected


@pytest.mark.parametrize(
    ("call", "receiver"),
    [
        (lambda: Scale.fit(np.ones((2, 3, 4))), Scale),
        (lambda: Sharper.fit(np.ones((2, 3))), Sharper),
        (lambda: Sharper(1.0).widen(np.ones(3)), Sharper),
    ],
    ids=["the base class", "a subclass", "an instance of a subclass"],
)
def test_receiver_type_returned_when_self_declared(
    call: Callable[[], Scale], receiver: type[Scale]
) -> None:
    # Act
    result = call()

    # Assert
    assert type(result) is receiver


def test_return_rejected_when_self_type_wrong() -> None:
    # Act and Assert
    with pytest.raises(ValidationError):
        Broken.fit(np.ones(3))


def test_keyword_argument_preserved_when_method_checked() -> None:
    # Act
    result = Scale.fit(np.ones(3), floor=0.5)

    # Assert
    assert result.mean == 1.5


def test_annotations_unchanged_when_function_decorated() -> None:
    # Arrange
    def widen(values: Float[Shape["rows"]]) -> Float[Shape["rows"]]:
        return values

    written = dict(widen.__annotations__)

    # Act
    checked(widen)

    # Assert
    assert widen.__annotations__ == written


def test_annotations_unchanged_when_self_specialized() -> None:
    # Arrange
    class Written(Scale):
        """A caller class whose Self-returning method this test decorates itself."""

        def shift(self, values: Float[Shape["rows"]]) -> Self:
            self.mean += float(values.sum())
            return self

    original = Written.shift
    written = dict(original.__annotations__)

    class Cold(Written):
        """A receiver that has never specialized that method."""

        shift = checked(original)

    # Act
    Cold(1.0).shift(np.ones(3))

    # Assert
    assert original.__annotations__ == written


def test_receiver_type_returned_when_cold_calls_concurrent() -> None:
    # Arrange
    class Cold(Scale):
        """A class whose first checked calls all arrive at once."""

    callers = 4
    start = threading.Barrier(callers)
    seen: list[type[Scale]] = []

    def run() -> None:
        start.wait(timeout=5)
        seen.append(type(Cold.fit(np.ones(3))))

    threads = [threading.Thread(target=run) for _ in range(callers)]

    # Act
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    # Assert
    assert seen == [Cold] * callers


def test_default_copied_again_when_its_declaration_names_a_transform() -> None:
    # Arrange
    shared = np.ones(2)

    @checked
    def measure(values: Float64[Shape["rows"], CopyReadOnly] = shared) -> NDArray[np.float64]:
        return values

    first = measure()

    # Act
    second = measure()

    # Assert  A default is checked like a supplied argument, so this call sealed its own
    # copy. Nothing caches a transformed value between calls.
    assert not second.flags.writeable
    assert second is not first
    assert not np.shares_memory(second, first)
    assert np.array_equal(second, shared)
    assert shared.flags.writeable


def test_refuses_property_outside_the_alias() -> None:
    # Act & Assert
    with pytest.raises(AnnotationError, match="Finite is a Tenspec property"):

        @checked
        def count(values: Annotated[Float[Shape["rows"]], Finite]) -> int:
            return values.shape[0]


def test_refuses_transform_outside_the_alias() -> None:
    # Act & Assert
    with pytest.raises(AnnotationError, match="CopyReadOnly is a Tenspec transform"):

        @checked
        def store(values: Annotated[Float[Shape["rows"]], CopyReadOnly]) -> int:
            return values.shape[0]
