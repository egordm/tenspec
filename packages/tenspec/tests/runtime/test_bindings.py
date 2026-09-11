"""Scope isolation, and what a dimension lookup reports when it cannot give one extent."""

import threading
from typing import Literal as Shape

import numpy as np
import pytest

from pydantic import ValidationError
from tenspec import axis_size, checked
from tenspec.errors import BindingError, BindingReason
from tenspec.numpy import Float
from tenspec.runtime.bindings import Bindings, validation_scope


def test_lookup_fails_when_no_scope() -> None:
    # Act and Assert
    with pytest.raises(BindingError) as refusal:
        axis_size("rows")

    assert refusal.value.reason is BindingReason.NO_SCOPE


def test_lookup_fails_when_name_unbound() -> None:
    # Act and Assert
    with validation_scope(), pytest.raises(BindingError) as refusal:
        axis_size("rows")

    assert refusal.value.reason is BindingReason.UNBOUND
    assert refusal.value.name == "rows"


def test_lookup_fails_when_name_variadic() -> None:
    # Arrange
    bindings = Bindings()
    bindings.bind_dimension("batch", (2, 3))

    # Act and Assert
    with pytest.raises(BindingError) as refusal:
        bindings.axis_size("batch")

    assert refusal.value.reason is BindingReason.NOT_SCALAR


def test_outer_scope_restored_when_inner_scope_fails() -> None:
    # Act and Assert
    with validation_scope() as outer:
        outer.bind_dimension("rows", 2)

        with pytest.raises(RuntimeError), validation_scope():
            raise RuntimeError("the body failed")

        assert axis_size("rows") == 2

    with pytest.raises(BindingError):
        axis_size("rows")


def test_outer_binding_preserved_when_inner_call_fails() -> None:
    # Arrange
    @checked
    def inner(first: Float[Shape["rows"]], second: Float[Shape["rows"]]) -> int:
        return first.shape[0] + second.shape[0]

    @checked
    def outer(values: Float[Shape["rows"]]) -> tuple[int, int]:
        with pytest.raises(ValidationError, match="rows=5"):
            inner(np.ones(5, dtype=np.float32), np.ones(6, dtype=np.float32))
        return axis_size("rows"), values.shape[0]

    # Act
    result = outer(np.ones(2, dtype=np.float32))

    # Assert
    assert result == (2, 2)


def test_inner_binding_independent_when_name_reused() -> None:
    # Arrange
    @checked
    def inner(values: Float[Shape["rows"]]) -> tuple[int, int]:
        return axis_size("rows"), values.shape[0]

    @checked
    def outer(values: Float[Shape["rows"]]) -> tuple[tuple[int, int], int, int]:
        return inner(np.ones(5, dtype=np.float32)), axis_size("rows"), values.shape[0]

    # Act
    result = outer(np.ones(2, dtype=np.float32))

    # Assert
    assert result == ((5, 5), 2, 2)


def test_bindings_independent_when_threads_overlap() -> None:
    # Arrange
    started = threading.Barrier(2)
    seen: dict[str, int] = {}

    @checked
    def measure(values: Float[Shape["rows"]]) -> None:
        started.wait(timeout=5)
        seen[str(values.shape[0])] = axis_size("rows")

    threads = [
        threading.Thread(target=measure, args=(np.ones(rows, dtype=np.float32),), name=str(rows))
        for rows in (2, 7)
    ]

    # Act
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    # Assert
    assert seen == {"2": 2, "7": 7}
