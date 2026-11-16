import inspect
from types import ModuleType

import pytest

from tenspec import numpy, torch


@pytest.mark.parametrize("backend", [numpy, torch], ids=["numpy", "torch"])
def test_overrides_are_marked_when_backend_implements_abstract_base(backend: ModuleType) -> None:
    # Arrange
    implementations = [
        (owner, name, getattr(owner, name))
        for _, owner in inspect.getmembers(backend, inspect.isclass)
        if owner.__module__ == backend.__name__ and not inspect.isabstract(owner)
        for name in vars(owner)
        if inspect.isroutine(getattr(owner, name))
        and any(inspect.isabstract(base) and hasattr(base, name) for base in owner.__mro__[1:])
    ]

    # Act
    unmarked = [
        f"{owner.__name__}.{name}"
        for owner, name, method in implementations
        if not getattr(method, "__override__", False)
    ]

    # Assert
    assert implementations
    assert unmarked == []
