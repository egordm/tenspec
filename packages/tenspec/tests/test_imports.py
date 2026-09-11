"""The import boundary between the common surface and each array backend.

Tenspec's own modules import one array library each. It cannot promise the reverse
isolation for Torch, because Torch itself imports NumPy when NumPy is installed.
"""

import subprocess
import sys

import pytest


def loaded_modules(source: str) -> set[str]:
    """The top-level modules a fresh interpreter holds after it runs `source`.

    Returns:
        Every module name in that interpreter's own sys.modules.
    """
    program = f"{source}\nimport sys\nprint(' '.join(sys.modules))"
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    return set(result.stdout.split())


def test_backends_unloaded_when_common_module_imported() -> None:
    # Act
    modules = loaded_modules("import tenspec")

    # Assert
    assert "numpy" not in modules
    assert "torch" not in modules


@pytest.mark.parametrize(
    ("source", "library"),
    [("import tenspec.numpy", "numpy"), ("import tenspec.torch", "torch")],
    ids=["the numpy backend", "the torch backend"],
)
def test_array_library_loaded_when_backend_imported(source: str, library: str) -> None:
    # Act
    modules = loaded_modules(source)

    # Assert
    assert library in modules


def test_torch_unloaded_when_numpy_backend_imported() -> None:
    # Act
    modules = loaded_modules("import tenspec.numpy")

    # Assert
    assert "torch" not in modules
