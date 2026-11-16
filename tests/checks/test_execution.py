from pathlib import Path

import pytest
from checks.execution import CheckRuns
from checks.ruff import Ruff


def test_selects_base_checks_when_base_contains_package() -> None:
    # Arrange
    paths = {"checks/__init__.py", "checks/suppressions.py"}

    # Act
    selected = CheckRuns.has_base_checks(paths)

    # Assert
    assert selected


def test_uses_bootstrap_when_base_has_no_checks() -> None:
    # Arrange
    paths = {"library/example.py", "ruff.toml"}

    # Act
    selected = CheckRuns.has_base_checks(paths)

    # Assert
    assert not selected


def test_reports_error_when_explicit_source_inventory_is_empty() -> None:
    # Arrange
    paths: list[str] = []

    # Act & Assert
    with pytest.raises(ValueError, match="must not be empty"):
        Ruff(Path("/"), paths)


def test_reports_error_when_tool_output_is_invalid() -> None:
    # Arrange
    output = "not JSON"

    # Act & Assert
    with pytest.raises(ValueError, match="Expecting value"):
        Ruff.parse_diagnostics(output)


def test_reports_error_when_ruff_reports_file_outside_inventory() -> None:
    # Arrange
    ruff = Ruff(Path("/"), ["tracked.py"])
    diagnostic = {"filename": "/hidden.py"}

    # Act & Assert
    with pytest.raises(ValueError, match="no inventoried source"):
        ruff.validate_paths([diagnostic])
