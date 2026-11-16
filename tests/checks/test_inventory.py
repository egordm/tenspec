import pytest
from checks.inventory import InventoryCheck


def test_passes_when_tracked_paths_are_checked_source_and_documents() -> None:
    # Arrange
    check = InventoryCheck(["library/module.py", "docs/guide.md", "tests/test_module.py"], [])

    # Act
    findings = check.findings()

    # Assert
    assert findings == []


@pytest.mark.parametrize("path", ["build/hidden.py", "ignored/readme.md", "data/cache.bin"])
def test_reports_path_when_tracked_file_matches_ignore_rules(path: str) -> None:
    # Arrange
    check = InventoryCheck(["tracked.py", path], [path])

    # Act
    findings = check.findings()

    # Assert
    assert [(finding.path, finding.rule) for finding in findings] == [(path, "tracked-ignored")]


@pytest.mark.parametrize("path", ["examples/notebook.ipynb", "examples/NOTEBOOK.IPYNB"])
def test_reports_notebook_when_tracked_path_has_notebook_suffix(path: str) -> None:
    # Arrange
    check = InventoryCheck(["tracked.py", path], [])

    # Act
    findings = check.findings()

    # Assert
    assert [(finding.path, finding.rule) for finding in findings] == [(path, "tracked-notebook")]


def test_reports_both_rules_when_tracked_notebook_is_ignored() -> None:
    # Arrange
    path = "ignored/notebook.ipynb"
    check = InventoryCheck([path], [path])

    # Act
    findings = check.findings()

    # Assert
    assert {finding.rule for finding in findings} == {"tracked-ignored", "tracked-notebook"}


def test_reports_empty_subject_when_tracked_inventory_is_empty() -> None:
    # Arrange
    check = InventoryCheck([], [])

    # Act
    findings = check.findings()

    # Assert
    assert [finding.rule for finding in findings] == ["empty-subject"]
