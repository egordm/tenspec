import pytest
from checks.configuration import ConfigurationCheck


@pytest.mark.parametrize(
    "path",
    [
        "checks/growth.py",
        "checks/banned-terms.toml",
        "tests/checks/test_growth.py",
        "quality-budget.toml",
        "ruff.toml",
        "nested/ruff.toml",
        "src/ruff.toml",
        "ty.toml",
        "nested/ty.toml",
        ".importlinter",
        ".pre-commit-config.yaml",
        "poe_tasks.toml",
        ".github/workflows/ci.yml",
        "nested/pyproject.toml",
    ],
)
def test_reports_mixed_change_when_enforcement_file_is_added(path: str) -> None:
    # Arrange
    branch = {path: b"", "src/example.py": b"value = 2\n"}

    # Act
    findings = ConfigurationCheck({}, branch).findings()

    # Assert
    assert any(item.rule == "configuration" for item in findings)


def test_passes_when_enforcement_changes_stand_alone() -> None:
    # Arrange
    base = {"ruff.toml": b"select = ['ALL']\n"}
    branch = {"ruff.toml": b"select = ['E4']\n"}

    # Act
    findings = ConfigurationCheck(base, branch).findings()

    # Assert
    assert findings == []


def test_passes_when_non_enforcement_files_change_together() -> None:
    # Arrange
    base = {"src/example.py": b"value = 1\n"}
    branch = {"src/example.py": b"value = 2\n", "docs/example.md": b"Explain the value.\n"}

    # Act
    findings = ConfigurationCheck(base, branch).findings()

    # Assert
    assert findings == []


def test_reports_mixed_change_when_existing_enforcement_bytes_differ() -> None:
    # Arrange
    base = {"ruff.toml": b"# Policy\n", "src/example.py": b"value = 1\n"}
    branch = {"ruff.toml": b"# Updated policy\n", "src/example.py": b"value = 2\n"}

    # Act
    findings = ConfigurationCheck(base, branch).findings()

    # Assert
    assert [item.path for item in findings] == ["ruff.toml"]


@pytest.mark.parametrize("table", ["ruff", "ty", "pytest", "poe", "importlinter"])
def test_reports_mixed_change_when_protected_project_table_changes(table: str) -> None:
    # Arrange
    before = f"[tool.{table}]\nsetting = 'old'\n".encode()
    after = f"[tool.{table}]\nsetting = 'new'\n".encode()
    base = {"pyproject.toml": before, "src/example.py": b"value = 1\n"}
    branch = {"pyproject.toml": after, "src/example.py": b"value = 2\n"}

    # Act
    findings = ConfigurationCheck(base, branch).findings()

    # Assert
    assert len(findings) == 1


def test_passes_when_budget_falls_alongside_code() -> None:
    # Arrange
    base = {"quality-budget.toml": b"[structure]\nlong-modules = 2\n"}
    branch = {
        "quality-budget.toml": b"[structure]\nlong-modules = 1\n",
        "src/example.py": b"value = 2\n",
    }

    # Act
    findings = ConfigurationCheck(base, branch).findings()

    # Assert
    assert findings == []


def test_reports_mixed_change_when_budget_rises_alongside_code() -> None:
    # Arrange
    base = {"quality-budget.toml": b"[structure]\nlong-modules = 2\n"}
    branch = {
        "quality-budget.toml": b"[structure]\nlong-modules = 3\n",
        "src/example.py": b"value = 2\n",
    }

    # Act
    findings = ConfigurationCheck(base, branch).findings()

    # Assert
    assert len(findings) == 1


def test_passes_when_testpaths_gain_entry_alongside_code() -> None:
    # Arrange
    base = {
        "pyproject.toml": b"[tool.pytest.ini_options]\ntestpaths = ['tests']\naddopts = '-ra'\n"
    }
    branch = {
        "pyproject.toml": b"[tool.pytest.ini_options]\ntestpaths = ['tests', 'examples']\n"
        b"addopts = '-ra'\n",
        "src/example.py": b"value = 2\n",
    }

    # Act
    findings = ConfigurationCheck(base, branch).findings()

    # Assert
    assert findings == []


def test_reports_mixed_change_when_testpaths_lose_entry() -> None:
    # Arrange
    base = {"pyproject.toml": b"[tool.pytest.ini_options]\ntestpaths = ['tests', 'examples']\n"}
    branch = {
        "pyproject.toml": b"[tool.pytest.ini_options]\ntestpaths = ['examples']\n",
        "src/example.py": b"value = 2\n",
    }

    # Act
    findings = ConfigurationCheck(base, branch).findings()

    # Assert
    assert len(findings) == 1


def test_reports_mixed_change_when_dependency_and_policy_share_project_file() -> None:
    # Arrange
    base = {"pyproject.toml": b"[project]\nname = 'example'\n[tool.ty.rules]\nrule = 'error'\n"}
    branch = {"pyproject.toml": b"[project]\nname = 'renamed'\n[tool.ty.rules]\nrule = 'ignore'\n"}

    # Act
    findings = ConfigurationCheck(base, branch).findings()

    # Assert
    assert len(findings) == 1


def test_reports_empty_subject_when_no_candidate_files_are_read() -> None:
    # Arrange
    check = ConfigurationCheck({}, {})

    # Act
    findings = check.findings()

    # Assert
    assert findings[0].rule == "empty-subject"


def test_passes_when_first_testpath_table_is_added_with_code() -> None:
    # Arrange
    base = {"pyproject.toml": b"[project]\nname = 'example'\n"}
    branch = {
        "pyproject.toml": base["pyproject.toml"]
        + b"[tool.pytest.ini_options]\ntestpaths = ['tests']\n",
        "src/example.py": b"value = 2\n",
    }

    # Act
    findings = ConfigurationCheck(base, branch).findings()

    # Assert
    assert findings == []
