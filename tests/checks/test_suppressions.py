import pytest
from checks.suppressions import SuppressionCheck


@pytest.mark.parametrize(
    "directive",
    [
        "noqa: C901",
        "NOQA",
        "ruff: noqa",
        "ruff: ignore",
        "ruff: isort: skip",
        "type: ignore",
        "ty: ignore[rule]",
        "pyright: ignore",
        "pragma: no cover",
        "ast-grep-ignore",
        "fmt: off",
        "fmt: skip",
    ],
)
def test_reports_suppression_when_comment_is_added(directive: str) -> None:
    # Arrange
    before = {"library/example.py": "value = 1\n"}
    after = {"library/example.py": f"value = 1  # {directive}\n"}

    # Act
    findings = SuppressionCheck(before, after).findings()

    # Assert
    assert len(findings) == 1
    assert findings[0].line == 1


def test_passes_when_existing_comment_stays_on_unchanged_line() -> None:
    # Arrange
    original = "value = 1  # " + "noqa: C901\n"
    before = {"example.py": original}
    after = {"example.py": "other = 2\n" + original}

    # Act
    findings = SuppressionCheck(before, after).findings()

    # Assert
    assert findings == []


def test_reports_suppression_when_waived_function_moves_between_files() -> None:
    # Arrange
    text = "def calculate():  # " + "noqa: C901\n    return 1\n"
    before = {"old.py": text}
    after = {"new.py": text}

    # Act
    findings = SuppressionCheck(before, after).findings()

    # Assert
    assert len(findings) == 1
    assert findings[0].path == "new.py"


@pytest.mark.parametrize(
    "directive",
    [
        "isort: skip_file",
        "ruff: isort: skip_file",
        "isort: skip",
        "ruff: isort: skip",
        "isort: off",
        "ruff: isort: off",
    ],
)
def test_reports_suppression_when_isort_directive_is_added(directive: str) -> None:
    # Arrange
    branch = {"example.pyi": f"# {directive}\nvalue: int\n"}

    # Act
    findings = SuppressionCheck({}, branch).findings()

    # Assert
    assert len(findings) == 1


def test_passes_when_string_quotes_directive() -> None:
    # Arrange
    branch = {"example.py": 'description = "# noqa: C901"\n'}

    # Act
    findings = SuppressionCheck({}, branch).findings()

    # Assert
    assert findings == []


def test_reports_empty_subject_when_no_python_files_are_read() -> None:
    # Arrange
    check = SuppressionCheck({}, {})

    # Act
    findings = check.findings()

    # Assert
    assert findings[0].rule == "empty-subject"
