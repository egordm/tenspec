import pytest
from checks.budgets import BudgetCheck, structure_counts
from checks.ruff import Ruff


def test_reports_increase_when_budget_is_raised() -> None:
    # Arrange
    check = BudgetCheck("[suppressed]\nC901 = 2\n", "[suppressed]\nC901 = 3\n")

    # Act
    findings = check.findings({"suppressed.C901": 2})

    # Assert
    assert any("raised" in item.message for item in findings)


def test_passes_when_budget_and_measurement_fall() -> None:
    # Arrange
    check = BudgetCheck("[suppressed]\nC901 = 3\n", "[suppressed]\nC901 = 2\n")

    # Act
    findings = check.findings({"suppressed.C901": 2})

    # Assert
    assert findings == []


def test_reports_unbudgeted_rule_when_diagnostic_is_new() -> None:
    # Arrange
    check = BudgetCheck(None, "[suppressed]\n")

    # Act
    findings = check.findings({"suppressed.C901": 1})

    # Assert
    assert len(findings) == 1
    assert "missing" in findings[0].message


def test_reports_count_when_measurement_exceeds_budget() -> None:
    # Arrange
    check = BudgetCheck(None, "[structure]\nlong-functions = 1\n")

    # Act
    findings = check.findings({"structure.long-functions": 2})

    # Assert
    assert len(findings) == 1


def test_reports_new_budget_when_positive_row_is_added_after_introduction() -> None:
    # Arrange
    check = BudgetCheck("[suppressed]\nC901 = 1\n", "[suppressed]\nC901 = 1\nBLE001 = 2\n")

    # Act
    findings = check.findings({"suppressed.C901": 1, "suppressed.BLE001": 2})

    # Assert
    assert len(findings) == 1


def test_counts_structure_when_boundaries_are_exceeded() -> None:
    # Arrange
    dense = "def calculate():\n" + "    value = 1\n" * 81
    files = {f"packages/tenspec/src/tenspec/domain/module{number}.py": "" for number in range(11)}
    files["packages/tenspec/src/tenspec/domain/large.py"] = dense + "\n" * 420
    files["packages/tenspec/src/tenspec/domain/__init__.py"] = ""

    # Act
    counts = structure_counts(files)

    # Assert
    assert counts == {
        "structure.wide-directories": 1,
        "structure.long-modules": 1,
        "structure.long-functions": 1,
        "structure.dense-functions": 1,
    }


def test_excludes_nested_body_when_measuring_parent() -> None:
    # Arrange
    source = (
        'def parent():\n    """Explain.\n\n    More.\n    """\n    def child():\n'
        + "        value = 1\n" * 81
        + "    return child\n"
    )

    # Act
    counts = structure_counts({"packages/tenspec/src/tenspec/domain.py": source})

    # Assert
    assert counts["structure.long-functions"] == 1
    assert counts["structure.dense-functions"] == 1


def test_reports_error_when_ruff_statistics_output_is_empty() -> None:
    # Arrange
    output = ""

    # Act & Assert
    with pytest.raises(ValueError, match="Expecting value"):
        Ruff.parse_diagnostics(output)


def test_preserves_string_lines_when_comment_marker_is_inside_literal() -> None:
    # Arrange
    source = (
        'def render():\n    value = """first\n'
        + "# text inside literal\n" * 40
        + 'last"""\n    return value\n'
    )

    # Act
    counts = structure_counts({"packages/tenspec/src/tenspec/render.py": source})

    # Assert
    assert counts["structure.dense-functions"] == 1


def test_reports_exact_number_when_budget_exceeds_measurement() -> None:
    # Arrange
    check = BudgetCheck(None, "[suppressed]\nC901 = 3\n")

    # Act
    findings = check.findings({"suppressed.C901": 1})

    # Assert
    assert len(findings) == 1
    assert "write 1" in findings[0].message


def test_reports_empty_subject_when_budget_measurements_are_empty() -> None:
    # Arrange
    check = BudgetCheck(None, "[structure]\nlong-modules = 0\n")

    # Act
    findings = check.findings({})

    # Assert
    assert findings[0].rule == "empty-subject"


def test_reports_error_when_structure_inventory_is_empty() -> None:
    # Arrange
    texts: dict[str, str] = {}

    # Act & Assert
    with pytest.raises(ValueError, match="must not be empty"):
        structure_counts(texts)


def test_resets_run_when_nested_definition_separates_regions() -> None:
    # Arrange
    source = (
        "def parent():\n"
        + "    value = 1\n" * 20
        + "    def child():\n        return 1\n"
        + "    value = 2\n" * 20
    )

    # Act
    counts = structure_counts({"packages/tenspec/src/tenspec/example.py": source})

    # Assert
    assert counts["structure.dense-functions"] == 0
