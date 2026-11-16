from pathlib import Path

import pytest
from checks.growth import GrowthCheck, Measurement, StructureGrowth, measurements
from checks.repository import SourceRoots
from checks.source import PythonSource


def test_reports_growth_when_over_limit_function_adds_statements() -> None:
    # Arrange
    base = {("module.py", "Owner.calculate", 0): Measurement(2, 53, {"PLR0915": 51})}
    branch = {("module.py", "Owner.calculate", 0): Measurement(2, 56, {"PLR0915": 54})}

    # Act
    findings = GrowthCheck(base, branch).findings()

    # Assert
    assert {item.rule for item in findings} == {"PLR0915", "function-nodes"}


def test_passes_when_over_limit_function_shrinks() -> None:
    # Arrange
    base = {("module.py", "calculate", 0): Measurement(1, 53, {"PLR0915": 51})}
    branch = {("module.py", "calculate", 0): Measurement(1, 52, {"PLR0915": 50})}

    # Act
    findings = GrowthCheck(base, branch).findings()

    # Assert
    assert findings == []


def test_reports_new_function_when_no_counterpart_exists() -> None:
    # Arrange
    branch = {("new.py", "calculate", 0): Measurement(1, 12, {"PLR0913": 6})}

    # Act
    findings = GrowthCheck({}, branch).findings()

    # Assert
    assert len(findings) == 1
    assert findings[0].rule == "PLR0913"


def test_reports_argument_growth_when_another_metric_is_over_limit() -> None:
    # Arrange
    base = {("module.py", "calculate", 0): Measurement(1, 80, {"PLR0915": 60, "PLR0913": 2})}
    branch = {("module.py", "calculate", 0): Measurement(1, 80, {"PLR0915": 60, "PLR0913": 3})}

    # Act
    findings = GrowthCheck(base, branch).findings()

    # Assert
    assert [item.rule for item in findings] == ["PLR0913"]


def test_reads_qualified_identity_when_ruff_reports_function_on_method_line() -> None:
    # Arrange
    text = "class Owner:\n    async def calculate(self):\n        return 1\n"
    diagnostics = [
        {
            "filename": "/module.py",
            "location": {"row": 2},
            "code": "C901",
            "message": "calculate is too complex (11 > 10)",
        }
    ]

    # Act
    result = measurements({"module.py": text}, diagnostics, Path("/"))

    # Assert
    assert result["module.py", "Owner.calculate", 0] == Measurement(2, 2, {"C901": 11})


def test_measures_each_occurrence_when_qualified_name_has_duplicate_definitions() -> None:
    # Arrange
    text = "def calculate(value): ...\ndef calculate(value):\n    return value + 1\n"
    diagnostics = [
        {
            "filename": "/module.py",
            "location": {"row": 1},
            "code": "C901",
            "message": "calculate is too complex (1 > 0)",
        },
        {
            "filename": "/module.py",
            "location": {"row": 2},
            "code": "C901",
            "message": "calculate is too complex (2 > 0)",
        },
    ]

    # Act
    result = measurements({"module.py": text}, diagnostics, Path("/"))

    # Assert
    assert result["module.py", "calculate", 1].values == {"C901": 2}
    assert result["module.py", "calculate", 0].values == {"C901": 1}


def test_reports_added_occurrence_when_name_already_has_allowance() -> None:
    # Arrange
    base = {("module.py", "calculate", 0): Measurement(1, 100, {"PLR0913": 6})}
    branch = base | {("module.py", "calculate", 1): Measurement(10, 100, {"PLR0913": 6})}

    # Act
    findings = GrowthCheck(base, branch).findings()

    # Assert
    assert len(findings) == 1
    assert findings[0].line == 10


def test_reports_growth_when_statements_are_merged_without_reducing_nodes() -> None:
    # Arrange
    before = PythonSource("def calculate():\n    first = 1\n    second = 2\n").functions[1]
    after = PythonSource("def calculate():\n    first, second = 1, 2\n").functions[1]
    base = {("module.py", "calculate", 0): Measurement(1, before.body_nodes, {"PLR0915": 51})}
    branch = {("module.py", "calculate", 0): Measurement(1, after.body_nodes, {"PLR0915": 50})}

    # Act
    findings = GrowthCheck(base, branch).findings()

    # Assert
    assert after.body_nodes >= before.body_nodes
    assert any(item.rule == "function-nodes" for item in findings)


def test_reports_size_growth_when_module_and_directory_are_already_large(
    source_roots: SourceRoots,
) -> None:
    # Arrange
    base = {f"library/domain/module{index}.py": "" for index in range(11)}
    base["library/domain/module0.py"] = "value = 1\n" * 501
    branch = base | {
        "library/domain/new.py": "",
        "library/domain/module0.py": "value = 1\n" * 502,
    }

    # Act
    findings = StructureGrowth(base, branch, source_roots).findings()

    # Assert
    assert {item.rule for item in findings} == {"module-lines", "directory-modules"}


def test_reports_empty_subject_when_no_functions_are_measured() -> None:
    # Arrange
    check = GrowthCheck({}, {})

    # Act
    findings = check.findings()

    # Assert
    assert findings[0].rule == "empty-subject"


def test_reports_error_when_diagnostic_has_no_source() -> None:
    # Arrange
    diagnostic = {
        "filename": "/missing.py",
        "location": {"row": 1},
        "code": "C901",
        "message": "complexity (1 > 0)",
    }

    # Act & Assert
    with pytest.raises(ValueError, match="no mapped source"):
        measurements({"module.py": "def calculate(): return 1"}, [diagnostic], Path("/"))


def test_reports_error_when_function_file_has_no_functions() -> None:
    # Arrange
    diagnostic = {"filename": "/module.py", "location": {"row": 1}}

    # Act & Assert
    with pytest.raises(ValueError, match="Function inventory must not be empty"):
        measurements({"module.py": "value = 1"}, [diagnostic], Path("/"))


def test_passes_when_large_test_directory_gains_module(source_roots: SourceRoots) -> None:
    # Arrange
    base = {"library/module.py": ""} | {
        f"tests/core/test_case{index}.py": "" for index in range(133)
    }
    branch = base | {"tests/core/test_new_case.py": ""}

    # Act
    findings = StructureGrowth(base, branch, source_roots).findings()

    # Assert
    assert findings == []


@pytest.mark.parametrize("remove", [False, True])
@pytest.mark.parametrize(
    ("imports", "decorator"),
    [
        ("from typing import overload", "overload"),
        ("from typing_extensions import overload as signature", "signature"),
        ("import typing", "typing.overload"),
        ("import typing_extensions as types", "types.overload"),
    ],
)
def test_preserves_allowance_when_overload_declarations_change(
    imports: str, decorator: str, *, remove: bool
) -> None:
    # Arrange
    implementation = "def calculate(a, b, c, d, e, f):\n    return a\n"
    declarations = f"@{decorator}\ndef calculate(a: int) -> int: ...\n" * 2
    versions = [f"{imports}\n{implementation}", f"{imports}\n{declarations}{implementation}"]
    measured = []
    for text in versions:
        rows = [row for row, line in enumerate(text.splitlines(), 1) if line.startswith("def ")]
        diagnostics = [
            {
                "filename": "/module.py",
                "location": {"row": row},
                "code": "PLR0913",
                "message": f"Too many arguments ({6 if row == rows[-1] else 1} > 0)",
            }
            for row in rows
        ]
        measured.append(measurements({"module.py": text}, diagnostics, Path("/")))
    before, after = reversed(measured) if remove else measured

    # Act
    findings = GrowthCheck(before, after).findings()

    # Assert
    assert findings == []
    assert list(before) == list(after) == [("module.py", "calculate", 0)]


@pytest.mark.parametrize(
    ("nested", "indent", "statement_increase"),
    [
        ("    def child():\n        value = 1\n", "        ", 1),
        ("    async def child():\n        value = 1\n", "        ", 1),
        ("    class Child:\n        value = 1\n", "        ", 0),
        ("    class Child:\n        def method(self):\n            value = 1\n", "            ", 0),
    ],
)
def test_preserves_parent_measurements_when_nested_definition_gains_docstring(
    nested: str, indent: str, statement_increase: int
) -> None:
    # Arrange
    before = f"def parent(a, b, c, d, e, f):\n{nested}    return 1\n"
    after = before.replace(
        f"{indent}value = 1",
        f'{indent}"""Explain.\n\n{indent}More.\n{indent}"""\n{indent}value = 1',
    )
    measured = []
    for text, statements in ((before, 2), (after, 2 + statement_increase)):
        diagnostics = [
            {
                "filename": "/module.py",
                "location": {"row": 1},
                "code": code,
                "message": f"measurement ({value} > 0)",
            }
            for code, value in (("PLR0913", 6), ("PLR0915", statements))
        ]
        measured.append(measurements({"module.py": text}, diagnostics, Path("/")))

    # Act
    findings = GrowthCheck(*measured).findings()

    # Assert
    assert findings == []
    identity = ("module.py", "parent", 0)
    assert measured[0][identity].ast_nodes == measured[1][identity].ast_nodes
    assert measured[0][identity].values == measured[1][identity].values


def test_preserves_statement_excess_when_nested_class_has_docstring() -> None:
    # Arrange
    text = (
        'def parent():\n    class Child:\n        """Explain."""\n        pass\n'
        + "    value = 1\n" * 50
    )
    diagnostic = {
        "filename": "/module.py",
        "location": {"row": 1},
        "code": "PLR0915",
        "message": "Too many statements (51 > 0)",
    }
    branch = measurements({"module.py": text}, [diagnostic], Path("/"))

    # Act
    findings = GrowthCheck({}, branch).findings()

    # Assert
    assert [finding.rule for finding in findings] == ["PLR0915"]


def test_reports_error_when_diagnostic_matches_neither_function_nor_overload() -> None:
    # Arrange
    text = (
        "from typing import overload\n@overload\ndef calculate(a): ...\n"
        "def calculate(a): return a\n"
    )
    diagnostic = {
        "filename": "/module.py",
        "location": {"row": 2},
        "code": "PLR0913",
        "message": "Too many arguments (6 > 0)",
    }

    # Act & Assert
    with pytest.raises(ValueError, match=r"No function at module\.py:2"):
        measurements({"module.py": text}, [diagnostic], Path("/"))


def test_passes_when_large_documentation_module_grows(source_roots: SourceRoots) -> None:
    # Arrange
    path = "docs/example.py"
    base: dict[str, str] = {"library/module.py": "", path: "value = 1\n" * 501}
    branch: dict[str, str] = base | {path: "value = 1\n" * 502}

    # Act
    findings = StructureGrowth(base, branch, source_roots).findings()

    # Assert
    assert findings == []


@pytest.mark.parametrize("path", ["library/grown.py", "outside/grown.py"])
def test_reports_module_growth_when_file_is_under_configured_root(
    path: str, source_roots: SourceRoots
) -> None:
    # Arrange
    base = {"library/anchor.py": "", path: "value = 1\n" * 500}
    branch = base | {path: "value = 1\n" * 501}

    # Act
    findings = StructureGrowth(base, branch, source_roots).findings()

    # Assert
    assert [item.path for item in findings] == ([path] if path.startswith("library/") else [])


def test_reports_both_modules_when_two_roots_grow() -> None:
    # Arrange
    roots = SourceRoots.from_project(
        '[tool.quality]\nsource-roots = ["library/one", "library/two"]\n'
    )
    paths = ["library/one/module.py", "library/two/module.py"]
    base: dict[str, str] = dict.fromkeys(paths, "value = 1\n" * 500)
    branch: dict[str, str] = dict.fromkeys(paths, "value = 1\n" * 501)

    # Act
    findings = StructureGrowth(base, branch, roots).findings()

    # Assert
    assert {item.path for item in findings} == set(paths)


def test_reports_empty_subject_when_no_configured_source_is_read(source_roots: SourceRoots) -> None:
    # Arrange
    branch = {"outside/module.py": "value = 1\n"}

    # Act
    findings = StructureGrowth({}, branch, source_roots).findings()

    # Assert
    assert [item.rule for item in findings] == ["empty-subject"]


def test_passes_when_over_limit_function_keeps_its_size_in_renamed_file() -> None:
    # Arrange
    base = {("old.py", "calculate", 0): Measurement(1, 53, {"PLR0913": 7})}
    branch = {("new.py", "calculate", 0): Measurement(4, 53, {"PLR0913": 7})}

    # Act
    findings = GrowthCheck(base, branch, renamed={"new.py": "old.py"}).findings()

    # Assert
    assert findings == []


def test_reports_growth_when_over_limit_function_grows_in_renamed_file() -> None:
    # Arrange
    base = {("old.py", "calculate", 0): Measurement(1, 53, {"PLR0913": 7})}
    branch = {("new.py", "calculate", 0): Measurement(1, 53, {"PLR0913": 8})}

    # Act
    findings = GrowthCheck(base, branch, renamed={"new.py": "old.py"}).findings()

    # Assert
    assert [(item.path, item.rule) for item in findings] == [("new.py", "PLR0913")]


def test_passes_when_large_module_keeps_its_length_in_renamed_file(
    source_roots: SourceRoots,
) -> None:
    # Arrange
    base: dict[str, str] = {"library/old.py": "value = 1\n" * 600}
    branch: dict[str, str] = {"library/new.py": "value = 1\n" * 600}
    renamed = {"library/new.py": "library/old.py"}

    # Act
    findings = StructureGrowth(base, branch, source_roots, renamed=renamed).findings()

    # Assert
    assert findings == []


def test_reports_module_growth_when_large_module_grows_in_renamed_file(
    source_roots: SourceRoots,
) -> None:
    # Arrange
    base: dict[str, str] = {"library/old.py": "value = 1\n" * 600}
    branch: dict[str, str] = {"library/new.py": "value = 1\n" * 601}
    renamed = {"library/new.py": "library/old.py"}

    # Act
    findings = StructureGrowth(base, branch, source_roots, renamed=renamed).findings()

    # Assert
    assert [item.message for item in findings] == ["module grew to 601 lines from 600"]


def test_passes_when_wide_directory_is_renamed_with_its_modules(source_roots: SourceRoots) -> None:
    # Arrange
    base: dict[str, str] = {f"library/old/module{index}.py": "" for index in range(12)}
    branch: dict[str, str] = {f"library/new/module{index}.py": "" for index in range(12)}
    renamed = {path: path.replace("/new/", "/old/") for path in branch}

    # Act
    findings = StructureGrowth(base, branch, source_roots, renamed=renamed).findings()

    # Assert
    assert findings == []


def test_reports_directory_growth_when_renamed_wide_directory_gains_a_module(
    source_roots: SourceRoots,
) -> None:
    # Arrange
    base: dict[str, str] = {f"library/old/module{index}.py": "" for index in range(12)}
    branch: dict[str, str] = {f"library/new/module{index}.py": "" for index in range(12)}
    renamed = {path: path.replace("/new/", "/old/") for path in branch}
    branch["library/new/added.py"] = ""

    # Act
    findings = StructureGrowth(base, branch, source_roots, renamed=renamed).findings()

    # Assert
    assert [item.message for item in findings] == ["directory grew to 13 modules from 12"]
