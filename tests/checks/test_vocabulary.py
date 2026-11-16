import pytest
from checks.repository import Repository, SourceRoots
from checks.vocabulary import Term, VocabularyCheck, read_terms


@pytest.mark.parametrize(
    "text",
    [
        "batchSidecarPath",
        "batch_sidecar_path",
        "SIDECAR",
        "SidecarPath",
        "batchSIDECARPath",
        "SiDeCaR",
        "sIdecar",
        "sidecar2",
        "2sidecar",
        "sidecarpet",
    ],
)
def test_reports_banned_term_when_identifier_has_word_component(
    text: str, source_roots: SourceRoots
) -> None:
    # Arrange
    check = VocabularyCheck([Term("sidecar", "Describe the contents", "manifest")], source_roots)

    # Act
    findings = check.findings({"example.py": text})

    # Assert
    assert len(findings) == 1
    assert findings[0].rule == "vocabulary.sidecar"


def test_passes_when_text_contains_no_banned_substring(source_roots: SourceRoots) -> None:
    # Arrange
    check = VocabularyCheck([Term("sidecar", "Describe the contents", "manifest")], source_roots)

    # Act
    findings = check.findings({"example.py": "manifest = 1"})

    # Assert
    assert findings == []


def test_counts_file_name_when_name_contains_banned_term(source_roots: SourceRoots) -> None:
    # Arrange
    check = VocabularyCheck([Term("sidecar", "Describe the contents", "manifest")], source_roots)

    # Act
    counts = check.counts({"batch_sidecar.py": ""})

    # Assert
    assert counts["vocabulary.sidecar"] == 1


def test_skips_definitions_when_scanning_terms_and_own_fixtures(source_roots: SourceRoots) -> None:
    # Arrange
    check = VocabularyCheck([Term("sidecar", "Describe the contents", "manifest")], source_roots)

    # Act
    findings = check.findings(
        {
            "checks/banned-terms.toml": "sidecar",
            "quality-budget.toml": "sidecar",
            "tests/checks/test_terms.py": "sidecar",
            "library/example.py": "manifest = 1",
        }
    )

    # Assert
    assert findings == []


def test_counts_process_identifiers_when_source_mentions_records(source_roots: SourceRoots) -> None:
    # Arrange
    check = VocabularyCheck([Term("forbidden", "Describe contents", "manifest")], source_roots)

    # Act
    counts = check.counts(
        {"library/example.py": "# M12-S2 D12 D123 B123 G2 gap M1 port\n", "docs/example.md": "D12"}
    )

    # Assert
    assert counts["vocabulary.process-identifiers"] == 6


def test_loads_term_metadata_when_configuration_is_valid() -> None:
    # Arrange
    text = '[[terms]]\nword = "sidecar"\nreason = "Describe contents"\nreplacement = "manifest"\n'

    # Act
    terms = read_terms(text)

    # Assert
    assert terms == [Term("sidecar", "Describe contents", "manifest")]


@pytest.mark.parametrize(
    "path",
    ["example.txt", "example.rst", "example.json", "example.yaml", "example.ipynb", "SIDECAR.bin"],
)
def test_reports_banned_term_when_other_tracked_formats_contain_it(
    path: str, source_roots: SourceRoots
) -> None:
    # Arrange
    check = VocabularyCheck([Term("sidecar", "Describe contents", "manifest")], source_roots)
    text = "" if "SIDECAR" in path else "sidecar"

    # Act
    findings = check.findings({path: text})

    # Assert
    assert len(findings) == 1


def test_reports_empty_subject_when_no_vocabulary_files_are_read(source_roots: SourceRoots) -> None:
    # Arrange
    check = VocabularyCheck([Term("sidecar", "Describe contents", "manifest")], source_roots)

    # Act
    findings = check.findings({})

    # Assert
    assert findings[0].rule == "empty-subject"


def test_reports_error_when_term_definitions_are_empty(source_roots: SourceRoots) -> None:
    # Arrange
    terms: list[Term] = []

    # Act & Assert
    with pytest.raises(ValueError, match="must not be empty"):
        VocabularyCheck(terms, source_roots)


def test_reports_path_when_tracked_entry_has_no_utf8_file_contents(
    source_roots: SourceRoots,
) -> None:
    # Arrange
    check = VocabularyCheck([Term("sidecar", "Describe contents", "manifest")], source_roots)
    files = {"binary.dat": b"\xff"}
    paths = ["binary.dat", "sidecar-submodule"]
    texts = Repository.decode_utf8(files, paths)

    # Act
    findings = check.findings(texts)

    # Assert
    assert len(findings) == 1
    assert findings[0].path == "sidecar-submodule"


def test_counts_source_use_when_definitions_and_budgets_also_contain_term(
    source_roots: SourceRoots,
) -> None:
    # Arrange
    check = VocabularyCheck([Term("sidecar", "Describe contents", "manifest")], source_roots)
    texts = {
        "checks/banned-terms.toml": 'word = "sidecar"',
        "quality-budget.toml": "[vocabulary]\nsidecar = 0",
        "tests/checks/test_terms.py": 'value = "sidecar"',
        "library/module.py": 'value = "sidecar"',
    }

    # Act
    counts = check.counts(texts)

    # Assert
    assert counts["vocabulary.sidecar"] == 1


@pytest.mark.parametrize("path", ["library/module.py", "outside/module.py"])
def test_counts_process_identifier_when_file_is_under_configured_root(
    path: str, source_roots: SourceRoots
) -> None:
    # Arrange
    check = VocabularyCheck([Term("forbidden", "Describe contents", "manifest")], source_roots)

    # Act
    counts = check.counts({path: "# D12\n"})

    # Assert
    assert counts["vocabulary.process-identifiers"] == int(path.startswith("library/"))


def test_counts_process_identifiers_when_two_roots_are_configured() -> None:
    # Arrange
    roots = SourceRoots.from_project(
        '[tool.quality]\nsource-roots = ["library/one", "library/two"]\n'
    )
    check = VocabularyCheck([Term("forbidden", "Describe contents", "manifest")], roots)
    texts = {"library/one/module.py": "# D12\n", "library/two/module.pyi": "# D13\n"}

    # Act
    counts = check.counts(texts)

    # Assert
    assert counts["vocabulary.process-identifiers"] == 2


def test_reports_empty_subject_when_only_excluded_vocabulary_files_are_read(
    source_roots: SourceRoots,
) -> None:
    # Arrange
    check = VocabularyCheck([Term("forbidden", "Describe contents", "manifest")], source_roots)

    # Act
    findings = check.findings({"quality-budget.toml": "[vocabulary]\nforbidden = 0\n"})

    # Assert
    assert [item.rule for item in findings] == ["empty-subject"]


def test_reports_error_when_only_excluded_vocabulary_files_are_counted(
    source_roots: SourceRoots,
) -> None:
    # Arrange
    check = VocabularyCheck([Term("forbidden", "Describe contents", "manifest")], source_roots)

    # Act & Assert
    with pytest.raises(ValueError, match="empty after exclusions"):
        check.counts({"quality-budget.toml": "[vocabulary]\nforbidden = 0\n"})
