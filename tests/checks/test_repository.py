import pytest
from checks.repository import Repository, SourceRoots


@pytest.mark.parametrize(
    "text",
    [
        "",
        "[tool]\nquality = 1\n",
        "[tool.quality]\n",
        "[tool.quality]\nsource-roots = []\n",
        '[tool.quality]\nsource-roots = "library"\n',
        "[tool.quality]\nsource-roots = [1]\n",
        '[tool.quality]\nsource-roots = [""]\n',
        '[tool.quality]\nsource-roots = ["."]\n',
        '[tool.quality]\nsource-roots = ["../library"]\n',
        '[tool.quality]\nsource-roots = ["/library"]\n',
    ],
)
def test_reports_configuration_fix_when_source_roots_are_invalid(text: str) -> None:
    # Arrange: each project file has an absent, empty or invalid source setting.

    # Act & Assert
    with pytest.raises(
        ValueError,
        match=r'write \[tool\.quality\] with source-roots = \["library"\] in pyproject\.toml',
    ):
        SourceRoots.from_project(text)


def test_reports_configuration_fix_when_source_roots_are_constructed_empty() -> None:
    # Arrange: no source directories are configured.

    # Act & Assert
    with pytest.raises(ValueError, match=r"tool\.quality\.source-roots must be a nonempty list"):
        SourceRoots(paths=())


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("library/module.py", True),
        ("library/domain/module.py", True),
        ("library-copy/module.py", False),
        ("outside/library/module.py", False),
    ],
)
def test_matches_directory_boundary_when_testing_source_membership(
    path: str, *, expected: bool, source_roots: SourceRoots
) -> None:
    # Arrange: source_roots reads the fixture's library directory setting.

    # Act
    contains = source_roots.contains(path)

    # Assert
    assert contains is expected


def test_maps_renamed_python_file_to_origin_when_git_reports_renames() -> None:
    # Arrange
    name_status = "R097\0library/old.py\0library/new.py\0R100\0donor.txt\0library/gained.py\0"

    # Act
    renamed = Repository.renames(name_status, (".py", ".pyi"))

    # Assert
    assert renamed == {"library/new.py": "library/old.py"}
