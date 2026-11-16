from pathlib import Path

import pytest
from checks.configuration import ConfigurationCheck, main
from checks.repository import command


def contract_text() -> str:
    return """[importlinter]
root_package = library

[importlinter:contract:boundary]
name = Module boundary
type = forbidden
source_modules =
    library.alpha
    library.stable
forbidden_modules =
    library.beta
    library.peer
ignore_imports =
    library.stable -> library.peer
"""


@pytest.fixture
def moved_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    files = {
        "pyproject.toml": '[tool.quality]\nsource-roots = ["library"]\n',
        ".importlinter": contract_text(),
        "library/__init__.py": "",
        "library/alpha.py": "value = 1\n",
        "library/beta.py": "value = 2\n",
        "library/stable.py": "value = 3\n",
        "library/peer.py": "value = 4\n",
    }
    for path, text in files.items():
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
    command(["git", "init", "--initial-branch=main"], cwd=tmp_path)
    command(["git", "add", "."], cwd=tmp_path)
    command(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--no-verify",
            "-s",
            "-m",
            "Baseline",
        ],
        cwd=tmp_path,
    )
    for name in ("alpha", "beta"):
        (tmp_path / f"library/{name}.py").rename(tmp_path / f"library/moved_{name}.py")
    (tmp_path / ".importlinter").write_text(
        contract_text()
        .replace("library.alpha", "library.moved_alpha")
        .replace("library.beta", "library.moved_beta")
    )
    command(["git", "add", "--all"], cwd=tmp_path)
    return tmp_path


def test_passes_when_contract_follows_two_renamed_modules(moved_repository: Path) -> None:
    # Arrange: the temporary repository moves alpha and beta and updates their existing contract.
    assert (moved_repository / "library/moved_alpha.py").is_file()

    # Act
    result = main(["--base", "HEAD"])

    # Assert
    assert result == 0


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(
            "",
            "\n[importlinter:contract:new]\nname = New\ntype = forbidden\n"
            "source_modules = library.moved_alpha\nforbidden_modules = library.moved_beta\n",
            id="new-contract",
        ),
        pytest.param(
            "[importlinter:contract:boundary]", "[importlinter:contract:renamed]", id="contract-id"
        ),
        pytest.param("name = Module boundary", "name = Other boundary", id="contract-name"),
        pytest.param("type = forbidden", "type = independence", id="contract-type"),
        pytest.param("root_package = library", "root_package = other", id="root-package"),
        pytest.param(
            "type = forbidden", "type = forbidden\nas_packages = False", id="policy-option"
        ),
        pytest.param("    library.stable\n", "", id="existing-module-dropped"),
        pytest.param("    library.stable\n", "    library.peer\n", id="existing-module-changed"),
        pytest.param(
            "ignore_imports =\n",
            "ignore_imports =\n    library.peer -> library.stable\n",
            id="existing-import-ignored",
        ),
        pytest.param("    library.stable -> library.peer\n", "", id="existing-ignore-removed"),
        pytest.param(
            "    library.stable -> library.peer",
            "    library.moved_alpha -> library.peer",
            id="ignore-with-unchanged-endpoint",
        ),
        pytest.param(
            "ignore_imports =\n",
            "ignore_imports =\n    library.moved_alpha -> library.stable\n",
            id="new-ignore-with-unchanged-endpoint",
        ),
        pytest.param("    library.moved_alpha\n", "    library.missing\n", id="nonexistent-module"),
        pytest.param("    library.stable\n", "    library.stable.**\n", id="existing-wildcard"),
        pytest.param("type = forbidden", "type = custom", id="unknown-type"),
        pytest.param("type = forbidden", "type = forbidden\nunknown =", id="empty-policy-option"),
        pytest.param("    library.peer\n", "    external.module\n", id="external-module"),
        pytest.param(
            "    library.moved_alpha\n", "    library.moved_*\n", id="unanchored-wildcard"
        ),
        pytest.param("", "\n[DEFAULT]\nhidden = value\n", id="defaults"),
        pytest.param("", "# New explanation\n", id="comment-change"),
        pytest.param("    library.stable\n", "\tlibrary.stable\n", id="stable-line-format"),
        pytest.param("source_modules =", "source_modules:", id="header-separator"),
        pytest.param("forbidden_modules =", "FORBIDDEN_MODULES =", id="header-case"),
        pytest.param("ignore_imports =", "ignore_imports  =", id="header-spacing"),
    ],
)
def test_reports_mixed_change_when_policy_changes_during_move(
    before: str, after: str, moved_repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    config = moved_repository / ".importlinter"
    text = config.read_text()
    config.write_text(text.replace(before, after) if before else text + after)

    # Act
    result = main(["--base", "HEAD"])

    # Assert
    assert result == 1
    assert (
        ".importlinter:1: configuration: enforcement changes must stand alone"
        in capsys.readouterr().out
    )


def test_reports_mixed_change_when_contract_is_removed_during_move(moved_repository: Path) -> None:
    # Arrange
    (moved_repository / ".importlinter").write_text("[importlinter]\nroot_package = library\n")

    # Act
    result = main(["--base", "HEAD"])

    # Assert
    assert result == 1


@pytest.mark.parametrize("path", ["ruff.toml", "checks/policy.py", "nested/.importlinter"])
def test_reports_mixed_change_when_other_enforcement_changes_during_move(
    path: str, moved_repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    destination = moved_repository / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("# Policy\n")
    command(["git", "add", "--all"], cwd=moved_repository)

    # Act
    result = main(["--base", "HEAD"])

    # Assert
    assert result == 1
    assert f"{path}:1: configuration:" in capsys.readouterr().out


@pytest.mark.parametrize("change", ["add", "remove"])
def test_passes_when_contract_tracks_module_existence(change: str, moved_repository: Path) -> None:
    # Arrange
    config = moved_repository / ".importlinter"
    if change == "add":
        (moved_repository / "library/added.py").write_text("value = 5\n")
        config.write_text(
            config.read_text().replace("source_modules =", "source_modules =\n    library.added")
        )
    else:
        (moved_repository / "library/stable.py").unlink()
        (moved_repository / "library/peer.py").unlink()
        config.write_text(
            "\n".join(
                line
                for line in config.read_text().splitlines()
                if "library.stable" not in line and "library.peer" not in line
            )
        )
    command(["git", "add", "--all"], cwd=moved_repository)

    # Act
    result = main(["--base", "HEAD"])

    # Assert
    assert result == 0


def test_passes_when_ignore_is_added_between_two_new_modules(moved_repository: Path) -> None:
    # Arrange
    config = moved_repository / ".importlinter"
    config.write_text(config.read_text() + "    library.moved_alpha -> library.moved_beta\n")

    # Act
    result = main(["--base", "HEAD"])

    # Assert
    assert result == 0


@pytest.mark.parametrize("arrow", ["->", "  ->  ", "\t->\t"], ids=["bare", "spaces", "tabs"])
def test_passes_when_new_ignore_uses_another_arrow_spacing(
    arrow: str, moved_repository: Path
) -> None:
    # Arrange
    config = moved_repository / ".importlinter"
    config.write_text(
        config.read_text().replace(
            "ignore_imports =\n",
            f"ignore_imports =\n    library.moved_alpha{arrow}library.moved_beta\n",
        )
    )

    # Act
    result = main(["--base", "HEAD"])

    # Assert
    assert result == 0


def test_reports_mixed_change_when_inline_layers_header_is_respelled() -> None:
    # Arrange
    base, branch = layered_trees("old", "new")
    for files, header in ((base, "layers = "), (branch, "LAYERS: ")):
        files[".importlinter"] = files[".importlinter"].replace(b"layers =\n    ", header.encode())

    # Act
    findings = ConfigurationCheck(base, branch).findings()

    # Assert
    assert [finding.path for finding in findings] == [".importlinter"]


def test_reports_mixed_change_when_named_module_still_exists(moved_repository: Path) -> None:
    # Arrange
    (moved_repository / "library/alpha.py").write_text("value = 1\n")
    command(["git", "add", "--all"], cwd=moved_repository)

    # Act
    result = main(["--base", "HEAD"])

    # Assert
    assert result == 1


def test_reports_mixed_change_when_only_non_source_files_supply_module_paths(
    moved_repository: Path,
) -> None:
    # Arrange
    destination = moved_repository / "docs"
    destination.mkdir()
    for name in ("alpha", "beta"):
        (moved_repository / f"library/moved_{name}.py").rename(destination / f"moved_{name}.py")
    command(["git", "add", "--all"], cwd=moved_repository)

    # Act
    result = main(["--base", "HEAD"])

    # Assert
    assert result == 1


@pytest.mark.parametrize("root", ["library", "src/library", "packages/component/src/library"])
@pytest.mark.parametrize("pattern", ["", ".*", ".**"])
def test_passes_when_contract_follows_package_with_terminal_pattern(
    root: str, pattern: str
) -> None:
    # Arrange
    contract = (
        "[importlinter]\nroot_package = library\n"
        "[importlinter:contract:boundary]\nname = Boundary\ntype = independence\n"
        f"modules =\n    library.old{pattern}\n    library.stable\n"
    )
    base = {
        "pyproject.toml": f'[tool.quality]\nsource-roots = ["{root}"]\n'.encode(),
        ".importlinter": contract.encode(),
        f"{root}/__init__.py": b"",
        f"{root}/old/__init__.py": b"",
        f"{root}/old/child.py": b"",
        f"{root}/stable.py": b"",
    }
    branch = {
        path.replace("/old/", "/new/"): data
        for path, data in base.items()
        if path != ".importlinter"
    }
    branch[".importlinter"] = contract.replace("library.old", "library.new").encode()

    # Act
    findings = ConfigurationCheck(base, branch).findings()

    # Assert
    assert findings == []


@pytest.mark.parametrize(
    ("kind", "field", "other"),
    [
        ("protected", "protected_modules", "allowed_importers = library.stable\n"),
        ("protected", "allowed_importers", "protected_modules = library.stable\n"),
        ("layers", "containers", "layers =\n    child\n"),
        ("acyclic_siblings", "ancestors", ""),
        ("acyclic_siblings", "skip_descendants", "ancestors = library\n"),
    ],
)
def test_passes_when_other_module_fields_follow_move(kind: str, field: str, other: str) -> None:
    # Arrange
    contract = (
        "[importlinter]\nroot_package = library\n"
        f"[importlinter:contract:boundary]\nname = Boundary\ntype = {kind}\n"
        f"{field} =\n    library.alpha\n{other}"
    )
    base = {
        "pyproject.toml": b'[tool.quality]\nsource-roots = ["library"]\n',
        ".importlinter": contract.encode(),
        "library/__init__.py": b"",
        "library/alpha/__init__.py": b"",
        "library/alpha/child.py": b"",
        "library/stable.py": b"",
    }
    branch = {
        path.replace("/alpha/", "/moved_alpha/"): data
        for path, data in base.items()
        if path != ".importlinter"
    }
    branch[".importlinter"] = contract.replace("library.alpha", "library.moved_alpha").encode()

    # Act
    findings = ConfigurationCheck(base, branch).findings()

    # Assert
    assert findings == []


@pytest.mark.parametrize(
    "text",
    [
        "invalid INI",
        "[DEFAULT]\nname = hidden\n",
        "[importlinter]\nroot_package = library\nroot_package = other\n",
    ],
)
def test_reports_mixed_change_when_contract_file_is_unusable(
    text: str, moved_repository: Path
) -> None:
    # Arrange
    (moved_repository / ".importlinter").write_text(text)

    # Act
    result = main(["--base", "HEAD"])

    # Assert
    assert result == 1


def layered_trees(
    before: str,
    after: str,
    *,
    containers: tuple[str, ...] = ("library",),
    target: str = "new",
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    settings = "[importlinter]\nroot_package = library\n[importlinter:contract:boundary]\n"
    settings += "name = Layer boundary\ntype = layers\n"
    if containers:
        settings += "containers =\n" + "".join(f"    {name}\n" for name in containers)
    base = {
        "pyproject.toml": b'[tool.quality]\nsource-roots = ["library"]\n',
        "library/__init__.py": b"",
    }
    for container in containers or ("library",):
        prefix = container.replace(".", "/")
        base.update(
            {f"{prefix}/{name}.py": b"" for name in ("__init__", "old", "stable", "peer", "bottom")}
        )
    branch = {
        path.replace("/old.py", f"/{target.replace('.', '/')}.py"): data
        for path, data in base.items()
    }
    for files, layers in ((base, before), (branch, after)):
        files[".importlinter"] = (
            settings + "layers =\n" + "".join(f"    {line}\n" for line in layers.splitlines())
        ).encode()
    return base, branch


@pytest.mark.parametrize(
    ("before", "after", "containers", "target"),
    [
        ("library.old\nlibrary.stable", "library.new\nlibrary.stable", (), "new"),
        ("old\nstable", "new\nstable", ("library",), "new"),
        ("old | stable | peer", "new | stable | peer", ("library",), "new"),
        ("stable : old : peer", "stable : new : peer", ("library",), "new"),
        ("(old) | (stable) | peer", "(new) | (stable) | peer", ("library",), "new"),
        ("old\nstable", "sub.new\nstable", ("library",), "sub.new"),
        ("old\nstable", "new\nstable", ("library.first", "library.second"), "new"),
        ("old | stable\npeer", "stable\nnew | peer", ("library",), "new"),
        ("old\nstable\npeer", "stable\nnew\npeer", ("library",), "new"),
    ],
)
def test_passes_when_layered_module_follows_rename(
    before: str, after: str, containers: tuple[str, ...], target: str
) -> None:
    # Arrange
    base, branch = layered_trees(before, after, containers=containers, target=target)

    # Act
    findings = ConfigurationCheck(base, branch).findings()

    # Assert
    assert findings == []


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param("old | stable\npeer", "new\nstable | peer", id="changed-layer-membership"),
        pytest.param("old\nstable\npeer", "new\npeer\nstable", id="reordered-layers"),
        pytest.param("old | stable | peer", "new : stable : peer", id="changed-separator"),
        pytest.param("old | stable | peer", "new | peer | stable", id="reordered-members"),
        pytest.param("old | (stable)", "new | stable", id="changed-optionality"),
    ],
)
def test_reports_mixed_change_when_unchanged_layer_policy_changes(before: str, after: str) -> None:
    # Arrange
    base, branch = layered_trees(before, after)

    # Act
    findings = ConfigurationCheck(base, branch).findings()

    # Assert
    assert [finding.path for finding in findings] == [".importlinter"]


def test_reports_mixed_change_when_relative_layer_still_exists_in_another_container() -> None:
    # Arrange
    base, branch = layered_trees(
        "old\nstable", "new\nstable", containers=("library.first", "library.second")
    )
    branch["library/second/old.py"] = branch.pop("library/second/new.py")

    # Act
    findings = ConfigurationCheck(base, branch).findings()

    # Assert
    assert [finding.path for finding in findings] == [".importlinter"]


@pytest.mark.parametrize("option", ["exhaustive = true", "exhaustive_ignores = stable"])
def test_reports_mixed_change_when_layer_flag_changes(option: str) -> None:
    # Arrange
    base, branch = layered_trees("old\nstable", "new\nstable")
    branch[".importlinter"] += (option + "\n").encode()

    # Act
    findings = ConfigurationCheck(base, branch).findings()

    # Assert
    assert [finding.path for finding in findings] == [".importlinter"]


def test_passes_when_layered_module_is_split_into_two_new_modules() -> None:
    # Arrange
    base, branch = layered_trees("old\nstable", "new | extra\nstable")
    branch["library/extra.py"] = b""

    # Act
    findings = ConfigurationCheck(base, branch).findings()

    # Assert
    assert findings == []


@pytest.mark.parametrize("change", ["add", "remove"])
def test_passes_when_layer_entry_tracks_module_existence(change: str) -> None:
    # Arrange
    before, after = (
        ("old\nstable", "old | new\nstable") if change == "add" else ("old\nstable", "stable")
    )
    base, branch = layered_trees(before, after)
    if change == "add":
        branch["library/old.py"] = base["library/old.py"]
    else:
        branch.pop("library/new.py")

    # Act
    findings = ConfigurationCheck(base, branch).findings()

    # Assert
    assert findings == []
