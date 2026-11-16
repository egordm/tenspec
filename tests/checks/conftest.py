from pathlib import Path

import pytest
from checks.repository import SourceRoots


@pytest.fixture
def source_roots(tmp_path: Path) -> SourceRoots:
    project = tmp_path / "pyproject.toml"
    project.write_text('[tool.quality]\nsource-roots = ["library"]\n')
    return SourceRoots.from_project(project.read_text())
