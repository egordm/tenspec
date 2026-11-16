"""Keep changes to enforcement files separate from other repository changes."""

import tomllib
from pathlib import PurePosixPath
from typing import Any, ClassVar

from checks.budgets import BudgetCheck
from checks.repository import Finding, Repository, comparison_parser, report


class ConfigurationCheck:
    names: ClassVar[frozenset[str]] = frozenset(
        {
            "ruff.toml",
            ".ruff.toml",
            "ty.toml",
            ".importlinter",
            ".pre-commit-config.yaml",
            "poe_tasks.toml",
            "quality-budget.toml",
        }
    )
    tables = ("ruff", "ty", "pytest", "poe", "importlinter")

    def __init__(self, base: dict[str, bytes], branch: dict[str, bytes]) -> None:
        self.base = base
        self.branch = branch

    def findings(self) -> list[Finding]:
        if not self.branch:
            return [Finding.empty("configuration")]
        protected, outside = self.changed_groups()
        if protected and outside:
            message = (
                f"enforcement changes must stand alone; other changes: {', '.join(sorted(outside))}"
            )
            return [Finding(path, 1, "configuration", message) for path in sorted(protected)]
        return []

    def changed_groups(self) -> tuple[set[str], set[str]]:
        protected: set[str] = set()
        outside: set[str] = set()
        for path in self.base.keys() | self.branch.keys():
            before, after = self.base.get(path), self.branch.get(path)
            if before == after:
                continue
            enforcement, other = self.classify(path, before, after)
            if enforcement:
                protected.add(path)
            if other:
                outside.add(path)
        return protected, outside

    def classify(self, path: str, before: bytes | None, after: bytes | None) -> tuple[bool, bool]:
        if path == "quality-budget.toml" and self.lower_budget(before, after):
            return False, False
        if PurePosixPath(path).name == "pyproject.toml":
            return self.project_changes(before, after)
        protected = (
            path.startswith(("checks/", "tests/checks/", ".github/"))
            or PurePosixPath(path).name in self.names
        )
        return protected, not protected

    @staticmethod
    def lower_budget(before: bytes | None, after: bytes | None) -> bool:
        if before is None or after is None:
            return False
        old, new = BudgetCheck.parse(before.decode()), BudgetCheck.parse(after.decode())
        return all(value <= old.get(key, 0) for key, value in new.items())

    def project_changes(self, before: bytes | None, after: bytes | None) -> tuple[bool, bool]:
        old = tomllib.loads(before.decode()) if before is not None else {}
        new = tomllib.loads(after.decode()) if after is not None else {}
        old_tables, new_tables = self.project_tables(old), self.project_tables(new)
        self.allow_testpath_additions(old_tables, new_tables)
        protected = before is None or after is None or old_tables != new_tables
        return protected, old != new

    def project_tables(self, project: dict[str, Any]) -> dict[str, Any]:
        tools = project.get("tool", {})
        tables = {name: tools.pop(name) for name in self.tables if name in tools}
        if not tools:
            project.pop("tool", None)
        return tables

    @staticmethod
    def allow_testpath_additions(old: dict[str, Any], new: dict[str, Any]) -> None:
        old_options = old.get("pytest", {}).get("ini_options", {})
        new_options = new.get("pytest", {}).get("ini_options", {})
        before, after = old_options.get("testpaths", []), new_options.get("testpaths", [])
        if (
            isinstance(before, list)
            and isinstance(after, list)
            and [item for item in after if item in before] == before
        ):
            old_options.pop("testpaths", None)
            new_options.pop("testpaths", None)
            for tables in (old, new):
                pytest = tables.get("pytest")
                if pytest is None:
                    continue
                if not pytest.get("ini_options", {}):
                    pytest.pop("ini_options", None)
                if not pytest:
                    tables.pop("pytest", None)


def main(arguments: list[str] | None = None) -> int:
    args = comparison_parser(__doc__ or "").parse_args(arguments)
    repository = Repository(args.base)
    with repository.base_tree() as root:
        base = {
            path: (root / path).read_bytes()
            for path in repository.base_paths()
            if (root / path).is_file()
        }
    return report(ConfigurationCheck(base, repository.files()).findings())


if __name__ == "__main__":
    raise SystemExit(main())
