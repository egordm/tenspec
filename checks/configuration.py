"""Keep changes to enforcement files separate from other repository changes."""

import re
import tomllib
from configparser import ConfigParser, Error
from pathlib import PurePosixPath
from typing import Any, ClassVar

from checks.budgets import BudgetCheck
from checks.repository import Finding, Repository, SourceRoots, comparison_parser, report


class LayerContractChanges:
    """Project absolute or container-relative entries while preserving surviving layer policy."""

    def __init__(self, options: dict[str, str], changed: set[str]) -> None:
        self.enabled = options.get("type") == "layers"
        self.containers = tuple(options.get("containers", "").split()) or ("",)
        self.changed = changed
        self.option_indent: int | None = None

    def remaining(self, text: str) -> list[str] | None:
        layers = []
        for line in text.splitlines():
            if not line.strip():
                continue
            remaining = self.remaining_line(line)
            if remaining is None:
                return None
            if remaining:
                layers.append(remaining)
        return layers

    def remaining_line(self, line: str) -> str | None:
        if "|" in line and ":" in line:
            return None
        separator = "|" if "|" in line else ":"
        entries = line.split(separator)
        if any(not entry.strip() for entry in entries):
            return None
        return separator.join(
            entry for entry in entries if not self.changed_module(entry.strip())
        ).strip()

    def changed_module(self, entry: str) -> bool:
        name = entry[1:-1] if entry.startswith("(") and entry.endswith(")") else entry
        resolved = [f"{container}.{name}" if container else name for container in self.containers]
        return all(
            all(part.isidentifier() for part in module.split(".")) and module in self.changed
            for module in resolved
        )

    def raw_line(self, line: str) -> tuple[str, bool]:
        entry = line.strip()
        if not self.enabled or not entry or entry.startswith(("#", ";")):
            return line, False
        indent = len(line) - len(line.lstrip())
        if self.option_indent is not None and indent > self.option_indent:
            return self.raw_projection(line, entry), True
        self.option_indent = None
        header = re.fullmatch(r"layers\s*[=:]\s*(.*)", entry, flags=re.IGNORECASE)
        if header is None:
            return line, False
        self.option_indent = indent
        if not header[1]:
            return line, False
        written = line.rstrip()
        return written[: len(written) - len(header[1])] + self.raw_projection(line, header[1]), True

    def raw_projection(self, line: str, entry: str) -> str:
        remaining = self.remaining_line(entry)
        if remaining is None:
            return line
        if not remaining:
            return ""
        indent = line[: len(line) - len(line.lstrip())]
        ending = line[len(line.rstrip("\r\n")) :]
        body = line.rstrip("\r\n")
        trailing = body[len(body.rstrip()) :]
        return f"{indent}{remaining}{trailing}{ending}"

    @classmethod
    def raw_lines(
        cls, data: bytes, sections: dict[str, dict[str, str]], changed: set[str]
    ) -> list[tuple[str, bool]]:
        contract = cls({}, changed)
        lines = []
        for line in data.decode().splitlines(keepends=True):
            section = re.fullmatch(r"\[(.+)\]", line.strip())
            if section is not None:
                contract = cls(sections.get(section[1], {}), changed)
            lines.append(contract.raw_line(line))
        return lines


class ImportContractChanges:
    module_fields: ClassVar[dict[str, frozenset[str]]] = {
        "forbidden": frozenset({"source_modules", "forbidden_modules", "ignore_imports"}),
        "independence": frozenset({"modules", "ignore_imports"}),
        "protected": frozenset({"protected_modules", "allowed_importers", "ignore_imports"}),
        "layers": frozenset({"containers", "ignore_imports", "layers"}),
        "acyclic_siblings": frozenset({"ancestors", "skip_descendants", "ignore_imports"}),
    }

    def __init__(self, base: dict[str, bytes], branch: dict[str, bytes]) -> None:
        self.base = base
        self.branch = branch

    def follows_modules(self, before: bytes, after: bytes) -> bool:
        """Preserve policy and stable entries while paths appear or disappear.

        A changed ignored-import entry requires both endpoints to qualify.

        Returns:
            Whether only module entries follow additions or removals in the source trees.
        """
        try:
            old, new = self.parse(before), self.parse(after)
            base_modules, branch_modules = self.modules(self.base), self.modules(self.branch)
        except (Error, UnicodeDecodeError, ValueError):
            return False
        if old == new or old.keys() != new.keys():
            return False
        removed, added = base_modules - branch_modules, branch_modules - base_modules
        sections_follow = all(
            self.section_follows_modules(section, options, new[section], removed, added)
            for section, options in old.items()
        )
        return sections_follow and self.remaining_policy(
            before, removed, old
        ) == self.remaining_policy(after, added, new)

    @staticmethod
    def parse(data: bytes) -> dict[str, dict[str, str]]:
        parser = ConfigParser()
        parser.read_string(data.decode())
        if parser.defaults() or "importlinter" not in parser:
            raise ValueError("import contracts must have an importlinter section and no defaults")
        return {section: dict(parser[section]) for section in parser.sections()}

    @staticmethod
    def modules(files: dict[str, bytes]) -> set[str]:
        project = files.get("pyproject.toml", b"")
        roots = SourceRoots.from_project(project.decode())
        modules: set[str] = set()
        for root in roots.paths:
            package = PurePosixPath(root)
            for name in files:
                path = PurePosixPath(name)
                if path.suffix != ".py" or not path.is_relative_to(package):
                    continue
                parts = (package.name, *path.relative_to(package).with_suffix("").parts)
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                modules.update(".".join(parts[:length]) for length in range(1, len(parts) + 1))
        return modules

    def section_follows_modules(
        self,
        section: str,
        old: dict[str, str],
        new: dict[str, str],
        removed: set[str],
        added: set[str],
    ) -> bool:
        if not section.startswith("importlinter:contract:") or old.get("type") != new.get("type"):
            return old == new
        fields = self.module_fields.get(old.get("type", ""), frozenset())
        if {key: value for key, value in old.items() if key not in fields} != {
            key: value for key, value in new.items() if key not in fields
        }:
            return False
        for key in fields:
            before, after = old.get(key, ""), new.get(key, "")
            if key == "layers":
                old_layers = LayerContractChanges(old, removed).remaining(before)
                new_layers = LayerContractChanges(new, added).remaining(after)
                if old_layers is None or old_layers != new_layers:
                    return False
                continue
            if before != after and self.remaining_lines(
                before, removed, key
            ) != self.remaining_lines(after, added, key):
                return False
        return True

    @classmethod
    def remaining_lines(cls, text: str, changed: set[str], key: str) -> list[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return [line for line in lines if not cls.changed_entry(line, changed, key)]

    @classmethod
    def remaining_policy(
        cls, data: bytes, changed: set[str], sections: dict[str, dict[str, str]]
    ) -> list[str]:
        """Retain raw non-module lines, including comments and stable formatting.

        Returns:
            Lines left after removing eligible entries and module-field headers.
        """
        fields = frozenset().union(*cls.module_fields.values())
        remaining = []
        for line, layer_value in LayerContractChanges.raw_lines(data, sections, changed):
            if layer_value:
                if line:
                    remaining.append(line)
                continue
            entry = line.strip()
            header = re.fullmatch(r"(\w+)\s*[=:]\s*(.*)", entry)
            if header is not None and header[1] in fields:
                keys = (header[1],)
                entry = header[2]
                # The header's own spelling is policy text: keep it for the comparison.
                written = line.rstrip()
                remaining.append(written[: len(written) - len(entry)])
                if not entry:
                    continue
            else:
                keys = fields
            if not any(cls.changed_entry(entry, changed, key) for key in keys):
                remaining.append(line)
        return remaining

    @staticmethod
    def changed_entry(line: str, changed: set[str], key: str) -> bool:
        expressions = (
            [endpoint.strip() for endpoint in line.split("->")]
            if key == "ignore_imports"
            else [line]
        )
        if key == "ignore_imports" and len(expressions) != 2:
            return False
        for expression in expressions:
            parts = expression.split(".")
            if parts[-1] in {"*", "**"}:
                parts.pop()
            if not parts or not all(part.isidentifier() for part in parts):
                return False
            if ".".join(parts) not in changed:
                return False
        return True


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
    tables = ("ruff", "ty", "pytest", "poe", "importlinter", "quality")

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
        if (
            path == ".importlinter"
            and before is not None
            and after is not None
            and ImportContractChanges(self.base, self.branch).follows_modules(before, after)
        ):
            return False, True
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
