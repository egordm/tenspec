"""Prevent AST, complexity, module and directory growth beyond default limits."""

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from checks.repository import Finding, Repository, comparison_parser, report
from checks.ruff import Ruff
from checks.source import PythonSource


@dataclass(frozen=True)
class Measurement:
    limits: ClassVar[dict[str, int]] = {"C901": 10, "PLR0912": 12, "PLR0915": 50, "PLR0913": 5}
    line: int
    ast_nodes: int
    values: dict[str, int]

    def excess(self) -> dict[str, int]:
        return {code: value for code, value in self.values.items() if value > self.limits[code]}


class GrowthCheck:
    def __init__(
        self,
        base: dict[tuple[str, str, int], Measurement],
        branch: dict[tuple[str, str, int], Measurement],
    ) -> None:
        self.base = base
        self.branch = branch

    def findings(self) -> list[Finding]:
        if not self.branch:
            return [Finding.empty("growth")]
        findings = []
        for identity, current in self.branch.items():
            path, name, occurrence = identity
            previous = self.base.get(identity)
            label = f"{name} occurrence {occurrence + 1}"
            if previous is None or not previous.excess():
                findings.extend(
                    Finding(path, current.line, code, f"new over-limit function {label}: {value}")
                    for code, value in current.excess().items()
                )
                continue
            for code, value in current.values.items():
                old = previous.values.get(code, 0)
                if value > old:
                    findings.append(
                        Finding(path, current.line, code, f"{label} grew to {value} from {old}")
                    )
            if current.ast_nodes > previous.ast_nodes:
                message = f"{label} grew to {current.ast_nodes} AST nodes from {previous.ast_nodes}"
                findings.append(Finding(path, current.line, "function-nodes", message))
        return findings


class StructureGrowth:
    def __init__(self, base: dict[str, str], branch: dict[str, str]) -> None:
        self.base = base
        self.branch = branch

    def findings(self) -> list[Finding]:
        if not self.branch:
            return [Finding.empty("structure-growth")]
        findings = []
        for path, text in self.branch.items():
            if not path.startswith("packages/tenspec/src/tenspec/"):
                continue
            current, previous = len(text.splitlines()), len(self.base.get(path, "").splitlines())
            if current > max(500, previous):
                findings.append(
                    Finding(
                        path, 1, "module-lines", f"module grew to {current} lines from {previous}"
                    )
                )
        before, after = self.directory_counts(self.base), self.directory_counts(self.branch)
        for path, count in after.items():
            previous = before.get(path, 0)
            if count > max(10, previous):
                findings.append(
                    Finding(
                        path,
                        1,
                        "directory-modules",
                        f"directory grew to {count} modules from {previous}",
                    )
                )
        return findings

    @staticmethod
    def directory_counts(texts: dict[str, str]) -> dict[str, int]:
        directories: dict[str, set[str]] = defaultdict(set)
        for name in texts:
            if not name.startswith("packages/tenspec/src/tenspec/"):
                continue
            path = Path(name)
            if path.stem != "__init__":
                directories[str(path.parent)].add(path.stem)
        return {path: len(modules) for path, modules in directories.items()}


def measurements(
    texts: dict[str, str], diagnostics: list[dict[str, Any]], root: Path
) -> dict[tuple[str, str, int], Measurement]:
    if not texts or not diagnostics:
        raise ValueError("Function files and Ruff measurements must not be empty")
    sources = {path: PythonSource(text) for path, text in texts.items()}
    result = {
        (path, function.name, function.occurrence): Measurement(
            function.line, function.body_nodes, {}
        )
        for path, source in sources.items()
        for function in source.functions.values()
    }
    if not result:
        raise ValueError("Function inventory must not be empty")
    for item in diagnostics:
        path = Path(item["filename"]).resolve().relative_to(root.resolve()).as_posix()
        if path not in sources:
            raise ValueError(f"Ruff diagnostic has no mapped source: {path}")
        if item["location"]["row"] in sources[path].overload_lines:
            continue
        function = sources[path].functions.get(item["location"]["row"])
        if function is None:
            raise ValueError(f"No function at {path}:{item['location']['row']}")
        match = re.search(r"\((\d+)\s*>\s*\d+\)", item["message"])
        if match is None:
            raise ValueError(f"Cannot read Ruff measurement: {item['message']}")
        value = int(match[1])
        if item["code"] == "PLR0915":
            value -= function.docstring_statements
        result[path, function.name, function.occurrence].values[item["code"]] = value
    return result


def tree_texts(root: Path, paths: set[str]) -> dict[str, str]:
    return {
        path: (root / path).read_text()
        for path in paths
        if path.endswith((".py", ".pyi")) and (root / path).is_file()
    }


def main(arguments: list[str] | None = None) -> int:
    args = comparison_parser(__doc__ or "").parse_args(arguments)
    repository = Repository(args.base)
    with repository.base_tree() as root:
        base_texts = tree_texts(root, repository.base_paths())
        base = measurements(base_texts, Ruff(root, list(base_texts)).complexity(), root)
    branch_texts = repository.texts((".py", ".pyi"))
    branch = measurements(
        branch_texts, Ruff(repository.root, list(branch_texts)).complexity(), repository.root
    )
    return report(
        [
            *GrowthCheck(base, branch).findings(),
            *StructureGrowth(base_texts, branch_texts).findings(),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
