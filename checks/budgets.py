"""Ratchet suppressed diagnostics, structural counts and vocabulary counts."""

import tomllib
from collections import Counter
from pathlib import Path
from typing import Self

from checks.repository import Finding, Repository, SourceRoots, comparison_parser, report
from checks.ruff import Ruff
from checks.source import PythonSource


class BudgetCheck:
    def __init__(self, base: str | None, branch: str) -> None:
        self.base = self.parse(base) if base is not None else {}
        self.branch = self.parse(branch)

    @staticmethod
    def parse(text: str) -> dict[str, int]:
        tables = tomllib.loads(text)
        values: dict[str, int] = {}
        for group, entries in tables.items():
            if not isinstance(entries, dict):
                raise TypeError(f"Budget group {group} must be a table")
            for key, value in entries.items():
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError(f"Budget {group}.{key} must be a nonnegative integer")
                values[f"{group}.{key}"] = value
        return values

    @classmethod
    def from_repository(cls, repository: Repository) -> Self:
        path = "quality-budget.toml"
        base = repository.base_text(path) if path in repository.base_paths() else None
        return cls(base, (repository.root / path).read_text())

    def findings(
        self, measured: dict[str, int], *, groups: tuple[str, ...] | None = None
    ) -> list[Finding]:
        if not measured:
            return [Finding.empty("budgets")]
        findings = []
        relevant = {key for key in self.branch if groups is None or key.split(".", 1)[0] in groups}
        for key in measured.keys() | relevant:
            count, budget = measured.get(key, 0), self.branch.get(key)
            if budget != count:
                findings.append(
                    Finding(
                        "quality-budget.toml",
                        1,
                        key,
                        f"measured {count}; budget {budget if budget is not None else 'missing'}; "
                        f"write {count}",
                    )
                )
        for key, budget in self.branch.items():
            if key in self.base and budget > self.base[key]:
                findings.append(
                    Finding(
                        "quality-budget.toml",
                        1,
                        key,
                        f"budget raised from {self.base[key]} to {budget}",
                    )
                )
            elif self.base and key not in self.base and budget > 0:
                findings.append(
                    Finding("quality-budget.toml", 1, key, "new budget must start at zero")
                )
        return findings


def structure_counts(texts: dict[str, str], source_roots: SourceRoots) -> dict[str, int]:
    population = {
        path: text
        for path, text in texts.items()
        if source_roots.contains(path) and path.endswith(".py")
    }
    if not population:
        raise ValueError("Structure source inventory must not be empty")
    directories: Counter[str] = Counter()
    counts = {
        "structure.wide-directories": 0,
        "structure.long-modules": 0,
        "structure.long-functions": 0,
        "structure.dense-functions": 0,
    }
    for path, text in population.items():
        if Path(path).name != "__init__.py":
            directories[str(Path(path).parent)] += 1
        counts["structure.long-modules"] += len(text.splitlines()) > 500
        for function in PythonSource(text).functions.values():
            counts["structure.long-functions"] += function.code_lines > 80
            counts["structure.dense-functions"] += function.longest_run >= 40
    counts["structure.wide-directories"] = sum(count > 10 for count in directories.values())
    return counts


def measure(repository: Repository) -> dict[str, int]:
    texts = repository.texts((".py", ".pyi"))
    return {
        **{
            f"suppressed.{code}": count
            for code, count in Ruff(repository.root, list(texts)).suppressed_counts().items()
        },
        **structure_counts(texts, repository.source_roots),
    }


def main(arguments: list[str] | None = None) -> int:
    args = comparison_parser(__doc__ or "").parse_args(arguments)
    repository = Repository(args.base)
    return report(
        BudgetCheck.from_repository(repository).findings(
            measure(repository), groups=("suppressed", "structure")
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
