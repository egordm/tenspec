"""Measure Ruff findings over an explicit tracked Python and stub inventory."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from checks.repository import command


class Ruff:
    def __init__(self, root: Path, paths: list[str]) -> None:
        self.root = root.resolve()
        self.paths = paths
        if not paths:
            raise ValueError("Ruff source inventory must not be empty")

    @staticmethod
    def parse_diagnostics(output: str) -> list[dict[str, Any]]:
        rows = json.loads(output)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Ruff output must be a JSON list of diagnostics")
        return rows

    def diagnostics(self, options: list[str]) -> list[dict[str, Any]]:
        rows = []
        for start in range(0, len(self.paths), 200):
            arguments = [
                "ruff",
                "check",
                "--no-cache",
                "--no-force-exclude",
                "--no-respect-gitignore",
                "--output-format",
                "json",
                *options,
                "--",
                *self.paths[start : start + 200],
            ]
            output = command(arguments, cwd=self.root, accepts_findings=True)
            batch = self.parse_diagnostics(output)
            self.validate_paths(batch)
            rows.extend(batch)
        return rows

    def validate_paths(self, diagnostics: list[dict[str, Any]]) -> None:
        inventory = {str((self.root / path).resolve()) for path in self.paths}
        for item in diagnostics:
            if str(Path(item["filename"]).resolve()) not in inventory:
                raise ValueError(f"Ruff diagnostic has no inventoried source: {item['filename']}")

    def statistics(self, *, ignore_suppressions: bool) -> dict[str, int]:
        options = ["--ignore-noqa", "--extend-select", "PLR0913"] if ignore_suppressions else []
        return dict(Counter(item["code"] for item in self.diagnostics(options)))

    def suppressed_counts(self) -> dict[str, int]:
        waived = self.statistics(ignore_suppressions=True)
        visible = self.statistics(ignore_suppressions=False)
        return {
            code: count - visible.get(code, 0)
            for code, count in waived.items()
            if count > visible.get(code, 0)
        }

    def complexity(self) -> list[dict[str, Any]]:
        options = [
            "--isolated",
            "--extension",
            "pyi:python",
            "--ignore-noqa",
            "--select",
            "C901,PLR0912,PLR0915,PLR0913",
        ]
        for setting in (
            "mccabe.max-complexity",
            "pylint.max-branches",
            "pylint.max-statements",
            "pylint.max-args",
        ):
            options.extend(["--config", f"lint.{setting}=0"])
        rows = self.diagnostics(options)
        if not rows:
            raise ValueError("Ruff complexity output must contain function measurements")
        return rows
