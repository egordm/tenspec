"""Count prohibited vocabulary in text, identifiers and tracked file names."""

import re
import tomllib
from dataclasses import dataclass

from checks.budgets import BudgetCheck
from checks.repository import Finding, Repository, SourceRoots, comparison_parser, report


@dataclass(frozen=True)
class Term:
    word: str
    reason: str
    replacement: str


class VocabularyCheck:
    process_pattern = re.compile(
        r"M\d+-S\d+|\bD\d{2,3}\b|\bB\d{3}\b|\bG\d\b gap|M1 port", re.IGNORECASE
    )

    def __init__(self, terms: list[Term], source_roots: SourceRoots) -> None:
        if not terms:
            raise ValueError("Banned term definitions must not be empty")
        self.terms = terms
        self.source_roots = source_roots

    def findings(self, texts: dict[str, str]) -> list[Finding]:
        texts = {
            path: text
            for path, text in texts.items()
            if path not in {"checks/banned-terms.toml", "quality-budget.toml"}
            and not path.startswith("tests/checks/")
        }
        if not texts:
            return [Finding.empty("vocabulary")]
        findings = []
        for path, text in texts.items():
            findings.extend(self.terms_in(path, path, line=1))
            process_source = self.source_roots.contains(path) and path.endswith((".py", ".pyi"))
            for row, line in enumerate(text.splitlines(), 1):
                findings.extend(self.terms_in(path, line, line=row))
                if process_source:
                    findings.extend(
                        Finding(
                            path,
                            row,
                            "vocabulary.process-identifiers",
                            f"process identifier {match[0]!r} belongs in project records",
                        )
                        for match in self.process_pattern.finditer(line)
                    )
        return findings

    def terms_in(self, path: str, text: str, *, line: int) -> list[Finding]:
        return [
            Finding(path, line, f"vocabulary.{term.word}", f"{term.reason}; use {term.replacement}")
            for term in self.terms
            for _ in re.finditer(re.escape(term.word), text, re.IGNORECASE)
        ]

    def counts(self, texts: dict[str, str]) -> dict[str, int]:
        if not texts:
            raise ValueError("Vocabulary file inventory must not be empty")
        counts = {f"vocabulary.{term.word}": 0 for term in self.terms}
        counts["vocabulary.process-identifiers"] = 0
        for finding in self.findings(texts):
            if finding.rule == "empty-subject":
                raise ValueError("Vocabulary file inventory must not be empty after exclusions")
            counts[finding.rule] += 1
        return counts


def read_terms(text: str) -> list[Term]:
    return [
        Term(row["word"], row["reason"], row["replacement"]) for row in tomllib.loads(text)["terms"]
    ]


def main(arguments: list[str] | None = None) -> int:
    args = comparison_parser(__doc__ or "").parse_args(arguments)
    repository = Repository(args.base)
    terms = read_terms((repository.root / "checks/banned-terms.toml").read_text())
    check = VocabularyCheck(terms, repository.source_roots)
    findings = check.findings(repository.utf8_texts())
    counts = check.counts(repository.utf8_texts())
    budgets = BudgetCheck.from_repository(repository)
    excess = budgets.findings(counts, groups=("vocabulary",))
    rules = {finding.rule for finding in excess}
    locations = [item for item in findings if item.rule in rules]
    return report([*excess, *locations])


if __name__ == "__main__":
    raise SystemExit(main())
