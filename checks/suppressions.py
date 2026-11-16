"""Refuse newly added Python suppression comments, including moved comments."""

import difflib
import io
import re
import tokenize

from checks.repository import Finding, Repository, comparison_parser, report


class SuppressionCheck:
    pattern = re.compile(
        r"#\s*(?:noqa\b|ruff\s*:\s*(?:noqa\b|ignore\b|isort\s*:\s*(?:skip_file|skip|off)\b)"
        r"|isort\s*:\s*(?:skip_file|skip|off)\b"
        r"|(?:type|ty|pyright)\s*:\s*ignore\b|pragma\s*:\s*no\s+cover\b"
        r"|ast-grep-ignore\b|fmt\s*:\s*(?:off|skip)\b)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        base: dict[str, str],
        branch: dict[str, str],
        changed: set[str] | None = None,
        renamed: dict[str, str] | None = None,
    ) -> None:
        self.base = base
        self.branch = branch
        self.changed = set(branch) if changed is None else changed
        self.renamed = renamed or {}

    def findings(self) -> list[Finding]:
        if not self.branch:
            return [Finding.empty("suppressions")]
        findings = []
        for path, text in self.branch.items():
            if path not in self.changed:
                continue
            added = self.added_rows(self.base.get(self.renamed.get(path, path), ""), text)
            findings.extend(
                Finding(path, token.start[0], "suppression", "new suppression is forbidden")
                for token in tokenize.generate_tokens(io.StringIO(text).readline)
                if token.type == tokenize.COMMENT
                and token.start[0] in added
                and self.pattern.search(token.string)
            )
        return findings

    @staticmethod
    def added_rows(before: str, after: str) -> set[int]:
        matcher = difflib.SequenceMatcher(
            a=before.splitlines(), b=after.splitlines(), autojunk=False
        )
        return {
            row + 1
            for tag, _, _, start, stop in matcher.get_opcodes()
            if tag in {"insert", "replace"}
            for row in range(start, stop)
        }


def main(arguments: list[str] | None = None) -> int:
    args = comparison_parser(__doc__ or "").parse_args(arguments)
    repository = Repository(args.base)
    branch = repository.texts((".py", ".pyi"))
    changed = repository.changed_paths()
    paths = repository.base_paths()
    renamed = repository.renamed_paths((".py", ".pyi"))
    origins = {renamed.get(path, path) for path in branch if path in changed}
    base = {path: repository.base_text(path) for path in origins if path in paths}
    return report(SuppressionCheck(base, branch, changed, renamed).findings())


if __name__ == "__main__":
    raise SystemExit(main())
