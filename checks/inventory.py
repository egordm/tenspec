"""Reject tracked paths that bypass the ordinary source checks."""

from checks.repository import Finding, Repository, comparison_parser, report


class InventoryCheck:
    def __init__(self, paths: list[str], ignored: list[str]) -> None:
        self.paths = paths
        self.ignored = ignored

    def findings(self) -> list[Finding]:
        if not self.paths:
            return [Finding.empty("tracked-inventory")]
        return [
            *(
                Finding(path, 1, "tracked-ignored", "tracked file matches repository ignore rules")
                for path in sorted(self.ignored)
            ),
            *(
                Finding(path, 1, "tracked-notebook", "tracked notebooks bypass source checks")
                for path in sorted(self.paths)
                if path.lower().endswith(".ipynb")
            ),
        ]


def main(arguments: list[str] | None = None) -> int:
    args = comparison_parser(__doc__ or "").parse_args(arguments)
    repository = Repository(args.base)
    return report(InventoryCheck(repository.paths(), repository.ignored_paths()).findings())


if __name__ == "__main__":
    raise SystemExit(main())
