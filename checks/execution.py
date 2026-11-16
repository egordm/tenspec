"""The candidate controls this launcher and its --base argument, so this aggregate alone cannot
guarantee enforcement against the intended base.

That guarantee requires a trusted copy started with a base chosen outside the candidate, at the
lead's merge step now and in a trusted workflow later.
"""

import subprocess
import sys
from typing import ClassVar

from checks import budgets, configuration, growth, inventory, suppressions, vocabulary
from checks.repository import Finding, Repository, comparison_parser, report


class CheckRuns:
    baseline_script: ClassVar[str] = """
import sys
from pathlib import Path
root = Path(sys.argv.pop(1))
sys.path.insert(0, str(root))
if (root / "checks/execution.py").is_file():
    from checks.execution import CheckRuns
    from checks.repository import Repository
    raise SystemExit(CheckRuns(Repository(sys.argv[-1])).candidate())
from checks import budgets, configuration, growth, suppressions, vocabulary
results = [check(sys.argv[1:]) for check in (
    suppressions.main, budgets.main, growth.main, configuration.main, vocabulary.main
)]
raise SystemExit(int(any(results)))
"""

    def __init__(self, repository: Repository) -> None:
        self.repository = repository

    @staticmethod
    def has_base_checks(paths: set[str]) -> bool:
        return any(path.startswith("checks/") for path in paths)

    def baseline(self) -> int:
        if not self.has_base_checks(self.repository.base_paths()):
            return 0
        with self.repository.base_tree() as root:
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    self.baseline_script,
                    str(root),
                    "--base",
                    self.repository.base,
                ],
                cwd=self.repository.root,
                capture_output=True,
                text=True,
                check=False,
            )
        sys.stdout.write(result.stdout)
        if result.returncode not in (0, 1) or result.stderr:
            message = (
                " ".join(result.stderr.splitlines()) or f"base checks exited {result.returncode}"
            )
            report([Finding("checks", 1, "baseline-execution", message)])
        return int(result.returncode != 0 or bool(result.stderr))

    def candidate(self) -> int:
        arguments = ["--base", self.repository.base]
        results = [
            check(arguments)
            for check in (
                inventory.main,
                suppressions.main,
                budgets.main,
                growth.main,
                configuration.main,
                vocabulary.main,
            )
        ]
        return int(any(results))

    def run(self) -> int:
        baseline = self.baseline()
        candidate = self.candidate()
        return int(bool(baseline or candidate))


def main() -> int:
    args = comparison_parser(__doc__ or "").parse_args()
    try:
        return CheckRuns(Repository(args.base)).run()
    except (OSError, ValueError, RuntimeError, SyntaxError) as error:
        return report(
            [Finding("checks", 1, "measurement-error", " ".join(str(error).splitlines()))]
        )
