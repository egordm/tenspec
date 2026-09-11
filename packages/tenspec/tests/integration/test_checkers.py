"""Three real type checkers read the caller fixture, and each of them must accept it.

Each checker runs once, against the interpreter that runs this test, and reads
`checked_caller.py`. That file declares only valid Tenspec uses and asserts the type of each
result, so a checker that loses an array type, a signature or a return exits nonzero here.
The exit status carries the result, and the command with its output arrives with the failure.

These are ordinary tests. `uv run poe check` runs them with the rest of the suite, and the
normal development environment carries all three tools.

Two settings are not defaults:

- Pyright takes `--project pyrightconfig.json`, the file beside the fixture, which turns on
  `enableExperimentalFeatures`. Pyright gates its PEP 747 `TypeForm` support behind that
  setting, and the `validate` call in the fixture needs it.
- Pyrefly takes `--preset default`. With no `pyrefly.toml` in reach it falls back to the
  `basic` preset, which reports nothing at all, so a pass under it would prove nothing.
"""

import subprocess
import sys
from pathlib import Path

import pytest

CALLER = (Path(__file__).parent / "checked_caller.py").resolve()
"""The one file every checker reads."""

PYRIGHT_CONFIG = Path(__file__).parent / "pyrightconfig.json"
"""Pyright's settings for that file alone."""

TIMEOUT_SECONDS = 300
"""How long one checker may take. Pyright downloads its own Node on the first run."""

BIN = Path(sys.executable).parent
"""Where a synced environment puts the three executables."""

CHECKERS = {
    "ty": (str(BIN / "ty"), "check", "--python", sys.executable, str(CALLER)),
    "pyright": (
        str(BIN / "pyright"),
        "--project",
        str(PYRIGHT_CONFIG),
        "--pythonpath",
        sys.executable,
        str(CALLER),
    ),
    "pyrefly": (
        str(BIN / "pyrefly"),
        "check",
        "--preset",
        "default",
        "--python-interpreter-path",
        sys.executable,
        str(CALLER),
    ),
}
"""How each checker runs, pointed at the interpreter that runs this test."""

ACCEPTED = 0
"""The exit status all three tools use when they report no diagnostic."""


@pytest.mark.parametrize("argv", CHECKERS.values(), ids=CHECKERS)
def test_a_checker_accepts_the_caller(argv: tuple[str, ...]) -> None:
    # Act
    run = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, check=False)

    # Assert
    assert run.returncode == ACCEPTED, (
        f"{' '.join(argv)}\nstdout:\n{run.stdout}\nstderr:\n{run.stderr}"
    )
