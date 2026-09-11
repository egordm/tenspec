# Type checkers and Ruff

A declaration is an ordinary annotation, so a checker reads it as the array type it carries.
**No checker compares two shapes.** A shape that disagrees is a run-time refusal, never a
static error.

The [repository README](https://github.com/egordm/tenspec#develop) holds the development
commands. One of them, `uv run poe check`, runs three checkers over one caller file,
`packages/tenspec/tests/integration/checked_caller.py`, as part of the ordinary test suite.
Every checker must accept it. That file declares valid uses alone and asserts the type of each
result, so a checker that reads a declaration as `Any`, or that loses a signature or a return,
fails the run. It covers the spellings all three checkers support, not every public spelling. A
feature it does not exercise is untested there.

| Checker | Version measured | Setting the run needs |
|---|---|---|
| ty | 0.0.78 | none |
| Pyright | 1.1.411 | `enableExperimentalFeatures` |
| Pyrefly | 1.2.0 | `--preset default`, because `basic` reports nothing |

These are the versions this repository pins and measured, not a claim about today's releases.

Pyright gates its PEP 747 `TypeForm` support behind an experimental setting. Turn it on when
you call `validate` with a declaration:

```json
{ "enableExperimentalFeatures": true }
```

Pyrefly 1.2.0 refuses the direct spelling, `validate(array, Float[Shape["rows"]])`, with
`bad-argument-type`, because it does not read a subscripted type alias as a `TypeForm`
argument, and it then loses the type of the result. ty and Pyright accept that spelling.
Pyrefly accepts the tuple spelling, which is the one the caller file uses.

## Ruff and the `Shape` alias

`tenspec.Shape` is a public re-export of `typing.Literal`, and it works at run time. Ruff
0.16.6 does not follow the re-export, so it reads the shape text as a forward annotation and
reports a false diagnostic against it:

| Shape written as | Ruff 0.16.6 reports |
|---|---|
| `from tenspec import Shape`, one axis name | `F821 Undefined name` |
| `from tenspec import Shape`, several axes | `F722 Syntax error in forward annotation` |
| `from typing import Literal as Shape` | nothing |

Write `from typing import Literal as Shape` while that holds. The re-export stays public for a
later Ruff version. The workaround is narrow and keeps `F821` on for the rest of your file,
which is worth more than silencing the rule.

## The measured support matrix

The table holds the minimum tested version of each array library: the oldest release this
project installed and then exercised with the three entry points, the built-in properties and
the transform. `pyproject.toml` declares these same floors, with one marker per Python release.

| Python | Minimum tested NumPy | Minimum tested Torch |
|---|---|---|
| 3.13 | 2.1.0 | 2.6.0 |
| 3.14 | 2.3.2 | 2.9.0 |

This is a minimum tested claim, not a claim about the oldest release that can install. An
older patch release can carry a wheel that no measurement here covers.

The measurement ran on macOS, on an arm64 machine, on the CPU alone. No GPU and no trained
model took part in it. Another operating system or processor carries its own set of wheels,
and no measurement covers one.

Python 3.13 and 3.14 are the tested interpreters. `requires-python` permits a later release,
which no measurement covers.

## Versions this repository exercises

| Package | Version |
|---|---|
| Python | 3.13.2 |
| NumPy | 2.5.2 |
| Torch | 2.14.0 |
| pydantic | 2.13.5 |
| Ruff | 0.16.6 |
