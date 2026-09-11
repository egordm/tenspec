# Getting started

Tenspec declares the shape and the dtype of a NumPy array or a PyTorch tensor, in the
annotation beside the value. A PyTorch declaration can also require a device. Tenspec checks
that declaration at one boundary: a function, a Pydantic model, or one standalone call.

A declaration is an ordinary annotation, so a type checker reads it and untouched code still
runs. Tenspec accepts or refuses the array the caller passed. It converts nothing, moves
nothing, and copies nothing unless the declaration names a transform that copies.

## Install

```{include} ../README.md
:start-after: "## Install"
:end-before: "## What it gives you"
:relative-docs: docs/
:relative-images:
```

## The three forms

```{include} ../README.md
:start-after: "## Quickstart"
:end-before: "## Documentation"
:relative-docs: docs/
:relative-images:
```

## Where to read next

| Page | What it answers |
|---|---|
| [Array annotations](array-annotations.md) | What a declaration says about shape, dtype and device |
| [Runtime validation](runtime-validation.md) | How `@checked` and `validate` check a function, a method and one value |
| [Pydantic models](pydantic-models.md) | How `TensorContracts` relates the fields of your own model |
| [Custom checks and transforms](custom-checks.md) | How to write a property or a transform of your own |
| {doc}`Validate NumPy arrays <examples/array_checks>` | A tutorial this site runs on every build |
| [Type checkers and Ruff](type-checkers.md) | The checkers, the settings they need, and the measured support matrix |
| [API reference](api.md) | The names you import, and the advanced interfaces |

```{toctree}
:hidden:
:maxdepth: 1

Getting started <self>
Array annotations <array-annotations>
Runtime validation <runtime-validation>
Pydantic models <pydantic-models>
Custom checks and transforms <custom-checks>
```

```{toctree}
:hidden:
:caption: Examples
:maxdepth: 1

Validate NumPy arrays <examples/array_checks>
Custom checks and read-only arrays <examples/custom_operations>
More examples <examples>
```

```{toctree}
:hidden:
:caption: Reference
:maxdepth: 1

Type checkers and Ruff <type-checkers>
API reference <api>
Errors and limits <errors-and-limits>
```
