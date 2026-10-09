# tenspec conventions

This file holds what a contributor cannot read from the code alone. The [README](README.md)
holds the purpose, the install command and the three quickstarts. The
[documentation pages](docs/index.md) hold the complete reference for every capability and
every limit.

## What this is

One package, `packages/tenspec`, in one uv workspace. It declares tensor requirements and
checks them at a boundary. It composes Pydantic. It owns no second validator.

## The module map is accepted, not a build choice

```
src/tenspec/
  types/      what a declaration permits: shapes, dtypes, properties, TensorType
              operations.py holds the executable interfaces a backend runs, ArrayProperty and
              ArrayTransform, and derives_from, the one nominal-ancestry rule
  runtime/    the array facts and the one check that compares them: arrays, bindings, validation
              validation.py holds the prepared ArrayValidator and its one-shot validate_array form
              arrays.py also holds TensorDeclaration, the carrier every backend alias embeds,
              and the one resolver that turns a declared operation into the class that runs it
  pydantic/   the framework integration: annotations, calls, models, standalone
  numpy.py    the NumPy backend and its declarations
  torch.py    the Torch backend and its declarations
  errors.py   the errors every layer raises
```

Dependencies run one way. `runtime` uses `types`. The backends implement `runtime.arrays`.
`pydantic` composes all of them. Nothing under `types` or `runtime` imports Pydantic's
integration, and nothing outside `numpy.py` and `torch.py` imports an array library.

## Rules a reader would otherwise learn the hard way

- **The common import loads neither array library.** `tests/test_imports.py` measures it.
  Import an array library only inside `numpy.py` or `torch.py`.
- **Write `from typing import Literal as Shape` in an example.** Ruff 0.16.6 cannot follow
  the `tenspec.Shape` re-export and reports F821 and F722 against it. The re-export stays
  public for a later Ruff version. [Ruff and the `Shape` alias](docs/type-checkers.md#ruff-and-the-shape-alias)
  holds the measured table.
- **A declaration runs its transforms before its properties.** `validate_array` checks the
  structure, runs each transform and verifies the concrete class, shape, native dtype and
  device after each one, runs the properties, and commits only then. Preparation splits the
  two sequences once, so the runtime sorts nothing per call.
- **The resolution cache holds classes and format sets, never a value.** It is a bounded
  `lru_cache` per backend instance, and it keys on no backend, so a backend needs no hash.
  A declared class that cannot be hashed takes the uncached route and behaves the same.
- **Test an operation class by identity, not with `issubclass`.** `types.operations.derives_from`
  reads the method resolution order, because `issubclass` against an abstract base class
  hashes its argument, and a caller may write a class that has no usable hash.
- **A refused array commits no binding.** `validate_array` proposes its facts and commits
  them through the public bind operations only after the whole check passes.
- **Only a selected union arm commits bindings.** Each tensor-bearing alternative checks
  with copied facts. It retains its accepted value and facts until Pydantic selects an arm;
  the union then merges that candidate's facts and returns its native value.
- **An unsupported declaration raises AnnotationError.** It never falls back to a check of
  the base array type alone. `TensorDeclaration.__get_pydantic_core_schema__` is that guard.
  The carrier sits in `runtime/arrays.py` so a backend module never imports the resolver.
- **The root exports the common surface only.** A lower operation keeps its defining module:
  `runtime.validation.ArrayValidator` and `validate_array`, `runtime.bindings.Bindings`,
  `runtime.arrays.ArrayBackend`, and `pydantic.annotations.prepare_annotation`.
  Tests and examples import them from there.
- **A tensor-bearing parameter or field checks its default.** A parameter takes
  `Field(validate_default=True)` through `prepare_declaration`. A model field takes
  `FieldInfo.validate_default`, which is the field-specific form Pydantic asks for.
- **`TensorContracts` prepares at Pydantic's schema hook.** Pydantic has already resolved
  each annotation in the caller's own namespace by then, so a declaration written as a
  string, as a name local to the function, or as a name defined after the model reaches
  preparation as a real type. The mixin changes no model configuration. It chains to a
  schema hook a caller's own base owns, and calls the handler otherwise, because
  `BaseModel.__get_pydantic_core_schema__` is deprecated.
- **A method that returns `Self` is checked once per receiver class.** The specialization is
  built under a lock and published whole. The original callable's annotations never change.

## Stack

uv, ruff (all rules), ty (strict), pytest, pydantic. One config and one gate at the root.
`uv run poe check` must be clean before a commit.
