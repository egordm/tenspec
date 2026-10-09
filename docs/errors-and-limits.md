# Failures and limits

## The error classes

| Error | Import from | Base | Raised when |
|---|---|---|---|
| `AnnotationError` | `tenspec` | `TypeError` | a declaration is invalid, unsupported, or reached Pydantic with no entry point |
| `TensorMismatch` | `tenspec.errors` | `ValueError` | an array disagrees with its declared requirement |
| `BindingError` | `tenspec.errors` | `ValueError` | a dimension lookup found no extent. Its `reason` is `no_scope`, `unbound` or `not_scalar` |
| `ConstraintEvaluationError` | `tenspec` | `RuntimeError` | a backend cannot inspect a declared property of this value, such as `Finite` on a meta tensor. It reports an unavailable inspection, not necessarily a defect |
| `TransformContractError` | `tenspec.errors` | `RuntimeError` | a transform returned an array that broke its preservation contract |

**Where the error is raised decides how you catch it.** Pydantic wraps a `ValueError` raised
inside its own validation callbacks, and Tenspec's check runs in one of those. So a
`TensorMismatch`, and a `BindingError` raised while a value is being checked, arrive inside a
`ValidationError`. A model validator is one of those callbacks too. An error raised in the body
of a checked function is not: it propagates as itself. `axis_size` in a function body, or
outside every boundary, therefore raises `BindingError` directly, as
[Axis scope and `axis_size`](runtime-validation.md#axis-scope-and-axis_size) shows.

`TensorMismatch` keeps its parts readable, and it holds no array:

```python
from typing import Literal as Shape

import numpy as np

from pydantic import ValidationError
from tenspec import NonEmpty, validate
from tenspec.errors import TensorMismatch
from tenspec.numpy import Float

try:
    validate((np.ones(3), np.ones(5)), tuple[Float[Shape["rows"]], Float[Shape["rows"]]])
except ValidationError as refusal:
    entry = refusal.errors()[0]
    original = entry["ctx"]["error"]
    print(entry["loc"], isinstance(original, TensorMismatch))
    print(original.requirement, "|", original.observed, "|", original.bindings)

try:
    validate(np.zeros((2, 0)), Float[Shape["rows cols"], NonEmpty])
except ValidationError as refusal:
    print(refusal.errors()[0]["msg"])
```

## What this release does not check

| Case | What happens |
|---|---|
| an `async` function | `checked` raises `AnnotationError` |
| a callable that is not a plain Python function | `checked` raises `AnnotationError` |
| a union mixing tensor and ordinary non-tensor alternatives | `AnnotationError`. Tensor-bearing alternatives and optional `None` are supported |
| a tensor alias that contains its own name | `AnnotationError` |
| a device requirement on NumPy | `AnnotationError`, because NumPy places no array |
| a Tenspec declaration on a model without the mixin | `AnnotationError` when Pydantic builds that model's schema, which is the class statement unless the caller defers the build |

A declaration relates values inside one boundary. It does not relate the length of a Python
sequence to an axis, and it does not relate an axis of a nested model to one of its parent.
Write those by hand, as in
[A relation the declaration cannot express](pydantic-models.md#a-relation-the-declaration-cannot-express).

## Strict JSON input for models

`TensorContracts` owns a model's binding scope through a Pydantic wrap validator. An upstream
Pydantic limitation means that strict `model_validate_json` can reject JSON representations of
tuples, datetimes, decimals, sets, frozen sets and bytes. Nested models can be affected even when
they declare no tensor. This behavior was reproduced with Pydantic 2.14.0.

The wrapper still reports JSON mode, but its handler receives Python objects. For example, a
strict tuple field receives a list and refuses it. See
[Pydantic issue 9204](https://github.com/pydantic/pydantic/issues/9204).

Validation of native Python values remains available. Array serialization and deserialization
need an explicit representation; Tenspec does not provide one. If an application already has a
separate storage model, parse its metadata with ordinary Pydantic, reconstruct native values and
arrays, then validate the computational model. Decoding JSON into a dictionary alone does not
perform that conversion.

The scope wrapper releases its bindings on success and failure. Changing its lifetime to work
around this JSON limitation must preserve that behavior.

## What a declaration does not promise

- **No immutability.** `CopyReadOnly` and `ReadOnly` control a copy and a flag, not access
  through a base array.
- **No promise about a later value.** A check happens at its boundary. See
  [An owned computed result](pydantic-models.md#an-owned-computed-result).
- **Nothing about what an outer validator returns.** An `AfterValidator` runs after the check
  and owns its own result.
- **No byte order.** A logical alias such as `Float64` accepts either byte order. Transform
  preservation compares the native dtype, so a transform may not change it.
- **No claim about operability.** A backend that represents a format says nothing about
  whether your installed NumPy or Torch can compute with it.
- **No static shape check.** No type checker compares two shapes.
