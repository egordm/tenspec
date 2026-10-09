# API reference

The public surface, in the order a caller meets it. This page is curated: it lists the names you
import, not every symbol in the package. The task pages explain how they compose, and
[Errors and limits](errors-and-limits.md) states what each failure means.

`import tenspec` loads neither array library. Import `tenspec.numpy` or `tenspec.torch` for the
declarations of the library you use.

## The three entry points

```{eval-rst}
.. autofunction:: tenspec.checked

.. autoclass:: tenspec.TensorContracts

.. autofunction:: tenspec.validate
```

## Backend aliases

Each alias takes a shape first, then any transforms, then any properties:
`Float[Shape["rows features"]]`, or `Float[Shape["rows features"], Finite]` with a property. The
spellings are the same in `tenspec.numpy` and `tenspec.torch`.

| Alias | Accepts |
|---|---|
| `Float` | any supported floating format |
| `Int` | any supported signed integer format |
| `UInt` | any supported unsigned integer format |
| `Complex` | any supported complex format |
| `Bool` | the boolean format |
| `Float16`, `Float32`, `Float64` | that exact floating format |
| `Int8`, `Int16`, `Int32`, `Int64` | that exact signed integer format |
| `UInt8`, `UInt16`, `UInt32`, `UInt64` | that exact unsigned integer format |
| `Complex64`, `Complex128` | that exact complex format |

`tenspec.torch` adds `BFloat16`, and the device markers a placement requirement names. See
[Dtype](array-annotations.md#dtype) for what each backend
represents, and [Device](array-annotations.md#device) for why NumPy takes no device.

These are parameterized type aliases. The table gives the spelling a caller writes, because the
expanded form carries Tenspec's internal declaration rather than anything a caller types.

### Writing a shape

`tenspec.Shape` is a public re-export of `typing.Literal`. A shape is its single string
argument, as `Shape["rows features"]`.

Write `from typing import Literal as Shape` in your own code. The re-export works at run time,
but Ruff 0.16.6 does not follow it and reports a false diagnostic against the shape text. See
[Ruff and the `Shape` alias](type-checkers.md#ruff-and-the-shape-alias).

## Common operations

A property checks a value and changes nothing. A transform replaces the value every later
operation and the caller receive. A declaration names every transform before its properties.

```{eval-rst}
.. autoclass:: tenspec.Finite

.. autoclass:: tenspec.NonEmpty

.. autoclass:: tenspec.Contiguous

.. autoclass:: tenspec.Materialized
```

These four import from `tenspec` and work on either backend. The next two are NumPy's own,
because they read a NumPy writeable flag:

```{eval-rst}
.. autoclass:: tenspec.numpy.ReadOnly

.. autoclass:: tenspec.numpy.CopyReadOnly
```

[`CopyReadOnly` against `ReadOnly`](custom-checks.md#copyreadonly-against-readonly) states what neither
one promises.

## Operation author bases

Derive your own property or transform from the author base of your backend, which fixes the
scalar domain the operation accepts.

```{eval-rst}
.. autoclass:: tenspec.numpy.ArrayProperty
   :members: validate

.. autoclass:: tenspec.numpy.ArrayTransform
   :members: transform
```

`tenspec.torch` carries the same two names for the Torch backend. Its author bases take no
scalar type parameter. [A transform of your own](custom-checks.md#a-transform-of-your-own) shows both,
with the rules a transform must keep.

## Reading a bound axis

```{eval-rst}
.. autofunction:: tenspec.axis_size
```

## Errors

```{eval-rst}
.. autoclass:: tenspec.AnnotationError

.. autoclass:: tenspec.ConstraintEvaluationError

.. autoclass:: tenspec.errors.TensorMismatch

.. autoclass:: tenspec.errors.BindingError

.. autoclass:: tenspec.errors.TransformContractError
```

[The error classes](errors-and-limits.md#the-error-classes) says where each one is raised, and which of them
a caller normally catches.

## The advanced interfaces

Each lower name keeps its defining module. Use them to check arrays outside an annotation, or
to add a backend.

| Name | Module | What it is |
|---|---|---|
| `ArrayValidator` | `tenspec.runtime.validation` | one resolved declaration, prepared once and reused for many arrays |
| `validate_array` | `tenspec.runtime.validation` | the one-shot form of the same check |
| `TensorType` | `tenspec.types.tensor` | the whole requirement: shape, dtype, device, transforms, properties |
| `parse_shape` | `tenspec.types.shapes` | the shape string as a `ShapeSpec` |
| `Bindings`, `validation_scope`, `active_bindings` | `tenspec.runtime.bindings` | the facts one boundary holds, and the boundary itself |
| `prepare_annotation`, `prepare_declaration` | `tenspec.pydantic.annotations` | a declaration as an ordinary Pydantic annotation |
| `ArrayBackend` | `tenspec.runtime.arrays` | the seam a new array library implements |

`Bindings.fork()` copies known dimension, dtype and device facts for an isolated attempt.
`Bindings.merge(candidate)` checks agreement before committing any candidate fact. A disagreement
raises `TensorMismatch` and leaves all existing facts unchanged. `validation_scope(bindings=...)`
temporarily exposes supplied bindings and restores the previous scope on success or failure;
without that argument, it starts empty.

```python
import numpy as np

from tenspec.numpy import NUMPY_ARRAYS
from tenspec.runtime.bindings import validation_scope
from tenspec.runtime.validation import ArrayValidator
from tenspec.types.dtypes import DTypeFamily, DTypeRequirement
from tenspec.types.properties import NonEmpty
from tenspec.types.shapes import parse_shape
from tenspec.types.tensor import TensorType

requirement = TensorType(
    shape=parse_shape("rows cols"),
    dtype=DTypeRequirement(permitted=DTypeFamily.FLOATING),
    properties=(NonEmpty,),
)
validator = ArrayValidator(tensor_type=requirement, backend=NUMPY_ARRAYS)

with validation_scope() as bindings:
    validator.validate(np.ones((3, 4)), bindings=bindings)
    print(dict(bindings.dimensions))
```

`ArrayValidator` resolves its operations once, at construction, and refuses a requirement the
backend cannot answer there. It holds no array and no bindings, so one validator serves many
calls. `validate_array` does the same work in one call, and pays the preparation each time.

An `ArrayBackend` reads facts and compares nothing: `shape`, `dtype`, `device`, `native_dtype`
and `operation_formats`. It receives neither a `TensorType` nor a `Bindings`, so a backend
cannot read a declaration or decide whether two values agree. The runtime does that.
