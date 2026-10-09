# Array annotations

An annotation names the array you expect: its dtype, its shape, and on PyTorch its device.

```python
from typing import Literal as Shape

import numpy as np

from tenspec import validate
from tenspec.numpy import Float

print(validate(np.ones((3, 4)), Float[Shape["rows features"]]).shape)
```

`Float` is the dtype. `Shape["rows features"]` is the shape, with one name per axis. A
declaration takes the shape first, then any transforms, then any properties:
`Float[Shape["rows features"], Finite]`. [Custom checks and
transforms](custom-checks.md) covers those operations, and [Runtime
validation](runtime-validation.md) covers the boundaries that run the check.

The names in a shape relate the axes of different arrays. That is what a declaration adds over
an ordinary array type, and it holds only inside one boundary.

## Shape

A shape is a string of whitespace-separated axis tokens.

| Spelling | Meaning |
|---|---|
| `rows cols` | two axes that bind the names `rows` and `cols`, or match what they hold |
| `4` | an axis of exactly four |
| `_` | one axis of any extent. It binds nothing |
| `*batch time` | any leading axes, bound together as `batch`, then one axis `time` |
| `...` or `*_` | any leading axes, bound to nothing |
| `#rows` | an extent of one passes and binds nothing. Any other extent binds `rows`, or matches what `rows` already holds |
| `2*width` | an axis computed from dimensions that are bound already |
| `""` | a scalar, which is an array of rank zero |

A shape declares at most one variadic group. A named group binds the whole tuple of extents it
covered, so `axis_size` cannot read it as one number. A shape expression evaluates; it never
solves. `2*width` therefore needs `width` bound by an earlier axis or an earlier value in the
same boundary. Its grammar accepts whole numbers, dimension names, `+`, `-`, `*` and
parentheses, and Tenspec evaluates that bounded tree itself. It calls no Python `eval`, and it
runs no other construct.

## Alternative array types

Use a union when a boundary accepts either backend:

```python
from typing import Literal as Shape

from tenspec import checked
from tenspec.numpy import Float
from tenspec.torch import Float as TorchFloat


@checked
def row_count(
    values: Float[Shape["rows hidden"]] | TorchFloat[Shape["rows hidden"]],
) -> int:
    return values.shape[0]
```

Each alternative checks against the facts already established at that boundary. Pydantic selects
the accepted alternative, and only its bindings become visible to later operands. An alternative
that fails after checking an earlier tuple item leaves no binding behind.

Every non-`None` arm must contain a tensor declaration. An optional `None` remains supported.
A union does not require different arguments to choose the same backend, and it does not convert
arrays. If several arms accept, Pydantic's normal selection policy applies. Checks and caller
validators may run in more than one arm; only binding facts are isolated, not external side effects.

## Dtype

Each backend module carries the same alias names. A family alias accepts every format of its
family. An exact alias accepts one format. Neither is a cast: a declaration accepts or refuses
the array it received.

| Family alias | Accepts |
|---|---|
| `Float` | float16, float32, float64, and bfloat16 on Torch |
| `Int` | int8, int16, int32, int64 |
| `UInt` | uint8, uint16, uint32, uint64 |
| `Complex` | complex64, complex128 |
| `Bool` | bool |

| Exact alias | Backend |
|---|---|
| `Float16`, `Float32`, `Float64` | both |
| `BFloat16` | Torch only, because NumPy represents no bfloat16 |
| `Int8`, `Int16`, `Int32`, `Int64` | both |
| `UInt8`, `UInt16`, `UInt32`, `UInt64` | both |
| `Complex64`, `Complex128` | both |

A backend representing a format is not the same as your installed NumPy or Torch being able to
compute with it. Tenspec answers the first question alone.

The generic NumPy declaration is `NDArray`, so it permits a supported `ndarray` subclass. An
exact concrete class is not a built-in requirement. A property of your own can express one,
and that stays your class.

## Device

Only a backend that places arrays takes a device requirement. PyTorch does. NumPy does not, and
it reports no placement rather than reporting the CPU. `tenspec.torch` names `CPU`, `CUDA` and
`MPS`. A device kind compares the kind alone, so `CUDA` accepts any CUDA device.

## Shared variables

A shared variable relates two values. It is optional: you do not need one to declare a floating
array, because `Float` already does that. Reach for one only when two values must carry the
*same* format, or sit on the *same* device.

- A variable bound to `DType` or to `Floating` relates the scalar format.
- A variable bound to `tenspec.torch.Device`, or to one of its kinds, relates the placement,
  and it compares the whole descriptor, ordinal included. Two CUDA tensors on different
  devices therefore disagree.

Write it as a type parameter of the function or the model. A module-level `TypeVar` works at
run time, but a type checker refuses it inside the body of a model that is not generic.

```python
from typing import Literal as Shape

import numpy as np
import torch

from pydantic import ValidationError
from tenspec import AnnotationError, Floating, checked, validate
from tenspec.numpy import Float
from tenspec.torch import CPU
from tenspec.torch import Float as TorchFloat


@checked
def combine[Format: Floating](
    left: Float[Shape["rows"], Format], right: Float[Shape["rows"], Format]
) -> np.ndarray:
    return left + right


print(combine(np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)).dtype)

try:
    combine(np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float64))
except ValidationError as refusal:
    print(refusal.errors()[0]["msg"])

print(validate(torch.ones(2, 3), TorchFloat[Shape["rows cols"], CPU]).shape)

try:
    validate(np.ones((2, 3)), Float[Shape["rows cols"], CPU])
except AnnotationError as refusal:
    print(refusal)
```
