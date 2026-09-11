# Custom checks and transforms

A property reads an array and returns `None` to accept, or raises to refuse. It changes
nothing. A transform returns the array every later operation and the caller receive. Both are
your own classes, and a declaration names them beside the shape.

## A property of your own

Derive a property from the author base of your backend, and write one `validate` static method:

```python
from typing import Any
from typing import Literal as Shape

import numpy as np
from numpy.typing import NDArray

from pydantic import ValidationError
from tenspec import validate
from tenspec.numpy import ArrayProperty, Float64


class NonPositive(ArrayProperty[np.floating[Any]]):
    """No entry of the array is above zero."""

    @staticmethod
    def validate(values: NDArray[np.floating[Any]]) -> None:
        if bool((values > 0).any()):
            raise ValueError("no entry may be positive")


type LogProbs = Float64[Shape["masks tokens"], NonPositive]

print(validate(-np.ones((2, 3)), LogProbs).shape)

try:
    validate(np.ones((2, 3)), LogProbs)
except ValidationError as refusal:
    print(refusal.errors()[0]["msg"])
```

The NumPy author base takes the scalar domain the method reads. A declaration that permits a
wider domain than the method accepts is refused when Tenspec prepares it, so
`Int[Shape["rows"], NonPositive]` raises `AnnotationError` there rather than at the first
positive value. A Torch property fixes no scalar domain, because a `Tensor` carries no scalar
type, so the Torch author base takes no parameter. Properties run in the order the declaration
names them.

## A transform of your own

A transform is for how a value is stored, not for what it means. It takes the array alone and
returns the array the caller receives:

```python
from typing import Any
from typing import Literal as Shape

import numpy as np
from numpy.typing import NDArray

from tenspec import validate
from tenspec.numpy import ArrayTransform, Float64


class ContiguousCopy(ArrayTransform[np.floating[Any]]):
    """Return a C-contiguous copy of the array."""

    @staticmethod
    def transform(values: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        return values.copy(order="C")


type StoredScores = Float64[Shape["masks tokens"], ContiguousCopy]

supplied = np.asfortranarray(np.ones((2, 3)))
print(supplied.flags.c_contiguous, validate(supplied, StoredScores).flags.c_contiguous)
```

Both kinds are `@staticmethod`, both are synchronous, and both take the array alone. A
declaration names every transform before its properties, and the other order raises
`AnnotationError` when Tenspec prepares it. A transform runs after the shape, the scalar format
and the placement already match, and before every property, so it receives a structurally
checked array and no property guarantee.

Tenspec reads four facts before a transform runs and compares them after it: the concrete
class, the exact shape, the native dtype and the device. A transform that changed one raises
`TransformContractError`. Those four are the whole guarantee. A transform may change anything
else, the writeable flag and the memory layout included, and the next transform receives what
it returned. Require what you need at the end of the declaration: name `Contiguous` after the
transform, and that property reads what the transform returned. The native dtype includes byte
order, so a transform must not byte-swap, and a logical alias such as `Float64` says nothing
about byte order on its own.

A transform must not change the caller's array, and must not change shared storage. Tenspec
does not copy the input and does not undo a side effect, so a transform owns what it does.

Keep general numerical work out of a declaration. A sort, a cast, an arithmetic step and a
device transfer belong where a reader can see them.

## The built-in properties

| Property | Import | It holds when |
|---|---|---|
| `Finite` | `tenspec` | every element is a finite number. A materialized integer array always passes |
| `NonEmpty` | `tenspec` | the array holds at least one element |
| `Contiguous` | `tenspec` | the array uses C-contiguous storage |
| `Materialized` | `tenspec` | the array holds data rather than metadata alone. Every NumPy array does; a Torch meta or fake tensor does not |
| `ReadOnly` | `tenspec.numpy` | this array's own writeable flag is off |

On Torch, `Finite` raises `ConstraintEvaluationError` for a tensor that holds metadata alone,
because the values do not exist to read. That is never a measured refusal.

## Name a declaration

Write a declaration once, name it, and use the name at every boundary that means the same
thing. A name changes nothing else. The axis names still bind inside the boundary that checks
the value, not where the alias is written, so two calls that take `Scores` relate nothing.

```python
from typing import Literal as Shape

import numpy as np

from tenspec import Finite, NonEmpty, validate
from tenspec.numpy import Float64

type Scores = Float64[Shape["layers heads"], Finite, NonEmpty]

supplied = np.ones((2, 3))
print(validate(supplied, Scores) is supplied)
```

A name may hold another name, an optional tensor, or a tuple. A tensor declaration that
contains its own name is refused, because such a declaration has no end. An ordinary alias
that contains its own name stays with Pydantic, and Tenspec returns it unchanged.

## `CopyReadOnly` against `ReadOnly`

| Name | Kind | What it does |
|---|---|---|
| `tenspec.numpy.CopyReadOnly` | transform | copies the array, turns the copy's writeable flag off, and returns a view of that copy |
| `tenspec.numpy.ReadOnly` | property | reads the array's own writeable flag, and refuses when it is on |

`CopyReadOnly` copies on every validation, so it costs time and memory each time, and no cache
holds the result. `ReadOnly` inspects a flag: it owns no storage and seals nothing. Neither
gives absolute immutability. A caller who reaches the base array of a read-only view can still
change the values.

## Pydantic composition

A Tenspec declaration is an ordinary annotation, so you can wrap it in `Annotated` with your
own Pydantic validators. This is interoperability, not a second transform engine: use it for
ordinary Pydantic work, and use a Tenspec transform for a Tenspec declaration.

**Keep a Tenspec property or transform inside the alias brackets, and never in the outer
`Annotated`.** The alias consumes its own operations while it prepares, so an outer Tenspec
operation would check nothing. Tenspec raises `AnnotationError` there, and names the term and
its correct place. Your own validators and ordinary metadata stay legal in the outer
`Annotated`, as the rest of this section shows.

The order is the ordinary Pydantic order, and it decides what each side sees:

- A `BeforeValidator` runs **before** Tenspec, so it decides what Tenspec receives. Tenspec
  then checks its result.
- An `AfterValidator` runs **after** Tenspec, so it receives the array Tenspec accepted or
  returned. Whatever it returns is what the caller gets, and **Tenspec does not check it
  again**. That validator owns every requirement the declaration established.

```python
from typing import Annotated, Any
from typing import Literal as Shape

import numpy as np
from numpy.typing import NDArray

from pydantic import AfterValidator, BeforeValidator
from tenspec import validate
from tenspec.numpy import Float64


def as_float64(values: Any) -> Any:
    return np.asarray(values, dtype=np.float64)


def double(values: NDArray[Any]) -> NDArray[Any]:
    return values * 2.0


type Composed = Annotated[
    Float64[Shape["rows"]], BeforeValidator(as_float64), AfterValidator(double)
]

accepted = validate([1.0, 2.0], Composed)
print(accepted, accepted.dtype)
```

The `BeforeValidator` turned a list into a float64 array, which Tenspec then accepted. The
`AfterValidator` replaced that array, so the returned value is not the one Tenspec checked.
