# Pydantic models

`TensorContracts` gives the tensor fields of one model a single boundary, so related fields
share their axes. Add it as the first base of your own model:

```python
from typing import Literal as Shape

import numpy as np

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tenspec import Finite, TensorContracts
from tenspec.numpy import Float


class ProjectBase(BaseModel):
    """The caller's own base, with the caller's own policy."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, strict=True)


class Attention(TensorContracts, ProjectBase):
    weights: Float[Shape["heads tokens tokens"], Finite] = Field(
        description="one attention map per head, over a square token grid"
    )
    bias: Float[Shape["heads"]] = Field(description="one bias per head")


block = Attention(weights=np.ones((2, 5, 5)), bias=np.zeros(2))
print(block.weights.shape, block.bias.shape)

try:
    Attention(weights=np.ones((2, 5, 5)), bias=np.zeros(3))
except ValidationError as refusal:
    print(refusal.errors()[0]["msg"])
```

`heads` appears in both fields, so one model validation relates them. No handwritten shape
validator does that work.

## What the mixin expects

Place `TensorContracts` first among the bases. It adds no base of its own, and it sets neither
strictness nor a frozen policy: the caller's own base owns those. The caller's configuration
must permit arbitrary types, because an array is not a Pydantic type.

Each model validation opens its own boundary. A nested model therefore binds independently,
and it shares no axis with the model that holds it.

The mixin keeps what the caller declared in `__tensor_declarations__`. Tenspec's own step
replaces a field value only when the declaration names a transform. A validator the caller
wrote around that field keeps its own behavior, as in
[Pydantic composition](custom-checks.md#pydantic-composition).

Strict JSON loading has an upstream Pydantic limitation that can affect non-array fields too.
See [Strict JSON input for models](errors-and-limits.md#strict-json-input-for-models) before using
`model_validate_json` for a model with this mixin.

## An owned computed result

A declaration checks a value at its boundary. It does not follow that value afterwards, and
arithmetic on accepted inputs can produce an output the same declaration refuses. So check a
derived result where you build it: compute it once, validate the new output once, store it,
and then read it as an ordinary attribute.

Declare `CopyReadOnly` on the computed field, and the model stores a read-only copy of the
output. Later reads return that retained array. They run no check and make no copy.

```python
from typing import Literal as Shape
from typing import Self

import numpy as np

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tenspec import Finite, NonEmpty, TensorContracts, checked
from tenspec.numpy import CopyReadOnly, Float64


class ProjectBase(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class RowMeans(TensorContracts, ProjectBase):
    """The mean of every row, computed once and retained."""

    means: Float64[Shape["rows"], CopyReadOnly, NonEmpty, Finite] = Field(
        description="one finite mean per row, stored as a read-only copy"
    )

    @classmethod
    @checked
    def of(cls, values: Float64[Shape["rows cols"], NonEmpty, Finite]) -> Self:
        with np.errstate(over="ignore"):
            return cls(means=values.mean(axis=1))


ordinary = RowMeans.of(np.array([[1.0, 3.0], [5.0, 7.0]]))
print(ordinary.means, ordinary.means.flags.writeable)
print(ordinary.means is ordinary.means)

huge = np.array([[1e308, 1e308]])
print("every input is finite:", bool(np.isfinite(huge).all()))
try:
    RowMeans.of(huge)
except ValidationError as refusal:
    print(refusal.errors()[0]["msg"])
```

The first call has two ordinary rows and passes. The second call has one huge row, whose two
entries sit near the float64 maximum: every entry is finite, so the input declaration accepts
it, and their sum overflows, so the mean is infinite and the output declaration refuses it.
That is the whole point of checking a derived value at its own boundary.

## A relation the declaration cannot express

Shared axis names relate arrays. They do not relate an array to the length of an ordinary
Python sequence. Write that one relation by hand, in an ordinary validator:

```python
from typing import Literal as Shape
from typing import Self

import numpy as np

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from tenspec import TensorContracts
from tenspec.numpy import Float


class ProjectBase(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class Labelled(TensorContracts, ProjectBase):
    values: Float[Shape["rows cols"]] = Field(description="one row per label")
    labels: tuple[str, ...] = Field(description="the name of every row")

    @model_validator(mode="after")
    def rows_match_labels(self) -> Self:
        if len(self.labels) != self.values.shape[0]:
            raise ValueError(
                f"the label count {len(self.labels)} does not match the row count "
                f"{self.values.shape[0]}"
            )
        return self


print(Labelled(values=np.ones((2, 3)), labels=("a", "b")).labels)

try:
    Labelled(values=np.ones((2, 3)), labels=("a",))
except ValidationError as refusal:
    print(refusal.errors()[0]["msg"])
```
