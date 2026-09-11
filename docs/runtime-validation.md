# Runtime validation

An annotation alone performs no runtime work. Add `@checked` to a function, and Tenspec checks
every declared argument and the declared return:

```python
from typing import Literal as Shape

import numpy as np

from tenspec import checked
from tenspec.numpy import Float


@checked
def weigh_columns(
    values: Float[Shape["rows features"]], weights: Float[Shape["features"]]
) -> Float[Shape["rows features"]]:
    return values * weights


print(weigh_columns(np.ones((3, 4)), np.ones(4)).shape)
```

`features` appears in both arguments, so the two arrays must agree on it. Remove `@checked`
and the declarations stay as ordinary annotations, with no check at run time.

Three boundaries run a check. Each one opens a scope of its own, so a name binds inside that
boundary alone, and two separate calls never share a binding.

| Boundary | Use it for |
|---|---|
| `@checked` | a function or method, with its arguments and its return |
| `TensorContracts` | a Pydantic model, whose related fields share one boundary. See [Pydantic models](pydantic-models.md) |
| `validate(value, annotation)` | one value, or one tuple of related values |

## Functions and methods

`checked` reads the declared arguments and the declared return, and checks both inside one
boundary. The argument checks, the body and the return check share one scope, so `features`
binds once and the return must agree with it. Every call starts fresh bindings.

`checked` is strict for ordinary values too, so a `str` does not become an `int`. A
tensor-bearing default is checked the way a supplied argument is.

Place `checked` under an ordinary `@classmethod` or `@staticmethod`, closest to the function.
A method that returns `Self` is checked against the class that received the call, so a
subclass call cannot return a base instance. Write `Self` rather than the class name in
quotes: `checked` resolves the annotations when it decorates, and the class does not exist
yet at that moment.

```python
from typing import Any, Self
from typing import Literal as Shape

import numpy as np

from pydantic import ValidationError
from tenspec import checked
from tenspec.numpy import Float


class Window:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    @classmethod
    @checked
    def of(cls, values: Float[Shape["rows cols"]], label: str = "window") -> Self:
        return cls(values)


print(type(Window.of(np.ones((2, 3)))).__name__)

from_untyped_data: Any = 7
try:
    Window.of(np.ones((2, 3)), label=from_untyped_data)
except ValidationError as refusal:
    print(refusal.errors()[0]["type"])
```

## One value with `validate`

`validate` checks one value against one declaration, with no decorator and no model. One
validation of a tuple relates several values inside one boundary. Two separate `validate`
calls relate nothing.

```python
from typing import Literal as Shape

import numpy as np

from pydantic import ValidationError
from tenspec import validate
from tenspec.numpy import Float

related = tuple[Float[Shape["rows"]], Float[Shape["rows cols"]]]

print(validate(np.ones(4), Float[Shape["rows"]]).shape)
print([item.shape for item in validate((np.ones(4), np.ones((4, 2))), related)])
print(validate(None, Float[Shape["rows"]] | None))

try:
    validate((np.ones(4), np.ones((5, 2))), related)
except ValidationError as refusal:
    print(refusal.errors()[0]["loc"], refusal.errors()[0]["msg"])
```

Tenspec returns the array by identity unless the declaration names a transform. An outer
Pydantic validator is your own code, and it may return something else. See
[Pydantic composition](custom-checks.md#pydantic-composition).

## Axis scope and `axis_size`

A boundary starts with no bindings. The first axis that carries a name binds it, and every
later axis with that name, in that boundary, must match. Nothing commits until the whole array
passes, so a refused array leaves the boundary as it found it.

`axis_size(name)` reads a bound dimension inside the boundary that is validating now. Call it
from the body of a checked function, or from any code that body reaches. There it raises
`BindingError` as itself, because a function body is not one of Pydantic's validation
callbacks. A call outside every boundary raises the same class the same way. Inside one of
those callbacks, a model validator included, the error stays under Pydantic's wrapper.

```python
from typing import Literal as Shape

import numpy as np

from pydantic import ValidationError
from tenspec import axis_size, checked, validate
from tenspec.errors import BindingError
from tenspec.numpy import Float

print(validate(np.array(1.5), Float[Shape[""]]).shape)
print(validate(np.ones((2, 3, 8)), Float[Shape["*_ width"]]).shape)
print(
    validate(
        (np.ones((1, 3)), np.ones((4, 3))),
        tuple[Float[Shape["#rows cols"]], Float[Shape["rows cols"]]],
    )[1].shape
)
print(
    validate((np.ones(3), np.ones(6)), tuple[Float[Shape["width"]], Float[Shape["2*width"]]])[
        1
    ].shape
)

try:
    validate(np.ones(6), Float[Shape["2*width"]])
except ValidationError as refusal:
    print(refusal.errors()[0]["msg"])


@checked
def row_count(values: Float[Shape["*batch rows cols"]]) -> int:
    return axis_size("rows")


print(row_count(np.ones((4, 6, 2))))


@checked
def batch_count(values: Float[Shape["*batch rows"]]) -> int:
    return axis_size("batch")


try:
    batch_count(np.ones((2, 3, 4)))
except BindingError as refusal:
    print(refusal.reason.value, "|", refusal)

try:
    axis_size("rows")
except BindingError as refusal:
    print(refusal.reason.value, "|", refusal)
```

One class, two locations: the unbound `2*width` is a `BindingError` raised while a value is
checked, so Pydantic wraps it in a `ValidationError`. The two calls after it sit in a function
body and outside every boundary, so they arrive as themselves. See
[The error classes](errors-and-limits.md#the-error-classes).
