# tenspec

Declare what an array must be, beside the value it describes, and check it at one boundary.

Tenspec moves the shape and dtype checks from the top of a function into its signature, where a
reader and a type checker both see them. It accepts or refuses the array the caller passed. It
converts nothing, moves nothing, and copies nothing, unless the declaration names a transform
that copies.

This is an early release, at version 0.1.0. The interface can change before version 1.0.

## Install

Install it with pip, or with uv. Either one is enough, so run the line for the tool you use:

```bash
pip install "tenspec[numpy]"   # or: uv add "tenspec[numpy]"
```

Name the array library you need as an extra: `numpy` or `torch`. `import tenspec` loads neither
one. It needs Python 3.13 or newer.

## What it gives you

- **Ordinary types.** A declaration is an ordinary annotation carrying your backend's array
  type, so a type checker reads it and untouched code still runs. The checkers, the settings
  they need and their measured limits are on
  [Type checkers and Ruff](https://egordm.github.io/tenspec/type-checkers.html).
- **Pydantic boundaries, natively.** `TensorContracts` relates the fields of your own model
  inside one model validation.
- **Runtime checks are opt in.** An annotation alone performs no runtime work.
- **Transforms are explicit.** A value changes only when a declaration names a transform, and
  the runtime verifies the result kept its class, shape, native dtype and device.

## The three entry points

| Entry point | Use it for |
|---|---|
| `@checked` | a function or a method, with its arguments and its return |
| `TensorContracts` | a Pydantic model, whose related fields share one boundary |
| `validate(value, annotation)` | one value, or one tuple of related values |

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


print(weigh_columns(np.ones((3, 4)), np.array([1.0, 2.0, 3.0, 4.0])).shape)
```

`weights` must cover as many features as `values` has columns. A disagreement is a run-time
refusal naming the axis, both extents, and the bindings the boundary held.

No type checker compares two shapes. A shape that disagrees is a run-time refusal, never a
static error.

## Documentation

[The documentation site](https://egordm.github.io/tenspec/) holds the guide, the API reference
and two executed tutorials. The
[repository](https://github.com/egordm/tenspec) holds the source.

## Attribution

The shape notation follows [jaxtyping](https://github.com/patrick-kidger/jaxtyping), which is
MIT licensed. jaxtyping is not a dependency of this package.

## License

MIT. See `LICENSE`.
