# tenspec

Declare what an array must be, beside the value it describes, and check it at one boundary.

The shape and dtype checks at the top of a function move into its signature, where a reader and
a type checker both see them. Tenspec accepts or refuses the array the caller passed: it
converts nothing, moves nothing, and copies nothing unless the declaration names a transform
that copies.

**[The documentation site](https://egordm.github.io/tenspec/) holds the guide, the API reference
and two executed tutorials.**

## Install

Install it with pip, or with uv. Either one is enough, so run the line for the tool you use:

```bash
pip install "tenspec[numpy]"   # or: uv add "tenspec[numpy]"
```

Name the array library you need as an extra: `numpy` or `torch`. `import tenspec` loads neither
one.

This is an early release, at version 0.2.0, and the interface can change before version 1.0. It
needs Python 3.13 or newer. The tested interpreter and array-library combinations are in
[the support matrix](docs/type-checkers.md#the-measured-support-matrix).

## What it gives you

**Ordinary types.** A declaration is an ordinary annotation carrying the array type of your
backend. A type checker reads it as that type, and code that never calls Tenspec still imports
and runs.

**Pydantic boundaries, natively.** `TensorContracts` relates the fields of your own model inside
one model validation, beside your own validators and your own configuration.

**Runtime checks are opt in.** An annotation alone performs no runtime work. A declaration
checks a value only where `@checked`, `TensorContracts` or `validate` prepares it.

**Array alternatives.** A union can accept a NumPy array or Torch tensor at one boundary.
Only the alternative Pydantic selects contributes bindings to later operands.

**Transforms are explicit and structure-preserving.** A declaration changes a value only when it
names a transform. The runtime verifies that the result kept its class, shape, native dtype and
device.

No type checker compares two shapes. A shape that disagrees is a run-time refusal, never a
static error. The checkers, their required settings and their measured limits are in
[Type checkers and Ruff](docs/type-checkers.md).

## Quickstart

Three forms declare the same relation: a matrix of `rows` by `features`, and one weight per
feature. Each one opens a boundary of its own, and each refuses the same disagreement.

### A checked function

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

`weights` must cover as many features as `values` has columns, and the returned matrix must keep
the shape it came with. Remove `@checked` and the declarations stay as ordinary annotations that
a type checker still reads, with no Tenspec check at run time.

### A model

```python
from typing import Literal as Shape

import numpy as np

from pydantic import BaseModel, ConfigDict, Field
from tenspec import TensorContracts
from tenspec.numpy import Float


class ProjectBase(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class WeighedColumns(TensorContracts, ProjectBase):
    values: Float[Shape["rows features"]] = Field(description="one row per observation")
    weights: Float[Shape["features"]] = Field(description="one weight per feature")


block = WeighedColumns(values=np.ones((3, 4)), weights=np.array([1.0, 2.0, 3.0, 4.0]))
print(block.values.shape, block.weights.shape)
```

The field annotations establish the relation at construction, with no handwritten shape
validator. The caller's own base owns the configuration, and it must permit arbitrary types,
because an array is not a Pydantic type.

### One standalone call

```python
from typing import Literal as Shape

import numpy as np

from pydantic import ValidationError
from tenspec import validate
from tenspec.numpy import Float

weighed = tuple[Float[Shape["rows features"]], Float[Shape["features"]]]

print([item.shape for item in validate((np.ones((3, 4)), np.ones(4)), weighed)])

try:
    validate((np.ones((3, 4)), np.ones(5)), weighed)
except ValidationError as refusal:
    print(refusal.errors()[0]["msg"])
```

One tuple is one boundary, which is what relates the two arrays. Two separate `validate` calls
relate nothing. The refusal names the axis, both extents, and the bindings the boundary held:

```text
Value error, expected features=4, but the array has features=5, with {'rows': 3, 'features': 4}
```

**Write `from typing import Literal as Shape`.** `tenspec.Shape` is a public re-export of
`typing.Literal` and works at run time, but Ruff 0.16.6 does not follow it. See
[Ruff and the `Shape` alias](docs/type-checkers.md#ruff-and-the-shape-alias).

## Documentation

| Page | What it answers |
|---|---|
| [Getting started](docs/index.md) | what Tenspec is, how to install it, and the three forms |
| [Array annotations](docs/array-annotations.md) | what a declaration says about shape, dtype and device |
| [Runtime validation](docs/runtime-validation.md) | how `@checked` and `validate` check a function, a method and one value |
| [Pydantic models](docs/pydantic-models.md) | how `TensorContracts` relates the fields of your own model |
| [Custom checks and transforms](docs/custom-checks.md) | how to write a property or a transform of your own |
| [Examples](docs/examples.md) | two tutorials the site executes, and two longer demonstrations |
| [Type checkers and Ruff](docs/type-checkers.md) | the checkers, their settings, and the measured support matrix |
| [API reference](docs/api.md) | the names you import, and the advanced interfaces |

## Develop

```bash
uv sync
uv run poe check
uv run --only-group docs poe docs
```

`poe check` runs lint, format-check, the strict type check and the whole test suite, which
includes reading one caller file with ty, Pyright and Pyrefly. `poe docs` builds the
documentation site into `artifacts/docs/html`, using the documentation dependencies alone.

A published release uploads the wheel and the source archive to PyPI. Its tag must name the
package version, as `v0.1.0` names `0.1.0`, and a tag that does not stops the build.
`.github/workflows/publish.yml` holds that step, and it uses GitHub OIDC rather than a stored
token.

## Attribution

The shape notation follows [jaxtyping](https://github.com/patrick-kidger/jaxtyping), which is
MIT licensed. The spellings a jaxtyping reader knows carry the same meaning here: a named or
fixed axis, the variadic `*name` and `...`, the anonymous `_`, and the broadcast `#`. jaxtyping
is not a dependency of this package.

## License

MIT. See `LICENSE`.
