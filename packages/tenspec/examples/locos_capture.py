"""LOCOS: a capture boundary, and related measurement fields.

A demonstration against the real operand relationships of
`packages/skllm/src/skllm/attribution/locos/torch.py` and `.../locos/models.py` at Nightjar
476e2743. No consumer file is read at run time and no consumer file changes. The numerical
bodies are the smallest work that shows the declarations fit the real boundary.

Run it with `uv run python packages/tenspec/examples/locos_capture.py`.
"""

from typing import Annotated, Any, TypeVar
from typing import Literal as Shape

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError
from tenspec import Finite, Floating, Materialized, NonEmpty, TensorContracts, checked, validate
from tenspec.numpy import Float64, ReadOnly
from tenspec.torch import Float
from tenspec.types.properties import Device

Scalar = TypeVar("Scalar", bound=Floating)
Placement = TypeVar("Placement", bound=Device)


@checked
def head_directions[Scalar: Floating, Placement: Device](
    output_projection: Float[
        Shape["heads head_dim hidden"], Scalar, Placement, NonEmpty, Materialized
    ],
    token_direction: Float[Shape["hidden"], Scalar, Placement, NonEmpty, Materialized],
) -> Tensor:
    """One direction per head, in the operands' own format and placement.

    The contraction needs a populated axis and actual values on both operands, so both
    say so. An empty head dimension would otherwise reach the einsum and return a
    direction of no length.
    """
    return torch.einsum("hdm,m->hd", output_projection, token_direction)


def measure_step(
    attention: Tensor, values: Tensor, directions: Tensor, source_span: tuple[int, int]
) -> tuple[Tensor, Tensor]:
    """The contribution of a source span, and the contribution outside it, per head.

    The capture boundary checks the related operands together, then the numerical body runs
    with no further check. Every operand must be populated and hold actual values, because
    the modulo and the contraction below read both. Grouped-head divisibility and the span
    bounds stay ordinary Python, because neither is a relation between axes.

    Returns:
        The source contribution and the background contribution, one value per head.

    Raises:
        ValidationError: when an operand is empty, or disagrees with a related operand.
        ValueError: when the head groups do not divide, or the span leaves no background.
    """
    attention, values, directions = validate(
        (attention, values, directions),
        tuple[
            Float[Shape["heads keys"], Scalar, Placement, NonEmpty, Materialized],
            Float[Shape["kv_heads keys head_dim"], Scalar, Placement, NonEmpty, Materialized],
            Float[Shape["heads head_dim"], Scalar, Placement, NonEmpty, Materialized],
        ],
    )
    heads, keys = attention.shape
    kv_heads = values.shape[0]
    if heads % kv_heads:
        raise ValueError(f"{heads} query heads do not divide over {kv_heads} key-value heads")
    start, stop = source_span
    if not 0 <= start < stop <= keys or stop - start == keys:
        raise ValueError(f"span {source_span} leaves no background inside {keys} keys")

    grouped = directions.reshape(kv_heads, heads // kv_heads, values.shape[2])
    per_key = torch.einsum("kjd,kgd->kgj", values, grouped).reshape(heads, keys)
    weighted = attention * per_key
    return weighted[:, start:stop].sum(dim=1), weighted.sum(dim=1) - weighted[:, start:stop].sum(
        dim=1
    )


def sealed(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """A read-only copy, the way the consumer's own base prepares a block.

    The result owns its storage. It is a copy, not a view of what the caller passed.

    Returns:
        The read-only copy.
    """
    copy = np.array(values, dtype=np.float64, copy=True)
    copy.setflags(write=False)
    return copy


Block = Annotated[
    Float64[Shape["selected_steps layers heads"], Finite, ReadOnly], BeforeValidator(sealed)
]


class FrozenArrays(BaseModel):
    """The consumer's own base. Tenspec adds none of its own and changes no setting."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class LocosAttribution(TensorContracts, FrozenArrays):
    """Two representative measurement fields, on the same axes, stated once per field."""

    source_attention: Block = Field(description="Source attention per step, layer, and head.")
    background_attention: Block = Field(description="Background attention on the same axes.")


def main() -> None:
    """Run the capture boundary and the measurement model."""
    heads, kv_heads, keys, head_dim, hidden = 4, 2, 6, 3, 5
    projection = torch.ones(heads, head_dim, hidden)
    token = torch.ones(hidden)
    directions = head_directions(projection, token)
    print(f"Directions: {tuple(directions.shape)}, dtype {directions.dtype}")

    attention = torch.full((heads, keys), 1.0 / keys)
    values = torch.ones(kv_heads, keys, head_dim)
    source, background = measure_step(attention, values, directions, (1, 3))
    print(f"Source contribution {source.tolist()}")
    print(f"Background contribution {background.tolist()}")

    try:
        measure_step(attention, torch.ones(3, keys, head_dim), directions, (1, 3))
    except ValueError as refusal:
        print(f"Grouped heads stay ordinary Python: {refusal}")

    try:
        measure_step(attention, torch.ones(kv_heads, keys + 1, head_dim), directions, (1, 3))
    except ValidationError as refusal:
        print(f"The shared key axis is a declaration: {refusal.errors()[0]['msg']}")

    try:
        measure_step(attention, torch.ones(0, keys, head_dim), directions, (1, 3))
    except ValidationError as refusal:
        print(f"An empty operand refuses before the modulo: {refusal.errors()[0]['msg']}")

    measurements = LocosAttribution(
        source_attention=np.ones((2, 3, heads)), background_attention=np.zeros((2, 3, heads))
    )
    stored: Any = measurements.source_attention
    print(f"Stored block {stored.shape}, writeable {stored.flags.writeable}")

    try:
        LocosAttribution(
            source_attention=np.ones((2, 3, heads)), background_attention=np.zeros((5, 3, heads))
        )
    except ValidationError as refusal:
        print(f"Related fields share their axes: {refusal.errors()[0]['msg']}")


if __name__ == "__main__":
    main()
