"""A valid caller that three type checkers read, and that each of them must accept.

`test_checkers.py` runs ty, Pyright and Pyrefly over this file, and every one must exit zero.
The positive `assert_type` checks are what makes that meaningful: a checker that reads a
declaration as `Any`, or that loses a signature or a return, fails one of them.

Nothing imports this file and nothing runs it. No checker here compares a shape, so a shape
that disagrees is not a static error.

The file holds the spellings all three checkers support, not every public spelling.
`validate` takes its tuple form here. Pyright needs `enableExperimentalFeatures` for a
`TypeForm` argument, which `pyrightconfig.json` beside this file sets.
"""

from typing import Any, Self, assert_type
from typing import Literal as Shape

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from pydantic import BaseModel, ConfigDict, Field
from tenspec import (
    Contiguous,
    Finite,
    Floating,
    NonEmpty,
    TensorContracts,
    axis_size,
    checked,
    validate,
)
from tenspec.numpy import (
    NUMPY_ARRAYS,
    ArrayProperty,
    ArrayTransform,
    Bool,
    CopyReadOnly,
    Float,
    Float32,
    Float64,
    Int,
    ReadOnly,
)
from tenspec.runtime.bindings import Bindings
from tenspec.runtime.validation import ArrayValidator
from tenspec.torch import CPU, Device
from tenspec.torch import ArrayProperty as TorchProperty
from tenspec.torch import ArrayTransform as TorchTransform
from tenspec.torch import Float as TorchFloat
from tenspec.types.dtypes import DTypeId, DTypeRequirement
from tenspec.types.operations import ArrayProperty as PropertyInterface
from tenspec.types.operations import ArrayTransform as TransformInterface
from tenspec.types.shapes import parse_shape
from tenspec.types.tensor import TensorType

type Scores = Float[Shape["layers heads"], Finite, NonEmpty]
"""A declaration the caller names once and writes wherever it means the same thing."""


@checked
def named(scores: Scores) -> int:
    """Read the base type of a named declaration, inside the body it declares.

    Returns:
        The head count.
    """
    assert_type(scores, NDArray[np.floating])
    return scores.shape[1]


@checked
def weigh(values: Float[Shape["rows features"]], weights: Float[Shape["features"]]) -> int:
    """Read the base type of a NumPy declaration, inside the body it declares.

    Returns:
        The row count, plus the extent the name `rows` holds.
    """
    assert_type(values, NDArray[np.floating])
    return values.shape[0] * weights.shape[0] + axis_size("rows")


@checked
def masked(
    values: Float[Shape["*batch time"]], mask: Bool[Shape["*batch time"]]
) -> NDArray[np.floating]:
    """Take the variadic spelling on both operands.

    Returns:
        The values the mask keeps.
    """
    return np.where(mask, values, 0.0)


@checked
def counted(exact: Int[Shape["4"]], anonymous: Float32[Shape["_ 3"]]) -> int:
    """Take a fixed axis, an anonymous axis, and a narrow scalar format.

    Returns:
        The total extent of both operands.
    """
    return exact.shape[0] + anonymous.shape[1]


@checked
def broadcast(values: Float[Shape["#rows"]], other: Float[Shape["#rows"]]) -> int:
    """Take the broadcast spelling on both operands.

    Returns:
        The row count of the wider operand.
    """
    return max(values.shape[0], other.shape[0])


@checked
def sealed(values: Float[Shape["rows"], ReadOnly, NonEmpty, Finite]) -> int:
    """Take three properties beside the shape.

    Returns:
        The row count.
    """
    return values.shape[0]


@checked
def related[Format: Floating](
    left: Float[Shape["rows"], Format], right: Float[Shape["rows"], Format]
) -> int:
    """Relate the scalar format of two operands through one variable.

    Returns:
        The total row count.
    """
    return left.shape[0] + right.shape[0]


@checked
def placed(values: TorchFloat[Shape["rows"], CPU]) -> int:
    """Read the base type of a Torch declaration, and name a placement kind.

    Returns:
        The row count.
    """
    assert_type(values, torch.Tensor)
    return values.shape[0]


@checked
def paired[Placement: Device](
    operands: tuple[TorchFloat[Shape["a"], Placement], TorchFloat[Shape["b"], Placement]],
) -> int:
    """Relate the placement of two tensors inside one tuple declaration.

    Returns:
        The total row count.
    """
    left, right = operands
    return left.shape[0] + right.shape[0]


@checked
def optional_mask(
    values: Float[Shape["rows"]], mask: Bool[Shape["rows"]] | None = None
) -> NDArray[np.floating]:
    """Take an optional tensor beside a required one.

    Returns:
        The values the mask keeps, or the values themselves.
    """
    return values if mask is None else np.where(mask, values, 0.0)


class CallerBase(BaseModel):
    """The caller's own base, which the mixin keeps."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class Block[Format: Floating](TensorContracts, CallerBase):
    """Two fields on one axis and in one scalar format, and one optional field.

    The model declares the scalar format as its own type parameter. A module-level
    TypeVar in this body is unbound, and every checker refuses it.
    """

    left: Float[Shape["rows"], Format]
    right: Float[Shape["rows"], Format]
    mask: Bool[Shape["rows"]] | None = None


class NonPositive(ArrayProperty[np.floating[Any]]):
    """A caller's own NumPy check, whose method the backend runs."""

    @staticmethod
    def validate(values: NDArray[np.floating[Any]]) -> None:
        if bool((values > 0).any()):
            raise ValueError("no entry may be positive")


class ContiguousCopy(ArrayTransform[np.floating[Any]]):
    """A caller's own NumPy transform, for storage rather than for values.

    `ndarray.copy` keeps the class, the shape and the native dtype, and NumPy places no
    array. The C order it produces is a layout, which a transform does not guarantee, so
    a declaration that needs it requires `Contiguous` after the transforms.
    """

    @staticmethod
    def transform(values: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        return values.copy(order="C")


class NonPositiveTensor(TorchProperty):
    """A caller's own Torch check, on the tensor the backend supplies."""

    @staticmethod
    def validate(values: Tensor) -> None:
        if bool((values > 0).any().item()):
            raise ValueError("no entry may be positive")


class Detached(TorchTransform):
    """A caller's own Torch transform, for a plain tensor of no graph.

    `Tensor.detach` returns a Tensor of the same shape, dtype and device. It preserves
    the concrete class of a plain Tensor. It does not return every subclass it receives,
    so a declaration that carries a Tensor subclass needs its own transform.
    """

    @staticmethod
    def transform(values: Tensor) -> Tensor:
        return values.detach()


type LogProbs = Float64[Shape["masks tokens"], NonEmpty, Finite, NonPositive]
"""A named declaration carrying a caller's own check."""


@checked
def drops(logprobs: LogProbs) -> NDArray[np.float64]:
    """Read a caller-defined property through a named declaration."""
    assert_type(logprobs, NDArray[np.float64])
    return logprobs.sum(axis=1)


type StoredBlock = Float64[Shape["rows cols"], CopyReadOnly, Finite, NonPositive]
"""The library copy, before two properties that read the copy it returns."""


@checked
def stored(block: StoredBlock) -> Float64[Shape["rows cols"], ReadOnly]:
    """Take the library copy and two properties, and return the sealed result."""
    assert_type(block, NDArray[np.float64])
    return block


@checked
def laid_out(values: Float64[Shape["rows cols"], ContiguousCopy, Contiguous]) -> int:
    """Take a caller's own storage transform, and require the layout it produces."""
    assert_type(values, NDArray[np.float64])
    return values.shape[0]


@checked
def detached(values: TorchFloat[Shape["rows"], Detached, NonPositiveTensor]) -> int:
    """Take a Torch transform and a Torch check on one tensor."""
    assert_type(values, torch.Tensor)
    return values.shape[0]


class Capture(TensorContracts, CallerBase):
    """A model field that stores the sealed copy of the block it received."""

    block: Float64[Shape["rows cols"], CopyReadOnly, Finite] = Field(
        description="One measurement block, stored read-only."
    )


class Scale:
    """A caller's own class, whose checked method returns the receiver's class."""

    def __init__(self, mean: float) -> None:
        self.mean = mean

    @classmethod
    @checked
    def fit(cls, values: Float[Shape["*batch time"]], *, floor: float = 1e-3) -> Self:
        """Take the plain classmethod route, and return the receiver's class.

        Returns:
            One instance of the class that received the call.
        """
        return cls(float(values.mean()) + floor)


class Sharper(Scale):
    """A subclass whose checked call returns the subclass."""


# The lower runtime, for a caller that keeps one validator and supplies its own bindings.
requirement = TensorType(
    shape=parse_shape("rows cols"),
    dtype=DTypeRequirement(permitted=DTypeId.FLOAT64),
    properties=(Finite,),
)
validator = ArrayValidator(tensor_type=requirement, backend=NUMPY_ARRAYS)

assert_type(
    validator.validate(np.ones((2, 3)), bindings=Bindings()), np.ndarray[Any, np.dtype[Any]]
)
assert_type(
    NUMPY_ARRAYS.resolve_property(Finite, permitted_formats=frozenset({DTypeId.FLOAT64})),
    type[PropertyInterface[np.ndarray[Any, np.dtype[Any]]]],
)
assert_type(
    NUMPY_ARRAYS.resolve_transform(CopyReadOnly, permitted_formats=frozenset({DTypeId.FLOAT64})),
    type[TransformInterface[np.ndarray[Any, np.dtype[Any]]]],
)

values = np.ones((2, 3))
weights = np.ones(3)
tensors = (torch.ones(2), torch.ones(3))
read_only = np.ones(3)
read_only.flags.writeable = False

assert_type(weigh(values, weights), int)
assert_type(masked(values, np.ones((2, 3), dtype=np.bool_)), NDArray[np.floating])
assert_type(counted(np.ones(4, dtype=np.int64), np.ones((2, 3), dtype=np.float32)), int)
assert_type(broadcast(np.ones(1), np.ones(3)), int)
assert_type(sealed(read_only), int)
assert_type(related(np.ones(2), np.ones(2)), int)
assert_type(placed(torch.ones(3)), int)
assert_type(paired(tensors), int)
assert_type(optional_mask(np.ones(3)), NDArray[np.floating])
assert_type(Sharper.fit(values), Sharper)
assert_type(Block(left=np.ones(3), right=np.ones(3)).left, NDArray[np.floating])
assert_type(named(np.ones((2, 3))), int)
assert_type(
    validate((np.ones(2), np.ones(2)), tuple[Float[Shape["r"]], Float[Shape["r"]]]),
    tuple[NDArray[np.floating], NDArray[np.floating]],
)
assert_type(
    validate((np.ones((2, 3)), np.ones((2, 3))), tuple[Scores, Scores]),
    tuple[NDArray[np.floating], NDArray[np.floating]],
)
assert_type(drops(-np.ones((2, 3))), NDArray[np.float64])
assert_type(stored(-np.ones((2, 3))), NDArray[np.float64])
assert_type(laid_out(np.ones((2, 3))), int)
assert_type(detached(-torch.ones(3)), int)
assert_type(Capture(block=np.ones((2, 3))).block, NDArray[np.float64])
