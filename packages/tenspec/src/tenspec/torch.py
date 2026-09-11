"""The Torch backend, and the declarations a Torch caller writes."""

from abc import ABC
from collections.abc import Hashable
from types import MappingProxyType
from typing import Annotated, Any

import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensor

from tenspec.errors import (
    AnnotationError,
    ConstraintEvaluationError,
    TensorMismatch,
    describe_empty,
)
from tenspec.runtime.arrays import ArrayBackend, TensorDeclaration
from tenspec.types import operations, properties
from tenspec.types.dtypes import DTypeFamily, DTypeId
from tenspec.types.operations import derives_from
from tenspec.types.properties import (
    Device as Device,  # noqa: PLC0414  A Torch caller names this bound, so this module exports it.
)

TORCH_FORMATS: frozenset[DTypeId] = frozenset(
    item for item in DTypeId if isinstance(getattr(torch, item.value, None), torch.dtype)
)
"""Every logical format this Torch build represents, read from Torch itself.

Torch 2.14.0 represents all of them, bfloat16 included.
"""


class ArrayProperty(operations.ArrayProperty[Tensor], ABC):
    """The base a Torch property derives from.

    Write `class NonPositive(ArrayProperty)` and implement `validate`. A Tensor carries
    no scalar type, so a Torch property accepts every format Torch represents. The method
    raises ConstraintEvaluationError for a tensor it cannot read, such as a meta tensor.
    """


class ArrayTransform(operations.ArrayTransform[Tensor], ABC):
    """The base a Torch transform derives from.

    Write `class Owned(ArrayTransform)` and implement `transform`. The result must keep
    the concrete class, the shape, the native dtype and the device of its input. An
    ordinary clone turns a Parameter into a Tensor, so it breaks that contract and
    belongs outside a declaration.
    """


def is_materialized(value: Tensor) -> bool:
    """True when the tensor holds values rather than metadata alone.

    Returns:
        False for a meta tensor and for a fake tensor, whose values no check can read.
    """
    return not value.is_meta and not isinstance(value, FakeTensor)


class Materialized(ArrayProperty):
    """The tensor holds actual data, rather than metadata alone."""

    @staticmethod
    def validate(values: Tensor) -> None:
        if not is_materialized(values):
            raise TensorMismatch(requirement="Materialized", observed="metadata alone")


class Finite(ArrayProperty):
    """Every element of the tensor is a finite number.

    A tensor that holds metadata alone raises ConstraintEvaluationError, because its
    values do not exist to be read. That is never a measured refusal, and it comes before
    the scalar format is read, so a meta integer tensor raises rather than passes. A
    materialized integer tensor always passes.
    """

    @staticmethod
    def validate(values: Tensor) -> None:
        if not is_materialized(values):
            raise ConstraintEvaluationError(
                "this tensor holds metadata alone, so Finite cannot be measured on it"
            )
        if (values.dtype.is_floating_point or values.dtype.is_complex) and not bool(
            torch.isfinite(values).all().item()
        ):
            raise TensorMismatch(requirement="Finite", observed="non-finite entries")


class NonEmpty(ArrayProperty):
    """The tensor holds at least one element."""

    @staticmethod
    def validate(values: Tensor) -> None:
        if values.numel() == 0:
            raise TensorMismatch(
                requirement="NonEmpty", observed=describe_empty(tuple(values.shape))
            )


class Contiguous(ArrayProperty):
    """The tensor uses C-contiguous storage."""

    @staticmethod
    def validate(values: Tensor) -> None:
        if not values.is_contiguous():
            raise TensorMismatch(
                requirement="Contiguous", observed="storage that is not C-contiguous"
            )


class CPU(Device):
    """The tensor sits in host memory."""

    kind = "cpu"


class CUDA(Device):
    """The tensor sits on a CUDA device."""

    kind = "cuda"


class MPS(Device):
    """The tensor sits on an Apple Metal device."""

    kind = "mps"


class TorchArrays(ArrayBackend[Tensor]):
    """Reads shape, scalar format, placement, and value properties from a Torch tensor."""

    supported_formats = TORCH_FORMATS
    builtin_properties = MappingProxyType(
        {
            properties.Finite: Finite,
            properties.NonEmpty: NonEmpty,
            properties.Contiguous: Contiguous,
            properties.Materialized: Materialized,
        }
    )
    reports_device = True

    def shape(self, value: Tensor) -> tuple[int, ...]:
        """The extent of every axis.

        Returns:
            One extent per axis, in order.
        """
        return tuple(value.shape)

    def dtype(self, value: Tensor) -> DTypeId:
        """The logical scalar format of the tensor.

        Returns:
            The logical scalar format.

        Raises:
            ConstraintEvaluationError: for a Torch dtype with no logical format.
        """
        name = str(value.dtype).removeprefix("torch.")
        try:
            return DTypeId(name)
        except ValueError as cause:
            raise ConstraintEvaluationError(
                f"Torch dtype {name!r} has no logical scalar format"
            ) from cause

    def device(self, value: Tensor) -> properties.DeviceId | None:
        """Where the tensor sits, with its ordinal when Torch reports one.

        Returns:
            The placement of the tensor.
        """
        return properties.DeviceId(kind=value.device.type, ordinal=value.device.index)

    def native_dtype(self, value: Tensor) -> Hashable:
        """Torch's own dtype object.

        Returns:
            The tensor's dtype object.
        """
        return value.dtype

    def operation_formats(self, operation: type[operations.Operation[Any]]) -> frozenset[DTypeId]:
        """Every logical format one Torch operation accepts.

        A Torch operation fixes no scalar domain, so it accepts every format Torch
        represents.

        Returns:
            The logical formats the operation accepts.

        Raises:
            AnnotationError: when the class is not a Torch operation.
        """
        if not derives_from(operation, ArrayProperty, ArrayTransform):
            raise AnnotationError(
                f"{operation.__name__} is not a Torch operation. Derive it from "
                f"tenspec.torch.ArrayProperty or tenspec.torch.ArrayTransform."
            )
        return self.supported_formats


TORCH_ARRAYS = TorchArrays()
"""The one Torch backend every Torch declaration carries."""


type Float[S, *C] = Annotated[
    Tensor, TensorDeclaration(TORCH_ARRAYS, DTypeFamily.FLOATING), S, tuple[*C]
]

type Int[S, *C] = Annotated[
    Tensor, TensorDeclaration(TORCH_ARRAYS, DTypeFamily.INTEGER), S, tuple[*C]
]

type UInt[S, *C] = Annotated[
    Tensor, TensorDeclaration(TORCH_ARRAYS, DTypeFamily.UNSIGNED), S, tuple[*C]
]

type Complex[S, *C] = Annotated[
    Tensor, TensorDeclaration(TORCH_ARRAYS, DTypeFamily.COMPLEX), S, tuple[*C]
]

type Bool[S, *C] = Annotated[
    Tensor, TensorDeclaration(TORCH_ARRAYS, DTypeFamily.BOOLEAN), S, tuple[*C]
]

type Float16[S, *C] = Annotated[
    Tensor, TensorDeclaration(TORCH_ARRAYS, DTypeId.FLOAT16), S, tuple[*C]
]

type Float32[S, *C] = Annotated[
    Tensor, TensorDeclaration(TORCH_ARRAYS, DTypeId.FLOAT32), S, tuple[*C]
]

type Float64[S, *C] = Annotated[
    Tensor, TensorDeclaration(TORCH_ARRAYS, DTypeId.FLOAT64), S, tuple[*C]
]

type Int8[S, *C] = Annotated[Tensor, TensorDeclaration(TORCH_ARRAYS, DTypeId.INT8), S, tuple[*C]]

type Int16[S, *C] = Annotated[Tensor, TensorDeclaration(TORCH_ARRAYS, DTypeId.INT16), S, tuple[*C]]

type Int32[S, *C] = Annotated[Tensor, TensorDeclaration(TORCH_ARRAYS, DTypeId.INT32), S, tuple[*C]]

type Int64[S, *C] = Annotated[Tensor, TensorDeclaration(TORCH_ARRAYS, DTypeId.INT64), S, tuple[*C]]

type UInt8[S, *C] = Annotated[Tensor, TensorDeclaration(TORCH_ARRAYS, DTypeId.UINT8), S, tuple[*C]]

type UInt16[S, *C] = Annotated[
    Tensor, TensorDeclaration(TORCH_ARRAYS, DTypeId.UINT16), S, tuple[*C]
]

type UInt32[S, *C] = Annotated[
    Tensor, TensorDeclaration(TORCH_ARRAYS, DTypeId.UINT32), S, tuple[*C]
]

type UInt64[S, *C] = Annotated[
    Tensor, TensorDeclaration(TORCH_ARRAYS, DTypeId.UINT64), S, tuple[*C]
]

type Complex64[S, *C] = Annotated[
    Tensor, TensorDeclaration(TORCH_ARRAYS, DTypeId.COMPLEX64), S, tuple[*C]
]

type Complex128[S, *C] = Annotated[
    Tensor, TensorDeclaration(TORCH_ARRAYS, DTypeId.COMPLEX128), S, tuple[*C]
]

type BFloat16[S, *C] = Annotated[
    Tensor, TensorDeclaration(TORCH_ARRAYS, DTypeId.BFLOAT16), S, tuple[*C]
]
