"""The NumPy backend, and the declarations a NumPy caller writes."""

from abc import ABC
from collections.abc import Hashable, Mapping
from types import MappingProxyType
from typing import Annotated, Any, get_args, get_origin

import numpy as np
from numpy.typing import NDArray

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

NUMPY_SCALARS: Mapping[DTypeId, type[np.generic]] = {
    DTypeId.BOOL: np.bool_,
    DTypeId.INT8: np.int8,
    DTypeId.INT16: np.int16,
    DTypeId.INT32: np.int32,
    DTypeId.INT64: np.int64,
    DTypeId.UINT8: np.uint8,
    DTypeId.UINT16: np.uint16,
    DTypeId.UINT32: np.uint32,
    DTypeId.UINT64: np.uint64,
    DTypeId.FLOAT16: np.float16,
    DTypeId.FLOAT32: np.float32,
    DTypeId.FLOAT64: np.float64,
    DTypeId.COMPLEX64: np.complex64,
    DTypeId.COMPLEX128: np.complex128,
}
"""The native scalar class of every logical format NumPy represents.

NumPy has no bfloat16, so that format is absent here and from `supported_formats`.
"""


class ArrayProperty[D: np.generic](operations.ArrayProperty[NDArray[D]], ABC):
    """The base a NumPy property derives from. D is the scalar domain it accepts.

    Write `class NonPositive(ArrayProperty[np.floating[Any]])` and implement `validate`.
    A declaration must guarantee D, so a float64-only check refuses a broad Float
    declaration. Write an exact scalar class, or a family with Any.
    """


class Finite(ArrayProperty[np.generic]):
    """Every element of the array is a finite number. An integer array always passes."""

    @staticmethod
    def validate(values: NDArray[np.generic]) -> None:
        if values.dtype.kind in "fc" and not bool(np.isfinite(values).all()):
            raise TensorMismatch(requirement="Finite", observed="non-finite entries")


class NonEmpty(ArrayProperty[np.generic]):
    """The array holds at least one element."""

    @staticmethod
    def validate(values: NDArray[np.generic]) -> None:
        if values.size == 0:
            raise TensorMismatch(requirement="NonEmpty", observed=describe_empty(values.shape))


class Contiguous(ArrayProperty[np.generic]):
    """The array uses C-contiguous storage."""

    @staticmethod
    def validate(values: NDArray[np.generic]) -> None:
        if not values.flags.c_contiguous:
            raise TensorMismatch(
                requirement="Contiguous", observed="storage that is not C-contiguous"
            )


class Materialized(ArrayProperty[np.generic]):
    """The array holds actual data. Every NumPy array does, so this accepts every array."""

    @staticmethod
    def validate(values: NDArray[np.generic]) -> None:
        """Accept every array."""


class ReadOnly(ArrayProperty[np.generic]):
    """This array's own writeable flag is off.

    It is a check of that flag. It is not a seal, and it never promises that another
    reference to the same storage cannot change the values.
    """

    @staticmethod
    def validate(values: NDArray[np.generic]) -> None:
        if values.flags.writeable:
            raise TensorMismatch(requirement="ReadOnly", observed="its writeable flag on")


class ArrayTransform[D: np.generic](operations.ArrayTransform[NDArray[D]], ABC):
    """The base a NumPy transform derives from. D is the scalar domain it accepts.

    Write `class Owned(ArrayTransform[np.generic])` and implement `transform`. The
    result must keep the concrete class, the shape, the native dtype and the device of
    its input, and must leave the caller's own array unchanged.
    """


class CopyReadOnly(ArrayTransform[np.generic]):
    """Replace the array with a read-only copy of it, on separate storage.

    The caller's array keeps its own values and its writeable flag. The stored copy
    shares no memory with it, and refuses an ordinary write. This is not immutability:
    a caller who reaches the copy's base array can still change the values.

    It copies on every validation, so it costs time and memory each time.
    """

    @staticmethod
    def transform[D: np.generic](values: NDArray[D]) -> NDArray[D]:
        """Copy the array, seal the copy, and return a view of it.

        Returns:
            A read-only view of a separate copy, with the same class, shape and dtype.
        """
        owned = values.copy()
        owned.setflags(write=False)
        return owned.view()


NUMPY_AUTHOR_BASES = (ArrayProperty, ArrayTransform)
"""The two bases a NumPy operation derives from. Each one fixes a scalar domain."""


def scalar_domain(operation: type[operations.Operation[Any]]) -> Any:
    """The scalar expression one NumPy operation names on its author base.

    The walk reads each class's own bases, so a subclass of a fixed-domain property
    inherits that domain rather than lose it.

    Returns:
        The scalar class or family the operation accepts.

    Raises:
        AnnotationError: when the class is not a NumPy operation, or when it names no
            scalar domain.
    """
    if not derives_from(operation, *NUMPY_AUTHOR_BASES):
        raise AnnotationError(
            f"{operation.__name__} is not a NumPy operation. Derive it from "
            f"tenspec.numpy.ArrayProperty or tenspec.numpy.ArrayTransform."
        )
    for ancestor in operation.__mro__:
        for base in ancestor.__dict__.get("__orig_bases__", ()):
            if get_origin(base) in NUMPY_AUTHOR_BASES:
                return get_args(base)[0]
    raise AnnotationError(
        f"{operation.__name__} names no scalar domain. Write the author base with an "
        f"exact scalar class, such as ArrayProperty[np.float64]."
    )


def scalar_formats(scalar: Any) -> frozenset[DTypeId]:
    """Every logical format one NumPy scalar class or family covers.

    np.float64 covers float64 alone. np.floating and np.floating[Any] cover every
    floating format NumPy represents. A family whose precision arguments are all Any
    covers the whole family, however many it takes: np.complexfloating takes two, and
    np.complexfloating[Any, Any] is its unrestricted form.

    A family that restricts any precision argument, such as np.floating[_64Bit], names a
    domain this release cannot read. It refuses, because widening it to the whole family
    would hand float32 values to a check written for float64.

    Returns:
        The logical formats of that scalar expression.

    Raises:
        AnnotationError: when the expression is not a NumPy scalar class, or when any
            precision argument is not Any.
    """
    native = scalar
    if get_origin(scalar) is not None:
        if not all(argument is Any for argument in get_args(scalar)):
            raise AnnotationError(
                f"{scalar!r} names a precision Tenspec does not read. Write the exact "
                f"scalar class, such as np.float64, or the family with Any."
            )
        native = get_origin(scalar)
    if not (isinstance(native, type) and issubclass(native, np.generic)):
        raise AnnotationError(f"{scalar!r} is not a NumPy scalar class, so its formats are unknown")
    return frozenset(
        item for item, scalar_type in NUMPY_SCALARS.items() if issubclass(scalar_type, native)
    )


class NumpyArrays(ArrayBackend[np.ndarray[Any, np.dtype[Any]]]):
    """Reads shape, scalar format, and value properties from a NumPy array."""

    supported_formats = frozenset(NUMPY_SCALARS)
    builtin_properties = MappingProxyType(
        {
            properties.Finite: Finite,
            properties.NonEmpty: NonEmpty,
            properties.Contiguous: Contiguous,
            properties.Materialized: Materialized,
        }
    )
    reports_device = False

    def shape(self, value: np.ndarray[Any, np.dtype[Any]]) -> tuple[int, ...]:
        """The extent of every axis.

        Returns:
            One extent per axis, in order.
        """
        return value.shape

    def dtype(self, value: np.ndarray[Any, np.dtype[Any]]) -> DTypeId:
        """The logical scalar format of the array.

        Returns:
            The logical scalar format.

        Raises:
            ConstraintEvaluationError: for a NumPy dtype with no logical format, such
                as a structured or object dtype.
        """
        name = value.dtype.name
        try:
            return DTypeId(name)
        except ValueError as cause:
            raise ConstraintEvaluationError(
                f"NumPy dtype {name!r} has no logical scalar format"
            ) from cause

    def device(self, value: np.ndarray[Any, np.dtype[Any]]) -> properties.DeviceId | None:  # noqa: ARG002  NumPy reports no placement, so it reads nothing.
        """NumPy places no array, so it reports no placement.

        Returns:
            None, which never stands for the CPU.
        """
        return None

    def native_dtype(self, value: np.ndarray[Any, np.dtype[Any]]) -> Hashable:
        """NumPy's own dtype object, which carries byte order as well as format.

        Returns:
            The array's dtype object.
        """
        return value.dtype

    def operation_formats(self, operation: type[operations.Operation[Any]]) -> frozenset[DTypeId]:
        """Every logical format one NumPy operation accepts, from its declared domain.

        Returns:
            The logical formats the operation accepts.

        Raises:
            AnnotationError: when the class is not a NumPy operation, or when its scalar
                domain has no logical format.
        """
        return scalar_formats(scalar_domain(operation))


NUMPY_ARRAYS = NumpyArrays()
"""The one NumPy backend every NumPy declaration carries."""


type Float[S, *C] = Annotated[
    NDArray[np.floating], TensorDeclaration(NUMPY_ARRAYS, DTypeFamily.FLOATING), S, tuple[*C]
]

type Int[S, *C] = Annotated[
    NDArray[np.signedinteger], TensorDeclaration(NUMPY_ARRAYS, DTypeFamily.INTEGER), S, tuple[*C]
]

type UInt[S, *C] = Annotated[
    NDArray[np.unsignedinteger], TensorDeclaration(NUMPY_ARRAYS, DTypeFamily.UNSIGNED), S, tuple[*C]
]

type Complex[S, *C] = Annotated[
    NDArray[np.complexfloating], TensorDeclaration(NUMPY_ARRAYS, DTypeFamily.COMPLEX), S, tuple[*C]
]

type Bool[S, *C] = Annotated[
    NDArray[np.bool_], TensorDeclaration(NUMPY_ARRAYS, DTypeFamily.BOOLEAN), S, tuple[*C]
]

type Float16[S, *C] = Annotated[
    NDArray[np.float16], TensorDeclaration(NUMPY_ARRAYS, DTypeId.FLOAT16), S, tuple[*C]
]

type Float32[S, *C] = Annotated[
    NDArray[np.float32], TensorDeclaration(NUMPY_ARRAYS, DTypeId.FLOAT32), S, tuple[*C]
]

type Float64[S, *C] = Annotated[
    NDArray[np.float64], TensorDeclaration(NUMPY_ARRAYS, DTypeId.FLOAT64), S, tuple[*C]
]

type Int8[S, *C] = Annotated[
    NDArray[np.int8], TensorDeclaration(NUMPY_ARRAYS, DTypeId.INT8), S, tuple[*C]
]

type Int16[S, *C] = Annotated[
    NDArray[np.int16], TensorDeclaration(NUMPY_ARRAYS, DTypeId.INT16), S, tuple[*C]
]

type Int32[S, *C] = Annotated[
    NDArray[np.int32], TensorDeclaration(NUMPY_ARRAYS, DTypeId.INT32), S, tuple[*C]
]

type Int64[S, *C] = Annotated[
    NDArray[np.int64], TensorDeclaration(NUMPY_ARRAYS, DTypeId.INT64), S, tuple[*C]
]

type UInt8[S, *C] = Annotated[
    NDArray[np.uint8], TensorDeclaration(NUMPY_ARRAYS, DTypeId.UINT8), S, tuple[*C]
]

type UInt16[S, *C] = Annotated[
    NDArray[np.uint16], TensorDeclaration(NUMPY_ARRAYS, DTypeId.UINT16), S, tuple[*C]
]

type UInt32[S, *C] = Annotated[
    NDArray[np.uint32], TensorDeclaration(NUMPY_ARRAYS, DTypeId.UINT32), S, tuple[*C]
]

type UInt64[S, *C] = Annotated[
    NDArray[np.uint64], TensorDeclaration(NUMPY_ARRAYS, DTypeId.UINT64), S, tuple[*C]
]

type Complex64[S, *C] = Annotated[
    NDArray[np.complex64], TensorDeclaration(NUMPY_ARRAYS, DTypeId.COMPLEX64), S, tuple[*C]
]

type Complex128[S, *C] = Annotated[
    NDArray[np.complex128], TensorDeclaration(NUMPY_ARRAYS, DTypeId.COMPLEX128), S, tuple[*C]
]
