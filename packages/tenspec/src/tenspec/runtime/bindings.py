"""The dimension, dtype and device facts of one validation boundary."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import TypeVar

from tenspec.errors import BindingError, BindingReason, TensorMismatch
from tenspec.types.dtypes import DTypeId
from tenspec.types.properties import DeviceId

type Extent = int | tuple[int, ...]


class Bindings:
    """The facts one validation boundary established, and the agreements they force.

    A fresh Bindings starts empty. Two boundaries never share one object, so a name bound
    in one call places no requirement on the next call. A named axis binds one extent. A
    named variadic group binds the whole tuple of extents it covered.
    """

    def __init__(self) -> None:
        self.committed_dimensions: dict[str, Extent] = {}
        self.committed_dtypes: dict[TypeVar, DTypeId] = {}
        self.committed_devices: dict[TypeVar, DeviceId] = {}

    @property
    def dimensions(self) -> Mapping[str, Extent]:
        """The committed extent of every bound dimension name."""
        return MappingProxyType(self.committed_dimensions)

    @property
    def dtypes(self) -> Mapping[TypeVar, DTypeId]:
        """The committed scalar format of every bound dtype variable."""
        return MappingProxyType(self.committed_dtypes)

    @property
    def devices(self) -> Mapping[TypeVar, DeviceId]:
        """The committed placement of every bound device variable."""
        return MappingProxyType(self.committed_devices)

    def bind_dimension(self, name: str, extent: Extent) -> None:
        """Bind `name` to `extent`, or match the extent it already holds.

        The binding stays unchanged after a refusal.

        Raises:
            TensorMismatch: when the name already holds another extent.
        """
        bound = self.committed_dimensions.get(name)
        if bound is None:
            self.committed_dimensions[name] = extent
        elif bound != extent:
            raise TensorMismatch(
                requirement=f"{name}={bound}",
                observed=f"{name}={extent}",
                bindings=self.dimensions,
            )

    def bind_dtype(self, variable: TypeVar, dtype: DTypeId) -> None:
        """Bind `variable` to `dtype`, or match the format it already holds.

        Raises:
            TensorMismatch: when the variable already holds another format.
        """
        bound = self.committed_dtypes.get(variable)
        if bound is None:
            self.committed_dtypes[variable] = dtype
        elif bound is not dtype:
            raise TensorMismatch(
                requirement=f"dtype {bound.value} for {variable.__name__}",
                observed=f"dtype {dtype.value}",
            )

    def bind_device(self, variable: TypeVar, device: DeviceId) -> None:
        """Bind `variable` to `device`, or match the placement it already holds.

        Raises:
            TensorMismatch: when the variable already holds another placement.
        """
        bound = self.committed_devices.get(variable)
        if bound is None:
            self.committed_devices[variable] = device
        elif bound != device:
            raise TensorMismatch(
                requirement=f"device {describe_device(bound)} for {variable.__name__}",
                observed=f"device {describe_device(device)}",
            )

    def axis_size(self, name: str) -> int:
        """The extent bound to `name`.

        Returns:
            The committed extent.

        Raises:
            BindingError: with reason UNBOUND when this boundary never bound the name, and
                NOT_SCALAR when the name holds a variadic group.
        """
        extent = self.committed_dimensions.get(name)
        if extent is None:
            raise BindingError(BindingReason.UNBOUND, name)
        if not isinstance(extent, int):
            raise BindingError(BindingReason.NOT_SCALAR, name)
        return extent


def describe_device(device: DeviceId) -> str:
    """A placement written the way a caller writes it.

    Returns:
        The kind, with its ordinal when the backend reported one.
    """
    return device.kind if device.ordinal is None else f"{device.kind}:{device.ordinal}"


ACTIVE_BINDINGS: ContextVar[Bindings | None] = ContextVar("tenspec_bindings", default=None)
"""The bindings of the innermost open scope. Each thread and each task holds its own."""


@contextmanager
def validation_scope() -> Iterator[Bindings]:
    """Open one boundary with fresh bindings, and restore the previous scope on exit.

    The previous scope returns after success and after failure alike.

    Yields:
        The fresh bindings of the boundary this scope opened.
    """
    bindings = Bindings()
    token = ACTIVE_BINDINGS.set(bindings)
    try:
        yield bindings
    finally:
        ACTIVE_BINDINGS.reset(token)


def active_bindings() -> Bindings:
    """The bindings of the innermost open scope.

    Returns:
        The bindings of the innermost open scope.

    Raises:
        BindingError: with reason NO_SCOPE when no scope is open.
    """
    bindings = ACTIVE_BINDINGS.get()
    if bindings is None:
        raise BindingError(BindingReason.NO_SCOPE)
    return bindings


def axis_size(name: str) -> int:
    """The extent bound to `name` in the boundary that is validating now.

    Call it inside a checked function, a model validator, or a standalone validation.
    It reads a bound dimension and never selects a tensor or reparses a shape.

    Returns:
        The extent bound to the name.

    Raises:
        BindingError: with reason NO_SCOPE outside a scope, UNBOUND for a name this
            boundary never bound, and NOT_SCALAR for a variadic group.
    """
    return active_bindings().axis_size(name)
