"""The facts one array library reports, and the carrier that names that library."""

import functools
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, cast

from pydantic_core import CoreSchema

from pydantic import GetCoreSchemaHandler
from tenspec.errors import AnnotationError
from tenspec.types import operations
from tenspec.types.dtypes import DTypeFamily, DTypeId
from tenspec.types.operations import derives_from
from tenspec.types.properties import ArrayProperty, DeviceId

RESOLUTION_CACHE_SIZE = 128
"""How many resolved operation classes one backend remembers."""


class ArrayBackend[A](ABC):
    """Reads facts from one array library. It compares nothing and changes nothing.

    A backend receives neither a TensorType nor a Bindings object, so it cannot read a
    shape declaration or decide whether two operands agree. The runtime does that.

    It also resolves a declared operation into the class that runs it.
    """

    supported_formats: ClassVar[frozenset[DTypeId]] = frozenset()
    """Every logical scalar format this library represents.

    It must agree with the formats `dtype` returns.
    """

    builtin_properties: ClassVar[
        Mapping[type[ArrayProperty], type[operations.ArrayProperty[Any]]]
    ] = MappingProxyType({})
    """The class that runs each common property name, such as Finite, on this library."""

    reports_device: ClassVar[bool] = False
    """Whether this library places arrays on a device that a declaration can require."""

    @abstractmethod
    def shape(self, value: A) -> tuple[int, ...]:
        """The extent of every axis, read from metadata alone.

        Returns:
            One extent per axis, in order.
        """

    @abstractmethod
    def dtype(self, value: A) -> DTypeId:
        """The logical scalar format of the array.

        Returns:
            The logical scalar format.

        Raises:
            ConstraintEvaluationError: when this library's format has no DTypeId.
        """

    @abstractmethod
    def device(self, value: A) -> DeviceId | None:
        """Where the array actually sits.

        Returns:
            The placement, or None for a library with no device contract.
        """

    @abstractmethod
    def native_dtype(self, value: A) -> Hashable:
        """This library's own dtype descriptor, which a logical DTypeId cannot replace.

        Two arrays of one logical format can hold different native dtypes: a byte-swapped
        float64 is still float64 logically. Transform preservation compares this
        descriptor, so it must be immutable and comparable inside this backend.

        Returns:
            The native dtype descriptor.
        """

    @abstractmethod
    def operation_formats(self, operation: type[operations.Operation[Any]]) -> frozenset[DTypeId]:
        """Every scalar format one operation class accepts, read from its author base.

        A backend that fixes no scalar domain on its author base reports every format
        it represents.

        Returns:
            The scalar formats the operation accepts.

        Raises:
            AnnotationError: when the class is not an operation of this backend, or when
                its declared scalar domain has no logical format.
        """

    @functools.cached_property
    def cached_resolver(
        self,
    ) -> Callable[
        [type[Any], frozenset[DTypeId], type[Any]],
        type[operations.Operation[A]],
    ]:
        """This backend's bounded cache of resolved operation classes.

        It holds operation classes and format sets, never an array, a result, or a
        binding, so every call still checks its own values. The cache belongs to one
        backend instance and never keys on it, so a backend needs no hash of its own.

        Returns:
            The cached form of `resolve_operation`.
        """
        return functools.lru_cache(maxsize=RESOLUTION_CACHE_SIZE)(self.resolve_operation)

    def resolve_operation(
        self,
        declared: type[Any],
        permitted_formats: frozenset[DTypeId],
        interface: type[Any],
    ) -> type[operations.Operation[A]]:
        """Resolve one declared operation against `interface`, reading no cache.

        A common property name resolves to this backend's implementation. A caller's own
        class resolves to itself. `permitted_formats` is
        `DTypeRequirement.permitted_formats`, and it must guarantee the operation's own
        scalar domain.

        Returns:
            The concrete class the runtime calls.

        Raises:
            AnnotationError: when this backend cannot run the operation, when the class
                is abstract, or when the declaration does not guarantee its scalar domain.
        """
        kind = "check" if interface is operations.ArrayProperty else "transform"
        executable = builtin_implementation(self.builtin_properties, declared)
        if not derives_from(executable, interface):
            raise AnnotationError(
                f"{executable.__name__} implements no {kind}. Derive it from this "
                f"backend's {interface.__name__}."
            )
        if inspect.isabstract(executable):
            raise AnnotationError(
                f"{executable.__name__} is abstract, so it would do nothing. "
                f"Name a class that implements its method."
            )
        effective = effective_formats(permitted_formats, self)
        accepted = self.operation_formats(executable)
        if not effective <= accepted:
            raise AnnotationError(
                f"this declaration does not guarantee the scalar domain of "
                f"{executable.__name__}. It permits {format_names(effective)}, and "
                f"{executable.__name__} accepts {format_names(accepted)}"
            )
        return cast("type[operations.Operation[A]]", executable)

    def resolve_with_cache(
        self, declared: type[Any], permitted_formats: frozenset[DTypeId], interface: type[Any]
    ) -> type[operations.Operation[A]]:
        """Resolve one declared operation, through the cache when its key has a hash.

        The route comes from hashing the real cache key, because a class can carry a
        `__hash__` that raises rather than none at all. The catch covers that one hash
        call, so a failure inside resolution or inside a caller's own code keeps its own
        identity and never picks a route. The uncached resolver hashes nothing, so a
        class the cache cannot hold still resolves the same way.

        Returns:
            The concrete class the runtime calls.

        Raises:
            AnnotationError: for every reason `resolve_operation` refuses.
        """
        try:
            hash((declared, permitted_formats, interface))
        except TypeError:
            return self.resolve_operation(declared, permitted_formats, interface)
        return self.cached_resolver(declared, permitted_formats, interface)

    def resolve_property(
        self, property_type: type[ArrayProperty], *, permitted_formats: frozenset[DTypeId]
    ) -> type[operations.ArrayProperty[A]]:
        """The class that runs one declared property on this library.

        Returns:
            The concrete class whose `validate` the runtime calls.

        Raises:
            AnnotationError: for every reason `resolve_operation` refuses.
        """
        return cast(
            "type[operations.ArrayProperty[A]]",
            self.resolve_with_cache(property_type, permitted_formats, operations.ArrayProperty),
        )

    def resolve_transform(
        self,
        transform_type: type[operations.ArrayTransform[Any]],
        *,
        permitted_formats: frozenset[DTypeId],
    ) -> type[operations.ArrayTransform[A]]:
        """The class that runs one declared transform on this library.

        Returns:
            The concrete class whose `transform` the runtime calls.

        Raises:
            AnnotationError: for every reason `resolve_operation` refuses.
        """
        return cast(
            "type[operations.ArrayTransform[A]]",
            self.resolve_with_cache(transform_type, permitted_formats, operations.ArrayTransform),
        )


def builtin_implementation(
    builtins: Mapping[type[ArrayProperty], type[operations.ArrayProperty[Any]]],
    declared: type[Any],
) -> type[Any]:
    """This backend's implementation of a common property name, or the class itself.

    The match is by identity, so a class a caller made unhashable never needs a hash to
    reach resolution. A common name is one exact class, so identity is the same answer a
    mapping lookup gives.

    Returns:
        The class resolution continues with.
    """
    return next(
        (implementation for common, implementation in builtins.items() if common is declared),
        declared,
    )


def effective_formats(
    permitted: frozenset[DTypeId], backend: ArrayBackend[Any], subject: str = "this declaration"
) -> frozenset[DTypeId]:
    """The formats one declaration permits that this library also represents.

    Both entry routes compute the domain here, and it reads no array.

    Returns:
        The formats both the declaration and the library allow.

    Raises:
        AnnotationError: when no format remains, so no array can satisfy the declaration.
    """
    if not permitted:
        raise AnnotationError(f"{subject} permits no scalar format, so no array can satisfy it")
    effective = permitted & backend.supported_formats
    if not effective:
        raise AnnotationError(
            f"{type(backend).__name__} represents no scalar format {subject} permits"
        )
    return effective


def format_names(formats: frozenset[DTypeId]) -> str:
    """The scalar formats of one set, in a fixed order, for a refusal message.

    Returns:
        The format names, separated by commas.
    """
    return ", ".join(sorted(item.value for item in formats))


def refuse_device(backend: ArrayBackend[Any], subject: str = "this declaration") -> None:
    """Refuse a device requirement on a library that places no array.

    Annotation preparation and the direct runtime share this one rule, so a requirement
    the annotation path refuses cannot reach a value check through the other route.

    Raises:
        AnnotationError: when this library places no array.
    """
    if not backend.reports_device:
        raise AnnotationError(
            f"{type(backend).__name__} places no array, so {subject} takes no device"
        )


@dataclass(frozen=True, slots=True)
class TensorDeclaration:
    """The fixed part of one backend alias: its array library and its scalar format.

    Every Tenspec alias carries one of these in its metadata, so preparation reads the
    backend from the declaration rather than from a registry or from the incoming value.
    It sits beside ArrayBackend so a backend module never reaches the annotation resolver.
    """

    backend: ArrayBackend[Any]
    dtype: DTypeId | DTypeFamily

    def __get_pydantic_core_schema__(
        self, source: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Refuse a declaration that reached Pydantic with no Tenspec preparation.

        This carrier implements Pydantic's schema protocol only to refuse. Without it
        Pydantic would ignore the tensor metadata and check the base array type alone,
        which reports a pass the declaration never earned.

        Raises:
            AnnotationError: always, because no entry point resolved this declaration.
        """
        raise AnnotationError(
            f"{source} carries a Tenspec declaration that no entry point prepared. "
            f"Use checked, TensorContracts, or validate."
        )
