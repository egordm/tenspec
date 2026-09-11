"""Alias resolution: turn a Tenspec declaration into an ordinary Pydantic annotation."""

import functools
import operator
from dataclasses import dataclass
from types import GenericAlias, NoneType, UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    TypeAliasType,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

from typing_extensions import TypeForm

from pydantic import AfterValidator, Field
from tenspec.errors import AnnotationError
from tenspec.runtime.arrays import (
    TensorDeclaration,
    effective_formats,
    refuse_device,
)
from tenspec.runtime.bindings import active_bindings
from tenspec.runtime.validation import ArrayValidator
from tenspec.types import operations
from tenspec.types.dtypes import DType, DTypeRequirement
from tenspec.types.operations import derives_from
from tenspec.types.properties import ArrayProperty, Device, DeviceRequirement
from tenspec.types.shapes import parse_shape
from tenspec.types.tensor import TensorType


@dataclass(frozen=True, slots=True)
class TensorCheck:
    """One prepared validator, ready to run inside ordinary Pydantic validation.

    It holds no bindings. It reads the boundary that is open when it runs, so one
    prepared check serves every later call and resolves no operation again.
    """

    validator: ArrayValidator[Any]

    def __call__(self, value: Any) -> Any:
        """Check `value` against the requirement.

        Returns:
            The identical array the caller supplied, or the replacement its declared
            transforms returned.
        """
        return self.validator.validate(value, bindings=active_bindings())


def declaration_of(alias: TypeAliasType) -> TensorDeclaration | None:
    """The Tenspec declaration inside an alias body.

    Returns:
        The declaration, or None when the alias is not one of ours.
    """
    return next(
        (item for item in get_args(alias.__value__) if isinstance(item, TensorDeclaration)),
        None,
    )


def carries_declaration(annotation: Any, *, entered: frozenset[Any] = frozenset()) -> bool:
    """True when a Tenspec declaration sits anywhere inside this annotation.

    It looks through the wrappers this release supports: a union, a tuple, and a caller's
    own alias. `entered` holds the caller aliases this walk already opened, so an alias
    that names itself ends the walk instead of repeating it.

    Returns:
        True when the annotation declares a tensor.
    """
    if isinstance(annotation, TypeAliasType):
        if declaration_of(annotation) is not None:
            return True
        return annotation not in entered and carries_declaration(
            annotation.__value__, entered=entered | {annotation}
        )
    metadata = getattr(annotation, "__metadata__", None)
    if metadata is not None:
        return carries_declaration(annotation.__origin__, entered=entered)
    origin = get_origin(annotation)
    if isinstance(origin, TypeAliasType):
        return declaration_of(origin) is not None
    if origin in {UnionType, Union, tuple}:
        return any(carries_declaration(item, entered=entered) for item in get_args(annotation))
    return False


def prepare_declaration[T](annotation: TypeForm[T]) -> TypeForm[T]:
    """Prepare one parameter or field declaration, and check its default with it.

    A declaration that carries a tensor takes Pydantic's own default validation, so an
    omitted default meets the same requirement as a supplied argument. An ordinary
    parameter keeps the default behavior of its caller.

    Returns:
        A Pydantic annotation whose default is checked when it declares a tensor.
    """
    resolved: Any = prepare_annotation(annotation)
    if not carries_declaration(annotation):
        return cast("TypeForm[T]", resolved)
    return cast("TypeForm[T]", Annotated[resolved, Field(validate_default=True)])


def prepare_annotation[T](
    annotation: TypeForm[T], *, entered: frozenset[Any] = frozenset()
) -> TypeForm[T]:
    """Resolve every Tenspec declaration inside `annotation` into a Pydantic annotation.

    The result is an ordinary annotation with an AfterValidator, so Pydantic's own
    argument, field, return, and default validation still applies in its usual order.
    An annotation that carries no Tenspec declaration comes back unchanged.

    prepare_annotation(Float[Shape["rows cols"]]) returns the base array type with one
    validator attached. The original declaration stays unchanged.

    `entered` holds the caller aliases this resolution already opened, and only the walk
    through those aliases passes it.

    Returns:
        An ordinary Pydantic annotation, or the input when it carries no declaration.

    Raises:
        AnnotationError: for a declaration this release does not support.
    """
    wrapped: Any = annotation
    metadata = getattr(annotation, "__metadata__", None)
    if metadata is not None:
        inner: Any = prepare_annotation(wrapped.__origin__, entered=entered)
        return cast("TypeForm[T]", Annotated[(inner, *metadata)])
    origin = get_origin(annotation)
    if origin is UnionType or origin is Union:
        return cast("TypeForm[T]", prepare_union(get_args(annotation), entered=entered))
    if origin is tuple:
        items = tuple(prepare_annotation(item, entered=entered) for item in get_args(annotation))
        return cast("TypeForm[T]", GenericAlias(tuple, items))
    if isinstance(origin, TypeAliasType):
        return prepare_alias(origin, get_args(annotation))
    if isinstance(annotation, TypeAliasType):
        return cast("TypeForm[T]", prepare_named_alias(annotation, entered))
    return annotation


def prepare_named_alias(alias: TypeAliasType, entered: frozenset[Any]) -> Any:
    """Resolve an alias the caller wrote without a subscript.

    A Tenspec alias still needs its own shape, so it takes the subscripted path and fails
    there. A caller's own alias names a whole declaration, and its body carries the shape
    and every constraint. An alias that declares no tensor comes back unchanged, and
    Pydantic reads it as it always did.

    Returns:
        The prepared body, or the alias itself when it declares no tensor.

    Raises:
        AnnotationError: when a Tenspec alias names no shape, or when a caller's alias
            reaches itself, because a tensor declaration that contains its own name has
            no end.
    """
    if declaration_of(alias) is not None:
        return prepare_alias(alias, ())
    if alias in entered:
        raise AnnotationError(
            f"{alias.__name__} contains itself, and a recursive tensor alias is not supported"
        )
    if not carries_declaration(alias, entered=entered):
        return alias
    return prepare_annotation(alias.__value__, entered=entered | {alias})


def prepare_union(arms: tuple[Any, ...], *, entered: frozenset[Any] = frozenset()) -> Any:
    """Resolve a union that holds at most one tensor arm beside None.

    An optional tensor is supported. A union of alternative tensor structures is not,
    because a refused arm can leave a binding behind that a later operand then matches.
    An arm that names a caller's own alias counts as the declaration that alias holds.

    Returns:
        The union with every arm prepared.

    Raises:
        AnnotationError: for a union of alternative tensor structures.
    """
    classified = [(arm, carries_declaration(arm, entered=entered)) for arm in arms]
    tensor_arms = [arm for arm, carries in classified if carries]
    others = [arm for arm, carries in classified if not carries]
    if tensor_arms and not (len(tensor_arms) == 1 and all(arm is NoneType for arm in others)):
        raise AnnotationError(
            "a union of alternative tensor structures is not supported. Write one tensor "
            "declaration, and add None for an optional tensor."
        )
    return functools.reduce(
        operator.or_, [prepare_annotation(arm, entered=entered) for arm in arms]
    )


def prepare_alias(alias: TypeAliasType, supplied: tuple[Any, ...]) -> Any:
    """Resolve one subscripted Tenspec alias into its checked Pydantic annotation.

    The alias body holds the parts every use shares: the base array type and the
    declaration. The subscript holds the parts the caller writes.

    Returns:
        An ordinary Pydantic annotation with one tensor check attached.

    Raises:
        AnnotationError: when the subscript names no shape, or an unsupported argument.
    """
    body = get_args(alias.__value__)
    declaration = declaration_of(alias) if body else None
    if declaration is None:
        return alias[supplied] if supplied else alias
    if not supplied:
        raise AnnotationError(
            f"{alias.__name__} needs a shape, such as {alias.__name__}[Shape['rows cols']]"
        )
    base = body[0]
    shape_argument, *constraints = supplied
    dtype_variable, device, transforms, properties = declared_constraints(
        alias, declaration, constraints
    )
    tensor_type = TensorType(
        shape=parse_shape(shape_text(alias, shape_argument)),
        dtype=DTypeRequirement(permitted=declaration.dtype, variable=dtype_variable),
        device=device,
        transforms=transforms,
        properties=properties,
    )
    effective_formats(
        tensor_type.dtype.permitted_formats, declaration.backend, subject=alias.__name__
    )
    validator = ArrayValidator(tensor_type=tensor_type, backend=declaration.backend)
    return Annotated[base, AfterValidator(TensorCheck(validator))]


def shape_text(alias: TypeAliasType, argument: Any) -> str:
    """The declared shape string inside a Shape[...] argument.

    Returns:
        The shape text the caller wrote.

    Raises:
        AnnotationError: when the argument is not a single-string Literal.
    """
    values = get_args(argument)
    if get_origin(argument) is not Literal or len(values) != 1 or not isinstance(values[0], str):
        raise AnnotationError(
            f"the first argument of {alias.__name__} must be Shape['...'], not {argument!r}"
        )
    return values[0]


def shares_scalar_format(variable: TypeVar) -> bool:
    """Read what one shared variable relates, from the bound the caller gave it.

    The bound of a dtype variable also narrows the declaration, and
    `DTypeRequirement.permitted_formats` owns that one calculation.

    Returns:
        True when the variable relates scalar formats, False when it relates devices.

    Raises:
        AnnotationError: for a bound that is neither DType nor Device.
    """
    bound = variable.__bound__
    if isinstance(bound, type) and issubclass(bound, DType):
        return True
    if isinstance(bound, type) and issubclass(bound, Device):
        return False
    raise AnnotationError(
        f"the shared variable {variable.__name__} must be bound to DType or Device, "
        f"or to one of their subsets"
    )


def device_kind_from_bound(variable: TypeVar | None, declared_kind: str | None) -> str | None:
    """The placement kind a shared device variable carries through its own bound.

    A variable bound to CUDA requires a CUDA tensor, whatever else the declaration says.

    Returns:
        The kind the declaration requires, from the bound or from an explicit marker.

    Raises:
        AnnotationError: when an explicit kind contradicts the variable's bound.
    """
    if variable is None:
        return declared_kind
    bound_kind = cast("type[Device]", variable.__bound__).kind
    if bound_kind is None:
        return declared_kind
    if declared_kind is not None and declared_kind != bound_kind:
        raise AnnotationError(
            f"{variable.__name__} is bound to {bound_kind}, which contradicts the "
            f"declared {declared_kind}"
        )
    return bound_kind


def device_kind_of(marker: type[Device]) -> str:
    """The placement kind one device marker names.

    Returns:
        The kind, such as cuda.

    Raises:
        AnnotationError: for the bare Device bound, which names no kind.
    """
    if marker.kind is None:
        raise AnnotationError(
            "Device is the bound of a shared device variable. Name a kind, such as CUDA."
        )
    return marker.kind


def shared_variables(constraints: list[Any]) -> tuple[TypeVar | None, TypeVar | None]:
    """The dtype variable and the device variable one declaration names.

    A declaration relates at most one scalar format and at most one placement, so a
    second variable of either kind is a declaration error.

    Returns:
        The dtype variable and the device variable, each None when absent.

    Raises:
        AnnotationError: for a second variable of either kind, or for a bound that is
            neither DType nor Device.
    """
    dtype_variable: TypeVar | None = None
    device_variable: TypeVar | None = None
    for item in constraints:
        if not isinstance(item, TypeVar):
            continue
        if shares_scalar_format(item):
            if dtype_variable is not None:
                raise AnnotationError("a declaration takes at most one dtype variable")
            dtype_variable = item
        elif device_variable is not None:
            raise AnnotationError("a declaration takes at most one device variable")
        else:
            device_variable = item
    return dtype_variable, device_variable


def declared_constraints(
    alias: TypeAliasType, declaration: TensorDeclaration, constraints: list[Any]
) -> tuple[
    TypeVar | None,
    DeviceRequirement | None,
    tuple[type[operations.ArrayTransform[Any]], ...],
    tuple[type[ArrayProperty], ...],
]:
    """Sort every subscript argument after the shape into what it requires.

    An argument is a transform, a property marker, a device kind, or a shared variable.
    A variable's bound says whether it shares a scalar format or a placement. The two
    operation sequences split here, once, and each keeps the order the caller wrote.

    Returns:
        The shared dtype variable, the device requirement, the transforms, and the
        property markers.

    Raises:
        AnnotationError: for a repeated constraint, a bound that is neither DType nor
            Device, a transform written after a property, a requirement this backend
            cannot check, or an unknown argument.
    """
    backend = declaration.backend
    dtype_variable, device_variable = shared_variables(constraints)
    device_kind: str | None = None
    transforms: list[type[operations.ArrayTransform[Any]]] = []
    properties: list[type[ArrayProperty]] = []
    for item in constraints:
        if isinstance(item, TypeVar):
            continue
        if isinstance(item, type) and derives_from(item, operations.ArrayTransform):
            if properties:
                raise AnnotationError(
                    f"{item.__name__} is a transform, and a declaration names every "
                    f"transform before its properties"
                )
            transforms.append(item)
        elif isinstance(item, type) and issubclass(item, ArrayProperty):
            properties.append(item)
        elif isinstance(item, type) and issubclass(item, Device):
            if device_kind is not None:
                raise AnnotationError("a declaration takes at most one device kind")
            device_kind = device_kind_of(item)
        else:
            raise AnnotationError(f"declaration argument {item!r} is not supported")
    if device_kind is not None or device_variable is not None:
        refuse_device(backend, alias.__name__)
    device_kind = device_kind_from_bound(device_variable, device_kind)
    device = (
        DeviceRequirement(kind=device_kind, variable=device_variable)
        if device_kind is not None or device_variable is not None
        else None
    )
    return dtype_variable, device, tuple(transforms), tuple(properties)
