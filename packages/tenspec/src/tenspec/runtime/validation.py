"""The one check that compares an array with its declared requirement."""

from collections import ChainMap
from collections.abc import Mapping
from dataclasses import dataclass, field

from tenspec.errors import TensorMismatch, TransformContractError
from tenspec.runtime.arrays import (
    ArrayBackend,
    effective_formats,
    format_names,
    refuse_device,
)
from tenspec.runtime.bindings import Bindings, Extent, describe_device
from tenspec.types import operations
from tenspec.types.properties import DeviceId, DeviceRequirement
from tenspec.types.shapes import (
    AnonymousAxis,
    Axis,
    ExpressionAxis,
    FixedAxis,
    ShapeSpec,
    evaluate,
)
from tenspec.types.tensor import TensorType


def match_axis(
    axis: Axis, extent: int, known: Mapping[str, Extent], proposed: dict[str, Extent]
) -> None:
    """Compare one observed extent with one declared axis, and propose what it binds.

    A broadcastable axis passes on an extent of one, and establishes nothing.

    Raises:
        TensorMismatch: when the extent disagrees with the axis.
        BindingError: when an expression reads a name that has no binding.
    """
    if isinstance(axis, AnonymousAxis):
        return
    if axis.broadcastable and extent == 1:
        return
    if isinstance(axis, FixedAxis):
        if extent != axis.extent:
            raise TensorMismatch(
                requirement=f"an axis of {axis.extent}",
                observed=f"an axis of {extent}",
                bindings=known,
            )
        return
    if isinstance(axis, ExpressionAxis):
        wanted = evaluate(axis.expression, known)
        if extent != wanted:
            raise TensorMismatch(
                requirement=f"{axis.source}={wanted}",
                observed=f"an axis of {extent}",
                bindings=known,
            )
        return
    expected = known.get(axis.name)
    if expected is None:
        proposed[axis.name] = extent
    elif expected != extent:
        raise TensorMismatch(
            requirement=f"{axis.name}={expected}", observed=f"{axis.name}={extent}", bindings=known
        )


def proposed_dimensions(
    shape: ShapeSpec, observed: tuple[int, ...], bindings: Bindings
) -> dict[str, Extent]:
    """The dimension bindings this shape adds, once every axis agrees.

    A named axis matches the extent the boundary already committed, or the extent an
    earlier axis of this same shape proposed. Nothing commits here.

    Returns:
        The name and extent of every dimension this shape would newly bind.

    Raises:
        TensorMismatch: when the rank or any axis disagrees.
    """
    if shape.group is None and len(observed) != shape.fixed_rank:
        raise TensorMismatch(
            requirement=f"rank {shape.fixed_rank}",
            observed=f"rank {len(observed)} {observed}",
            bindings=bindings.dimensions,
        )
    if shape.group is not None and len(observed) < shape.fixed_rank:
        raise TensorMismatch(
            requirement=f"rank {shape.fixed_rank} or more",
            observed=f"rank {len(observed)} {observed}",
            bindings=bindings.dimensions,
        )

    tail_start = len(observed) - len(shape.suffix)
    named_axes = (*shape.prefix, *shape.suffix)
    named_extents = (*observed[: len(shape.prefix)], *observed[tail_start:])
    proposed: dict[str, Extent] = {}
    known: Mapping[str, Extent] = ChainMap(proposed, dict(bindings.dimensions))
    for axis, extent in zip(named_axes, named_extents, strict=True):
        match_axis(axis, extent, known, proposed)

    if shape.group is not None and shape.group.name is not None:
        covered = observed[len(shape.prefix) : tail_start]
        expected = known.get(shape.group.name)
        if expected is None:
            proposed[shape.group.name] = covered
        elif expected != covered:
            raise TensorMismatch(
                requirement=f"{shape.group.name}={expected}",
                observed=f"{shape.group.name}={covered}",
                bindings=known,
            )
    return proposed


def check_device[A](
    placement: DeviceRequirement,
    value: A,
    backend: ArrayBackend[A],
    bindings: Bindings,
) -> DeviceId:
    """Compare where the array actually sits with where the declaration requires it.

    A kind compares the kind alone. A shared variable compares the whole descriptor.

    Returns:
        The placement the backend reported.

    Raises:
        TensorMismatch: when the placement disagrees with the requirement.
    """
    observed = backend.device(value)
    if observed is None:
        raise TensorMismatch(
            requirement=f"device {placement.kind}", observed="an array with no placement"
        )
    if placement.kind is not None and observed.kind != placement.kind:
        raise TensorMismatch(
            requirement=f"device {placement.kind}",
            observed=f"device {describe_device(observed)}",
        )
    if placement.variable is not None:
        shared = bindings.devices.get(placement.variable)
        if shared is not None and shared != observed:
            raise TensorMismatch(
                requirement=f"device {describe_device(shared)} for {placement.variable.__name__}",
                observed=f"device {describe_device(observed)}",
            )
    return observed


def prepared_properties[A](
    tensor_type: TensorType, backend: ArrayBackend[A]
) -> tuple[type[operations.ArrayProperty[A]], ...]:
    """Every declared property, as the class this backend runs for it.

    Annotation preparation and the direct runtime both call this function, so one
    sequence carries a common name and a caller's own class alike.

    Returns:
        The resolved properties, in the order the declaration named them.

    Raises:
        AnnotationError: when this backend cannot run one of them, or when the
            declaration does not guarantee its scalar domain.
    """
    permitted = tensor_type.dtype.permitted_formats
    return tuple(
        backend.resolve_property(item, permitted_formats=permitted)
        for item in tensor_type.properties
    )


def prepared_transforms[A](
    tensor_type: TensorType, backend: ArrayBackend[A]
) -> tuple[type[operations.ArrayTransform[A]], ...]:
    """Every declared transform, as the class this backend runs for it.

    Preparation already split the transforms from the properties and fixed their order.
    This reads that sequence and never regroups it.

    Returns:
        The resolved transforms, in the order the declaration named them.

    Raises:
        AnnotationError: when this backend cannot run one of them, or when the
            declaration does not guarantee its scalar domain.
    """
    permitted = tensor_type.dtype.permitted_formats
    return tuple(
        backend.resolve_transform(item, permitted_formats=permitted)
        for item in tensor_type.transforms
    )


def apply_transform[A](
    value: A, transform: type[operations.ArrayTransform[A]], backend: ArrayBackend[A]
) -> A:
    """Run one transform, and refuse a result that changed a structural fact.

    Every fact of the input is read before the transform runs, so a transform that
    changes its input in place cannot move both sides of the comparison. The comparison
    reads those facts, not the declaration, so a result that still fits a broad
    declaration is refused when it changed an unconstrained extent or the byte order of
    its dtype.

    It neither copies the input nor undoes a side effect, so a transform still owns what
    it does to the values.

    Returns:
        The replacement array, which every later operation receives.

    Raises:
        TransformContractError: when the result changed its class, shape, native dtype
            or device.
    """
    supplied_class = type(value)
    supplied = (
        ("shape", backend.shape(value)),
        ("native dtype", backend.native_dtype(value)),
        ("device", backend.device(value)),
    )
    result = transform.transform(value)
    if type(result) is not supplied_class:
        raise TransformContractError(
            transform=transform.__name__,
            fact="array class",
            before=supplied_class.__name__,
            after=type(result).__name__,
        )
    observed = (backend.shape(result), backend.native_dtype(result), backend.device(result))
    for (fact, before), after in zip(supplied, observed, strict=True):
        if before != after:
            raise TransformContractError(
                transform=transform.__name__, fact=fact, before=repr(before), after=repr(after)
            )
    return result


@dataclass(frozen=True, eq=False, slots=True, kw_only=True)
class ArrayValidator[A]:
    """One resolved declaration, ready to check many arrays.

    Construction resolves the declared transforms and properties once and refuses a
    requirement this backend cannot answer. A later check reads the resolved sequences
    and never resolves again.

    The prepared state is frozen, so no caller can disable a resolved check by assignment.
    It holds no array, no result and no bindings, so every check reads its own values and
    commits into the bindings its caller supplies. A new requirement needs a new
    validator. Equality and hashing stay the identity of the object, so a caller's own
    operation class never has to be hashable.

    Raises:
        AnnotationError: when this backend cannot answer the requirement at all.
    """

    tensor_type: TensorType
    backend: ArrayBackend[A]
    transforms: tuple[type[operations.ArrayTransform[A]], ...] = field(init=False)
    properties: tuple[type[operations.ArrayProperty[A]], ...] = field(init=False)

    def __post_init__(self) -> None:
        """Refuse an unanswerable requirement, then resolve both operation sequences.

        Raises:
            AnnotationError: when this backend cannot answer the requirement at all.
        """
        if self.tensor_type.device is not None:
            refuse_device(self.backend)
        effective_formats(self.tensor_type.dtype.permitted_formats, self.backend)
        object.__setattr__(self, "transforms", prepared_transforms(self.tensor_type, self.backend))
        object.__setattr__(self, "properties", prepared_properties(self.tensor_type, self.backend))

    def validate(self, value: A, *, bindings: Bindings) -> A:
        """Check one array against the prepared requirement, and return the array to use.

        It reads the structure of the supplied array, runs every declared transform in
        order and verifies preservation after each one, runs every declared property on
        the result, and only then commits new facts. A declaration that names no
        transform returns the identical object.

        On success every new dimension, dtype and device fact commits to `bindings`. On
        any failure nothing commits, so a refused array leaves the boundary as it found
        it, and an earlier accepted operand keeps the facts it established.

        Returns:
            The identical array the caller supplied, or the result of the declared
            transforms.

        Raises:
            TensorMismatch: when the array disagrees with the requirement.
            TransformContractError: when a transform changed a structural fact.
        """
        tensor_type = self.tensor_type
        backend = self.backend
        proposed = proposed_dimensions(tensor_type.shape, backend.shape(value), bindings)

        requirement = tensor_type.dtype
        observed_dtype = backend.dtype(value)
        if not requirement.accepts(observed_dtype):
            narrowing = (
                ""
                if requirement.variable is None
                else f" that {requirement.variable.__name__} narrows to "
                f"{format_names(requirement.permitted_formats)}"
            )
            raise TensorMismatch(
                requirement=f"dtype {requirement.permitted.value}{narrowing}",
                observed=f"dtype {observed_dtype.value}",
            )
        dtype_variable = requirement.variable
        if dtype_variable is not None:
            shared = bindings.dtypes.get(dtype_variable)
            if shared is not None and shared is not observed_dtype:
                raise TensorMismatch(
                    requirement=f"dtype {shared.value} for {dtype_variable.__name__}",
                    observed=f"dtype {observed_dtype.value}",
                )

        placement = tensor_type.device
        observed_device = (
            None if placement is None else check_device(placement, value, backend, bindings)
        )

        accepted = value
        for change in self.transforms:
            accepted = apply_transform(accepted, change, backend)

        for check in self.properties:
            check.validate(accepted)

        for name, extent in proposed.items():
            bindings.bind_dimension(name, extent)
        if dtype_variable is not None:
            bindings.bind_dtype(dtype_variable, observed_dtype)
        if placement is not None and placement.variable is not None and observed_device is not None:
            bindings.bind_device(placement.variable, observed_device)
        return accepted


def validate_array[A](
    value: A,
    *,
    tensor_type: TensorType,
    backend: ArrayBackend[A],
    bindings: Bindings,
) -> A:
    """Check one array against one requirement, in one call.

    This is the one-shot form of `ArrayValidator`. It prepares the requirement and checks
    the array, so it refuses an unanswerable requirement before it reads any value. A
    caller that checks many arrays against one requirement keeps an `ArrayValidator`
    instead, and pays that preparation once.

    Returns:
        The identical array the caller supplied, or the result of the declared
        transforms.

    Raises:
        AnnotationError: when this backend cannot answer the requirement at all.
        TensorMismatch: when the array disagrees with the requirement.
        TransformContractError: when a transform changed a structural fact.
    """
    validator = ArrayValidator(tensor_type=tensor_type, backend=backend)
    return validator.validate(value, bindings=bindings)
