"""The array properties and the device placements a declaration can require."""

from typing import Any, ClassVar, Never, TypeVar

from pydantic import Field
from tenspec.errors import AnnotationError
from tenspec.types.base import Frozen


class ArrayProperty:
    """A property of an array value. A caller names a subclass in a declaration."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Never:
        """Refuse a property that reached Pydantic as ordinary metadata.

        A tensor alias consumes its own properties while it prepares, so Pydantic never
        sees a correctly placed one. This hook therefore runs only for a property written
        outside the alias brackets, where it would check nothing.

        Raises:
            AnnotationError: always.
        """
        raise AnnotationError(
            f"{cls.__name__} is a Tenspec property. It reached Pydantic as ordinary "
            f"metadata, where it checks nothing. Write it inside the tensor alias "
            f"brackets, after the shape: Alias[Shape['...'], {cls.__name__}]. Alias is "
            f"the backend alias whose scalar domain the property accepts."
        )


class Finite(ArrayProperty):
    """Every element of the array is a finite number."""


class NonEmpty(ArrayProperty):
    """The array holds at least one element."""


class Contiguous(ArrayProperty):
    """The array uses C-contiguous storage."""


class Materialized(ArrayProperty):
    """The array holds actual data, rather than metadata alone."""


class Device:
    """The bound of a shared device variable. Every placement satisfies it.

    A backend names its own kinds by deriving from it, as `tenspec.torch.CUDA` does.
    """

    kind: ClassVar[str | None] = None


class DeviceId(Frozen):
    """Where one array actually sits, as its backend reports it."""

    kind: str = Field(min_length=1, description="the placement kind, such as cpu or cuda")
    ordinal: int | None = Field(default=None, description="which device of that kind, if any")


class DeviceRequirement(Frozen):
    """What a declaration requires of one array's placement.

    A kind compares the kind alone. A shared variable compares the whole descriptor, so
    two operands must sit on the same device of the same kind.
    """

    kind: str | None = Field(default=None, description="the required placement kind")
    variable: TypeVar | None = Field(
        default=None, description="the shared variable every operand carrying it must match"
    )
