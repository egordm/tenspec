"""The two interfaces a backend runs an operation behind, and the ancestry rule for them."""

from abc import ABC, abstractmethod
from typing import Annotated, Any, Never, cast

from pydantic import PlainValidator
from tenspec.errors import AnnotationError
from tenspec.types import properties


class ArrayProperty[A](properties.ArrayProperty, ABC):
    """A check one backend runs over one array, with no result and no replacement.

    A caller derives from the author base of its backend, such as
    `tenspec.numpy.ArrayProperty`, which fixes A and the scalar formats the check accepts.
    """

    @staticmethod
    @abstractmethod
    def validate(values: A) -> None:
        """Refuse `values` when they do not have this property.

        Return None to accept. This method changes nothing and returns no replacement.

        Raises:
            ValueError: when the values measurably lack the property. TensorMismatch
                carries structured detail, and every built-in check raises it.
            ConstraintEvaluationError: when this value's state makes the check
                impossible, which is never a measured refusal.
        """


class ArrayTransform[A](ABC):
    """A replacement one backend runs over one array, keeping every structural fact.

    A caller derives from the author base of its backend, such as
    `tenspec.numpy.ArrayTransform`, which fixes A and the scalar formats it accepts.

    A transform is not a property, so it carries no declaration marker. A declaration
    names every transform before its properties.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Never:
        """Refuse a transform that reached Pydantic as ordinary metadata.

        A tensor alias consumes its own transforms while it prepares, so Pydantic never
        sees a correctly placed one. This hook therefore runs only for a transform written
        outside the alias brackets, where it would replace nothing.

        Raises:
            AnnotationError: always.
        """
        raise AnnotationError(
            f"{cls.__name__} is a Tenspec transform. It reached Pydantic as ordinary "
            f"metadata, where it replaces nothing. Write it inside the tensor alias "
            f"brackets, after the shape and before every property: "
            f"Alias[Shape['...'], {cls.__name__}]. Alias is the backend alias whose "
            f"scalar domain the transform accepts."
        )

    @staticmethod
    @abstractmethod
    def transform(values: A) -> A:
        """Return a replacement for `values`, with the same structure.

        The result keeps the concrete class, the shape, the native dtype and the device
        of its input. The runtime checks each of those and refuses a result that changed
        one. Do not change the caller's own array, and do not change shared storage.

        Returns:
            The replacement array.
        """


type Operation[A] = ArrayProperty[A] | ArrayTransform[A]
"""Either kind of operation a declaration can name."""


def derives_from(candidate: type[Any], *bases: type[Any]) -> bool:
    """True when `candidate` really derives from one of `bases`.

    It reads the method resolution order by identity rather than call `issubclass`,
    because an abstract base class hashes its argument to cache the answer, and a caller
    may define a class that has no usable hash. Every route that tests the ancestry of a
    declared operation calls this function, so one rule answers for all of them.

    Returns:
        True when one of the bases is an ancestor of the candidate.
    """
    return any(base is ancestor for ancestor in candidate.__mro__ for base in bases)


def checked_transform(candidate: Any) -> type[ArrayTransform[Any]]:
    """The declared class, once it really derives from ArrayTransform.

    Returns:
        The identical class the caller named.

    Raises:
        ValueError: when the value is no class, or derives from no transform interface.
    """
    if not (isinstance(candidate, type) and derives_from(candidate, ArrayTransform)):
        raise ValueError(f"{candidate!r} is not a transform")
    return cast(type[ArrayTransform[Any]], candidate)


type TransformClass = Annotated[type[ArrayTransform], PlainValidator(checked_transform)]
"""A declared transform class, as a Pydantic field reads it.

The field stays typed, and this validator replaces the check Pydantic runs for a
`type[...]` field. That check calls `issubclass` against an abstract base class, which
hashes the candidate, so a class a caller made unhashable never reached the runtime.
"""
