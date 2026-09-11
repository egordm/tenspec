"""The one rule that decides whether a backend can run a declared property."""

from abc import ABCMeta

import numpy as np
import pytest
from numpy.typing import NDArray

from tenspec import Finite
from tenspec.errors import AnnotationError
from tenspec.numpy import NUMPY_ARRAYS, ArrayProperty, NumpyArrays
from tenspec.numpy import Finite as NumpyFinite
from tenspec.types.dtypes import DTypeId
from tenspec.types.properties import ArrayProperty as PropertyMarker

FLOATING = frozenset({DTypeId.FLOAT16, DTypeId.BFLOAT16, DTypeId.FLOAT32, DTypeId.FLOAT64})
"""Every floating format a `Float` declaration permits, the Torch-only bfloat16 included."""


class Unimplemented(PropertyMarker):
    """A common marker that names a check no backend implements."""


class Float64Only(ArrayProperty[np.float64]):
    """A caller's property that can only read float64 values."""

    @staticmethod
    def validate(values: NDArray[np.float64]) -> None:
        """Accept every float64 array."""


def test_common_name_resolved_when_only_another_library_has_a_permitted_format() -> None:
    # Arrange  NumPy has no bfloat16, so the backend inventory must drop it before the
    # comparison. Otherwise a valid NumPy floating declaration would fail.
    assert DTypeId.BFLOAT16 not in NUMPY_ARRAYS.supported_formats

    # Act
    resolved = NUMPY_ARRAYS.resolve_property(Finite, permitted_formats=FLOATING)

    # Assert
    assert resolved is NumpyFinite


def test_caller_property_resolved_to_itself() -> None:
    # Act
    resolved = NUMPY_ARRAYS.resolve_property(
        Float64Only, permitted_formats=frozenset({DTypeId.FLOAT64})
    )

    # Assert
    assert resolved is Float64Only


@pytest.mark.parametrize(
    ("declared", "permitted", "refusal"),
    [
        (Unimplemented, FLOATING, "implements no check"),
        # An abstract validate returns None, so an abstract class would pass as a no-op.
        (ArrayProperty, FLOATING, "is abstract"),
        (Float64Only, FLOATING, "does not guarantee the scalar domain"),
        (Finite, frozenset({DTypeId.BFLOAT16}), "represents no scalar format"),
    ],
    ids=["no check", "abstract", "a wider declared domain", "no format this library has"],
)
def test_property_refused_when_this_backend_cannot_run_it(
    declared: type[PropertyMarker], permitted: frozenset[DTypeId], refusal: str
) -> None:
    # Act and Assert
    with pytest.raises(AnnotationError, match=refusal):
        NUMPY_ARRAYS.resolve_property(declared, permitted_formats=permitted)


def test_scalar_domain_inherited_when_subclass_names_no_argument() -> None:
    # Arrange
    class Stricter(Float64Only):
        """A caller's subclass that keeps the domain of the class it extends."""

    # Act
    accepted = NUMPY_ARRAYS.operation_formats(Stricter)

    # Assert
    assert accepted == {DTypeId.FLOAT64}


class UnhashableMeta(ABCMeta):
    """A metaclass that leaves every class it makes without a hash."""

    __hash__ = None


class RaisesHashMeta(ABCMeta):
    """A metaclass whose hash raises rather than being absent."""

    def __hash__(cls) -> int:
        """Refuse to hash.

        Raises:
            TypeError: always, which is what the cache route must survive.
        """
        raise TypeError("hash deliberately unavailable")


class NoHash(ArrayProperty[np.generic], metaclass=UnhashableMeta):
    """A caller's property whose class carries no hash."""

    @staticmethod
    def validate(values: NDArray[np.generic]) -> None:
        """Accept every array."""


class HashRaises(ArrayProperty[np.generic], metaclass=RaisesHashMeta):
    """A caller's property whose class raises when it is hashed."""

    @staticmethod
    def validate(values: NDArray[np.generic]) -> None:
        """Accept every array."""


@pytest.mark.parametrize(
    "declared", [NoHash, HashRaises], ids=["no hash at all", "a hash that raises"]
)
def test_property_resolved_when_the_cache_cannot_hold_its_class(
    declared: type[PropertyMarker],
) -> None:
    # Act
    resolved = NUMPY_ARRAYS.resolve_property(declared, permitted_formats=FLOATING)

    # Assert  A hash that raises is not a missing hash, and both take the uncached route.
    assert resolved is declared


def test_property_resolved_when_the_backend_carries_no_hash() -> None:
    # Arrange
    class Unhashable(NumpyArrays):
        """A backend a caller made unhashable."""

        __hash__ = None

    # Act
    resolved = Unhashable().resolve_property(Finite, permitted_formats=FLOATING)

    # Assert  The cache belongs to the instance and never keys on it.
    assert resolved is NumpyFinite
