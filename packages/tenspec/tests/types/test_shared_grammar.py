"""Where the shape grammar agrees with its upstream, and where it deliberately does not.

The verdicts come from jaxtyping 0.3.11 at revision
40cf683c2b6964ecbe7637889356985dbd9888ea, run against the same declarations, not from
Tenspec's own saved output. jaxtyping is not a dependency of this package, so the
comparison runs outside the suite and this module locks the result it produced.
"""

import pytest

from tenspec.errors import AnnotationError
from tenspec.types.shapes import parse_shape

SHARED = [
    "",
    "rows cols",
    "4",
    "2 rows",
    "*batch time",
    "... height width",
    "_",
    "_ rows",
    "#rows cols",
    "*_ time",
    "rows+1",
    "2*width",
    "(rows+1)*2",
]

REFUSED_BY_BOTH = [
    "*batch *other",
    "... ...",
    "*4",
    "_4",
    "#_",
    "rows#",
    "rows,cols",
    "##rows",
    "...batch",
]

REFUSED_BY_TENSPEC_ALONE = {
    "#*batch": "the accepted design leaves a broadcastable variadic group out of version 1",
    "(False)": "Tenspec reads an integer literal by its type, and excludes bool",
    "min(a,b)": "Tenspec's expression grammar holds no call",
    "rows/2": "Tenspec's expression grammar holds no division",
}


@pytest.mark.parametrize("declaration", SHARED)
def test_shape_accepted_when_in_shared_grammar(declaration: str) -> None:
    # Act
    shape = parse_shape(declaration)

    # Assert
    assert shape is not None


@pytest.mark.parametrize("declaration", REFUSED_BY_BOTH)
def test_shape_rejected_when_both_grammars_exclude_it(declaration: str) -> None:
    # Act and Assert
    with pytest.raises(AnnotationError):
        parse_shape(declaration)


@pytest.mark.parametrize("declaration", sorted(REFUSED_BY_TENSPEC_ALONE))
def test_shape_rejected_when_outside_supported_subset(
    declaration: str,
) -> None:
    # Act and Assert
    with pytest.raises(AnnotationError):
        parse_shape(declaration)
