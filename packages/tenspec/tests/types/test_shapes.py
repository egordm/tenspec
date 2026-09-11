"""What the shape grammar accepts, and what it refuses."""

import pytest

from tenspec.errors import AnnotationError
from tenspec.types.shapes import (
    AnonymousAxis,
    ExpressionAxis,
    FixedAxis,
    NamedAxis,
    VariadicGroup,
    parse_shape,
)


def test_scalar_shape_when_declaration_empty() -> None:
    # Act
    shape = parse_shape("")

    # Assert
    assert shape.prefix == ()
    assert shape.group is None


def test_axes_partitioned_when_group_present() -> None:
    # Act
    shape = parse_shape("channels *batch height width")

    # Assert
    assert shape.prefix == (NamedAxis(name="channels"),)
    assert shape.group == VariadicGroup(name="batch")
    assert shape.suffix == (NamedAxis(name="height"), NamedAxis(name="width"))
    assert shape.fixed_rank == 3


@pytest.mark.parametrize("declaration", ["... time", "*_ time"], ids=["three dots", "star _"])
def test_group_unnamed_when_anonymous(declaration: str) -> None:
    # Act
    shape = parse_shape(declaration)

    # Assert
    assert shape.group == VariadicGroup()


def test_modifiers_preserved_when_shape_parsed() -> None:
    # Act
    shape = parse_shape("#rows _ 4")

    # Assert
    assert shape.prefix == (
        NamedAxis(name="rows", broadcastable=True),
        AnonymousAxis(),
        FixedAxis(extent=4),
    )


def test_expression_text_preserved_when_shape_parsed() -> None:
    # Act
    axis = parse_shape("(rows+1)*2").prefix[0]

    # Assert
    assert isinstance(axis, ExpressionAxis)
    assert axis.source == "(rows+1)*2"


@pytest.mark.parametrize(
    ("shape", "refusal"),
    [
        ("*batch *other", "at most one variadic group"),
        ("... ...", "at most one variadic group"),
        ("*4", "no \\* and no _"),
        ("_4", "no \\* and no _"),
        ("#_", "never broadcastable"),
        ("#*batch", "broadcastable variadic group"),
        ("*rows+1", "no \\* and no _"),
        ("rows#", "# at its start"),
        ("rows,cols", "not commas"),
        ("...batch", "stands on its own"),
        ("##rows", "at most once"),
        ("rows/2", "outside whole numbers"),
        ("min(a,b)", "outside whole numbers"),
        ("(False)", "outside whole numbers"),
        ("(True)", "outside whole numbers"),
    ],
)
def test_shape_rejected_when_syntax_unsupported(shape: str, refusal: str) -> None:
    # Act and Assert
    with pytest.raises(AnnotationError, match=refusal):
        parse_shape(shape)
