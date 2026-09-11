"""Axis requirements, shape expressions, and the parser that reads one declaration."""

import ast
from collections.abc import Mapping
from enum import StrEnum

from pydantic import Field
from tenspec.errors import AnnotationError, BindingError, BindingReason
from tenspec.types.base import Frozen


class ArithmeticOperator(StrEnum):
    """The operators a shape expression can use."""

    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"


class IntegerLiteral(Frozen):
    """A whole number written inside a shape expression."""

    value: int = Field(description="the number itself")


class DimensionReference(Frozen):
    """A dimension name a shape expression reads. The name must be bound already."""

    name: str = Field(min_length=1, description="the dimension name to read")


class Arithmetic(Frozen):
    """One operation over two shape expressions."""

    operator: ArithmeticOperator = Field(description="the operation to apply")
    left: "Expression" = Field(description="the left operand")
    right: "Expression" = Field(description="the right operand")


type Expression = IntegerLiteral | DimensionReference | Arithmetic

Arithmetic.model_rebuild()


class FixedAxis(Frozen):
    """An axis that must have one exact extent."""

    extent: int = Field(ge=0, description="the extent this axis must have")
    broadcastable: bool = Field(default=False, description="whether an extent of one passes")


class NamedAxis(Frozen):
    """An axis that binds a dimension name, or matches the name's committed extent."""

    name: str = Field(min_length=1, description="the dimension name this axis carries")
    broadcastable: bool = Field(default=False, description="whether an extent of one passes")


class AnonymousAxis(Frozen):
    """An axis of any extent. It binds nothing and relates to nothing."""


class ExpressionAxis(Frozen):
    """An axis whose extent is computed from dimensions that are bound already."""

    expression: Expression = Field(description="the expression that gives the extent")
    source: str = Field(description="the expression as the caller wrote it")
    broadcastable: bool = Field(default=False, description="whether an extent of one passes")


type Axis = FixedAxis | NamedAxis | AnonymousAxis | ExpressionAxis


class VariadicGroup(Frozen):
    """The axes between the prefix and the suffix, taken together.

    A named group binds the whole tuple of extents. An anonymous group binds nothing.
    """

    name: str | None = Field(default=None, description="the name the tuple binds, or None")


class ShapeSpec(Frozen):
    """The axes one array must have: a fixed prefix, one optional group, a fixed suffix."""

    prefix: tuple[Axis, ...] = Field(description="the fixed-rank axes before the group")
    group: VariadicGroup | None = Field(default=None, description="the variadic group, if any")
    suffix: tuple[Axis, ...] = Field(default=(), description="the fixed-rank axes after it")

    @property
    def fixed_rank(self) -> int:
        """The number of axes this declaration names one at a time."""
        return len(self.prefix) + len(self.suffix)


def evaluate(expression: Expression, dimensions: Mapping[str, int | tuple[int, ...]]) -> int:
    """The extent one shape expression gives, from dimensions that are bound already.

    It evaluates. It never solves for an unknown dimension.

    Returns:
        The computed extent.

    Raises:
        BindingError: when a referenced name has no binding, or holds a variadic tuple.
    """
    if isinstance(expression, IntegerLiteral):
        return expression.value
    if isinstance(expression, DimensionReference):
        extent = dimensions.get(expression.name)
        if extent is None:
            raise BindingError(BindingReason.UNBOUND, expression.name)
        if not isinstance(extent, int):
            raise BindingError(BindingReason.NOT_SCALAR, expression.name)
        return extent
    left = evaluate(expression.left, dimensions)
    right = evaluate(expression.right, dimensions)
    if expression.operator is ArithmeticOperator.ADD:
        return left + right
    if expression.operator is ArithmeticOperator.SUBTRACT:
        return left - right
    return left * right


def parse_expression(source: str) -> Expression:
    """Read one shape expression into a typed tree, once, during preparation.

    Python's own parser reads the text. Tenspec then converts the parse into its own tree
    and refuses every construct outside whole numbers, dimension names, +, -, * and
    parentheses. Nothing is executed, here or later.

    Returns:
        The typed expression tree.

    Raises:
        AnnotationError: for text this grammar does not accept.
    """
    try:
        parsed = ast.parse(source, mode="eval")
    except SyntaxError as cause:
        raise AnnotationError(f"shape expression {source!r} is not readable") from cause
    return convert_expression(parsed.body, source)


def convert_expression(node: ast.expr, source: str) -> Expression:
    """One node of Python's parse tree as a Tenspec shape expression.

    Returns:
        The typed expression for this node.

    Raises:
        AnnotationError: for a construct this grammar does not accept.
    """
    operators: Mapping[type[ast.operator], ArithmeticOperator] = {
        ast.Add: ArithmeticOperator.ADD,
        ast.Sub: ArithmeticOperator.SUBTRACT,
        ast.Mult: ArithmeticOperator.MULTIPLY,
    }
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return IntegerLiteral(value=node.value)
    if isinstance(node, ast.Name):
        return DimensionReference(name=node.id)
    if isinstance(node, ast.BinOp) and type(node.op) in operators:
        return Arithmetic(
            operator=operators[type(node.op)],
            left=convert_expression(node.left, source),
            right=convert_expression(node.right, source),
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return Arithmetic(
            operator=ArithmeticOperator.SUBTRACT,
            left=IntegerLiteral(value=0),
            right=convert_expression(node.operand, source),
        )
    raise AnnotationError(
        f"shape expression {source!r} uses a construct outside whole numbers, dimension "
        f"names, +, - and *"
    )


class AxisModifiers(Frozen):
    """The prefixes one axis token carries before its name, extent, or expression."""

    broadcastable: bool = Field(default=False, description="the # prefix")
    variadic: bool = Field(default=False, description="the * prefix")
    anonymous: bool = Field(default=False, description="the _ prefix")


def read_modifiers(word: str) -> tuple[AxisModifiers, str]:
    """The prefixes of one axis token, and what remains after them.

    Returns:
        The modifiers, and the rest of the token.

    Raises:
        AnnotationError: when one prefix appears more than once.
    """
    seen = {"#": False, "*": False, "_": False}
    rest = word
    while rest and rest[0] in seen:
        prefix = rest[0]
        if seen[prefix]:
            raise AnnotationError(f"{prefix} appears at most once per axis: {word!r}")
        seen[prefix], rest = True, rest[1:]
    return AxisModifiers(broadcastable=seen["#"], variadic=seen["*"], anonymous=seen["_"]), rest


def named_or_group(word: str, modifiers: AxisModifiers, name: str) -> Axis | VariadicGroup:
    """The axis or group an identifier token declares.

    Returns:
        The named or anonymous axis, or the group it opens.

    Raises:
        AnnotationError: for a combination of modifiers this grammar refuses.
    """
    if modifiers.anonymous:
        if modifiers.broadcastable:
            raise AnnotationError(f"an anonymous axis is never broadcastable: {word!r}")
        return VariadicGroup() if modifiers.variadic else AnonymousAxis()
    if not name:
        raise AnnotationError(f"an axis needs a name, an extent, or _: {word!r}")
    if modifiers.variadic:
        return VariadicGroup(name=name)
    return NamedAxis(name=name, broadcastable=modifiers.broadcastable)


def parse_axis(word: str) -> Axis | VariadicGroup:
    """One whitespace-separated token of a shape declaration.

    Returns:
        The axis the token declares, or the variadic group it opens.

    Raises:
        AnnotationError: for a token or a combination of modifiers this grammar refuses.
    """
    if "," in word and "(" not in word:
        raise AnnotationError(f"axes are separated with spaces, not commas: {word!r}")
    if word.endswith("#"):
        raise AnnotationError(f"a broadcastable axis carries # at its start, not its end: {word!r}")
    if "..." in word:
        if word != "...":
            raise AnnotationError(f"the anonymous group ... stands on its own, but got {word!r}")
        return VariadicGroup()

    modifiers, rest = read_modifiers(word)
    if modifiers.variadic and modifiers.broadcastable:
        raise AnnotationError(f"a broadcastable variadic group is not supported: {word!r}")
    if rest.isdigit():
        if modifiers.variadic or modifiers.anonymous:
            raise AnnotationError(f"an exact extent takes no * and no _: {word!r}")
        return FixedAxis(extent=int(rest), broadcastable=modifiers.broadcastable)
    if not rest or rest.isidentifier():
        return named_or_group(word, modifiers, rest)
    if modifiers.variadic or modifiers.anonymous:
        raise AnnotationError(f"a shape expression takes no * and no _: {word!r}")
    return ExpressionAxis(
        expression=parse_expression(rest), source=rest, broadcastable=modifiers.broadcastable
    )


def parse_shape(declaration: str) -> ShapeSpec:
    """Read a shape declaration written as whitespace-separated axis tokens.

    An empty declaration is a scalar. A declaration holds at most one variadic group.

    parse_shape("*batch 2 rows") accepts any leading axes, then an axis of exactly two,
    then an axis that binds rows.

    Returns:
        The axes the declaration names, in order.

    Raises:
        AnnotationError: for a token this grammar refuses, or a second variadic group.
    """
    prefix: list[Axis] = []
    suffix: list[Axis] = []
    group: VariadicGroup | None = None
    for word in declaration.split():
        parsed = parse_axis(word)
        if isinstance(parsed, VariadicGroup):
            if group is not None:
                raise AnnotationError("a shape declares at most one variadic group")
            group = parsed
        elif group is None:
            prefix.append(parsed)
        else:
            suffix.append(parsed)
    return ShapeSpec(prefix=tuple(prefix), group=group, suffix=tuple(suffix))
