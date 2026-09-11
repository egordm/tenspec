"""Check one value, or a tuple of related values, without a decorator or a model."""

import functools
from typing import Any, TypeAliasType, cast, get_origin

from typing_extensions import TypeForm

from pydantic import ConfigDict, TypeAdapter
from tenspec.pydantic.annotations import carries_declaration, prepare_annotation
from tenspec.runtime.bindings import validation_scope

STANDALONE_CONFIG = ConfigDict(strict=True, arbitrary_types_allowed=True)
"""The setup a standalone check applies: strict ordinary values, and array types permitted."""

ADAPTER_CACHE_SIZE = 128
"""How many prepared declarations one process keeps.

The decorator and the model path already hold their prepared schemas, so this bound
serves the standalone route alone."""


def adapter_title(annotation: Any) -> str | None:
    """The heading a standalone refusal shows, or None when no tensor is declared.

    The title reads the outer name alone.

    Returns:
        A caller's own alias name, a subscripted Tenspec alias's own name, the
        fixed "Tenspec" for a form that names neither, or None.
    """
    if not carries_declaration(annotation):
        return None
    if isinstance(annotation, TypeAliasType):
        return annotation.__name__
    origin = get_origin(annotation)
    if isinstance(origin, TypeAliasType):
        return origin.__name__
    return "Tenspec"


def build_adapter(annotation: Any) -> TypeAdapter[Any]:
    """Prepare one annotation, and build the adapter that checks it.

    Both steps read the declaration alone. Neither reads a value, and neither opens or
    reads a boundary, so a later call can reuse the result.

    A tensor declaration takes its title from adapter_title.

    Returns:
        The adapter for this annotation.
    """
    title = adapter_title(annotation)
    config: ConfigDict = (
        STANDALONE_CONFIG if title is None else {**STANDALONE_CONFIG, "title": title}
    )
    return TypeAdapter(prepare_annotation(annotation), config=config)


prepared_adapter = functools.lru_cache(maxsize=ADAPTER_CACHE_SIZE)(build_adapter)
"""build_adapter, with the last ADAPTER_CACHE_SIZE annotations kept."""


def hashable(annotation: Any) -> bool:
    """True when this annotation can key the cache.

    A caller can attach metadata that no hash accepts, such as a list. This asks the
    question directly, so an unhashable annotation takes the uncached route, and no
    later failure stands in for the answer.

    Returns:
        True when hash(annotation) succeeds.
    """
    try:
        hash(annotation)
    except TypeError:
        return False
    return True


def validate[T](value: Any, annotation: TypeForm[T]) -> T:
    """Check `value` against `annotation`, and return the accepted value.

    One call owns one boundary, so every tensor inside a tuple declaration shares the
    dimension and dtype bindings. Two calls never relate their names. No binding stays
    active after the call returns.

    A repeated annotation reuses its prepared schema. Every declared check still runs
    against this call's own values, inside this call's own bindings.

    validate((a, b), tuple[Float[Shape["rows"]], Float[Shape["rows"]]]) refuses two
    arrays whose first axis disagrees.

    Returns:
        The accepted value. Tenspec returns an array by identity unless its declaration
        names a transform. A caller's own validator can still replace it.
    """
    adapter = prepared_adapter(annotation) if hashable(annotation) else build_adapter(annotation)
    with validation_scope():
        return cast("T", adapter.validate_python(value))
