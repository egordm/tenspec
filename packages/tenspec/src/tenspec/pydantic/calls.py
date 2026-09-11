"""Check the declared arguments and return of one function or method."""

import functools
import inspect
import threading
from collections.abc import Callable
from types import FunctionType
from typing import Any, Self

from pydantic import ConfigDict, validate_call
from tenspec.errors import AnnotationError
from tenspec.pydantic.annotations import prepare_annotation, prepare_declaration
from tenspec.runtime.bindings import validation_scope

CALL_CONFIG = ConfigDict(strict=True, arbitrary_types_allowed=True)
"""The setup a checked call applies: strict ordinary values, and array types permitted."""


def function_with_annotations(function: FunctionType, annotations: dict[str, Any]) -> FunctionType:
    """A copy of `function` that carries `annotations`, leaving the original unchanged.

    Pydantic reads a signature to build its schema. It reads the copy, so no first call
    changes the annotations a caller wrote, and two cold calls cannot race over them.

    Returns:
        A copy of the function that carries the prepared annotations.
    """
    copy = FunctionType(
        function.__code__,
        function.__globals__,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    copy.__kwdefaults__ = function.__kwdefaults__
    copy.__qualname__ = function.__qualname__
    copy.__doc__ = function.__doc__
    copy.__module__ = function.__module__
    copy.__dict__.update(function.__dict__)
    copy.__annotations__ = annotations
    return copy


def prepared_call(function: FunctionType, declared: dict[str, Any]) -> Callable[..., Any]:
    """One Pydantic call wrapper over a copy that carries the prepared declarations.

    Returns:
        A callable that checks the arguments and the return.
    """
    prepared = {
        name: prepare_annotation(value) if name == "return" else prepare_declaration(value)
        for name, value in declared.items()
    }
    return validate_call(config=CALL_CONFIG, validate_return=True)(
        function_with_annotations(function, prepared)
    )


def with_receiver(declared: dict[str, Any], receiver: type) -> dict[str, Any]:
    """The declarations of one method, with Self read as the runtime receiver class.

    Returns:
        The declarations, with every Self replaced.
    """
    return {name: receiver if value is Self else value for name, value in declared.items()}


def receiver_checked_call(function: FunctionType, declared: dict[str, Any]) -> Callable[..., Any]:
    """A method that names Self, checked once per runtime receiver class.

    A subclass call cannot reuse a base class's return check, because Self means the class
    that received the call. One specialization is built under a lock and published whole,
    so two cold calls cannot share a half-built schema.

    Returns:
        A callable that specializes on its receiver, then checks the call.
    """
    specializations: dict[type, Callable[..., Any]] = {}
    lock = threading.Lock()

    @functools.wraps(function)
    def call(*args: Any, **kwargs: Any) -> Any:
        if not args:
            raise AnnotationError(
                f"{function.__qualname__} names Self, so it needs a receiver as its first argument"
            )
        receiver = args[0] if isinstance(args[0], type) else type(args[0])
        validated = specializations.get(receiver)
        if validated is None:
            with lock:
                validated = specializations.get(receiver)
                if validated is None:
                    validated = prepared_call(function, with_receiver(declared, receiver))
                    specializations[receiver] = validated
        with validation_scope():
            return validated(*args, **kwargs)

    return call


def checked[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Check the declared arguments and the declared return of one call.

    One scope spans the argument checks, the body, and the return check, so the operands
    and the return share their dimension, dtype and device bindings. Every call starts
    fresh bindings, so a later call can use other extents. Tenspec replaces an accepted
    array only when its declaration names a transform. A caller's own validator keeps its
    own behavior.

    Place it under an ordinary @classmethod or @staticmethod, closest to the function. A
    method that returns Self is checked against the class that received the call.

    Returns:
        A function with the same signature, which checks its declarations on every call.

    Raises:
        AnnotationError: for an async function, and for a callable that is not a plain
            Python function. This release checks neither.
    """
    if not isinstance(function, FunctionType):
        raise AnnotationError(f"{function!r} is not a plain function, which checked requires.")
    if inspect.iscoroutinefunction(function):
        raise AnnotationError(
            f"{function.__qualname__} is async. Version 1 does not check an async call."
        )
    declared = inspect.get_annotations(function, eval_str=True)
    if Self in declared.values():
        return receiver_checked_call(function, declared)
    validated = prepared_call(function, declared)

    @functools.wraps(function)
    def call(*args: P.args, **kwargs: P.kwargs) -> R:
        with validation_scope():
            return validated(*args, **kwargs)

    return call
