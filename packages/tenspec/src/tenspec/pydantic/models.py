"""The mixin that gives one model validation one boundary for its related fields."""

from collections.abc import Callable
from typing import Annotated, Any, ClassVar, cast

from pydantic_core import CoreSchema
from pydantic_core.core_schema import ValidatorFunctionWrapHandler

from pydantic import BaseModel, GetCoreSchemaHandler, model_validator
from tenspec.pydantic.annotations import carries_declaration, prepare_annotation
from tenspec.runtime.bindings import validation_scope


class TensorContracts:
    """Check the tensor fields of one model together, in one boundary.

    Place it first among the bases, as `class M(TensorContracts, ProjectBase)`. It adds no
    base of its own and keeps the caller's model configuration, strictness, and validators.
    The caller permits arbitrary types for the array fields.

    Each model validation opens its own boundary, so a nested model binds independently.
    Tenspec replaces a field value only when its declaration names a transform, such as
    CopyReadOnly, and it copies or seals nothing else. A caller's own validator keeps its
    own behavior. A tensor field checks its default the way it checks a supplied value.

    Preparation runs where Pydantic builds the model's schema, so Pydantic has already
    resolved every annotation in the caller's own namespace. A declaration written as a
    string, as a name local to the function that defines the model, or as a name defined
    after the model, therefore reaches preparation as a real type. The subclass keeps what
    the caller declared in __tensor_declarations__.
    """

    __tensor_declarations__: ClassVar[dict[str, Any]] = {}

    @classmethod
    def caller_schema_hook(cls) -> Callable[..., CoreSchema] | None:
        """The schema hook a caller's own base owns, when one does.

        Pydantic's own BaseModel hook is deprecated, so Tenspec calls the handler rather
        than that one. A hook the caller wrote still runs.

        Returns:
            The caller's bound hook, or None when no base after this mixin owns one.
        """
        following = cls.__mro__[cls.__mro__.index(TensorContracts) + 1 :]
        for base in following:
            if base is BaseModel:
                return None
            hook = base.__dict__.get("__get_pydantic_core_schema__")
            if hook is not None:
                return cast("Callable[..., CoreSchema]", hook.__get__(None, cls))
        return None

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Prepare the resolved tensor fields, then let the remaining bases build.

        Returns:
            The schema the caller's own bases and Pydantic build for this model.
        """
        model = cast("type[BaseModel]", cls)
        declared = dict(cls.__tensor_declarations__)
        for name, field in model.model_fields.items():
            if field.annotation is not None and carries_declaration(field.annotation):
                declared[name] = field.annotation
                complete: Any = (
                    Annotated[(field.annotation, *field.metadata)]
                    if field.metadata
                    else field.annotation
                )
                field.annotation = prepare_annotation(complete)
                field.metadata = []
                field.validate_default = True
        cls.__tensor_declarations__ = declared
        chained = cls.caller_schema_hook()
        return chained(source, handler) if chained is not None else handler(source)

    # The name stays private because this hook lands in the caller's own model namespace.
    @model_validator(mode="wrap")
    @classmethod
    def _check_in_one_tensor_scope(cls, value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
        """Run every field check of this model inside one fresh boundary.

        Returns:
            The validated model.
        """
        with validation_scope():
            return handler(value)
