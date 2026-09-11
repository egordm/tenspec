"""The shared base of every immutable Tenspec value."""

from pydantic import BaseModel, ConfigDict


class Frozen(BaseModel):
    """A value that never changes after construction.

    Arbitrary types are permitted because these values hold type objects and type
    variables, which carry no Pydantic schema of their own.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
