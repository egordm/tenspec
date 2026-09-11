# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Custom checks and read-only arrays
#
# An attribution run masks parts of a context, scores the same answer under each mask, and reads
# how far each masked score falls below the unmasked one. This run requires its scores to be
# finite, never above zero, and stored on their own memory.
#
# "Never above zero" follows from what a log probability is. "Finite" is this run's own policy: a
# log probability of zero probability is negative infinity, which some runs keep.
#
# The declared form comes first. The same requirements with outer Pydantic validators follow it
# for comparison. The data here is synthetic.

# %%
from typing import Annotated, Any
from typing import Literal as Shape

import numpy as np
from numpy.typing import NDArray

from pydantic import AfterValidator, BeforeValidator, ValidationError
from tenspec import AnnotationError, Finite, NonEmpty, checked, validate
from tenspec.numpy import ArrayProperty, CopyReadOnly, Float64, Int

# %% [markdown]
# ## A property of your own
#
# Tenspec reads the shape, the scalar format, `NonEmpty` and `Finite`. The remaining requirement
# is a property of this run's own, derived from the NumPy author base. Its scalar type parameter
# is the domain its method reads.


# %%
class NonPositive(ArrayProperty[np.floating[Any]]):
    """No entry of the array is above zero."""

    @staticmethod
    def validate(values: NDArray[np.floating[Any]]) -> None:
        """Refuse a block holding a positive entry.

        Raises:
            ValueError: when any entry is above zero.
        """
        if bool((values > 0).any()):
            raise ValueError("no entry may be positive")


type LogProbs = Float64[Shape["masks tokens"], NonEmpty, Finite, NonPositive]


@checked
def ablation_drop(logprobs: LogProbs) -> NDArray[np.float64]:
    """How far each masked score falls below the unmasked score in the first row, in nats.

    Returns:
        One drop per mask.
    """
    totals = logprobs.sum(axis=1)
    return totals[0] - totals


scores = np.log(np.array([[0.9, 0.8], [0.5, 0.4], [0.2, 0.7]]))
print(ablation_drop(scores).round(3).tolist())

# %% [markdown]
# Every requirement is one name inside one declaration, and a block with a positive entry is
# refused through `ValidationError`.
#
# The properties run in the order this declaration names them, so `Finite` runs before
# `NonPositive` here. That order belongs to the declaration, not to the class.

# %%
positive = np.array([[-0.1, 0.2]])

try:
    ablation_drop(positive)
except ValidationError as refusal:
    print(refusal.errors()[0]["msg"])

# %% [markdown]
# ## Owned storage
#
# `CopyReadOnly` is a transform, so the declaration names it before every property. The stored
# block holds the same values on its own memory, refuses an ordinary write, and leaves the
# caller's array untouched.

# %%
type StoredLogProbs = Float64[Shape["masks tokens"], CopyReadOnly, NonEmpty, Finite, NonPositive]

stored = validate(scores, StoredLogProbs)
print(
    f"equal {np.array_equal(stored, scores)}, "
    f"own storage {not np.shares_memory(stored, scores)}, "
    f"writeable {stored.flags.writeable}, input still writeable {scores.flags.writeable}"
)

# %% [markdown]
# ## What the declaration refuses before a value arrives
#
# A declared property states the scalar formats its method reads. Preparation refuses a
# declaration that cannot guarantee them.

# %%
try:
    validate(np.ones(2, dtype=np.int64), Int[Shape["rows"], NonPositive])
except AnnotationError as refusal:
    print(refusal)

# %% [markdown]
# A declared transform runs after the shape and the scalar format already match, so an array of
# the wrong dtype is refused before the copy happens.

# %%
try:
    validate(np.ones((1, 2), dtype=np.float32), StoredLogProbs)
except ValidationError as refusal:
    print(refusal.errors()[0]["msg"])

# %% [markdown]
# ## The same requirements with wrapper validators
#
# The same run is possible with ordinary Pydantic validators around a shorter declaration. The
# check becomes a function attached with an `AfterValidator`, and the copy becomes a function
# attached with a `BeforeValidator`.


# %%
def refuse_positive(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Refuse a block holding a positive entry.

    Returns:
        The accepted block.

    Raises:
        ValueError: when any entry is above zero.
    """
    if bool((values > 0).any()):
        raise ValueError("no entry may be positive")
    return values


def seal(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Copy a block and seal the copy against writes.

    Returns:
        A read-only view of a separate copy.
    """
    owned = values.copy()
    owned.setflags(write=False)
    return owned.view()


LogProbsByHand = Annotated[
    Float64[Shape["masks tokens"], NonEmpty, Finite], AfterValidator(refuse_positive)
]

StoredByHand = Annotated[
    Float64[Shape["masks tokens"], NonEmpty, Finite],
    BeforeValidator(seal),
    AfterValidator(refuse_positive),
]


@checked
def ablation_drop_by_hand(logprobs: LogProbsByHand) -> NDArray[np.float64]:
    """How far each masked score falls below the unmasked score in the first row, in nats.

    Returns:
        One drop per mask.
    """
    totals = logprobs.sum(axis=1)
    return totals[0] - totals


print(ablation_drop_by_hand(scores).round(3).tolist())

try:
    ablation_drop_by_hand(positive)
except ValidationError as refusal:
    print(refusal.errors()[0]["msg"])

stored_by_hand = validate(scores, StoredByHand)
print(
    f"equal {np.array_equal(stored_by_hand, scores)}, "
    f"own storage {not np.shares_memory(stored_by_hand, scores)}, "
    f"writeable {stored_by_hand.flags.writeable}, "
    f"input still writeable {scores.flags.writeable}"
)

# %% [markdown]
# ## What the pair shows
#
# Both forms reach the same result, so the choice is not about capability.
#
# The declared form keeps every requirement in one place, as names a reader can follow, instead
# of splitting them between a declaration and two wrapper functions. It gives this run's own
# check a typed interface, so a declaration that cannot feed it is refused at preparation rather
# than at the first array.
#
# The two run in different orders. `CopyReadOnly` runs after the shape and the scalar format
# already match. The `BeforeValidator` above runs first, so it copies and seals an array that
# the next check was going to refuse anyway.
#
# Tenspec also compares the class, the shape, the native dtype and the device across its own
# transform, and refuses a transform that changed one. A wrapper validator carries no such
# check, so its author owns that guarantee.
#
# `CopyReadOnly` copies on every check, which costs time and memory. The copy refuses an ordinary
# write. That is not immutability against deliberate access through its base array.
