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
# # Validate NumPy arrays
#
# Two operations, each declared with Tenspec and then written again by hand. The declaration
# comes first, and the handwritten block follows it for comparison. The numerical body is
# identical in every pair.
#
# The data here is synthetic.

# %%
from typing import Any
from typing import Literal as Shape

import numpy as np
from numpy.typing import NDArray

from pydantic import ValidationError
from tenspec import checked
from tenspec.numpy import Bool, Float

# %% [markdown]
# ## One weight per feature
#
# The operation scales every column of a feature matrix by its own weight. `features` appears in
# both declarations, so the two arrays must agree on it. The body checks nothing.


# %%
@checked
def weigh_columns(
    values: Float[Shape["rows features"]], weights: Float[Shape["features"]]
) -> NDArray[np.floating]:
    """Scale every column of a feature matrix by its own weight.

    Returns:
        The weighed matrix.
    """
    return values * weights


features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
weights = np.array([10.0, 0.5], dtype=np.float32)

print(weigh_columns(features, weights).tolist())

# %% [markdown]
# A weight vector of the wrong length is refused. The message names the axis, both extents, and
# the bindings the boundary held.

# %%
short = np.array([10.0, 0.5, 1.0], dtype=np.float32)

try:
    weigh_columns(features, short)
except ValidationError as refusal:
    print(refusal.errors()[0]["msg"])

# %% [markdown]
# ### The same operation, written by hand
#
# Three statements state the requirements that the declaration above states in the signature.


# %%
def weigh_columns_by_hand(
    values: NDArray[np.floating], weights: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Scale every column of a feature matrix by its own weight.

    Returns:
        The weighed matrix.

    Raises:
        ValueError: when the ranks or the feature count disagree.
    """
    if values.ndim != 2:
        raise ValueError(f"values must have two axes, but it has {values.ndim}")
    if weights.ndim != 1:
        raise ValueError(f"weights must have one axis, but it has {weights.ndim}")
    if weights.shape[0] != values.shape[1]:
        raise ValueError(
            f"weights must cover {values.shape[1]} features, but it covers {weights.shape[0]}"
        )
    return values * weights


print(weigh_columns_by_hand(features, weights).tolist())

try:
    weigh_columns_by_hand(features, short)
except ValueError as refusal:
    print(refusal)

# %% [markdown]
# ## A mask over a batch of series
#
# This operation averages over the last axis, counting only the observations the mask keeps.
# `*batch` names any number of leading axes, and `time` names the last one. Writing the same
# group in both declarations is what requires the mask to match the values.


# %%
@checked
def masked_mean(
    values: Float[Shape["*batch time"]], mask: Bool[Shape["*batch time"]]
) -> np.floating[Any] | NDArray[np.floating]:
    """Average over the last axis, counting only the observations the mask keeps.

    Returns:
        One mean per leading position, or one NumPy scalar for a single series.
    """
    return np.where(mask, values, 0.0).sum(axis=-1) / mask.sum(axis=-1)


values = np.array([[1.0, np.nan, 3.0], [2.0, 4.0, 6.0]])
mask = np.array([[True, False, True], [True, True, False]])

print(masked_mean(values, mask).tolist())

# %% [markdown]
# The same declaration covers a single series, where the batch group is empty. A mask of the
# wrong shape is refused.

# %%
series = np.array([1.0, np.nan, 3.0])
kept = np.array([True, False, True])

print(masked_mean(series, kept))

try:
    masked_mean(values, np.array([True, False, True]))
except ValidationError as refusal:
    print(refusal.errors()[0]["msg"])

# %% [markdown]
# ### The same operation, written by hand
#
# The declaration accepts any number of leading axes, so the handwritten form checks the dtypes,
# the rank and the whole mask shape.


# %%
def masked_mean_by_hand(
    values: NDArray[np.floating], mask: NDArray[np.bool_]
) -> np.floating[Any] | NDArray[np.floating]:
    """Average over the last axis, counting only the observations the mask keeps.

    Returns:
        One mean per leading position, or one NumPy scalar for a single series.

    Raises:
        ValueError: when the dtypes, the rank, or the mask shape disagree.
    """
    if values.dtype.kind != "f":
        raise ValueError(f"values must be floating, but it is {values.dtype}")
    if mask.dtype != np.bool_:
        raise ValueError(f"mask must be boolean, but it is {mask.dtype}")
    if values.ndim < 1:
        raise ValueError("values must have a time axis")
    if mask.shape != values.shape:
        raise ValueError(f"mask must be shaped {values.shape}, but it is {mask.shape}")
    return np.where(mask, values, 0.0).sum(axis=-1) / mask.sum(axis=-1)


print(masked_mean_by_hand(values, mask).tolist())
print(masked_mean_by_hand(series, kept))

try:
    masked_mean_by_hand(values, np.array([True, False, True]))
except ValueError as refusal:
    print(refusal)

# %% [markdown]
# ## What the pairs show
#
# The declared form states each requirement once, in the signature, where a reader and a type
# checker both see it. The handwritten form states the same requirements as statements inside
# the body, where a caller reading the signature does not see them.
#
# The refusals are not weaker for being generated. Each one names the axis, both extents and the
# bindings the boundary held, which is more than the handwritten messages carry.
#
# The second pair is the clearer win. `*batch` states "any number of leading axes, the same in
# both arrays" in one token. The handwritten equivalent needs a rank test and a full shape
# comparison to say it.
#
# Neither form checks that the mask selects the observations the caller meant. Both check
# declared structure alone, so a mask of the right shape and dtype passes either one.
