"""Forecasting: a NaN-aware scaler whose method shape the declarations must fit.

A demonstration against the accepted method shape of Cairn's ContextScale, in
`packages/cairn-nn/src/cairn/nn/normalization.py` at Cairn b869c260. No consumer file is
read at run time and no consumer file changes. The body holds only the statistics needed to
show that the declarations fit the real boundary.

Run it with `uv run python packages/tenspec/examples/forecast_scale.py`.
"""

from typing import Any, Self
from typing import Literal as Shape

import numpy as np
from numpy.typing import NDArray

from pydantic import ValidationError
from tenspec import checked
from tenspec.numpy import Bool, Float, Float64


class ContextScale:
    """A mean and a floored scale along the last axis, fitted once per context."""

    def __init__(self, location: NDArray[np.float64], scale: NDArray[np.float64]) -> None:
        self.location = location
        self.scale = scale

    @classmethod
    @checked
    def fit(
        cls,
        values: Float[Shape["*batch time"]],
        mask: Bool[Shape["*batch time"]] | None = None,
        *,
        flatness_floor: float = 1e-3,
    ) -> Self:
        """Fit over the last axis. A NaN and a False mask step are both missing.

        Returns:
            A scaler of the class that received the call.
        """
        observed = values if mask is None else np.where(mask, values, np.nan)
        location = np.nanmean(observed, axis=-1)
        spread = np.nanstd(observed, axis=-1)
        floor = flatness_floor * (np.abs(location) + 1.0)
        return cls(np.asarray(location, np.float64), np.asarray(np.maximum(spread, floor)))

    @checked
    def normalize(self, values: Float[Shape["*batch time"]]) -> Float64[Shape["*batch time"]]:
        """Centre and scale, and let a missing observation pass through.

        Returns:
            The normalized block, in float64.
        """
        return (values - self.location[..., None]) / self.scale[..., None]


class RobustScale(ContextScale):
    """A subclass. Its fit must return a RobustScale, never a ContextScale."""


def main() -> None:
    """Fit and normalize a target history and a covariate history of different ranks."""
    target = np.array([1.0, np.nan, 3.0, 5.0], dtype=np.float32)
    covariates = np.array([[2.0, 4.0, 6.0, 8.0], [1.0, 1.0, np.nan, 1.0]], dtype=np.float32)
    observed = np.array([[True, True, True, False], [True, True, True, True]])

    target_scale = ContextScale.fit(target)
    covariate_scale = ContextScale.fit(covariates, observed, flatness_floor=0.05)
    print(f"Target location {target_scale.location.round(3).tolist()}")
    print(f"Covariate location {covariate_scale.location.round(3).tolist()}")

    normalized: Any = covariate_scale.normalize(covariates)
    print(f"Normalized {normalized.shape}, dtype {normalized.dtype}")
    print(f"A missing observation passes through: {bool(np.isnan(normalized[1, 2]))}")

    print(f"A subclass fit returns its own class: {type(RobustScale.fit(target)).__name__}")

    try:
        ContextScale.fit(covariates, np.array([True, False, True, True]))
    except ValidationError as refusal:
        print(f"The mask must cover the same axes: {refusal.errors()[0]['msg']}")

    try:
        ContextScale.fit(np.array([1, 2, 3]))
    except ValidationError as refusal:
        print(f"An integer history is refused: {refusal.errors()[0]['msg']}")


if __name__ == "__main__":
    main()
