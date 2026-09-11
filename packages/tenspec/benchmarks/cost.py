"""What Tenspec costs on a CPU, measured per boundary rather than per claim.

Run it with `uv run python packages/tenspec/benchmarks/cost.py`. It reports the median of
several samples and the spread across them, so a difference under that spread reads as
noise rather than as a result.
"""

import platform
import timeit
from typing import Any
from typing import Literal as Shape

import numpy as np
from numpy.typing import NDArray

import pydantic
from pydantic import BaseModel, ConfigDict
from tenspec import Finite, TensorContracts, checked, validate
from tenspec.numpy import NUMPY_ARRAYS, Float
from tenspec.pydantic.annotations import prepare_annotation
from tenspec.pydantic.standalone import build_adapter
from tenspec.runtime.bindings import Bindings, validation_scope
from tenspec.runtime.validation import validate_array
from tenspec.types.dtypes import DTypeFamily, DTypeRequirement
from tenspec.types.properties import Finite as FiniteProperty
from tenspec.types.shapes import parse_shape
from tenspec.types.tensor import TensorType

REPEATS = 2000
"""How many times one measured operation runs inside a single sample."""

SAMPLES = 7
"""How many samples the report takes the median and the spread from."""

METADATA_ONLY = TensorType(
    shape=parse_shape("rows cols"), dtype=DTypeRequirement(permitted=DTypeFamily.FLOATING)
)
WITH_VALUE_SCAN = TensorType(
    shape=parse_shape("rows cols"),
    dtype=DTypeRequirement(permitted=DTypeFamily.FLOATING),
    properties=(FiniteProperty,),
)


@checked
def checked_call(values: Float[Shape["rows cols"]]) -> Float[Shape["rows cols"]]:
    """A boundary that does nothing but check.

    Returns:
        The array it received.
    """
    return values


class Base(BaseModel):
    """The caller's own base."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Measured(TensorContracts, Base):
    """One tensor field, checked at model construction."""

    values: Float[Shape["rows cols"]]


def uncached_validate(values: Any, annotation: Any) -> Any:
    """One standalone check that prepares its own adapter, as the route did before the cache.

    It is the measured baseline of the cache, so the report states what the cache buys
    on this machine rather than on the machine of an earlier run.

    Returns:
        The accepted value.
    """
    adapter = build_adapter(annotation)
    with validation_scope():
        return adapter.validate_python(values)


def microseconds(operation: Any) -> tuple[float, float]:
    """The median and the spread of one operation, in microseconds per call.

    Returns:
        The median sample and the difference between the slowest and fastest sample.
    """
    samples = timeit.repeat(operation, repeat=SAMPLES, number=REPEATS)
    ordered = sorted(sample / REPEATS * 1e6 for sample in samples)
    return ordered[len(ordered) // 2], ordered[-1] - ordered[0]


def report(label: str, operation: Any) -> None:
    """Print one measured line."""
    median, spread = microseconds(operation)
    print(f"  {label:<44} {median:9.2f} us   spread {spread:8.2f} us")


def main() -> None:
    """Measure preparation, metadata checks, and a value scan on two array sizes."""
    print(
        f"Python {platform.python_version()}, NumPy {np.__version__}, "
        f"Pydantic {pydantic.__version__}"
    )
    print(f"{REPEATS} calls per sample, {SAMPLES} samples, median and spread below\n")

    print("Once per declaration:")
    report("prepare_annotation, two named axes", lambda: prepare_annotation(Float[Shape["a b"]]))
    report("parse_shape, two named axes", lambda: parse_shape("rows cols"))

    for label, shape in (("small (10, 10)", (10, 10)), ("larger (1000, 1000)", (1000, 1000))):
        values: NDArray[np.float64] = np.ones(shape)
        print(f"\nPer check, {label}:")
        report(
            "validate_array, metadata only",
            lambda values=values: validate_array(
                values, tensor_type=METADATA_ONLY, backend=NUMPY_ARRAYS, bindings=Bindings()
            ),
        )
        report(
            "validate_array, with Finite",
            lambda values=values: validate_array(
                values, tensor_type=WITH_VALUE_SCAN, backend=NUMPY_ARRAYS, bindings=Bindings()
            ),
        )
        report("checked call, warm", lambda values=values: checked_call(values))
        report("model validation", lambda values=values: Measured(values=values))
        report(
            "standalone validate, schema built per call",
            lambda values=values: uncached_validate(values, Float[Shape["rows cols"]]),
        )
        report(
            "standalone validate, schema from the cache",
            lambda values=values: validate(values, Float[Shape["rows cols"]]),
        )
        report("the operation alone, values * 2", lambda values=values: values * 2)
        report("np.isfinite(values).all()", lambda values=values: bool(np.isfinite(values).all()))
        report(
            "Finite through the alias",
            lambda values=values: validate(values, Float[Shape["rows cols"], Finite]),
        )


if __name__ == "__main__":
    main()
