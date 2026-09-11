"""Tenspec: declare what an array must be, and check it at one boundary.

Write the declaration once, beside the value it describes. A checked function, a model
mixin, or a standalone call then holds it. Import a backend module for the array library
you use: `tenspec.numpy` or `tenspec.torch`. This module loads neither one.

This is the common surface. Everything below it keeps its defining module, including
`tenspec.errors.TensorMismatch`, `tenspec.errors.BindingError`,
`tenspec.runtime.validation.validate_array`, `tenspec.runtime.bindings.Bindings`,
`tenspec.runtime.arrays.ArrayBackend`, and `tenspec.pydantic.annotations.prepare_annotation`.
"""

from typing import Literal as Shape

from tenspec.errors import AnnotationError, ConstraintEvaluationError
from tenspec.pydantic.calls import checked
from tenspec.pydantic.models import TensorContracts
from tenspec.pydantic.standalone import validate
from tenspec.runtime.bindings import axis_size
from tenspec.types.dtypes import DType, Floating
from tenspec.types.properties import Contiguous, Finite, Materialized, NonEmpty

__version__ = "0.1.0"

__all__ = [
    "AnnotationError",
    "ConstraintEvaluationError",
    "Contiguous",
    "DType",
    "Finite",
    "Floating",
    "Materialized",
    "NonEmpty",
    "Shape",
    "TensorContracts",
    "axis_size",
    "checked",
    "validate",
]
