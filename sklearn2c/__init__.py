"""sklearn2c: transpile fitted scikit-learn models to dependency-free C."""

from .codegen import SUPPORTED_MODELS, ModelSpec, inspect_model, transpile
from .errors import (
    CompilationError,
    Sklearn2CError,
    UnsupportedLabelError,
    UnsupportedModelError,
)
from .parity import ParityReport, probe_inputs, verify_predictions

__version__ = "0.2.0"

__all__ = [
    "SUPPORTED_MODELS",
    "ModelSpec",
    "inspect_model",
    "transpile",
    "verify_predictions",
    "probe_inputs",
    "ParityReport",
    "Sklearn2CError",
    "UnsupportedModelError",
    "UnsupportedLabelError",
    "CompilationError",
    "__version__",
]
