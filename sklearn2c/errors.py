"""Exceptions raised when a model cannot be transpiled faithfully."""


class Sklearn2CError(Exception):
    """Base class for all sklearn2c errors."""


class UnsupportedModelError(Sklearn2CError, TypeError):
    """The estimator type (or one of its settings) has no C backend."""


class UnsupportedLabelError(Sklearn2CError, ValueError):
    """The classifier's ``classes_`` cannot be represented as C ``int32_t`` labels."""


class CompilationError(Sklearn2CError, RuntimeError):
    """The C compiler rejected the generated code."""
