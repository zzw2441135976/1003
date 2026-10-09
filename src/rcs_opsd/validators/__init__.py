"""Validator adapter interfaces and the framework-agnostic runner."""

from .base import ValidatorAdapter, ValidatorResult
from .reporting import aggregate_validation_results, classify_world_result
from .runner import ValidatorRunner

__all__ = [
    "ValidatorAdapter",
    "ValidatorResult",
    "ValidatorRunner",
    "aggregate_validation_results",
    "classify_world_result",
]
