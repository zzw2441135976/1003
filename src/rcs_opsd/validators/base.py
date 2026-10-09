"""Unified validator result and adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rcs_opsd.dataset.finding import Finding, ValidatorSpec, ValidatorType
from rcs_opsd.runtime.workspace import Workspace


@dataclass(frozen=True)
class ValidatorResult:
    """Normalized result for compile, functionality, and security validators."""

    success: bool
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    command: tuple[str, ...]
    validator_type: ValidatorType
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def not_run(cls, validator_type: ValidatorType, reason: str) -> "ValidatorResult":
        return cls(
            success=False,
            exit_code=None,
            stdout="",
            stderr="",
            duration_seconds=0.0,
            command=(),
            validator_type=validator_type,
            status="NOT_RUN",
            metadata={"reason": reason},
        )


class ValidatorAdapter(ABC):
    """Owns how one validator entry point is invoked, never model logic."""

    name: str

    @abstractmethod
    def run(
        self,
        finding: Finding,
        spec: ValidatorSpec,
        workspace: Workspace,
        *,
        timeout_seconds: int,
        trace: bool,
    ) -> ValidatorResult:
        """Execute one validator in the supplied isolated workspace."""

