"""Stable, execution-free descriptions of dataset findings.

`Finding` deliberately contains paths and metadata only.  Workspace creation,
patch application, project inspection, and validator execution stay in their
own modules so an RCS or OPSD component cannot acquire execution behaviour by
depending on this data model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ValidatorType(str, Enum):
    COMPILE = "compile"
    FUNCTIONALITY = "functionality"
    SECURITY = "security"


class ValidatorProtocol(str, Enum):
    """Observed dataset validator layouts, independent of project framework."""

    SANDBOX_TEST_FILE = "sandbox_test_file"
    SELF_CONTAINED_WRAPPER = "self_contained_wrapper"


class ProjectFramework(str, Enum):
    FOUNDRY = "foundry"
    HARDHAT = "hardhat"
    FOUNDRY_HARDHAT = "foundry-hardhat"
    TRUFFLE = "truffle"
    BROWNIE = "brownie"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProjectInfo:
    """Evidence-based project classification; no build command is inferred here."""

    framework: ProjectFramework
    source_root: Path
    configuration_files: tuple[Path, ...] = ()
    compiler_configuration_files: tuple[Path, ...] = ()
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidatorSpec:
    """One concrete dataset validator entry point.

    `command_path` is an executable shell artifact supplied by the dataset.
    `test_path` is set only for the legacy project-sandbox protocol.  The
    adapter owns the distinction; callers always pass a `ValidatorType`.
    """

    validator_type: ValidatorType
    protocol: ValidatorProtocol
    adapter_name: str
    entry_kind: str
    command_path: Path
    test_path: Path | None = None
    expected_vulnerable: str = "PASS"
    expected_repaired: str = "PASS"


@dataclass(frozen=True)
class Finding:
    """A stable boundary for one repair problem, without execution methods."""

    id: str
    dataset_root: Path
    finding_root: Path
    task_dir: Path
    vulnerable_repository: Path
    source_link_target: str
    audit_finding: Path | None
    reference_patch: Path
    compile_validator: ValidatorSpec
    functionality_validator: ValidatorSpec
    security_validator: ValidatorSpec
    project: ProjectInfo
    compiler_configuration_files: tuple[Path, ...] = ()
    metadata_path: Path | None = None
    trusted_validation_path: Path | None = None
    alternate_validator_protocols: tuple[ValidatorProtocol, ...] = ()
    discovery_issues: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def validator(self, validator_type: ValidatorType) -> ValidatorSpec:
        return {
            ValidatorType.COMPILE: self.compile_validator,
            ValidatorType.FUNCTIONALITY: self.functionality_validator,
            ValidatorType.SECURITY: self.security_validator,
        }[validator_type]

    def to_manifest_record(self) -> dict[str, Any]:
        """Serialize paths relative to the dataset where possible.

        Source snapshots are sometimes intentionally symlinked outside the
        dataset root.  Their resolved location is therefore kept explicitly,
        while all owned artifacts remain portable dataset-relative paths.
        """

        def path_value(path: Path | None) -> str | None:
            if path is None:
                return None
            try:
                return str(path.relative_to(self.dataset_root))
            except ValueError:
                return str(path)

        def validator_value(spec: ValidatorSpec) -> dict[str, Any]:
            return {
                "type": spec.validator_type.value,
                "protocol": spec.protocol.value,
                "adapter": spec.adapter_name,
                "entry_kind": spec.entry_kind,
                "command_path": path_value(spec.command_path),
                "test_path": path_value(spec.test_path),
                "expected_vulnerable": spec.expected_vulnerable,
                "expected_repaired": spec.expected_repaired,
            }

        return {
            "schema_version": "rcs-opsd-finding-manifest/v1",
            "finding_id": self.id,
            "finding_root": path_value(self.finding_root),
            "task_dir": path_value(self.task_dir),
            "audit_finding": path_value(self.audit_finding),
            "repository": {
                "source_link": path_value(self.task_dir / "source"),
                "source_link_target": self.source_link_target,
                "resolved_path": str(self.vulnerable_repository),
            },
            "reference_patch": path_value(self.reference_patch),
            "validators": {
                "compile": validator_value(self.compile_validator),
                "functionality": validator_value(self.functionality_validator),
                "security": validator_value(self.security_validator),
            },
            "project": {
                "framework": self.project.framework.value,
                "source_root": str(self.project.source_root),
                "configuration_files": [path_value(path) for path in self.project.configuration_files],
                "compiler_configuration_files": [
                    path_value(path) for path in self.project.compiler_configuration_files
                ],
                "evidence": list(self.project.evidence),
            },
            "compiler_configuration_files": [
                path_value(path) for path in self.compiler_configuration_files
            ],
            "metadata_path": path_value(self.metadata_path),
            "trusted_validation_path": path_value(self.trusted_validation_path),
            "alternate_validator_protocols": [item.value for item in self.alternate_validator_protocols],
            "discovery_issues": list(self.discovery_issues),
            "metadata": self.metadata,
        }
