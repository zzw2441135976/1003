"""Inspect the observed dataset layout and build structured Finding records."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from rcs_opsd.dataset.finding import Finding, ProjectInfo, ValidatorProtocol, ValidatorSpec, ValidatorType
from rcs_opsd.projects.base import ProjectAdapterRegistry


class DatasetInspectionError(RuntimeError):
    """Raised only when the requested dataset root itself is unusable."""


@dataclass(frozen=True)
class DatasetManifest:
    dataset_root: Path
    findings: tuple[Finding, ...]

    def validator_protocol_counts(self) -> Counter[str]:
        return Counter(finding.compile_validator.protocol.value for finding in self.findings)

    def framework_counts(self) -> Counter[str]:
        return Counter(finding.project.framework.value for finding in self.findings)

    def write_jsonl(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for finding in self.findings:
                handle.write(json.dumps(finding.to_manifest_record(), sort_keys=True) + "\n")


class DatasetLoader:
    """Loader for the two validator protocols actually present in dataset_root."""

    def __init__(self, dataset_root: Path | str, registry: ProjectAdapterRegistry | None = None) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.findings_root = self.dataset_root / "findings"
        self.project_assets_root = self.dataset_root / "project_assets"
        self.registry = registry or ProjectAdapterRegistry()
        self._project_info_by_source: dict[Path, ProjectInfo] = {}
        if not self.findings_root.is_dir():
            raise DatasetInspectionError(f"Missing dataset findings directory: {self.findings_root}")

    def finding_ids(self) -> tuple[str, ...]:
        return tuple(path.name for path in self._finding_directories())

    def inspect(self) -> DatasetManifest:
        return DatasetManifest(
            dataset_root=self.dataset_root,
            findings=tuple(self.load(path.name) for path in self._finding_directories()),
        )

    def load(self, finding_id: str) -> Finding:
        finding_root = self.findings_root / finding_id
        task_dir = finding_root / "task"
        if not task_dir.is_dir():
            raise DatasetInspectionError(f"Missing task directory for {finding_id}: {task_dir}")
        source = task_dir / "source"
        if not source.is_symlink() or not source.exists():
            raise DatasetInspectionError(f"Missing or unresolved task/source for {finding_id}: {source}")
        source_link_target = str(source.readlink())
        vulnerable_repository = source.resolve()
        if not vulnerable_repository.is_dir():
            raise DatasetInspectionError(f"Resolved task/source is not a directory for {finding_id}")
        reference_patch = task_dir / "reference_patch.diff"
        if not reference_patch.is_file():
            raise DatasetInspectionError(f"Missing reference patch for {finding_id}: {reference_patch}")

        protocol, alternate_protocols = self._validator_protocol(task_dir)
        validators = self._validator_specs(task_dir, protocol, source_link_target)
        metadata_path = self._optional_file(task_dir / "metadata.json")
        trusted_validation_path = self._optional_file(task_dir / "trusted_validation.json")
        metadata = self._read_json(metadata_path) if metadata_path else {}
        audit_finding = self._first_existing(task_dir / name for name in ("full_prompt.md", "finding.md"))
        issues = self._issues(task_dir, audit_finding)
        project = self._project_info(vulnerable_repository)
        return Finding(
            id=finding_id,
            dataset_root=self.dataset_root,
            finding_root=finding_root,
            task_dir=task_dir,
            vulnerable_repository=vulnerable_repository,
            source_link_target=source_link_target,
            audit_finding=audit_finding,
            reference_patch=reference_patch,
            compile_validator=validators[ValidatorType.COMPILE],
            functionality_validator=validators[ValidatorType.FUNCTIONALITY],
            security_validator=validators[ValidatorType.SECURITY],
            project=project,
            compiler_configuration_files=self._compiler_configurations(task_dir, protocol, project),
            metadata_path=metadata_path,
            trusted_validation_path=trusted_validation_path,
            alternate_validator_protocols=alternate_protocols,
            discovery_issues=issues,
            metadata=metadata,
        )

    def _finding_directories(self) -> Iterable[Path]:
        return tuple(sorted(path for path in self.findings_root.iterdir() if path.is_dir()))

    @staticmethod
    def _optional_file(path: Path) -> Path | None:
        return path if path.is_file() else None

    @staticmethod
    def _first_existing(paths: Iterable[Path]) -> Path | None:
        return next((path for path in paths if path.is_file()), None)

    @staticmethod
    def _read_json(path: Path) -> dict:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {"_inspection_error": f"Cannot parse {path.name}"}
        return value if isinstance(value, dict) else {"_inspection_error": f"{path.name} is not an object"}

    @staticmethod
    def _single_test_file(directory: Path) -> Path | None:
        matches = sorted(path for path in directory.glob("test.*") if path.is_file())
        return matches[0] if len(matches) == 1 else None

    def _validator_protocol(self, task_dir: Path) -> tuple[ValidatorProtocol, tuple[ValidatorProtocol, ...]]:
        wrapper = all(
            path.is_file()
            for path in (
                task_dir / "capsule" / "compile.sh",
                task_dir / "functionality" / "run.sh",
                task_dir / "security" / "run.sh",
            )
        )
        test_file = all(
            self._single_test_file(task_dir / validator_type.value) is not None
            for validator_type in (ValidatorType.FUNCTIONALITY, ValidatorType.SECURITY)
        )
        if wrapper:
            alternate = (ValidatorProtocol.SANDBOX_TEST_FILE,) if test_file else ()
            return ValidatorProtocol.SELF_CONTAINED_WRAPPER, alternate
        if test_file:
            return ValidatorProtocol.SANDBOX_TEST_FILE, ()
        raise DatasetInspectionError(f"No supported validator protocol under {task_dir}")

    def _validator_specs(
        self, task_dir: Path, protocol: ValidatorProtocol, source_link_target: str
    ) -> dict[ValidatorType, ValidatorSpec]:
        if protocol is ValidatorProtocol.SELF_CONTAINED_WRAPPER:
            return {
                ValidatorType.COMPILE: ValidatorSpec(
                    ValidatorType.COMPILE,
                    protocol,
                    "self_contained_wrapper",
                    "forge_bash_wrapper",
                    task_dir / "capsule" / "compile.sh",
                ),
                ValidatorType.FUNCTIONALITY: ValidatorSpec(
                    ValidatorType.FUNCTIONALITY,
                    protocol,
                    "self_contained_wrapper",
                    "forge_bash_wrapper",
                    task_dir / "functionality" / "run.sh",
                ),
                ValidatorType.SECURITY: ValidatorSpec(
                    ValidatorType.SECURITY,
                    protocol,
                    "self_contained_wrapper",
                    "forge_bash_wrapper",
                    task_dir / "security" / "run.sh",
                    expected_vulnerable="VULNERABILITY_DETECTED",
                ),
            }

        project_key = Path(source_link_target).name
        sandbox = self.project_assets_root / project_key / "sandbox"
        expected = {
            ValidatorType.COMPILE: "compile.sh",
            ValidatorType.FUNCTIONALITY: "run_functionality.sh",
            ValidatorType.SECURITY: "run_security.sh",
        }
        missing = [name for name in expected.values() if not (sandbox / name).is_file()]
        if missing:
            raise DatasetInspectionError(
                f"Missing legacy sandbox scripts for {task_dir.parent.name} ({project_key}): {', '.join(missing)}"
            )
        adapter_name = self._legacy_adapter_name(sandbox, tuple(expected.values()))
        return {
            validator_type: ValidatorSpec(
                validator_type,
                protocol,
                adapter_name,
                self._test_entry_kind(self._single_test_file(task_dir / validator_type.value))
                if validator_type is not ValidatorType.COMPILE
                else "project_compile_script",
                sandbox / script_name,
                self._single_test_file(task_dir / validator_type.value)
                if validator_type is not ValidatorType.COMPILE
                else None,
                expected_vulnerable="VULNERABILITY_DETECTED"
                if validator_type is ValidatorType.SECURITY
                else "PASS",
            )
            for validator_type, script_name in expected.items()
        }

    @staticmethod
    def _legacy_adapter_name(sandbox: Path, script_names: tuple[str, ...]) -> str:
        """Select a validator backend from the actual sandbox entry points.

        This classifies a validator invocation, not the project as a whole. A
        legacy test file can be a Solidity test while still use a custom shell
        backend; only scripts that explicitly invoke the Docker Foundry image
        with ``HOME=/tmp`` receive the cache-aware adapter.
        """

        try:
            scripts = [
                (sandbox / script_name).read_text(encoding="utf-8", errors="replace")
                for script_name in script_names
            ]
        except OSError:
            return "sandbox_test_file"
        if all(
            "docker run" in script and "HOME=/tmp" in script and "forge" in script
            for script in scripts
        ):
            return "docker_foundry_sandbox"
        if all(
            "docker run" in script and "hardhat" in script and ".hardhat-home" in script
            for script in scripts
        ):
            return "docker_hardhat_sandbox"
        return "sandbox_test_file"

    @staticmethod
    def _test_entry_kind(test_path: Path | None) -> str:
        if test_path is None:
            return "unknown_test"
        suffix = test_path.suffix.lower()
        if suffix == ".sol":
            return "solidity_foundry_test"
        if suffix == ".js":
            return "javascript_test"
        if suffix == ".ts":
            return "typescript_test"
        if suffix == ".sh":
            return "shell_test"
        if suffix == ".py":
            return "python_test"
        return f"other:{test_path.name}"

    @staticmethod
    def _compiler_configurations(
        task_dir: Path, protocol: ValidatorProtocol, project: ProjectInfo
    ) -> tuple[Path, ...]:
        """Identify the configuration that controls the actual compile entry point."""

        if protocol is ValidatorProtocol.SELF_CONTAINED_WRAPPER:
            capsule = task_dir / "capsule"
            names = (
                "foundry.toml",
                "hardhat.config.js",
                "hardhat.config.ts",
                "hardhat.config.cjs",
                "hardhat.config.mjs",
            )
            return tuple(path for name in names if (path := capsule / name).is_file())
        return project.compiler_configuration_files

    @staticmethod
    def _issues(task_dir: Path, audit_finding: Path | None) -> tuple[str, ...]:
        issues: list[str] = []
        if audit_finding is None:
            issues.append("Missing full_prompt.md or finding.md audit artifact.")
        return tuple(issues)

    def _project_info(self, source_root: Path):
        """Memoize inspection because many findings share one repository snapshot."""

        cached = self._project_info_by_source.get(source_root)
        if cached is None:
            cached = self.registry.detect(source_root)
            self._project_info_by_source[source_root] = cached
        return cached
