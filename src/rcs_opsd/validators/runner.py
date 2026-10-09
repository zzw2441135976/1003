"""Validator dispatch with all toolchain details confined to adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import signal
import shutil
import subprocess
import tempfile
import time
from typing import Iterator, Mapping

from rcs_opsd.dataset.finding import Finding, ValidatorProtocol, ValidatorSpec, ValidatorType
from rcs_opsd.runtime.workspace import Workspace, WorkspaceManager
from rcs_opsd.validators.base import ValidatorAdapter, ValidatorResult


def _finish_process(process: subprocess.Popen[str]) -> tuple[str, str]:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    try:
        return process.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        return process.communicate()


def _command_result(
    command: list[str],
    cwd: Path,
    validator_type: ValidatorType,
    timeout_seconds: int,
    metadata: dict,
    environment: Mapping[str, str] | None = None,
) -> ValidatorResult:
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env=environment,
        )
    except OSError as exc:
        return ValidatorResult(
            False, None, "", str(exc), time.monotonic() - started, tuple(command), validator_type, "ERROR", metadata
        )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        exit_code = process.returncode
        if validator_type is ValidatorType.SECURITY:
            status = "PASS" if exit_code == 0 else "VULNERABILITY_DETECTED" if exit_code == 1 else "ERROR"
        else:
            status = "PASS" if exit_code == 0 else "FAIL"
        return ValidatorResult(
            status == "PASS",
            exit_code,
            stdout or "",
            stderr or "",
            time.monotonic() - started,
            tuple(command),
            validator_type,
            status,
            metadata,
        )
    except subprocess.TimeoutExpired:
        stdout, stderr = _finish_process(process)
        return ValidatorResult(
            False,
            None,
            stdout or "",
            (stderr or "") + f"\nCommand timed out after {timeout_seconds} seconds.\n",
            time.monotonic() - started,
            tuple(command),
            validator_type,
            "TIMEOUT",
            metadata,
        )


class SandboxTestFileAdapter(ValidatorAdapter):
    """Legacy project sandbox scripts, which run from the copied source root.

    A few Hardhat projects use ``proper-lockfile`` for OpenZeppelin's generated
    validation cache.  A completed compiler process can very occasionally
    leave its workspace-local ``cache/validations.json.lock`` directory behind.
    This adapter recognises only that exact failure signature, removes only
    that generated lock from the disposable workspace, then makes one retry.
    The retry is deliberately adapter-local and fully recorded in metadata;
    it never changes a dataset snapshot or a validator command.
    """

    name = "sandbox_test_file"
    _validation_lock_relative_path = Path("cache/validations.json.lock")

    def run(
        self,
        finding: Finding,
        spec: ValidatorSpec,
        workspace: Workspace,
        *,
        timeout_seconds: int,
        trace: bool,
    ) -> ValidatorResult:
        command = ["bash", str(spec.command_path)]
        if spec.test_path is not None:
            command.append(str(spec.test_path))
        metadata = {
            "adapter": self.name,
            "protocol": spec.protocol.value,
            "trace_requested": trace,
            "trace_available": False,
        }
        result = _command_result(
            command,
            workspace.root,
            spec.validator_type,
            timeout_seconds,
            metadata,
        )
        if not self._is_retryable_validation_lock(result):
            return result

        lock_removed = self._clear_workspace_validation_lock(workspace)
        retried = _command_result(
            command,
            workspace.root,
            spec.validator_type,
            timeout_seconds,
            metadata,
        )
        return replace(
            retried,
            metadata={
                **retried.metadata,
                "transient_validation_lock_retry": {
                    "attempts": 2,
                    "initial_exit_code": result.exit_code,
                    "initial_status": result.status,
                    "lock_path": str(self._validation_lock_relative_path),
                    "lock_removed": lock_removed,
                    "signature": "proper-lockfile:ELOCKED:cache/validations.json",
                },
            },
        )

    @staticmethod
    def _is_retryable_validation_lock(result: ValidatorResult) -> bool:
        output = f"{result.stdout}\n{result.stderr}"
        return (
            result.status == "FAIL"
            and "proper-lockfile" in output
            and "ELOCKED" in output
            and "cache/validations.json" in output
        )

    def _clear_workspace_validation_lock(self, workspace: Workspace) -> bool:
        """Remove one known generated lock, never a dataset path or source file."""

        lock_path = workspace.root / self._validation_lock_relative_path
        if not lock_path.exists():
            return False
        if lock_path.is_dir():
            shutil.rmtree(lock_path)
        else:
            lock_path.unlink()
        return True


class DockerFoundrySandboxAdapter(ValidatorAdapter):
    """Run Docker-Foundry legacy scripts with an optional read-only SVM cache.

    The dataset owns its validator scripts, which explicitly run Docker with
    ``HOME=/tmp`` and execute Forge, regardless of whether they use a public
    Foundry image or a dataset-specific toolchain image. This adapter leaves
    those scripts untouched. When an operator provides an SVM cache
    (``RCS_OPSD_SOLC_CACHE``) or has the normal ``~/.svm`` cache, a short-lived
    Docker proxy mounts that cache at ``/tmp/.svm`` and sets Foundry offline.
    Without a usable cache the exact original Docker command still runs,
    allowing normal registry-based use.
    """

    name = "docker_foundry_sandbox"
    cache_environment_variable = "RCS_OPSD_SOLC_CACHE"
    cache_mount_target = "/tmp/.svm"

    def __init__(self, cache_dir: Path | str | None = None, docker_executable: str | None = None) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        self.docker_executable = docker_executable

    def run(
        self,
        finding: Finding,
        spec: ValidatorSpec,
        workspace: Workspace,
        *,
        timeout_seconds: int,
        trace: bool,
    ) -> ValidatorResult:
        command = ["bash", str(spec.command_path)]
        if spec.test_path is not None:
            command.append(str(spec.test_path))
        cache_dir = self._cache_directory()
        metadata = {
            "adapter": self.name,
            "protocol": spec.protocol.value,
            "trace_requested": trace,
            "trace_available": False,
            "solc_cache_mode": "registry_default",
        }
        if cache_dir is None:
            return _command_result(command, workspace.root, spec.validator_type, timeout_seconds, metadata)
        docker = self.docker_executable or shutil.which("docker")
        if docker is None:
            metadata["solc_cache_mode"] = "cache_available_docker_unavailable"
            return _command_result(command, workspace.root, spec.validator_type, timeout_seconds, metadata)
        with tempfile.TemporaryDirectory(prefix="rcs-opsd-docker-proxy-") as temporary:
            proxy_dir = Path(temporary)
            self._write_docker_proxy(proxy_dir / "docker")
            environment = os.environ.copy()
            environment["PATH"] = f"{proxy_dir}{os.pathsep}{environment.get('PATH', '')}"
            environment["RCS_OPSD_REAL_DOCKER"] = docker
            environment["RCS_OPSD_SOLC_CACHE"] = str(cache_dir)
            metadata.update(
                {
                    "solc_cache_mode": "mounted_read_only_offline",
                    "solc_cache_mount_target": self.cache_mount_target,
                }
            )
            return _command_result(
                command,
                workspace.root,
                spec.validator_type,
                timeout_seconds,
                metadata,
                environment,
            )

    def _cache_directory(self) -> Path | None:
        candidates = [self.cache_dir]
        configured = os.environ.get(self.cache_environment_variable)
        if configured:
            candidates.append(Path(configured).expanduser())
        candidates.append(Path.home() / ".svm")
        for candidate in candidates:
            if candidate is None:
                continue
            resolved = candidate.resolve()
            if resolved.is_dir():
                return resolved
        return None

    def _write_docker_proxy(self, path: Path) -> None:
        path.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "if [[ \"${1:-}\" != \"run\" ]]; then\n"
            "  exec \"$RCS_OPSD_REAL_DOCKER\" \"$@\"\n"
            "fi\n"
            "shift\n"
            "exec \"$RCS_OPSD_REAL_DOCKER\" run \\\n"
            "  --mount \"type=bind,src=$RCS_OPSD_SOLC_CACHE,dst=/tmp/.svm,readonly\" \\\n"
            "  --env FOUNDRY_OFFLINE=true \\\n"
            "  \"$@\"\n",
            encoding="utf-8",
        )
        path.chmod(0o700)


class DockerHardhatSandboxAdapter(ValidatorAdapter):
    """Expose a write-through Hardhat compiler cache to a Docker sandbox.

    Some legacy sandbox scripts re-exec themselves in a dataset-specific Docker
    image and deliberately set ``HOME=$PWD/.hardhat-home`` before invoking
    Hardhat. The image already contains the project's Node toolchain, but its
    compiler downloader otherwise cannot see the operator's cache. This
    adapter preserves the dataset command and injects only a cache mount at
    that explicitly configured Hardhat cache path. Hardhat 2.x performs a
    ``chmod`` on a cached compiler before use, so this mount must be writable;
    the mode is retained in result metadata and the directory is still owned by
    the operator rather than the dataset.
    """

    name = "docker_hardhat_sandbox"
    cache_environment_variable = "RCS_OPSD_HARDHAT_CACHE"
    cache_mount_target = "/workspace/.hardhat-home/.cache/hardhat-nodejs"

    def __init__(self, cache_dir: Path | str | None = None, docker_executable: str | None = None) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        self.docker_executable = docker_executable

    def run(
        self,
        finding: Finding,
        spec: ValidatorSpec,
        workspace: Workspace,
        *,
        timeout_seconds: int,
        trace: bool,
    ) -> ValidatorResult:
        command = ["bash", str(spec.command_path)]
        if spec.test_path is not None:
            command.append(str(spec.test_path))
        cache_dir = self._cache_directory()
        metadata = {
            "adapter": self.name,
            "protocol": spec.protocol.value,
            "trace_requested": trace,
            "trace_available": False,
            "hardhat_cache_mode": "registry_default",
        }
        if cache_dir is None:
            return _command_result(command, workspace.root, spec.validator_type, timeout_seconds, metadata)
        docker = self.docker_executable or shutil.which("docker")
        if docker is None:
            metadata["hardhat_cache_mode"] = "cache_available_docker_unavailable"
            return _command_result(command, workspace.root, spec.validator_type, timeout_seconds, metadata)
        self._prepare_cache_mount_target(workspace)
        with tempfile.TemporaryDirectory(prefix="rcs-opsd-docker-proxy-") as temporary:
            proxy_dir = Path(temporary)
            self._write_docker_proxy(proxy_dir / "docker")
            environment = os.environ.copy()
            environment["PATH"] = f"{proxy_dir}{os.pathsep}{environment.get('PATH', '')}"
            environment["RCS_OPSD_REAL_DOCKER"] = docker
            environment["RCS_OPSD_HARDHAT_CACHE"] = str(cache_dir)
            metadata.update(
                {
                    "hardhat_cache_mode": "mounted_write_through",
                    "hardhat_cache_mount_target": self.cache_mount_target,
                }
            )
            return _command_result(
                command,
                workspace.root,
                spec.validator_type,
                timeout_seconds,
                metadata,
                environment,
            )

    def _prepare_cache_mount_target(self, workspace: Workspace) -> Path:
        """Create the nested Docker mount point as the workspace owner.

        Docker otherwise creates the nested path as root under the bind-mounted
        workspace. That leaves a disposable workspace which the process owner
        cannot remove after validation. The directory is empty before Docker
        overlays the read-only cache and is contained in the copied workspace.
        """

        target = workspace.root / ".hardhat-home" / ".cache" / "hardhat-nodejs"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _cache_directory(self) -> Path | None:
        candidates = [self.cache_dir]
        configured = os.environ.get(self.cache_environment_variable)
        if configured:
            candidates.append(Path(configured).expanduser())
        candidates.append(Path.home() / ".cache" / "hardhat-nodejs")
        for candidate in candidates:
            if candidate is None:
                continue
            resolved = candidate.resolve()
            if (resolved / "compilers-v2").is_dir():
                return resolved
        return None

    def _write_docker_proxy(self, path: Path) -> None:
        path.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "if [[ \"${1:-}\" != \"run\" ]]; then\n"
            "  exec \"$RCS_OPSD_REAL_DOCKER\" \"$@\"\n"
            "fi\n"
            "shift\n"
            "exec \"$RCS_OPSD_REAL_DOCKER\" run \\\n"
            "  --mount \"type=bind,src=$RCS_OPSD_HARDHAT_CACHE,dst=/workspace/.hardhat-home/.cache/hardhat-nodejs\" \\\n"
            "  \"$@\"\n",
            encoding="utf-8",
        )
        path.chmod(0o700)


class SelfContainedWrapperAdapter(ValidatorAdapter):
    """Foundry capsules that need a staged task with source rebound to the world."""

    name = "self_contained_wrapper"

    def __init__(self, workspace_manager: WorkspaceManager) -> None:
        self.workspace_manager = workspace_manager

    def run(
        self,
        finding: Finding,
        spec: ValidatorSpec,
        workspace: Workspace,
        *,
        timeout_seconds: int,
        trace: bool,
    ) -> ValidatorResult:
        with self.workspace_manager.staged_task(finding, workspace) as task_copy:
            relative_command = spec.command_path.relative_to(finding.task_dir)
            command = ["bash", str(task_copy / relative_command)]
            return _command_result(
                command,
                task_copy,
                spec.validator_type,
                timeout_seconds,
                {
                    "adapter": self.name,
                    "protocol": spec.protocol.value,
                    "trace_requested": trace,
                    "trace_available": False,
                    "staged_task": True,
                },
            )


@dataclass(frozen=True)
class ValidationMatrix:
    compile: ValidatorResult
    functionality: ValidatorResult
    security: ValidatorResult

    def matches_world(self, finding: Finding, world: str) -> bool:
        if world not in {"vulnerable", "repaired"}:
            raise ValueError("world must be vulnerable or repaired")
        expected_attribute = "expected_vulnerable" if world == "vulnerable" else "expected_repaired"
        return all(
            result.status == getattr(finding.validator(result.validator_type), expected_attribute)
            for result in (self.compile, self.functionality, self.security)
        )


class ValidatorRunner:
    """Single validator dispatch surface; it has no Foundry/Hardhat conditionals."""

    def __init__(self, workspace_manager: WorkspaceManager | None = None) -> None:
        self.workspace_manager = workspace_manager or WorkspaceManager()
        self._adapters: dict[str, ValidatorAdapter] = {
            SandboxTestFileAdapter.name: SandboxTestFileAdapter(),
            DockerFoundrySandboxAdapter.name: DockerFoundrySandboxAdapter(),
            DockerHardhatSandboxAdapter.name: DockerHardhatSandboxAdapter(),
            SelfContainedWrapperAdapter.name: SelfContainedWrapperAdapter(self.workspace_manager),
        }

    def run_validator(
        self,
        finding: Finding,
        workspace: Workspace,
        validator_type: ValidatorType,
        *,
        trace: bool = False,
        timeout_seconds: int = 300,
    ) -> ValidatorResult:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        spec = finding.validator(validator_type)
        adapter = self._adapters.get(spec.adapter_name)
        if adapter is None:
            raise RuntimeError(f"No ValidatorAdapter registered for {spec.adapter_name}")
        return adapter.run(finding, spec, workspace, timeout_seconds=timeout_seconds, trace=trace)

    def run_world(
        self,
        finding: Finding,
        workspace: Workspace,
        *,
        trace: bool = False,
        timeout_seconds: int = 300,
        short_circuit: bool = True,
    ) -> ValidationMatrix:
        compile_result = self.run_validator(
            finding, workspace, ValidatorType.COMPILE, trace=trace, timeout_seconds=timeout_seconds
        )
        if short_circuit and compile_result.status != "PASS":
            return ValidationMatrix(
                compile_result,
                ValidatorResult.not_run(ValidatorType.FUNCTIONALITY, "compile did not pass"),
                ValidatorResult.not_run(ValidatorType.SECURITY, "compile did not pass"),
            )
        functionality_result = self.run_validator(
            finding, workspace, ValidatorType.FUNCTIONALITY, trace=trace, timeout_seconds=timeout_seconds
        )
        if short_circuit and functionality_result.status != "PASS":
            return ValidationMatrix(
                compile_result,
                functionality_result,
                ValidatorResult.not_run(ValidatorType.SECURITY, "functionality did not pass"),
            )
        security_result = self.run_validator(
            finding, workspace, ValidatorType.SECURITY, trace=trace, timeout_seconds=timeout_seconds
        )
        return ValidationMatrix(compile_result, functionality_result, security_result)
