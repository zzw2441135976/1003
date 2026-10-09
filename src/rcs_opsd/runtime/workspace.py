"""Copy-based workspaces that never modify a dataset source snapshot."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from typing import Iterator

from rcs_opsd.dataset.finding import Finding


_MARKER = ".rcs_opsd_workspace.json"
_FORMAT = "rcs-opsd-workspace/v1"


class WorkspaceError(RuntimeError):
    """Raised for workspace lifecycle invariant violations."""


@dataclass(frozen=True)
class Workspace:
    root: Path
    source_snapshot: Path
    finding_id: str
    world: str
    cleanup_root: Path


@dataclass(frozen=True)
class PatchApplicationResult:
    success: bool
    exit_code: int | None
    stdout: str
    stderr: str


class WorkspaceManager:
    """Creates copied worlds and safely stages self-contained validator tasks."""

    def __init__(self, temp_root: Path | str | None = None) -> None:
        self.temp_root = Path(temp_root).expanduser().resolve() if temp_root else None

    def create(self, finding: Finding, world: str = "vulnerable") -> Workspace:
        if world not in {"vulnerable", "repaired", "candidate"}:
            raise WorkspaceError(f"Unsupported workspace world: {world}")
        source = finding.vulnerable_repository.resolve()
        if not source.is_dir():
            raise WorkspaceError(f"Source snapshot is not available: {source}")
        cleanup_root = Path(tempfile.mkdtemp(prefix="rcs-opsd-workspace-", dir=self.temp_root))
        root = cleanup_root / "project"
        try:
            shutil.copytree(source, root, symlinks=False, copy_function=shutil.copy2)
            (root / _MARKER).write_text(
                json.dumps(
                    {
                        "format": _FORMAT,
                        "source_snapshot": str(source),
                        "finding_id": finding.id,
                        "world": world,
                        "cleanup_root": str(cleanup_root),
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except BaseException:
            shutil.rmtree(cleanup_root, ignore_errors=True)
            raise
        return Workspace(root=root, source_snapshot=source, finding_id=finding.id, world=world, cleanup_root=cleanup_root)

    def apply_reference_patch(self, workspace: Workspace, finding: Finding) -> PatchApplicationResult:
        self._validate(workspace)
        if workspace.finding_id != finding.id:
            raise WorkspaceError("A reference patch may only be applied to its own finding workspace.")
        check = self._git_apply(workspace.root, finding.reference_patch, check=True)
        if not check.success:
            return check
        return self._git_apply(workspace.root, finding.reference_patch, check=False)

    def reset(self, workspace: Workspace, finding: Finding) -> Workspace:
        """Discard a mutable world and return a fresh vulnerable copy."""

        self.cleanup(workspace)
        return self.create(finding, world="vulnerable")

    def collect_diff(self, workspace: Workspace) -> str:
        self._validate(workspace)
        completed = subprocess.run(
            ["git", "diff", "--no-index", "--", str(workspace.source_snapshot), str(workspace.root)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode not in {0, 1}:
            raise WorkspaceError(completed.stderr.strip() or "Unable to collect workspace diff.")
        return completed.stdout

    def cleanup(self, workspace: Workspace) -> None:
        self._validate(workspace)
        if workspace.cleanup_root.name.startswith("rcs-opsd-workspace-") and workspace.root.parent == workspace.cleanup_root:
            shutil.rmtree(workspace.cleanup_root)
            return
        raise WorkspaceError(f"Refusing to remove unmanaged workspace: {workspace.root}")

    @contextmanager
    def staged_task(self, finding: Finding, workspace: Workspace) -> Iterator[Path]:
        """Copy validator artifacts and rebind ``source`` to the isolated world.

        Self-contained wrappers are written to execute relative to `task/`.
        Staging prevents those scripts from reading the original task/source
        symlink or writing their generated cache/test artifacts into the
        dataset itself.
        """

        self._validate(workspace)
        stage_root = Path(tempfile.mkdtemp(prefix="rcs-opsd-validator-task-", dir=self.temp_root))
        task_copy = stage_root / "task"
        try:
            shutil.copytree(
                finding.task_dir,
                task_copy,
                symlinks=True,
                ignore=shutil.ignore_patterns("source"),
            )
            self._make_staged_task_writable(task_copy)
            (task_copy / "source").symlink_to(workspace.root, target_is_directory=True)
            yield task_copy
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)

    @staticmethod
    def _make_staged_task_writable(task_copy: Path) -> None:
        """Allow validator-owned staging only inside the disposable task copy.

        Dataset capsules intentionally preserve source artifact permissions;
        several compile wrappers then copy selected files into ``capsule/src``.
        ``copytree`` preserves a read-only source file's mode, which makes that
        legitimate validator-local staging fail.  The staged task is removed
        immediately after a validator invocation, so owner-write permission is
        restored only on this isolated copy and never on the dataset artifact
        or its rebound ``source`` workspace.
        """

        for path in (task_copy, *task_copy.rglob("*")):
            if path.is_symlink():
                continue
            mode = path.stat().st_mode
            writable_mode = mode | stat.S_IWUSR
            if path.is_dir():
                writable_mode |= stat.S_IXUSR
            path.chmod(writable_mode)

    @staticmethod
    def _git_apply(workspace: Path, patch: Path, *, check: bool) -> PatchApplicationResult:
        command = ["git", "-C", str(workspace), "apply", "--no-index"]
        if check:
            command.append("--check")
        command.extend(("--", str(patch)))
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60)
        except FileNotFoundError:
            return PatchApplicationResult(False, 127, "", "git executable is unavailable")
        except subprocess.TimeoutExpired:
            return PatchApplicationResult(False, None, "", "git apply timed out after 60 seconds")
        return PatchApplicationResult(
            completed.returncode == 0,
            completed.returncode,
            completed.stdout or "",
            completed.stderr or "",
        )

    @staticmethod
    def _validate(workspace: Workspace) -> None:
        marker = workspace.root / _MARKER
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkspaceError(f"Workspace has no valid RCS-OPSD marker: {workspace.root}") from exc
        if data.get("format") != _FORMAT or Path(data.get("cleanup_root", "")) != workspace.cleanup_root:
            raise WorkspaceError(f"Workspace marker does not match supplied workspace: {workspace.root}")
