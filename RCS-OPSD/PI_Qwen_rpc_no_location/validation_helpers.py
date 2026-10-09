"""Dataset validation helpers used by the isolated Pi rollout collector."""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rcs_opsd.runtime import WorkspaceManager
from rcs_opsd.validators import ValidatorRunner, classify_world_result
DEFAULT_VALIDATOR_PATH = os.environ.get("PATH", "")
MAX_INLINE_OUTPUT = 12_000

def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)

def git(work: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(work), *args])

def inline(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_INLINE_OUTPUT:
        return text, False
    return text[:MAX_INLINE_OUTPUT] + f"\n...[truncated {len(text) - MAX_INLINE_OUTPUT} characters; see log file]\n", True

def validator_record(result: Any, phase: str, directory: Path) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{phase}.stdout.log").write_text(result.stdout)
    (directory / f"{phase}.stderr.log").write_text(result.stderr)
    stdout, stdout_truncated = inline(result.stdout)
    stderr, stderr_truncated = inline(result.stderr)
    return {
        "success": result.success,
        "exit_code": result.exit_code,
        "status": result.status,
        "duration_seconds": result.duration_seconds,
        "command": list(result.command),
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "metadata": result.metadata,
    }

def matrix_record(matrix: Any, directory: Path) -> dict[str, dict[str, Any]]:
    return {
        phase: validator_record(getattr(matrix, phase), phase, directory)
        for phase in ("compile", "functionality", "security")
    }

def apply_candidate(workspace: Path, patch: Path) -> dict[str, Any]:
    base = ["git", "-C", str(workspace), "apply", "--no-index"]
    checked = subprocess.run([*base, "--check", "--", str(patch)], capture_output=True, text=True, timeout=60)
    if checked.returncode:
        return {"success": False, "exit_code": checked.returncode, "stdout": checked.stdout, "stderr": checked.stderr}
    applied = subprocess.run([*base, "--", str(patch)], capture_output=True, text=True, timeout=60)
    return {
        "success": applied.returncode == 0,
        "exit_code": applied.returncode,
        "stdout": applied.stdout,
        "stderr": applied.stderr,
    }

def validate_candidate(finding: Any, patch: Path, attempt: Path, timeout: int) -> dict[str, Any]:
    workspaces = WorkspaceManager()
    runner = ValidatorRunner(workspaces)
    baseline_workspace = workspaces.create(finding, world="vulnerable")
    try:
        baseline_matrix = runner.run_world(finding, baseline_workspace, timeout_seconds=timeout)
        baseline = {
            "matrix": matrix_record(baseline_matrix, attempt / "baseline_validation"),
            "matches_expected": baseline_matrix.matches_world(finding, "vulnerable"),
        }
        baseline["failure_class"] = classify_world_result("vulnerable", baseline)
    finally:
        workspaces.cleanup(baseline_workspace)
    if not baseline["matches_expected"]:
        return {"baseline": baseline, "candidate": {"status": "NOT_RUN", "reason": "baseline did not match"}}

    candidate_workspace = workspaces.create(finding, world="candidate")
    try:
        applied = apply_candidate(candidate_workspace.root, patch)
        if not applied["success"]:
            candidate = {"patch": applied, "matrix": None, "matches_expected": False, "failure_class": "CANDIDATE_PATCH_APPLY_FAILED"}
        else:
            candidate_matrix = runner.run_world(finding, candidate_workspace, timeout_seconds=timeout)
            candidate = {
                "patch": applied,
                "matrix": matrix_record(candidate_matrix, attempt / "validation"),
                "matches_expected": candidate_matrix.matches_world(finding, "repaired"),
            }
            candidate["failure_class"] = classify_world_result("repaired", candidate)
    finally:
        workspaces.cleanup(candidate_workspace)
    return {"baseline": baseline, "candidate": candidate}

def configure_validator_environment(validator_path: str) -> dict[str, str]:
    os.environ["PATH"] = validator_path
    tools = {name: shutil.which(name) for name in ("forge", "node", "npm", "corepack")}
    missing = [name for name, path in tools.items() if path is None]
    if missing:
        raise RuntimeError("validator toolchain missing from PATH: " + ", ".join(missing))
    versions = {
        name: subprocess.check_output([path, "--version"], text=True, timeout=10).splitlines()[0]
        for name, path in tools.items()
    }
    node_major = int(versions["node"].lstrip("v").split(".", 1)[0])
    if node_major < 18:
        raise RuntimeError(f"validator Node.js must be >=18, found {versions['node']}")
    return {"path": validator_path, **{f"{name}_path": path for name, path in tools.items()}, **versions}
