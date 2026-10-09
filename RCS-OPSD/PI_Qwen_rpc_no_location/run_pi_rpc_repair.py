#!/usr/bin/env python3
"""Run one isolated no-location PI RPC repair and validate its patch."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.request import ProxyHandler, build_opener


ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "dataset_root"
OUTPUT_ROOT = ROOT / "RCS-OPSD" / "PI_Qwen_rpc_no_location" / "runs"
PI_RUNTIME = ROOT / ".tools" / "pi-coding-agent"
PI_CLI = Path("node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js")
PI_NODE = Path(os.environ.get("RCS_OPSD_PI_NODE", shutil.which("node") or "node"))
TOOL_NODE_ROOT = Path(os.environ.get("RCS_OPSD_TOOL_NODE_ROOT", str(PI_NODE.resolve().parents[1])))
FORGE = Path(os.environ.get("RCS_OPSD_FORGE", shutil.which("forge") or "forge"))
BWRAP = Path("/usr/bin/bwrap")
ENDPOINT = os.environ.get("RCS_OPSD_MODEL_ENDPOINT", "http://127.0.0.1:9531/v1")
MODEL_ID = os.environ.get("RCS_OPSD_MODEL_ID", "qwen3.5-9b")
MODEL = f"local-qwen/{MODEL_ID}"
TOOLS = ["read", "bash", "edit", "write", "grep", "find", "ls"]

sys.path.insert(0, str(ROOT / "src"))
from rcs_opsd.dataset import DatasetLoader  # noqa: E402
from validation_helpers import (  # noqa: E402
    DEFAULT_VALIDATOR_PATH,
    atomic_json,
    configure_validator_environment,
    git,
    timestamp,
    validate_candidate,
)


def model_config(seed: int, thinking: str = "medium") -> dict[str, Any]:
    return {
        "providers": {
            "local-qwen": {
                "baseUrl": ENDPOINT,
                "api": "openai-completions",
                "apiKey": "EMPTY",
                "compat": {
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                    "maxTokensField": "max_tokens",
                },
                "models": [
                    {
                        "id": MODEL_ID,
                        "name": "Qwen3.5 9B local RPC",
                        "reasoning": True,
                        "input": ["text"],
                        "contextWindow": 131072,
                        "maxTokens": 32768,
                        "samplingParams": {
                            "temperature": 0.6,
                            "top_p": 0.95,
                            "top_k": 20,
                            "min_p": 0.0,
                            "presence_penalty": 0.0,
                            "repetition_penalty": 1.0,
                            "seed": seed,
                            "chat_template_kwargs": {"enable_thinking": thinking != "off"},
                        },
                        "cost": {
                            "input": 0,
                            "output": 0,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                        },
                    }
                ],
            }
        }
    }


def pi_command(
    work: Path,
    artifact: Path,
    scratch: Path,
    pi_runtime: Path,
    pi_node: Path,
    thinking: str = "medium",
) -> list[str]:
    config = scratch / "pi-config"
    config.mkdir()
    shutil.copyfile(artifact / "models.json", config / "models.json")
    shutil.copyfile(artifact / "settings.json", config / "settings.json")
    shutil.copyfile(Path(__file__).with_name("capture_requests.mjs"), config / "capture_requests.mjs")

    command = [
        str(BWRAP),
        "--unshare-user-try",
        "--unshare-pid",
        "--die-with-parent",
        "--clearenv",
    ]
    for path in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        command.extend(("--ro-bind", path, path))
    command.extend(
        (
            "--tmpfs", "/tmp",
            "--proc", "/proc",
            "--dev", "/dev",
            "--dir", "/opt",
            "--dir", "/opt/bin",
            "--ro-bind", str(pi_runtime), "/opt/pi",
            "--ro-bind", str(pi_node.parents[1]), "/opt/node-runtime",
            "--ro-bind", str(TOOL_NODE_ROOT), "/opt/node-tools",
            "--ro-bind", str(FORGE), "/opt/bin/forge",
            "--bind", str(config), "/config",
            "--bind", str(work), "/testbed",
            "--chdir", "/testbed",
        )
    )
    environment = {
        "HOME": "/tmp",
        "PATH": "/opt/node-tools/bin:/opt/bin:/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "PI_CODING_AGENT_DIR": "/config",
        "PI_OFFLINE": "1",
        "PI_SKIP_VERSION_CHECK": "1",
        "PI_TELEMETRY": "0",
        "NO_PROXY": "*",
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    for key, value in environment.items():
        command.extend(("--setenv", key, value))
    command.extend(
        (
            "/opt/node-runtime/bin/node",
            "/opt/pi/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js",
            "--mode", "rpc",
            "--provider", "local-qwen",
            "--model", MODEL_ID,
            "--thinking", thinking,
            "--no-session",
            "--tools", ",".join(TOOLS),
            "--no-extensions",
            "--extension", "/config/capture_requests.mjs",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
            "--no-context-files",
            "--no-approve",
            "--offline",
        )
    )
    return command


def pi_version(pi_runtime: Path, pi_node: Path) -> str:
    return subprocess.check_output(
        [str(pi_node), str(pi_runtime / PI_CLI), "--version"],
        text=True,
        timeout=10,
    ).strip()


def run_pi_rpc(
    command: list[str], prompt: str, artifact: Path, timeout: int
) -> dict[str, Any]:
    event_counts: Counter[str] = Counter()
    tool_names: set[str] = set()
    settled = False
    prompt_accepted = False
    invalid_event_lines = 0
    final_message: dict[str, Any] | None = None
    timed_out = False
    process_exit_code: int | None = None

    with (artifact / "pi-events.jsonl").open("wb") as events, (
        artifact / "pi.stderr.log"
    ).open("wb") as stderr:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            bufsize=0,
        )
        assert process.stdin is not None and process.stdout is not None
        request = json.dumps(
            {"id": "finding-prompt", "type": "prompt", "message": prompt},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        process.stdin.write(request)
        process.stdin.flush()

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        buffer = b""
        deadline = time.monotonic() + timeout
        try:
            while not settled:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                ready = selector.select(min(remaining, 1.0))
                if not ready:
                    if process.poll() is not None:
                        break
                    continue
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    raw, buffer = buffer.split(b"\n", 1)
                    events.write(raw + b"\n")
                    events.flush()
                    line = raw[:-1] if raw.endswith(b"\r") else raw
                    if not line:
                        continue
                    try:
                        event = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        invalid_event_lines += 1
                        continue
                    event_type = event.get("type")
                    if isinstance(event_type, str):
                        event_counts[event_type] += 1
                    if (
                        event_type == "response"
                        and event.get("id") == "finding-prompt"
                        and event.get("command") == "prompt"
                        and event.get("success") is True
                    ):
                        prompt_accepted = True
                    if event_type == "tool_execution_start":
                        tool_name = event.get("toolName")
                        if isinstance(tool_name, str):
                            tool_names.add(tool_name)
                    if event_type == "message_end":
                        message = event.get("message")
                        if isinstance(message, dict) and message.get("role") == "assistant":
                            final_message = message
                    if event_type == "agent_settled":
                        settled = True
                        break
        finally:
            selector.close()
            if buffer:
                events.write(buffer)
                events.flush()
            if timed_out:
                stderr.write(f"PI wall timeout after {timeout} seconds\n".encode())
            process.terminate()
            try:
                process_exit_code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process_exit_code = process.wait()
            process.stdin.close()
            process.stdout.close()

    if final_message is not None:
        atomic_json(artifact / "final-message.json", final_message)
    return {
        "settled": settled,
        "prompt_accepted": prompt_accepted,
        "timed_out": timed_out,
        "process_exit_code": process_exit_code,
        "event_counts": dict(sorted(event_counts.items())),
        "invalid_event_lines": invalid_event_lines,
        "tools_observed": sorted(tool_names),
        "final_message_saved": final_message is not None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finding", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--thinking", choices=["medium", "off"], default="medium")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pi-timeout", type=int, default=1800)
    parser.add_argument("--validator-timeout", type=int, default=600)
    parser.add_argument("--pi-runtime", type=Path, default=PI_RUNTIME)
    parser.add_argument("--pi-node", type=Path, default=PI_NODE)
    parser.add_argument("--pi-version")
    parser.add_argument(
        "--validator-path",
        default=os.environ.get("RCS_OPSD_VALIDATOR_PATH", DEFAULT_VALIDATOR_PATH),
    )
    args = parser.parse_args()
    if args.pi_timeout <= 0 or args.validator_timeout <= 0:
        parser.error("timeouts must be positive")
    return args


def main() -> int:
    args = parse_args()
    loader = DatasetLoader(DATASET_ROOT)
    if args.finding not in loader.finding_ids():
        raise SystemExit(f"unknown finding: {args.finding}")
    finding = loader.load(args.finding)
    output = (
        args.output_dir.resolve()
        if args.output_dir
        else OUTPUT_ROOT / f"{args.finding}-seed{args.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    pi_runtime = args.pi_runtime.resolve()
    pi_node = args.pi_node.resolve()
    version = args.pi_version or pi_version(pi_runtime, pi_node)
    validator_toolchain = configure_validator_environment(args.validator_path)
    with build_opener(ProxyHandler({})).open(ENDPOINT + "/models", timeout=10) as response:
        service = json.load(response)
    prompt = (
        (DATASET_ROOT / "dataset_tools" / "instruction_prompt.md").read_text().rstrip()
        + "\n\n"
        + (finding.task_dir / "full_prompt.md").read_text()
    )
    (output / "prompt.md").write_text(prompt)
    atomic_json(output / "models.json", model_config(args.seed, args.thinking))
    atomic_json(
        output / "settings.json",
        {
            "defaultProjectTrust": "never",
            "enableInstallTelemetry": False,
            "enableAnalytics": False,
            "compaction": {
                "enabled": True,
                "reserveTokens": 32768,
                "keepRecentTokens": 20000,
            },
        },
    )
    result: dict[str, Any] = {
        "schema_version": "pi-qwen-rpc-no-location-finding/v1",
        "finding_id": finding.id,
        "seed": args.seed,
        "started_at": timestamp(),
        "status": "RUNNING",
        "agent": "pi",
        "pi_mode": "rpc",
        "session_persistence": False,
        "pi_version": version,
        "pi_runtime": str(pi_runtime),
        "pi_node": str(pi_node),
        "model": MODEL,
        "model_config": model_config(args.seed, args.thinking),
        "endpoint": ENDPOINT,
        "model_service": service,
        "location_guidance_enabled": False,
        "initial_file_contents_included": False,
        "directory_ignore_enabled": False,
        "evaluation_feedback_to_agent": False,
        "project_context_enabled": False,
        "extensions_enabled": True,
        "request_capture_extension": "capture_requests.mjs (observation only)",
        "skills_enabled": False,
        "tools": TOOLS,
        "thinking_enabled": args.thinking != "off",
        "thinking_level": args.thinking,
        "validator_toolchain": validator_toolchain,
        "validator_protocol": finding.compile_validator.protocol.value,
        "validator_adapter": finding.compile_validator.adapter_name,
        "project_framework": finding.project.framework.value,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }
    atomic_json(output / "result.json", result)
    started = time.monotonic()

    try:
        with tempfile.TemporaryDirectory(prefix="pi-qwen-rpc-repair-") as temporary:
            scratch = Path(temporary)
            work = scratch / "project"
            shutil.copytree(
                finding.task_dir / "source",
                work,
                symlinks=False,
                ignore=shutil.ignore_patterns(".git"),
            )
            git(work, "init", "-q")
            git(work, "add", "-f", ".")
            git(
                work,
                "-c", "user.name=Experiment",
                "-c", "user.email=experiment@localhost",
                "commit", "-qm", "Original source baseline",
            )
            baseline_commit = git(work, "rev-parse", "HEAD").decode().strip()
            command = pi_command(work, output, scratch, pi_runtime, pi_node, args.thinking)
            atomic_json(output / "command.json", command)
            rpc = run_pi_rpc(command, prompt, output, args.pi_timeout)
            requests = scratch / "pi-config/provider-requests.jsonl"
            if requests.exists():
                shutil.copyfile(requests, output / "provider-requests.jsonl")
            session_files = [
                str(path.relative_to(scratch))
                for path in scratch.rglob("*.jsonl")
                if path.is_file()
            ]
            git(work, "add", "-A")
            patch = git(work, "diff", "--cached", "--binary", baseline_commit)
            (output / "patch.diff").write_bytes(patch)
            changed_files = git(
                work, "diff", "--cached", "--name-only", baseline_commit
            ).decode().splitlines()

        result.update(
            {
                "baseline_commit": baseline_commit,
                "rpc": rpc,
                "pi_exit_code": 124 if rpc["timed_out"] else 0 if rpc["settled"] else rpc["process_exit_code"],
                "session_files_created": session_files,
                "patch_bytes": len(patch),
                "patch_sha256": hashlib.sha256(patch).hexdigest(),
                "changed_files": changed_files,
                "temporary_directory_cleaned": True,
            }
        )
        if not patch:
            if rpc["timed_out"]:
                result["status"] = "PI_TIMEOUT"
            elif not rpc["settled"]:
                result["status"] = "PI_ERROR"
            else:
                result["status"] = "NO_PATCH"
            result["validation"] = {"status": "NOT_RUN", "reason": "no candidate patch"}
        else:
            validation = validate_candidate(
                finding, output / "patch.diff", output, args.validator_timeout
            )
            result["validation"] = validation
            candidate = validation["candidate"]
            if not validation["baseline"]["matches_expected"]:
                result["status"] = "BASELINE_" + validation["baseline"]["failure_class"]
            elif candidate.get("matches_expected"):
                result["status"] = "PASS"
            else:
                result["status"] = candidate.get(
                    "failure_class", candidate.get("status", "VALIDATION_FAILED")
                )
    except Exception as exc:
        result["status"] = "WORKER_ERROR"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}

    result["completed_at"] = timestamp()
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    atomic_json(output / "result.json", result)
    print(json.dumps({"status": result["status"], "output": str(output)}, ensure_ascii=False))
    return 0 if result["status"] != "WORKER_ERROR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
