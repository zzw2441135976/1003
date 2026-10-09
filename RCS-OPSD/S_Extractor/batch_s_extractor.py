#!/usr/bin/env python3
"""Run S Extractor for the completed M findings in parallel."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rcs_opsd.artifact_paths import M_RESULTS, S_ROOT
OUTPUT_ROOT = Path(__file__).with_name("batch_runs")
RUNNER = Path(__file__).with_name("run_s_extractor.py")
S_TEMPLATE = Path(__file__).with_name("S_Extractor.md")
AUDIT_ANCHOR_TEMPLATE = Path(__file__).with_name("Audit_Anchor_Extractor.md")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def targets() -> list[dict[str, Any]]:
    values = [json.loads(line) for line in M_RESULTS.read_text().splitlines() if line]
    return sorted(values, key=lambda value: value["finding_id"])


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def next_attempt(finding_dir: Path) -> Path:
    finding_dir.mkdir(parents=True, exist_ok=True)
    number = 1
    while (finding_dir / f"attempt-{number:02d}").exists():
        number += 1
    attempt = finding_dir / f"attempt-{number:02d}"
    attempt.mkdir()
    return attempt


def append_journal(path: Path, entry: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_journal(path: Path) -> tuple[set[str], list[dict[str, Any]]]:
    if not path.exists():
        return set(), []
    entries = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {entry["finding_id"] for entry in entries}, entries


def checkpoint(
    path: Path,
    metadata: dict[str, Any],
    entries: list[dict[str, Any]],
    total: int,
    status: str,
) -> None:
    write_json(
        path,
        {
            **metadata,
            "status": status,
            "updated_at": datetime.now().astimezone().isoformat(),
            "coverage": {
                "scheduled": total,
                "completed": len(entries),
                "remaining": total - len(entries),
            },
            "status_counts": dict(sorted(Counter(entry["status"] for entry in entries).items())),
            "category_counts": dict(sorted(Counter(entry["category"] for entry in entries).items())),
        },
    )


def run_one(
    output: str,
    target: dict[str, Any],
    temperature: float,
    seed: int,
    timeout: int,
) -> dict[str, Any]:
    finding_id = target["finding_id"]
    attempt = next_attempt(Path(output) / "findings" / finding_id)
    command = [
        sys.executable,
        str(RUNNER),
        "--finding", finding_id,
        "--output-dir", str(attempt),
        "--temperature", str(temperature),
        "--seed", str(seed),
        "--timeout", str(timeout),
    ]
    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True)
    duration = round(time.monotonic() - started, 3)
    (attempt / "runner.stdout.log").write_text(completed.stdout)
    (attempt / "runner.stderr.log").write_text(completed.stderr)
    result_path = attempt / "result.json"
    base = {
        "finding_id": finding_id,
        "category": target["category"],
        "repair_statuses": target["repair_statuses"],
        "artifact_dir": str(attempt),
        "duration_seconds": duration,
        "runner_exit_code": completed.returncode,
    }
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        return {
            **base,
            "status": result["status"],
            "selection": result.get("selection"),
            "audit_anchor_count": result["audit_anchor_count"],
            "patch_anchor_count": result["patch_anchor_count"],
            "initial_candidate_count": result["initial_candidate_count"],
            "iteration_count": len(result["iterations"]),
            "expanded": len(result["iterations"]) > 1,
        }
    return {
        **base,
        "status": "WORKER_ERROR",
        "error": completed.stderr.strip() or "runner did not produce result.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--retry-from",
        type=Path,
        help="Run only findings whose prior batch status was not EXTRACTED",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or args.timeout <= 0:
        parser.error("workers and timeout must be positive")
    if args.resume and args.output_dir is None:
        parser.error("--resume requires --output-dir")
    return args


def main() -> int:
    args = parse_args()
    selected = targets()
    category_totals = Counter(target["category"] for target in selected)
    if args.retry_from:
        prior = load_jsonl(args.retry_from)
        retry_ids = {
            entry["finding_id"] for entry in prior if entry["status"] != "EXTRACTED"
        }
        selected = [target for target in selected if target["finding_id"] in retry_ids]

    output = (
        args.output_dir.resolve()
        if args.output_dir
        else S_ROOT
    )
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise SystemExit(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    journal = output / "findings.jsonl"
    summary = output / "summary.json"
    completed_ids, entries = load_journal(journal)
    remaining = [target for target in selected if target["finding_id"] not in completed_ids]
    finding_ids = [target["finding_id"] for target in selected]

    metadata = {
        "schema_version": "qwen-s-extractor-batch/v1",
        "output_directory": str(output),
        "started_at": datetime.now().astimezone().isoformat(),
        "model": os.environ.get("RCS_OPSD_PI_MODEL", "qwen3.5-9b"),
        "endpoint": os.environ.get("RCS_OPSD_PI_ENDPOINT", "http://127.0.0.1:9531/v1").rstrip("/") + "/chat/completions",
        "thinking_enabled": True,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_tokens": 32768,
        "workers": args.workers,
        "timeout_seconds_per_model_call": args.timeout,
        "graph_expansion": {
            "direction": "bidirectional",
            "edge_types": ["CALL", "STATE_DEP"],
            "hops_per_iteration": 1,
        },
        "selection": {
            "stable_failure": sum(target["category"] == "STABLE_FAILURE" for target in selected),
            "unstable": sum(target["category"] == "UNSTABLE" for target in selected),
            "total": len(selected),
        },
        "retry_from": str(args.retry_from.resolve()) if args.retry_from else None,
        "finding_ids_sha256": hashlib.sha256("\n".join(finding_ids).encode()).hexdigest(),
        "m_results": str(M_RESULTS),
        "m_results_sha256": hashlib.sha256(M_RESULTS.read_bytes()).hexdigest(),
        "s_template": str(S_TEMPLATE),
        "s_template_sha256": hashlib.sha256(S_TEMPLATE.read_bytes()).hexdigest(),
        "audit_anchor_template": str(AUDIT_ANCHOR_TEMPLATE),
        "audit_anchor_template_sha256": hashlib.sha256(AUDIT_ANCHOR_TEMPLATE.read_bytes()).hexdigest(),
        "runner": str(RUNNER),
        "runner_sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
    }
    if args.resume and summary.is_file():
        previous = json.loads(summary.read_text())
        for key in (
            "finding_ids_sha256",
            "m_results_sha256",
            "s_template_sha256",
            "audit_anchor_template_sha256",
            "runner_sha256",
            "temperature",
            "seed",
        ):
            if previous.get(key) != metadata[key]:
                raise SystemExit(f"resume metadata differs for {key}")
        metadata["started_at"] = previous["started_at"]
    checkpoint(summary, metadata, entries, len(selected), "RUNNING")

    submitted = iter(remaining)
    futures: dict[Future[dict[str, Any]], str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while True:
            while len(futures) < args.workers:
                try:
                    target = next(submitted)
                except StopIteration:
                    break
                future = executor.submit(
                    run_one,
                    str(output),
                    target,
                    args.temperature,
                    args.seed,
                    args.timeout,
                )
                futures[future] = target["finding_id"]
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                finding_id = futures.pop(future)
                try:
                    entry = future.result()
                except Exception as exc:
                    target = next(item for item in selected if item["finding_id"] == finding_id)
                    entry = {
                        "finding_id": finding_id,
                        "category": target["category"],
                        "repair_statuses": target["repair_statuses"],
                        "status": "WORKER_ERROR",
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                append_journal(journal, entry)
                entries.append(entry)
                checkpoint(summary, metadata, entries, len(selected), "RUNNING")
                print(f"{len(entries)}/{len(selected)} {finding_id} {entry['status']}", flush=True)

    checkpoint(summary, metadata, entries, len(selected), "COMPLETED")
    latest = {entry["finding_id"]: entry for entry in entries}
    if set(latest) == set(finding_ids) and all(entry["status"] == "EXTRACTED" for entry in latest.values()):
        (output / "complete_findings.jsonl").write_text(
            "".join(json.dumps(latest[key], ensure_ascii=False) + "\n" for key in sorted(latest))
        )
    print(f"completed={len(entries)}/{len(selected)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
