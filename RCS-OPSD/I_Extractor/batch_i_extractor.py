#!/usr/bin/env python3
"""Extract repair invariants for the selected findings."""
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
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rcs_opsd.artifact_paths import M_RESULTS, S_RESULTS, I_ROOT
OUTPUT_ROOT = Path(__file__).with_name("batch_runs")
RUNNER = Path(__file__).with_name("run_i_extractor.py")
TEMPLATE = Path(__file__).with_name("I_Extractor.md")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def selected_findings() -> list[dict[str, Any]]:
    m_entries = [json.loads(line) for line in M_RESULTS.read_text().splitlines() if line]
    s_ids = {
        json.loads(line)["finding_id"]
        for line in S_RESULTS.read_text().splitlines()
        if line
    }
    targets = [
        {
            "finding_id": entry["finding_id"],
            "category": entry["category"],
            "repair_statuses": entry["repair_statuses"],
        }
        for entry in m_entries
    ]
    missing_s = sorted(target["finding_id"] for target in targets if target["finding_id"] not in s_ids)
    if missing_s:
        raise SystemExit(f"selected findings missing S results: {missing_s}")
    return sorted(targets, key=lambda target: target["finding_id"])


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
    completed = subprocess.run(command, capture_output=True, text=True)
    (attempt / "runner.stdout.log").write_text(completed.stdout)
    (attempt / "runner.stderr.log").write_text(completed.stderr)
    result_path = attempt / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        return {
            **target,
            "status": result["status"],
            "invariant": result["invariant"],
            "artifact_dir": str(attempt),
            "duration_seconds": result["duration_seconds"],
            "usage": result.get("usage"),
            "reasoning_chars": result.get("reasoning_chars", 0),
            "runner_exit_code": completed.returncode,
        }
    return {
        **target,
        "status": "WORKER_ERROR",
        "artifact_dir": str(attempt),
        "runner_exit_code": completed.returncode,
        "error": completed.stderr.strip() or "runner did not produce result.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or args.timeout <= 0:
        parser.error("workers and timeout must be positive")
    if args.resume and args.output_dir is None:
        parser.error("--resume requires --output-dir")
    return args


def main() -> int:
    args = parse_args()
    targets = selected_findings()
    category_totals = Counter(target["category"] for target in targets)

    output = (
        args.output_dir.resolve()
        if args.output_dir
        else I_ROOT
    )
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise SystemExit(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    journal = output / "findings.jsonl"
    summary = output / "summary.json"
    completed, entries = load_journal(journal)
    remaining = [target for target in targets if target["finding_id"] not in completed]

    finding_ids = [target["finding_id"] for target in targets]
    metadata = {
        "schema_version": "qwen-i-extractor-batch/v1",
        "output_directory": str(output),
        "started_at": datetime.now().astimezone().isoformat(),
        "model": os.environ.get("RCS_OPSD_PI_MODEL", "qwen3.5-9b"),
        "endpoint": os.environ.get("RCS_OPSD_PI_ENDPOINT", "http://127.0.0.1:9531/v1").rstrip("/") + "/chat/completions",
        "thinking_enabled": True,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_tokens": 32768,
        "workers": args.workers,
        "timeout_seconds": args.timeout,
        "selection": {
            "stable_failure": category_totals["STABLE_FAILURE"],
            "unstable": category_totals["UNSTABLE"],
            "total": len(targets),
            "criterion": "same finding cohort used by M and S extractors",
        },
        "finding_ids_sha256": hashlib.sha256("\n".join(finding_ids).encode()).hexdigest(),
        "template": str(TEMPLATE),
        "template_sha256": hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
        "m_results": str(M_RESULTS),
        "m_results_sha256": hashlib.sha256(M_RESULTS.read_bytes()).hexdigest(),
        "s_results": str(S_RESULTS),
        "s_results_sha256": hashlib.sha256(S_RESULTS.read_bytes()).hexdigest(),
        "runner": str(RUNNER),
        "runner_sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
    }
    if args.resume and summary.is_file():
        previous = json.loads(summary.read_text())
        for key in (
            "finding_ids_sha256", "template_sha256", "m_results_sha256",
            "s_results_sha256", "runner_sha256", "temperature", "seed",
        ):
            if previous.get(key) != metadata[key]:
                raise SystemExit(f"resume metadata differs for {key}")
        metadata["started_at"] = previous["started_at"]
    checkpoint(summary, metadata, entries, len(targets), "RUNNING")

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
                    target = next(item for item in targets if item["finding_id"] == finding_id)
                    entry = {
                        **target,
                        "status": "WORKER_ERROR",
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                append_journal(journal, entry)
                entries.append(entry)
                checkpoint(summary, metadata, entries, len(targets), "RUNNING")
                print(f"{len(entries)}/{len(targets)} {finding_id} {entry['status']}", flush=True)

    checkpoint(summary, metadata, entries, len(targets), "COMPLETED")
    latest = {entry["finding_id"]: entry for entry in entries}
    if set(latest) == set(finding_ids) and all(entry["status"] == "EXTRACTED" for entry in latest.values()):
        (output / "complete_findings.jsonl").write_text(
            "".join(json.dumps(latest[key], ensure_ascii=False) + "\n" for key in sorted(latest))
        )
    print(f"completed={len(entries)}/{len(targets)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
