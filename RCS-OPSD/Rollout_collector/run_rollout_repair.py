#!/usr/bin/env python3
"""Run one existing isolated PI RPC repair and normalize its recorded events."""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from recover_partial_rollouts import recover_turn

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "RCS-OPSD/PI_Qwen_rpc_no_location/run_pi_rpc_repair.py"
OUTPUT = Path(__file__).resolve().parent / "smoke_runs"


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def collect_rollout(output):
    result = json.loads((output / "result.json").read_text())
    turns, calls, counts = [], {}, Counter()
    current = None
    settled = False
    invalid_lines = 0
    events_path = output / "pi-events.jsonl"
    event_lines = events_path.read_text(errors="replace").splitlines() if events_path.exists() else []
    for line in event_lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            invalid_lines += 1
            continue
        kind = event["type"]
        counts[kind] += 1
        if kind == "turn_start":
            current = {"turn_id": len(turns), "assistant": None,
                       "tool_calls": [], "complete": False}
            turns.append(current)
        elif kind == "message_end":
            message = event["message"]
            if message["role"] == "assistant":
                current["assistant"] = message
        elif kind == "tool_execution_start":
            call = {"tool_call_id": event["toolCallId"],
                    "tool_name": event["toolName"], "args": event["args"],
                    "result": None, "is_error": None, "complete": False}
            current["tool_calls"].append(call)
            calls[call["tool_call_id"]] = call
        elif kind == "tool_execution_end":
            call = calls[event["toolCallId"]]
            call.update(result=event["result"], is_error=event["isError"], complete=True)
        elif kind == "turn_end":
            current["assistant"] = event["message"]
            current["complete"] = True
        elif kind == "agent_settled":
            settled = True
    for turn in turns:
        if not turn["complete"] and events_path.exists():
            recover_turn(events_path, turn)
    requests_path = output / "provider-requests.jsonl"
    requests = {}
    if requests_path.exists():
        for line in requests_path.read_text().splitlines():
            record = json.loads(line)
            requests[record["turn_id"]] = record["request"]
    for turn in turns:
        if turn["turn_id"] in requests:
            turn["student_request"] = requests[turn["turn_id"]]
    incomplete_calls = [call["tool_call_id"] for call in calls.values()
                        if not call["complete"]]
    incomplete_turns = [turn["turn_id"] for turn in turns if not turn["complete"]]
    capture_complete = bool(turns) and settled and not (
        incomplete_calls or incomplete_turns or invalid_lines)
    rollout = {
        "finding_id": result["finding_id"], "turns": turns,
        "seed": result.get("seed"), "model": result.get("model"),
        "endpoint": result.get("endpoint"),
        "thinking_enabled": result.get("thinking_enabled"),
        "thinking_level": result.get("thinking_level"),
        "success": result["status"] == "PASS",
        "repair_status": result["status"], "capture_complete": capture_complete,
        "agent_settled": settled,
    }
    save_json(output / "rollout.json", rollout)
    summary = {
        "finding_id": result["finding_id"], "repair_status": result["status"],
        "success": rollout["success"], "capture_complete": capture_complete,
        "turns": len(turns), "tool_calls": len(calls),
        "partial_assistant_messages": sum(turn.get("assistant_complete") is False for turn in turns),
        "pending_generated_tool_calls": sum(call.get("execution_started") is False
                                            for turn in turns for call in turn["tool_calls"]),
        "tool_counts": dict(Counter(call["tool_name"] for call in calls.values())),
        "tool_errors": sum(call["is_error"] is True for call in calls.values()),
        "incomplete_turns": incomplete_turns,
        "turns_with_student_request": sum("student_request" in turn for turn in turns),
        "incomplete_tool_calls": incomplete_calls,
        "invalid_event_lines": invalid_lines, "event_counts": dict(counts),
        "patch_bytes": result.get("patch_bytes", 0),
        "changed_files": result.get("changed_files", []),
        "duration_seconds": result.get("duration_seconds"),
        "endpoint": result.get("endpoint"),
        "raw_events_bytes": events_path.stat().st_size if events_path.exists() else 0,
        "rollout_bytes": (output / "rollout.json").stat().st_size,
        "artifact_dir": str(output),
    }
    save_json(output / "rollout_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finding", required=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--thinking", choices=["medium", "off"], default="medium")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:9531/v1")
    parser.add_argument("--model-id", default="qwen3.5-9b")
    parser.add_argument("--pi-timeout", type=int, default=1800)
    args = parser.parse_args()
    output = (args.output_dir or OUTPUT / (
        f"{args.finding}-seed{args.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    )).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty: {output}")
    command = [sys.executable, str(RUNNER), "--finding", args.finding,
               "--seed", str(args.seed), "--output-dir", str(output),
               "--pi-timeout", str(args.pi_timeout), "--thinking", args.thinking]
    print(json.dumps({"status": "STARTING", "artifact_dir": str(output)},
                     ensure_ascii=False), flush=True)
    environment = os.environ.copy()
    environment["RCS_OPSD_MODEL_ENDPOINT"] = args.endpoint
    environment["RCS_OPSD_MODEL_ID"] = args.model_id
    completed = subprocess.run(command, capture_output=True, text=True, env=environment)
    output.mkdir(parents=True, exist_ok=True)
    (output / "runner.stdout.log").write_text(completed.stdout)
    (output / "runner.stderr.log").write_text(completed.stderr)
    if not (output / "result.json").exists():
        save_json(output / "result.json", {
            "finding_id": args.finding, "status": "RUNNER_ERROR",
            "endpoint": args.endpoint, "runner_exit_code": completed.returncode,
            "error": completed.stderr,
        })
    summary = collect_rollout(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
