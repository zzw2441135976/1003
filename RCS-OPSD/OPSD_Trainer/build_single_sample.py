#!/usr/bin/env python3
"""Build and validate one OPSD training sample from a selected rollout turn."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ROLLOUT_ROOT = ROOT / "artifacts/opsd-v2/rollouts"
ORIGINAL_ROLLOUT_ROOT = ROLLOUT_ROOT
SELECTED = ROOT / "artifacts/opsd-v2/selected_turns.jsonl"
PI_ROOT = ROOT / "artifacts/opsd-v2/pi/turns"
DEFAULT_FINDING = "2023-12-ethereumcreditguild__H-01"
DEFAULT_TURN = 1


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def selected_turn(
    finding_id: str, turn_id: int, selected_path: Path = SELECTED
) -> dict[str, Any]:
    for line in selected_path.read_text().splitlines():
        record = json.loads(line)
        if record["finding_id"] == finding_id and record["turn_id"] == turn_id:
            return record
    raise KeyError(f"selected turn not found: {finding_id} turn {turn_id}")


def tool_call_ids(assistant: dict[str, Any]) -> list[str]:
    return [
        block["id"]
        for block in assistant.get("content", [])
        if block.get("type") == "toolCall" and block.get("id")
    ]


def build_sample(
    finding_id: str,
    turn_id: int,
    selected_path: Path = SELECTED,
    rollout_root: Path = ROLLOUT_ROOT,
    prompt_root: Path = ORIGINAL_ROLLOUT_ROOT,
    pi_root: Path = PI_ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = selected_turn(finding_id, turn_id, selected_path)
    rollout_path = rollout_root / finding_id / "rollout.json"
    rollout = load_json(rollout_path)
    turns = rollout["turns"]
    position = next(index for index, turn in enumerate(turns) if turn["turn_id"] == turn_id)
    current = turns[position]

    prompt_path = prompt_root / finding_id / "prompt.md"
    if not prompt_path.exists():
        prompt_path = prompt_root / finding_id / "attempt-01/prompt.md"
    initial_prompt = prompt_path.read_text()
    pi_dir = pi_root / f"{finding_id}__turn_{turn_id}"
    pi_result = load_json(pi_dir / "result.json")
    pi_input = load_json(pi_dir / "input.json")

    history = [
        {
            "role": "user",
            "content": initial_prompt,
            "source": str(prompt_path.relative_to(ROOT)),
        }
    ]
    history.extend(
        {
            "turn_id": turn["turn_id"],
            "assistant": turn["assistant"],
            "tool_calls": turn.get("tool_calls", []),
            "complete": turn.get("complete", False),
        }
        for turn in turns[:position]
    )

    sample = {
        "schema_version": "opsd-training-sample/v1",
        "finding_id": finding_id,
        "turn_id": turn_id,
        "state": selected["state"],
        "history": history,
        "privileged_information": pi_result["privileged_information"],
        "target": {
            "turn_id": current["turn_id"],
            "assistant_message": current["assistant"],
            "tool_calls": current.get("tool_calls", []),
            "complete": current.get("complete", False),
        },
        "metadata": {
            "focus": selected["function_focus"],
            "primary_focus": pi_result["current_focus"],
            "distance": selected["distance"],
            "path": pi_input.get("path"),
            "repair_action": selected["repair_action"],
            "selection_reasons": selected["selection_reasons"],
            "rollout": str(rollout_path.relative_to(ROOT)),
            "pi_result": str((pi_dir / "result.json").relative_to(ROOT)),
        },
    }
    if "student_request" in current:
        sample["student_request"] = current["student_request"]
    checks = validate_sample(sample, finding_id, turn_id, turns, position, pi_result)
    if "student_request" in current:
        checks["pi_context_matches_student_request"] = (
            pi_input["student_state"]["messages"] == current["student_request"]["messages"]
        )
        checks["all_pass"] = checks["all_pass"] and checks["pi_context_matches_student_request"]
    return sample, checks


def validate_sample(
    sample: dict[str, Any],
    finding_id: str,
    turn_id: int,
    turns: list[dict[str, Any]],
    position: int,
    pi_result: dict[str, Any],
) -> dict[str, Any]:
    history_turns = [item for item in sample["history"] if "turn_id" in item]
    history_ids = [item["turn_id"] for item in history_turns]
    target = sample["target"]
    target_call_ids = [call["tool_call_id"] for call in target["tool_calls"]]
    assistant_call_ids = tool_call_ids(target["assistant_message"])
    checks = {
        "finding_id_match": sample["finding_id"] == finding_id == pi_result["finding_id"],
        "turn_id_match": sample["turn_id"] == turn_id == pi_result["turn_id"] == target["turn_id"],
        "history_is_prefix": history_ids == [turn["turn_id"] for turn in turns[:position]],
        "target_excluded_from_history": turn_id not in history_ids,
        "target_is_next_turn": position == len(history_turns),
        "target_complete": target["complete"] is True,
        "tool_call_order_preserved": assistant_call_ids == target_call_ids,
        "tool_results_present_in_target": all("result" in call for call in target["tool_calls"]),
        "no_cross_rollout": all(item.get("turn_id", turn_id) != turn_id for item in history_turns),
    }
    checks["all_pass"] = all(checks.values())
    checks["history_turn_count"] = len(history_turns)
    checks["target_tool_call_count"] = len(target["tool_calls"])
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finding", default=DEFAULT_FINDING)
    parser.add_argument("--turn", type=int, default=DEFAULT_TURN)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "samples/ethereumcreditguild-h01-turn1.json",
    )
    args = parser.parse_args()
    sample, checks = build_sample(args.finding, args.turn)
    write_json(args.output, sample)
    write_json(args.output.with_name(args.output.stem + ".validation.json"), checks)
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(args.output.resolve())
    return 0 if checks["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
