#!/usr/bin/env python3
"""Recover recorded partial assistant messages without inventing an ending."""
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path


def recover_turn(events_path, turn):
    current_id = -1
    message = None
    blocks = {}
    argument_chunks = {}
    ended_calls = set()
    for line in events_path.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event["type"] == "turn_start":
            current_id += 1
        if current_id != turn["turn_id"]:
            continue
        if event["type"] == "message_start" and event["message"]["role"] == "assistant":
            message = copy.deepcopy(event["message"])
        if event["type"] != "message_update":
            continue
        update = event["assistantMessageEvent"]
        kind = update["type"]
        if "contentIndex" not in update:
            continue
        index = update["contentIndex"]
        if kind in {"thinking_start", "text_start"}:
            name = kind.removesuffix("_start")
            blocks[index] = {"type": name, name: ""}
        elif kind in {"thinking_delta", "text_delta"}:
            name = kind.removesuffix("_delta")
            blocks[index][name] += update["delta"]
        elif kind in {"thinking_end", "text_end"}:
            name = kind.removesuffix("_end")
            blocks[index][name] = update["content"]
        elif kind == "toolcall_start":
            blocks[index] = {"type": "toolCall", "id": update["id"],
                             "name": update["toolName"], "arguments": None}
            argument_chunks[index] = ""
        elif kind == "toolcall_delta":
            argument_chunks[index] += update["delta"]
        elif kind == "toolcall_end":
            blocks[index] = copy.deepcopy(update["toolCall"])
            ended_calls.add(index)
        if message is not None and event.get("usage") is not None:
            message["usage"] = event["usage"]

    for index, chunks in argument_chunks.items():
        block = blocks[index]
        block["arguments_complete"] = index in ended_calls
        if index not in ended_calls:
            block["arguments_partial"] = chunks
            try:
                block["arguments"] = json.loads(chunks)
            except json.JSONDecodeError:
                block["arguments"] = None
    needs_partial_message = turn["assistant"] is None or turn["assistant"].get("partial", False)
    recovered_assistant = needs_partial_message and message is not None
    if recovered_assistant:
        message["content"] = [blocks[index] for index in sorted(blocks)]
        message["partial"] = True
        turn["assistant"] = message
    turn["assistant_complete"] = not recovered_assistant and turn["assistant"] is not None
    existing = {call["tool_call_id"] for call in turn["tool_calls"]}
    pending = 0
    for block in blocks.values():
        if block["type"] == "toolCall" and block["id"] not in existing:
            call = {"tool_call_id": block["id"], "tool_name": block["name"],
                    "args": block["arguments"], "result": None, "is_error": None,
                    "complete": False, "execution_started": False,
                    "arguments_complete": block["arguments_complete"]}
            if "arguments_partial" in block:
                call["args_partial"] = block["arguments_partial"]
            turn["tool_calls"].append(call)
            pending += 1
    return {"assistant_recovered": recovered_assistant,
            "pending_generated_tool_calls": pending}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    args = parser.parse_args()
    batch = args.batch_dir.resolve()
    rows = [json.loads(line) for line in (batch / "findings.jsonl").read_text().splitlines()]
    entries = []
    for row in rows:
        if row["capture_complete"]:
            continue
        directory = Path(row["artifact_dir"])
        source = directory / "rollout.json"
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        original = json.loads(source.read_text())
        recovered = copy.deepcopy(original)
        changes = [recover_turn(directory / "pi-events.jsonl", turn)
                   for turn in recovered["turns"] if not turn["complete"]]
        recovered["partial_recovery"] = {
            "source": "pi-events.jsonl", "normal_end_reconstructed": False}
        assert recovered["capture_complete"] is False
        assert recovered["success"] == original["success"]
        assert all(a == b for a, b in zip(original["turns"], recovered["turns"]) if a["complete"])
        for before, after in zip(original["turns"], recovered["turns"]):
            assert after["complete"] == before["complete"]
            assert after["tool_calls"][:len(before["tool_calls"])] == before["tool_calls"]
        output = directory / "rollout.recovered.json"
        output.write_text(json.dumps(recovered, ensure_ascii=False, indent=2) + "\n")
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
        entries.append({"finding_id": row["finding_id"], "rollout_path": str(output),
                        "original_sha256": source_hash,
                        "assistant_recovered": any(c["assistant_recovered"] for c in changes),
                        "pending_generated_tool_calls": sum(c["pending_generated_tool_calls"] for c in changes),
                        "capture_complete": False, "success": recovered["success"]})
    output = batch / "partial_recovery"
    output.mkdir(exist_ok=True)
    (output / "findings.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries))
    summary = {"processed": len(entries), "assistant_messages_recovered": sum(e["assistant_recovered"] for e in entries),
               "pending_generated_tool_calls_recovered": sum(e["pending_generated_tool_calls"] for e in entries),
               "repair_statuses": dict(Counter(row["repair_status"] for row in rows if not row["capture_complete"])),
               "original_rollouts_unchanged": True, "completed_turns_and_existing_tool_results_unchanged": True,
               "all_capture_complete_remain_false": True}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
