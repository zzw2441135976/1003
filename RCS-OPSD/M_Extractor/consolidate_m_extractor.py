#!/usr/bin/env python3
"""Consolidate M extractions after recovering failed JSON parsing."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

from run_m_extractor import parse_extraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    batch = parse_args().batch_dir.resolve()
    original_journal = batch / "findings.jsonl"
    entries = [json.loads(line) for line in original_journal.read_text().splitlines() if line]
    retry_results = {
        result["finding_id"]: (path, result)
        for path in (batch / "recovery").glob("*/result.json")
        for result in [json.loads(path.read_text())]
    }
    methods: Counter[str] = Counter()
    complete = []

    for entry in entries:
        if entry["status"] == "EXTRACTED":
            methods["ORIGINAL"] += 1
            complete.append(entry)
            continue

        finding_id = entry["finding_id"]
        response_path = Path(entry["artifact_dir"]) / "response.json"
        response = json.loads(response_path.read_text())
        message = response["choices"][0]["message"]
        content = message["content"]
        recovered = {key: value for key, value in entry.items() if key != "error"}
        recovered["original_status"] = entry["status"]

        if isinstance(content, str):
            mechanism = parse_extraction(content)["mechanism"]
            recovered.update(
                {
                    "status": "EXTRACTED",
                    "mechanism": mechanism,
                    "usage": response.get("usage"),
                    "reasoning_chars": len(message.get("reasoning") or ""),
                    "finish_reason": response["choices"][0].get("finish_reason"),
                    "recovery": {
                        "method": "PARSE_JSON_CODE_FENCE",
                        "response_path": str(response_path),
                    },
                }
            )
            methods["PARSE_JSON_CODE_FENCE"] += 1
        else:
            retry_path, retry = retry_results[finding_id]
            recovered.update(
                {
                    "status": "EXTRACTED",
                    "mechanism": retry["mechanism"],
                    "artifact_dir": str(retry_path.parent),
                    "duration_seconds": retry["duration_seconds"],
                    "usage": retry.get("usage"),
                    "reasoning_chars": retry.get("reasoning_chars", 0),
                    "finish_reason": retry.get("finish_reason"),
                    "recovery": {
                        "method": "RERUN",
                        "seed": retry["seed"],
                        "original_artifact_dir": entry["artifact_dir"],
                    },
                }
            )
            methods["RERUN"] += 1
        complete.append(recovered)

    complete.sort(key=lambda item: item["finding_id"])
    if len({item["finding_id"] for item in complete}) != len(complete):
        raise SystemExit("consolidated finding set contains duplicates")
    if any(item["status"] != "EXTRACTED" for item in complete):
        raise SystemExit("consolidated journal still contains incomplete entries")

    output_journal = batch / "complete_findings.jsonl"
    output_journal.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in complete)
    )
    summary = {
        "schema_version": "qwen-m-extractor-complete/v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETED",
        "findings": len(complete),
        "unique_findings": len({item["finding_id"] for item in complete}),
        "category_counts": dict(sorted(Counter(item["category"] for item in complete).items())),
        "recovery_counts": dict(sorted(methods.items())),
        "original_journal": str(original_journal),
        "original_journal_sha256": hashlib.sha256(original_journal.read_bytes()).hexdigest(),
        "complete_journal": str(output_journal),
        "complete_journal_sha256": hashlib.sha256(output_journal.read_bytes()).hexdigest(),
    }
    (batch / "complete_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
