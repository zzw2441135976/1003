#!/usr/bin/env python3
"""Merge the successful S Extractor batches into one canonical cohort file."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rcs_opsd.artifact_paths import S_ROOT, S_RESULTS


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, action="append")
    args = parser.parse_args()
    batches = args.batch or [S_ROOT / "findings.jsonl"]
    merged = {}
    for batch in batches:
        for entry in load(batch):
            merged[entry["finding_id"]] = {
                **entry,
                "consolidated_from": str(batch.relative_to(ROOT)),
            }
    entries = [merged[finding_id] for finding_id in sorted(merged)]
    statuses = Counter(entry["status"] for entry in entries)
    if any(entry["status"] != "EXTRACTED" for entry in entries):
        raise SystemExit(f"non-complete statuses remain: {dict(statuses)}")
    categories = Counter(entry["category"] for entry in entries)
    S_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    S_RESULTS.write_text("".join(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n" for entry in entries))
    S_ROOT.joinpath("complete_summary.json").write_text(json.dumps({
        "schema_version": "qwen-s-extractor-complete/v1",
        "total": len(entries),
        "status_counts": dict(statuses),
        "category_counts": dict(categories),
        "source_batches": [str(path.relative_to(ROOT)) for path in batches],
        "output": str(S_RESULTS.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"complete={len(entries)} output={S_RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
