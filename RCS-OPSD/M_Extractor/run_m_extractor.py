#!/usr/bin/env python3
"""Extract one finding's vulnerability mechanism with local Qwen3.5-9B."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).with_name("M_Extractor.md")
MAPPING = ROOT / "data/finding_vulnerable_files.json"
DATASET_ROOT = ROOT / "dataset_root" / "findings"
OUTPUT_ROOT = Path(__file__).with_name("runs")
ENDPOINT = os.environ.get("RCS_OPSD_PI_ENDPOINT", "http://127.0.0.1:9531/v1").rstrip("/") + "/chat/completions"
MODEL = os.environ.get("RCS_OPSD_PI_MODEL", "qwen3.5-9b")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def parse_extraction(content: str) -> dict[str, object]:
    text = content.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[len("```json\n") : -len("\n```")]
    return json.loads(text)


def mapping_entry(finding_id: str) -> dict[str, object]:
    entries = json.loads(MAPPING.read_text())
    for entry in entries:
        if entry["finding_id"] == finding_id:
            return entry
    raise SystemExit(f"finding not present in vulnerable-file mapping: {finding_id}")


def build_prompt(finding_id: str) -> tuple[str, Path, list[Path]]:
    entry = mapping_entry(finding_id)
    audit_path = DATASET_ROOT / finding_id / "task" / "full_prompt.md"
    source_root = ROOT / str(entry["source_root"])
    source_paths = [
        source_root / str(file_entry["path"])
        for file_entry in entry["vulnerable_files"]
    ]
    vulnerable_code = "\n\n".join(
        f"===== BEGIN VULNERABLE FILE: {path.relative_to(source_root)} =====\n"
        f"{path.read_text()}\n"
        f"===== END VULNERABLE FILE: {path.relative_to(source_root)} ====="
        for path in source_paths
    )
    prompt = TEMPLATE.read_text().replace("{{AUDIT}}", audit_path.read_text()).replace(
        "{{VULNERABLE_CODE}}", vulnerable_code
    )
    return prompt, audit_path, source_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finding", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main() -> int:
    args = parse_args()
    output = (
        args.output_dir.resolve()
        if args.output_dir
        else OUTPUT_ROOT / f"{args.finding}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    prompt, audit_path, source_paths = build_prompt(args.finding)
    request_body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32768,
        "temperature": args.temperature,
        "seed": args.seed,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": True},
    }
    (output / "prompt.md").write_text(prompt)
    write_json(output / "request.json", request_body)

    started = time.monotonic()
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=args.timeout) as response:
        raw_response = json.loads(response.read().decode("utf-8"))
    write_json(output / "response.json", raw_response)

    content = raw_response["choices"][0]["message"]["content"]
    reasoning = raw_response["choices"][0]["message"].get("reasoning")
    extracted = parse_extraction(content)
    mechanism = extracted["mechanism"]
    if not isinstance(mechanism, str):
        raise ValueError("response mechanism is not a string")
    result = {
        "schema_version": "qwen-m-extractor-finding/v1",
        "status": "EXTRACTED",
        "finding_id": args.finding,
        "model": MODEL,
        "endpoint": ENDPOINT,
        "thinking_enabled": True,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_tokens": 32768,
        "audit_path": str(audit_path.relative_to(ROOT)),
        "vulnerable_files": [str(path.relative_to(ROOT)) for path in source_paths],
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "mechanism": mechanism,
        "usage": raw_response.get("usage"),
        "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0,
        "finish_reason": raw_response["choices"][0].get("finish_reason"),
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    write_json(output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
