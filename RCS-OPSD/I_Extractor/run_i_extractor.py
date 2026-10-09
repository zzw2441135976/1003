#!/usr/bin/env python3
"""Extract one finding's patch-independent repair invariant."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import os
from pathlib import Path
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rcs_opsd.artifact_paths import M_RESULTS, S_RESULTS
TEMPLATE = Path(__file__).with_name("I_Extractor.md")
DATASET_ROOT = ROOT / "dataset_root/findings"
MAPPING = ROOT / "data/finding_vulnerable_files.json"
OUTPUT_ROOT = Path(__file__).with_name("runs")
ENDPOINT = os.environ.get("RCS_OPSD_PI_ENDPOINT", "http://127.0.0.1:9531/v1").rstrip("/") + "/chat/completions"
MODEL = os.environ.get("RCS_OPSD_PI_MODEL", "qwen3.5-9b")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def load_record(path: Path, finding_id: str) -> dict[str, object]:
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("finding_id") == finding_id:
                return record
    raise SystemExit(f"finding not present in {path}: {finding_id}")


def reconstruct_diff_versions(diff: str) -> tuple[str, str]:
    """Return patch-local old/new code while preserving file and hunk boundaries."""
    vulnerable: list[str] = []
    fixed: list[str] = []
    current_file: str | None = None
    in_hunk = False
    old_remaining = new_remaining = 0

    for line in diff.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            current_file = parts[3][2:] if len(parts) >= 4 else line
            in_hunk = False
        elif line.startswith("--- ") and not in_hunk:
            current_file = line[4:].split("\t", 1)[0].removeprefix("a/")
            in_hunk = False
        elif line.startswith("+++ ") and not in_hunk:
            if current_file == "/dev/null":
                current_file = line[4:].split("\t", 1)[0].removeprefix("b/")
            in_hunk = False
        elif line.startswith("@@ "):
            match = re.match(r"@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@", line)
            old_remaining = int(match.group(1) or 1)
            new_remaining = int(match.group(2) or 1)
            label = f"===== FILE: {current_file or '<unknown>'} | {line} ====="
            vulnerable.append(label)
            fixed.append(label)
            in_hunk = True
        elif in_hunk and line.startswith(" "):
            vulnerable.append(line[1:])
            fixed.append(line[1:])
            old_remaining -= 1
            new_remaining -= 1
        elif in_hunk and line.startswith("-"):
            vulnerable.append(line[1:])
            old_remaining -= 1
        elif in_hunk and line.startswith("+"):
            fixed.append(line[1:])
            new_remaining -= 1
        elif in_hunk and line.startswith("\\ No newline at end of file"):
            continue
        if old_remaining == new_remaining == 0:
            in_hunk = False

    if not vulnerable or not fixed:
        raise ValueError("reference patch contains no reconstructable unified-diff hunks")
    return "\n".join(vulnerable), "\n".join(fixed)


def parse_extraction(content: str) -> dict[str, object]:
    text = content.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[len("```json\n") : -len("\n```")]
    parsed = json.loads(text)
    invariant = parsed.get("invariant")
    if not isinstance(invariant, str) or not invariant.strip():
        raise ValueError("response invariant is not a non-empty string")
    return {"invariant": invariant.strip()}


def build_prompt(finding_id: str) -> tuple[str, Path, str, dict[str, object]]:
    m_record = load_record(M_RESULTS, finding_id)
    s_record = load_record(S_RESULTS, finding_id)
    patch_path = DATASET_ROOT / finding_id / "task/reference_patch.diff"
    patch = patch_path.read_text()
    vulnerable_code, fixed_code = reconstruct_diff_versions(patch)
    causal_site = s_record["selection"]
    audit = (DATASET_ROOT / finding_id / "task/full_prompt.md").read_text()
    entry = next(
        item for item in json.loads(MAPPING.read_text()) if item["finding_id"] == finding_id
    )
    source_root = ROOT / entry["source_root"]
    vulnerable_context = "\n\n".join(
        f"===== VULNERABLE FILE: {file['path']} =====\n"
        + (source_root / file["path"]).read_text()
        for file in entry["vulnerable_files"]
    )

    prompt = TEMPLATE.read_text()
    replacements = {
        "{{AUDIT}}": audit,
        "{{MECHANISM}}": str(m_record["mechanism"]),
        "{{CAUSAL_SITE}}": json.dumps(causal_site, ensure_ascii=False, indent=2),
        "{{VULNERABLE_CONTEXT}}": vulnerable_context,
        "{{VULNERABLE_CODE}}": vulnerable_code,
        "{{FIXED_CODE}}": fixed_code,
        "{{PATCH_DIFF}}": patch,
    }
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    inputs = {
        "audit": audit,
        "mechanism": m_record["mechanism"],
        "causal_site": causal_site,
        "vulnerable_context": vulnerable_context,
        "vulnerable_code": vulnerable_code,
        "fixed_code": fixed_code,
        "patch_diff": patch,
    }
    return prompt, patch_path, patch, inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finding", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--timeout", type=int, default=1800)
    return parser.parse_args()


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

    prompt, patch_path, patch, inputs = build_prompt(args.finding)
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
    write_json(output / "inputs.json", inputs)
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

    message = raw_response["choices"][0]["message"]
    extracted = parse_extraction(message["content"])
    reasoning = message.get("reasoning")
    result = {
        "schema_version": "qwen-i-extractor-finding/v1",
        "status": "EXTRACTED",
        "finding_id": args.finding,
        "model": MODEL,
        "endpoint": ENDPOINT,
        "thinking_enabled": True,
        "temperature": args.temperature,
        "seed": args.seed,
        "patch_path": str(patch_path.relative_to(ROOT)),
        "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "invariant": extracted["invariant"],
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
