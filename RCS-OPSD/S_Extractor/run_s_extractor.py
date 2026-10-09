#!/usr/bin/env python3
"""Build one finding's minimal causal site with Qwen and the project graph."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rcs_opsd.artifact_paths import GRAPH_ROOT, M_RESULTS
AUDIT_ANCHOR_TEMPLATE = Path(__file__).with_name("Audit_Anchor_Extractor.md")
S_TEMPLATE = Path(__file__).with_name("S_Extractor.md")
MAPPING = ROOT / "data/finding_vulnerable_files.json"
DATASET_ROOT = ROOT / "dataset_root" / "findings"
OUTPUT_ROOT = Path(__file__).with_name("runs")
ENDPOINT = os.environ.get("RCS_OPSD_PI_ENDPOINT", "http://127.0.0.1:9531/v1").rstrip("/") + "/chat/completions"
MODEL = os.environ.get("RCS_OPSD_PI_MODEL", "qwen3.5-9b")
EDGE_TYPES = {"CALL", "STATE_DEP"}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[len("```json\n") : -len("\n```")]
    return json.loads(text)


def call_qwen(
    prompt: str,
    artifact: Path,
    temperature: float,
    seed: int,
    timeout: int,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    artifact.mkdir(parents=True, exist_ok=True)
    request_body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32768,
        "temperature": temperature,
        "seed": seed,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": True},
    }
    (artifact / "prompt.md").write_text(prompt)
    write_json(artifact / "request.json", request_body)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.monotonic()

    def send(body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            method="POST",
        )
        with opener.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    raw = send(request_body)
    write_json(artifact / "response.json", raw)
    content = raw["choices"][0]["message"].get("content")
    if content is None:
        retry_body = {
            **request_body,
            "messages": [{
                "role": "user",
                "content": prompt + "\n\nYour previous attempt exhausted the output budget. Think briefly and return the requested JSON now.",
            }],
            "seed": seed + 1,
        }
        write_json(artifact / "retry_request.json", retry_body)
        raw = send(retry_body)
        write_json(artifact / "retry_response.json", raw)
        content = raw["choices"][0]["message"].get("content")
    if content is None:
        raise ValueError("Qwen returned null content after one recovery call")
    duration = round(time.monotonic() - started, 3)
    parsed = parse_json_content(content)
    write_json(artifact / "parsed.json", parsed)
    return parsed, raw, duration


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def lookup(values: list[dict[str, Any]], finding_id: str) -> dict[str, Any]:
    for value in values:
        if value["finding_id"] == finding_id:
            return value
    raise SystemExit(f"missing data for finding: {finding_id}")


def extract_patch_anchors(patch: str) -> list[dict[str, Any]]:
    anchors = []
    current_file: str | None = None
    for line in patch.splitlines():
        file_match = re.match(r"diff --git a/(.+) b/(.+)$", line)
        if file_match:
            current_file = file_match.group(1)
            continue
        hunk_match = re.match(r"@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", line)
        if current_file and hunk_match:
            start = int(hunk_match.group(1))
            count = int(hunk_match.group(2) or "1")
            anchors.append(
                {
                    "source": "reference_patch",
                    "file": current_file,
                    "contract": None,
                    "function": None,
                    "signature": None,
                    "line_start": start,
                    "line_end": start + max(count, 1) - 1,
                    "code_snippet": None,
                }
            )
    return anchors


def extract_vulnerable_file_anchors(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    anchors = []
    for vulnerable_file in mapping.get("vulnerable_files", []):
        path = vulnerable_file["path"] if isinstance(vulnerable_file, dict) else vulnerable_file
        anchors.append({"source": "finding_vulnerable_files", "file": path})
        if not isinstance(vulnerable_file, dict):
            continue
        for location in vulnerable_file.get("locations", []):
            line_match = re.search(r"#L?(\d+)(?:-(\d+))?$", location)
            head = location[: line_match.start()] if line_match else location
            file_part, separator, symbol = head.partition("::")
            anchor_file = file_part if file_part.endswith((".sol", ".rs")) else path
            function = symbol or None
            if function == "__gap":
                function = None
            anchors.append({
                "source": "finding_vulnerable_files_location",
                "file": anchor_file,
                "function": function,
                "line_start": int(line_match.group(1)) if line_match else None,
                "line_end": int(line_match.group(2) or line_match.group(1)) if line_match else None,
            })
    return anchors


def clean_function_name(value: str) -> str:
    name = value.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
    return name.split("(", 1)[0].strip()


def clean_signature(value: str) -> str:
    return re.sub(r"\s+", "", value.rsplit("::", 1)[-1].rsplit(".", 1)[-1])


def file_matches(anchor_file: str, node_file: str) -> bool:
    def variants(value: str) -> set[str]:
        normalized = value.replace("\\", "/").removeprefix("a/").removeprefix("b/")
        normalized = normalized.split("#", 1)[0]
        parts = normalized.split("/")
        values = {normalized}
        for marker in ("contracts", "src"):
            if marker in parts:
                values.add("/".join(parts[parts.index(marker) :]))
        return values

    return any(
        left == right or left.endswith("/" + right) or right.endswith("/" + left)
        for left in variants(anchor_file)
        for right in variants(node_file)
    )


def resolve_source_path(node_file: str, source_root: Path) -> Path:
    direct = source_root / node_file
    if direct.is_file():
        return direct
    matches = [
        path
        for path in source_root.rglob(Path(node_file).name)
        if path.is_file() and file_matches(node_file, path.relative_to(source_root).as_posix())
    ]
    preferred = [
        path for path in matches
        if not {"artifacts", "cache", "out"}.intersection(path.relative_to(source_root).parts)
    ]
    choices = preferred or matches
    if not choices:
        raise FileNotFoundError(direct)
    return min(choices, key=lambda path: (len(path.relative_to(source_root).parts), str(path)))


def solidity_version_satisfies(version: tuple[int, int, int], expression: str) -> bool:
    for alternative in expression.split("||"):
        valid = True
        for operator, value in re.findall(r"(\^|>=|<=|>|<|=)?\s*(\d+\.\d+\.\d+)", alternative):
            target = tuple(int(part) for part in value.split("."))
            if operator == "^":
                upper = (target[0] + 1, 0, 0) if target[0] else (0, target[1] + 1, 0)
                condition = target <= version < upper
            elif operator == ">=":
                condition = version >= target
            elif operator == "<=":
                condition = version <= target
            elif operator == ">":
                condition = version > target
            elif operator == "<":
                condition = version < target
            else:
                condition = version == target
            valid = valid and condition
        if valid:
            return True
    return False


def source_ast_nodes(file_reference: str, source_root: Path) -> list[dict[str, Any]]:
    source_path = resolve_source_path(file_reference, source_root)
    source = source_path.read_text()
    relative = source_path.relative_to(source_root).as_posix()
    pragma = re.search(r"pragma\s+solidity\s+([^;]+);", source)
    expression = pragma.group(1) if pragma else ">=0.0.0"
    artifacts = Path.home() / ".solc-select" / "artifacts"
    installed = []
    for path in artifacts.glob("solc-*/solc-*"):
        match = re.fullmatch(r"solc-(\d+)\.(\d+)\.(\d+)", path.name)
        if match:
            version = tuple(int(part) for part in match.groups())
            installed.append((version, path))
    if not installed:
        return []
    compilers = [item for item in installed if solidity_version_satisfies(item[0], expression)]
    compiler = max(compilers or installed)[1]
    parser_source = source
    if not compilers and pragma:
        parser_source = source[: pragma.start(1)] + ">=0.0.0" + source[pragma.end(1) :]
    request = {
        "language": "Solidity",
        "sources": {relative: {"content": parser_source}},
        "settings": {
            "stopAfter": "parsing",
            "outputSelection": {"*": {"": ["ast"]}},
        },
    }
    completed = subprocess.run(
        [str(compiler), "--standard-json"],
        input=json.dumps(request),
        capture_output=True,
        text=True,
    )
    parsed = json.loads(completed.stdout)
    ast = parsed.get("sources", {}).get(relative, {}).get("ast")
    if ast is None:
        return []

    def type_name(node: dict[str, Any]) -> str:
        kind = node["nodeType"]
        if kind == "ElementaryTypeName":
            return node["name"]
        if kind in {"UserDefinedTypeName", "IdentifierPath"}:
            return (
                node.get("namePath")
                or node.get("name")
                or node.get("pathNode", {}).get("name", "unknown")
            )
        if kind == "ArrayTypeName":
            length = node.get("length")
            suffix = "" if length is None else str(length.get("value", ""))
            return f"{type_name(node['baseType'])}[{suffix}]"
        if kind == "Mapping":
            return f"mapping({type_name(node['keyType'])}=>{type_name(node['valueType'])})"
        return kind

    nodes = []
    for contract in ast.get("nodes", []):
        if contract.get("nodeType") != "ContractDefinition":
            continue
        for function in contract.get("nodes", []):
            if function.get("nodeType") != "FunctionDefinition":
                continue
            kind = function.get("kind", "function")
            name = function.get("name") or kind
            parameters = ",".join(
                type_name(parameter["typeName"])
                for parameter in function["parameters"]["parameters"]
            )
            signature = f"{name}({parameters})"
            start, length, _ = (int(part) for part in function["src"].split(":"))
            nodes.append(
                {
                    "id": f"{relative}::{contract['name']}::{signature}",
                    "file": relative,
                    "contract": contract["name"],
                    "contract_kind": contract.get("contractKind", "contract"),
                    "function": name,
                    "signature": signature,
                    "visibility": function.get("visibility", "internal"),
                    "mutability": function.get("stateMutability", "nonpayable"),
                    "implemented": function.get("implemented", True),
                    "special": kind if kind != "function" else None,
                    "start_line": source.count("\n", 0, start) + 1,
                    "end_line": source.count("\n", 0, start + length) + 1,
                    "state_variables_read": [],
                    "state_variables_written": [],
                    "source": "solc_ast_fallback",
                }
            )
    return nodes


def source_for_node(node: dict[str, Any], source_root: Path) -> str:
    lines = resolve_source_path(node["file"], source_root).read_text().splitlines()
    return "\n".join(lines[node["start_line"] - 1 : node["end_line"]])


def canonicalize_anchor(
    anchor: dict[str, Any],
    nodes: list[dict[str, Any]],
    source_root: Path,
) -> list[str]:
    anchor_file = anchor.get("file")
    file_candidates = []
    if isinstance(anchor_file, str) and anchor_file:
        file_candidates = [node for node in nodes if file_matches(anchor_file, node["file"])]
    contract = anchor.get("contract")
    contract_candidates = []
    if isinstance(contract, str) and contract:
        contract_candidates = [node for node in nodes if node["contract"] == contract]
    function = anchor.get("function")
    signature = anchor.get("signature")
    function_candidates = []
    if isinstance(function, str) and function:
        name = clean_function_name(function)
        function_candidates = [node for node in nodes if node["function"] == name]
    elif isinstance(signature, str) and signature:
        normalized = clean_signature(signature)
        function_candidates = [
            node for node in nodes if clean_signature(node["signature"]) == normalized
        ]

    if file_candidates:
        candidates = file_candidates
        narrowed = [node for node in candidates if node in contract_candidates]
        if narrowed:
            candidates = narrowed
        narrowed = [node for node in candidates if node in function_candidates]
        if narrowed:
            candidates = narrowed
    elif contract_candidates:
        candidates = contract_candidates
        narrowed = [node for node in candidates if node in function_candidates]
        if narrowed:
            candidates = narrowed
    elif len(function_candidates) == 1:
        candidates = function_candidates
    else:
        candidates = []
    line_start = anchor.get("line_start")
    line_end = anchor.get("line_end")
    if isinstance(anchor_file, str) and isinstance(line_start, int):
        end = line_end if isinstance(line_end, int) else line_start
        line_candidates = [
            node
            for node in candidates
            if node["start_line"] <= end and node["end_line"] >= line_start
        ]
        if line_candidates:
            candidates = line_candidates
    snippet = anchor.get("code_snippet")
    if not any((anchor_file, contract, function, signature, line_start)) and isinstance(snippet, str):
        normalized = re.sub(r"\s+", " ", snippet).strip()
        candidates = [
            node
            for node in nodes
            if normalized in re.sub(r"\s+", " ", source_for_node(node, source_root))
        ]
    return [node["id"] for node in candidates]


def canonicalize_selected_function(
    selected: str,
    candidate_ids: set[str],
    nodes_by_id: dict[str, dict[str, Any]],
) -> str | None:
    if selected in candidate_ids:
        return selected
    parts = selected.split("::")
    contract = parts[-2] if len(parts) >= 3 else None
    function = clean_function_name(parts[-1])
    matches = [
        candidate_id
        for candidate_id in candidate_ids
        if nodes_by_id[candidate_id]["function"] == function
        and (contract is None or nodes_by_id[candidate_id]["contract"] == contract)
    ]
    if not matches:
        matches = [
            candidate_id
            for candidate_id in candidate_ids
            if nodes_by_id[candidate_id]["function"] == function
        ]
    return matches[0] if len(matches) == 1 else None


def render_candidates(
    candidate_ids: set[str],
    nodes_by_id: dict[str, dict[str, Any]],
    source_root: Path,
) -> str:
    blocks = []
    for index, candidate_id in enumerate(sorted(candidate_ids), 1):
        node = nodes_by_id[candidate_id]
        blocks.append(
            f"[{index}] {candidate_id}\n"
            f"File: {node['file']}\n"
            f"Lines: {node['start_line']}-{node['end_line']}\n\n"
            f"```solidity\n{source_for_node(node, source_root)}\n```"
        )
    return "\n\n".join(blocks)


def source_location_candidates(
    anchors: list[dict[str, Any]], source_root: Path
) -> list[dict[str, Any]]:
    locations = []
    seen = set()
    for anchor in anchors:
        file_reference = anchor.get("file")
        if not isinstance(file_reference, str) or not file_reference:
            continue
        try:
            path = resolve_source_path(file_reference, source_root)
        except FileNotFoundError:
            continue
        lines = path.read_text().splitlines()
        has_line = isinstance(anchor.get("line_start"), int)
        start = anchor["line_start"] if has_line else 1
        end = anchor.get("line_end") if isinstance(anchor.get("line_end"), int) else start
        if not has_line:
            end = len(lines)
        start = max(1, start)
        end = min(len(lines), max(start, end))
        if path.suffix == ".rs":
            for index in range(start - 1, -1, -1):
                if not re.match(r"\s*impl(?:<[^>]+>)?\s+", lines[index]):
                    continue
                balance = 0
                opened = False
                for block_end in range(index, len(lines)):
                    balance += lines[block_end].count("{") - lines[block_end].count("}")
                    opened = opened or "{" in lines[block_end]
                    if opened and balance == 0:
                        if block_end + 1 >= end:
                            start, end = index + 1, block_end + 1
                        break
                break
        relative = path.relative_to(source_root).as_posix()
        key = (relative, start, end)
        if key in seen:
            continue
        seen.add(key)
        locations.append({
            "file": relative,
            "line_start": start,
            "line_end": end,
            "language": "rust" if path.suffix == ".rs" else "solidity",
            "code": "\n".join(lines[start - 1 : end]),
        })
    return locations


def render_source_locations(locations: list[dict[str, Any]]) -> str:
    blocks = []
    for index, location in enumerate(locations, 1):
        blocks.append(
            f"[Non-function source location {index}]\n"
            "This block has no canonical function ID. Do not put its label or file path "
            "in the functions array; use functions: [] if these locations are sufficient.\n"
            f"File: {location['file']}\n"
            f"Lines: {location['line_start']}-{location['line_end']}\n\n"
            f"```{location['language']}\n{location['code']}\n```"
        )
    return "\n\n".join(blocks)


def expand_bidirectionally(
    candidate_ids: set[str], edges: list[dict[str, Any]], node_ids: set[str]
) -> tuple[set[str], list[dict[str, Any]]]:
    expanded = set(candidate_ids)
    traversed = []
    for edge in edges:
        if edge["type"] not in EDGE_TYPES:
            continue
        source = edge["source"]
        target = edge["target"]
        if source in candidate_ids and target in node_ids:
            expanded.add(target)
            traversed.append(edge)
        if target in candidate_ids and source in node_ids:
            expanded.add(source)
            traversed.append(edge)
    return expanded, traversed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finding", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--temperature", type=float, default=0.6)
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

    mapping = lookup(json.loads(MAPPING.read_text()), args.finding)
    m_result = lookup(load_jsonl(M_RESULTS), args.finding)
    audit_path = DATASET_ROOT / args.finding / "task" / "full_prompt.md"
    patch_path = DATASET_ROOT / args.finding / "task" / "reference_patch.diff"
    graph_path = GRAPH_ROOT / f"{mapping['project_key']}.json"
    source_root = ROOT / mapping["source_root"]
    audit = audit_path.read_text()
    mechanism = m_result["mechanism"]
    graph = json.loads(graph_path.read_text())
    nodes = graph["nodes"]
    nodes_by_id = {node["id"]: node for node in nodes}
    node_ids = set(nodes_by_id)

    audit_anchor_prompt = AUDIT_ANCHOR_TEMPLATE.read_text().replace("{{AUDIT}}", audit)
    audit_result, audit_raw, audit_duration = call_qwen(
        audit_anchor_prompt,
        output / "audit_anchors",
        args.temperature,
        args.seed,
        args.timeout,
    )
    audit_anchors = [
        {"source": "audit", **anchor} for anchor in audit_result["anchors"]
    ]
    patch_anchors = extract_patch_anchors(patch_path.read_text())
    write_json(output / "patch_anchors.json", patch_anchors)

    all_anchors = [*audit_anchors, *patch_anchors]
    vulnerable_anchors = extract_vulnerable_file_anchors(mapping)
    write_json(output / "vulnerable_file_anchors.json", vulnerable_anchors)
    candidate_anchors = [*all_anchors, *vulnerable_anchors]
    mappings = []
    initial_candidates: set[str] = set()
    primary_candidates: set[str] = set()
    for anchor in candidate_anchors:
        matched = canonicalize_anchor(anchor, nodes, source_root)
        mappings.append({"anchor": anchor, "canonical_function_ids": matched})
        initial_candidates.update(matched)
        if anchor in all_anchors:
            primary_candidates.update(matched)
    candidate_fallback = None
    if initial_candidates and not primary_candidates:
        candidate_fallback = "finding_vulnerable_files"
    source_ast_node_count = 0
    unmatched_anchors = [
        item["anchor"] for item in mappings if not item["canonical_function_ids"]
    ]
    ast_anchors = unmatched_anchors if initial_candidates else candidate_anchors
    references = [
        anchor["file"]
        for anchor in ast_anchors
        if isinstance(anchor.get("file"), str) and anchor["file"]
    ]
    if references:
        supplemental: dict[str, dict[str, Any]] = {}
        for reference in dict.fromkeys(references):
            try:
                for node in source_ast_nodes(reference, source_root):
                    supplemental[node["id"]] = node
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        supplemental = {
            node_id: node for node_id, node in supplemental.items() if node_id not in nodes_by_id
        }
        source_ast_node_count = len(supplemental)
        nodes.extend(supplemental.values())
        nodes_by_id.update(supplemental)
        node_ids.update(supplemental)
        ast_candidates = set()
        for anchor in ast_anchors:
            matched = canonicalize_anchor(anchor, list(supplemental.values()), source_root)
            mappings.append({
                "anchor": {**anchor, "source": "source_ast_" + anchor["source"]},
                "canonical_function_ids": matched,
            })
            initial_candidates.update(matched)
            ast_candidates.update(matched)
        if ast_candidates:
            candidate_fallback = "source_ast"
    context_anchors = [
        *all_anchors,
        *(anchor for anchor in vulnerable_anchors if anchor["source"].endswith("_location")),
    ]
    if not audit_anchors:
        context_anchors.extend(
            anchor for anchor in vulnerable_anchors
            if anchor["source"] == "finding_vulnerable_files"
        )
    source_locations = []
    if not initial_candidates:
        source_locations = source_location_candidates(context_anchors, source_root)
        if source_locations:
            candidate_fallback = "source_locations"
    write_json(output / "anchor_mapping.json", mappings)
    write_json(output / "source_location_candidates.json", source_locations)
    if not initial_candidates and not source_locations:
        raise SystemExit("no anchors mapped to canonical candidates")

    candidate_ids = set(initial_candidates)
    iterations = []
    iteration_number = 0
    final_selection: dict[str, Any] | None = None
    while True:
        candidate_text = render_candidates(candidate_ids, nodes_by_id, source_root)
        location_text = render_source_locations(source_locations)
        if location_text:
            candidate_text = "\n\n".join(part for part in (candidate_text, location_text) if part)
        s_prompt = (
            S_TEMPLATE.read_text()
            .replace("{{AUDIT}}", audit)
            .replace("{{MECHANISM}}", mechanism)
            .replace("{{CANDIDATES}}", candidate_text)
        )
        iteration_dir = output / "iterations" / f"{iteration_number:02d}"
        selection, raw, duration = call_qwen(
            s_prompt,
            iteration_dir,
            args.temperature,
            args.seed,
            args.timeout,
        )
        sufficient = selection.get("sufficient")
        if not isinstance(sufficient, bool):
            raise ValueError("S Extractor sufficient field is not boolean")
        selected_functions = selection.get("functions")
        if not sufficient and selected_functions is None:
            selected_functions = []
        if not isinstance(selected_functions, list):
            raise ValueError("S Extractor returned a function outside the canonical candidates")
        if source_locations and not candidate_ids:
            selected_functions = []
        canonical_functions = [
            canonicalize_selected_function(function_id, candidate_ids, nodes_by_id)
            for function_id in selected_functions
        ]
        if any(function_id is None for function_id in canonical_functions):
            raise ValueError("S Extractor returned a function outside the canonical candidates")
        selection["functions"] = canonical_functions
        if not sufficient:
            selection.setdefault("state_variables", [])
            selection.setdefault("operations", [])
        iteration_record = {
            "iteration": iteration_number,
            "candidate_count": len(candidate_ids),
            "candidate_ids": sorted(candidate_ids),
            "source_location_count": len(source_locations),
            "sufficient": sufficient,
            "selection": selection,
            "duration_seconds": duration,
            "usage": raw.get("usage"),
            "reasoning_chars": len(raw["choices"][0]["message"].get("reasoning") or ""),
        }
        iterations.append(iteration_record)
        if sufficient:
            final_selection = selection
            break
        expanded, traversed = expand_bidirectionally(candidate_ids, graph["edges"], node_ids)
        new_ids = expanded - candidate_ids
        iteration_record["expansion"] = {
            "direction": "bidirectional",
            "edge_types": sorted(EDGE_TYPES),
            "new_candidate_ids": sorted(new_ids),
            "traversed_edges": traversed,
        }
        write_json(iteration_dir / "expansion.json", iteration_record["expansion"])
        if not new_ids:
            if not source_locations:
                source_locations = source_location_candidates(context_anchors, source_root)
                if source_locations:
                    candidate_fallback = "anchored_source_context"
                    write_json(output / "source_location_candidates.json", source_locations)
                    iteration_record["source_context_expansion"] = {
                        "source_location_count": len(source_locations),
                        "files": sorted({location["file"] for location in source_locations}),
                    }
                    write_json(
                        iteration_dir / "source_context_expansion.json",
                        iteration_record["source_context_expansion"],
                    )
                    iteration_number += 1
                    continue
            break
        candidate_ids = expanded
        iteration_number += 1

    result = {
        "schema_version": "qwen-s-extractor-finding/v1",
        "status": "EXTRACTED" if final_selection is not None else "INSUFFICIENT_NO_EXPANSION",
        "finding_id": args.finding,
        "project_key": mapping["project_key"],
        "model": MODEL,
        "endpoint": ENDPOINT,
        "thinking_enabled": True,
        "temperature": args.temperature,
        "seed": args.seed,
        "mechanism": mechanism,
        "audit_anchor_count": len(audit_anchors),
        "patch_anchor_count": len(patch_anchors),
        "initial_candidate_count": len(initial_candidates),
        "initial_candidate_ids": sorted(initial_candidates),
        "candidate_fallback": candidate_fallback,
        "source_ast_node_count": source_ast_node_count,
        "source_location_count": len(source_locations),
        "iterations": iterations,
        "selection": final_selection,
        "audit_anchor_duration_seconds": audit_duration,
        "audit_anchor_usage": audit_raw.get("usage"),
        "audit_path": str(audit_path.relative_to(ROOT)),
        "reference_patch_path": str(patch_path.relative_to(ROOT)),
        "project_graph_path": str(graph_path.relative_to(ROOT)),
        "m_results_path": str(M_RESULTS.relative_to(ROOT)),
        "s_template_sha256": hashlib.sha256(S_TEMPLATE.read_bytes()).hexdigest(),
        "audit_anchor_template_sha256": hashlib.sha256(AUDIT_ANCHOR_TEMPLATE.read_bytes()).hexdigest(),
    }
    write_json(output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
