#!/usr/bin/env python3
"""Build the repair region from patch functions and their one-hop neighbors."""
from __future__ import annotations

import argparse
import json
import re
from collections import deque
import sys
from pathlib import Path

from run_s_extractor import file_matches


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rcs_opsd.artifact_paths import GRAPH_ROOT, S_RESULTS
FINDINGS = ROOT / "dataset_root/findings"
EDGE_TYPES = {"CALL", "STATE_DEP"}


def changed_lines(patch: str) -> dict[str, set[int]]:
    changes: dict[str, set[int]] = {}
    file = None
    old_line = None
    old_remaining = new_remaining = 0
    for line in patch.splitlines():
        match = re.match(r"diff --git a/(.+) b/(.+)$", line)
        if match:
            file = match.group(1)
            old_line = None
            old_remaining = new_remaining = 0
            continue
        if line.startswith("--- ") and old_remaining == new_remaining == 0:
            file = line[4:].split("\t", 1)[0].removeprefix("a/")
            old_line = None
            continue
        if line.startswith("+++ ") and old_remaining == new_remaining == 0:
            if file == "/dev/null":
                file = line[4:].split("\t", 1)[0].removeprefix("b/")
            continue
        match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if match:
            old_line = int(match.group(1))
            old_remaining = int(match.group(2) or 1)
            new_remaining = int(match.group(4) or 1)
            continue
        if file is None or old_line is None or old_remaining == new_remaining == 0:
            continue
        if line.startswith("-"):
            changes.setdefault(file, set()).add(old_line)
            old_line += 1
            old_remaining -= 1
        elif line.startswith("+"):
            changes.setdefault(file, set()).add(max(1, old_line))
            new_remaining -= 1
        elif line.startswith(" "):
            old_line += 1
            old_remaining -= 1
            new_remaining -= 1
    return changes


def build_site(finding_id: str) -> dict:
    project = finding_id.rsplit("__", 1)[0]
    graph = json.loads((GRAPH_ROOT / f"{project}.json").read_text())
    patch = (FINDINGS / finding_id / "task/reference_patch.diff").read_text()
    changes = changed_lines(patch)
    patch_functions = sorted({
        node["id"]
        for node in graph["nodes"]
        for file, lines in changes.items()
        if file_matches(file, node["file"])
        and node["start_line"] is not None and node["end_line"] is not None
        and any(node["start_line"] <= line <= node["end_line"] for line in lines)
    })
    anchors = patch_functions
    anchor_source = "patch_changed_lines"
    # Inheritance edits, added functions, and missing patch-function nodes have
    # no original function overlap. Use the existing semantic site, not an
    # arbitrary nearby line or every function in the patched file.
    if not anchors:
        canonical_ids = {node["id"] for node in graph["nodes"]}
        with (S_RESULTS).open() as stream:
            for line in stream:
                record = json.loads(line)
                if record["finding_id"] == finding_id:
                    anchors = sorted(canonical_ids.intersection(record["selection"]["functions"]))
                    break
        anchor_source = "static_causal_site_without_patch_function_overlap"
    adjacent: dict[str, set[str]] = {node["id"]: set() for node in graph["nodes"]}
    for edge in graph["edges"]:
        if edge["type"] in EDGE_TYPES:
            adjacent[edge["source"]].add(edge["target"])
            adjacent[edge["target"]].add(edge["source"])
    core = sorted(set(anchors).union(
        *(adjacent[node] for node in anchors)
    ))
    visited = set(core)
    edges = [
        {"source": edge["source"], "target": edge["target"], "type": edge["type"]}
        for edge in graph["edges"]
        if edge["type"] in EDGE_TYPES
        and edge["source"] in visited and edge["target"] in visited
    ]
    return {
        "finding_id": finding_id,
        "project": project,
        "patch_functions": patch_functions,
        "anchor_functions": anchors,
        "anchor_source": anchor_source,
        "core_functions": core,
        "core_definition": "anchor_functions_plus_one_hop_CALL_STATE_DEP",
        "target_subgraph": {"nodes": sorted(visited), "edges": edges},
        "patch_changed_files": sorted(changes),
    }


def build_distance_index(site: dict) -> dict | None:
    if not site["core_functions"]:
        return None
    graph = json.loads((GRAPH_ROOT / f"{site['project']}.json").read_text())
    neighbors: dict[str, set[str]] = {node["id"]: set() for node in graph["nodes"]}
    for edge in graph["edges"]:
        if edge["type"] in EDGE_TYPES:
            neighbors[edge["source"]].add(edge["target"])
            neighbors[edge["target"]].add(edge["source"])
    nodes = {node: {"distance": None, "next": None} for node in neighbors}
    queue = deque(sorted(site["core_functions"]))
    for node in queue:
        nodes[node]["distance"] = 0
    while queue:
        current = queue.popleft()
        for neighbor in sorted(neighbors[current]):
            if nodes[neighbor]["distance"] is None:
                nodes[neighbor] = {
                    "distance": nodes[current]["distance"] + 1,
                    "next": current,
                }
                queue.append(neighbor)
    return {"project": site["project"], "targets": site["core_functions"], "nodes": nodes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("finding_id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_site(args.finding_id), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
