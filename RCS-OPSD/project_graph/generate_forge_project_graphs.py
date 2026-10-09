#!/usr/bin/env python3
"""Build source-grounded project graphs from nested forge finding capsules."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from generate_all_project_graphs import (
    atomic_write_json,
    discover_build_info,
    materialize_build_info_sources,
)


EXCLUDED_PARTS = {
    "lib",
    "mock",
    "mocks",
    "node_modules",
    "script",
    "scripts",
    "test",
    "tests",
}
PRINT_LOCK = Lock()
INDEPENDENT_UNITS = {
    "forge_2024-12-balancer-v3-securityreview": {
        "source": "a24ebf0141e9350a42639d8593c1436241deae59/balancer-v3-monorepo",
        "destination": ".",
        "source_dir": "pkg/vault/contracts",
        "target": "pkg/vault/contracts/Vault.sol",
        "solc": "0.8.24",
        "dependency": "project_assets/forge_ChainSecurity_StudioV_Neulock_Audit/sandbox/forge_ChainSecurity_StudioV_Neulock_Audit__0/capsule/lib/openzeppelin-contracts",
        "dependency_destination": "lib/openzeppelin-contracts",
        "remappings": [
            "@balancer-labs/v3-interfaces/=pkg/interfaces/",
            "@balancer-labs/v3-solidity-utils/=pkg/solidity-utils/",
            "@openzeppelin/=lib/openzeppelin-contracts/",
        ],
    },
    "forge_ChainSecurity_StudioV_Neulock_Audit": {
        "source": "neulock-onchain/contracts",
        "destination": "src",
        "source_dir": "src",
        "target": None,
        "solc": "0.8.28",
        "dependency": "project_assets/forge_ChainSecurity_StudioV_Neulock_Audit/sandbox/forge_ChainSecurity_StudioV_Neulock_Audit__0/capsule/lib",
        "dependency_destination": "lib",
        "remappings": [
            "@openzeppelin/contracts-upgradeable/=lib/openzeppelin-contracts-upgradeable/contracts/",
            "@openzeppelin/contracts/=lib/openzeppelin-contracts/contracts/",
        ],
    },
    "forge_USDT0_Polygon_Integration_Audit": {
        "source": "contracts",
        "destination": ".",
        "source_dir": "contracts",
        "target": "contracts/SpokePool.sol",
        "solc": "0.8.23",
        "dependency": "../data/graph_dependencies/blackhole-openzeppelin",
        "dependency_destination": "lib/openzeppelin",
        "remappings": [
            "@openzeppelin/contracts-upgradeable/=lib/openzeppelin/contracts-upgradeable/",
            "@openzeppelin/contracts/=lib/openzeppelin/contracts/",
        ],
    },
}

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rcs_opsd.artifact_paths import GRAPH_BASE_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset_root"))
    parser.add_argument(
        "--output-dir", type=Path, default=GRAPH_BASE_ROOT
    )
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--project", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_index(root: Path) -> tuple[dict[str, list[str]], int]:
    index: dict[str, list[str]] = defaultdict(list)
    count = 0
    for path in root.rglob("*.sol"):
        relative = path.relative_to(root)
        if source_file_excluded(relative):
            continue
        index[digest(path)].append(relative.as_posix())
        count += 1
    return index, count


def source_file_excluded(relative: Path) -> bool:
    for index, part in enumerate(relative.parts[:-1]):
        lowered = part.lower()
        if lowered == "lib" and relative.parts[index + 1] in {"openzeppelin-contracts", "openzeppelin-contracts-upgradeable", "v3-core", "v3-periphery", "forge-std"}:
            return True
        if lowered == "lib" and index > 0 and relative.parts[index - 1] in {"src", "contracts"}:
            continue
        if lowered in EXCLUDED_PARTS:
            return True
    return False


def preferred_source_files(finding_task: Path) -> list[str]:
    location = finding_task / "location.md"
    if not location.is_file():
        return []
    preferred: list[str] = []
    in_files = False
    for line in location.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "# Files":
            in_files = True
            continue
        if in_files and stripped.startswith("#"):
            break
        if in_files and stripped.startswith("-"):
            value = stripped[1:].strip().strip("`")
            if value.endswith(".sol"):
                preferred.append(value)
    capsule = finding_task / "capsule"
    for script in capsule.rglob("*.sh") if capsule.is_dir() else []:
        body = script.read_text(encoding="utf-8")
        preferred.extend(
            match.rstrip("/.")
            for match in re.findall(r"source/([A-Za-z0-9_./-]+)", body)
        )
    return preferred


def suffix_score(left: str, right: str) -> int:
    score = 0
    for a, b in zip(reversed(Path(left).parts), reversed(Path(right).parts)):
        if a != b:
            break
        score += 1
    return score


def preference_score(candidate: str, preferred: list[str]) -> int:
    candidate_parts = Path(candidate).parts
    best = 0
    for value in preferred:
        value_parts = Path(value).parts
        if candidate == value or candidate.endswith("/" + value):
            best = max(best, 1000 + len(value_parts))
            continue
        shared = 0
        for left, right in zip(candidate_parts, value_parts):
            if left != right:
                break
            shared += 1
        best = max(best, shared)
    return best


def choose_candidate(
    filename: str, candidates: list[str], preferred: list[str]
) -> str | None:
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    ranked = sorted(
        (
            (preference_score(candidate, preferred), suffix_score(filename, candidate)),
            candidate,
        )
        for candidate in candidates
    )
    if ranked[-1][0] > ranked[-2][0]:
        return ranked[-1][1]
    return None


def original_file(
    compiled_root: Path,
    filename: str,
    index: dict[str, list[str]],
    preferred: list[str],
) -> str | None:
    path = compiled_root / filename
    if not path.is_file():
        return None
    return choose_candidate(filename, index.get(digest(path), []), preferred)


def node_origin(
    node: dict[str, Any],
    compiled_root: Path,
    source_root: Path,
    index: dict[str, list[str]],
    preferred: list[str],
) -> tuple[str, str | None, int, int] | None:
    exact = original_file(compiled_root, node["file"], index, preferred)
    if exact is not None:
        return exact, node.get("contract"), node["start_line"], node["end_line"]

    compiled_path = compiled_root / node["file"]
    start = node.get("start_line")
    end = node.get("end_line")
    if not compiled_path.is_file() or not start or not end:
        return None
    compiled_lines = compiled_path.read_text(encoding="utf-8").splitlines(keepends=True)
    snippet = "".join(compiled_lines[start - 1 : end])
    if len(snippet.strip()) < 12:
        return None

    paths = set(source_root.rglob(compiled_path.name))
    for value in preferred:
        direct = source_root / value
        if direct.is_file():
            paths.add(direct)
        else:
            paths.update(path for path in source_root.rglob("*.sol") if path.as_posix().endswith("/" + value))

    candidates = []
    body_candidates = []
    body_offset = snippet.find("{")
    body = snippet[body_offset:] if body_offset >= 0 else ""
    for path in paths:
        relative = path.relative_to(source_root)
        if source_file_excluded(relative):
            continue
        content = path.read_text(encoding="utf-8")
        contract = node.get("contract")
        same_contract = not contract or contract == "<top-level>" or re.search(
            rf"\b(?:contract|library|interface)\s+{re.escape(contract)}\b", content
        )
        if same_contract:
            positions = [match.start() for match in re.finditer(re.escape(snippet), content)]
            if len(positions) == 1:
                actual_start = content.count("\n", 0, positions[0]) + 1
                actual_end = actual_start + len(snippet.splitlines()) - 1
                candidates.append(
                    (relative.as_posix(), contract, actual_start, actual_end, 1.0)
                )
                continue
            original_lines = content.splitlines(keepends=True)
            original_snippet = "".join(original_lines[start - 1 : end])
            similarity = difflib.SequenceMatcher(None, snippet, original_snippet).ratio()
            if similarity >= 0.95 and node.get("function", "") in original_snippet:
                candidates.append(
                    (relative.as_posix(), contract, start, end, similarity)
                )
                continue

        if len(body.strip()) < 40 or not any(
            relative.as_posix() == value
            or relative.as_posix().endswith("/" + value)
            for value in preferred
        ):
            continue
        positions = [match.start() for match in re.finditer(re.escape(body), content)]
        if len(positions) != 1:
            continue
        function_start = content.rfind("function", 0, positions[0])
        if function_start < 0 or positions[0] - function_start > 800:
            continue
        declarations = list(
            re.finditer(
                r"\b(?:contract|library|interface)\s+([A-Za-z_][A-Za-z0-9_]*)",
                content[:function_start],
            )
        )
        if not declarations:
            continue
        original_contract = declarations[-1].group(1)
        actual_start = content.count("\n", 0, function_start) + 1
        matched_function = content[function_start : positions[0] + len(body)]
        actual_end = actual_start + len(matched_function.splitlines()) - 1
        body_candidates.append(
            (relative.as_posix(), original_contract, actual_start, actual_end, 1.0)
        )

    candidates = candidates or body_candidates
    if not candidates:
        return None
    ranked = sorted(
        (
            (
                preference_score(candidate[0], preferred),
                suffix_score(node["file"], candidate[0]),
                candidate[4],
            ),
            candidate,
        )
        for candidate in candidates
    )
    if len(ranked) > 1 and ranked[-1][0] == ranked[-2][0]:
        return None
    filename, contract, actual_start, actual_end, _ = ranked[-1][1]
    return filename, contract, actual_start, actual_end


def ground_fragment(
    fragment: dict[str, Any],
    compiled_root: Path,
    source_root: Path,
    index: dict[str, list[str]],
    finding_id: str,
    preferred: list[str],
) -> tuple[dict[str, Any], int]:
    id_map: dict[str, str] = {}
    locations: dict[str, tuple[str, str, int]] = {}
    context_map: dict[str, str] = {}
    nodes: list[dict[str, Any]] = []
    ungrounded = 0
    for node in fragment["nodes"]:
        origin = node_origin(node, compiled_root, source_root, index, preferred)
        if origin is None:
            ungrounded += 1
            continue
        filename, original_contract, actual_start, actual_end = origin
        old_id = node["id"]
        compiled_file = node["file"]
        compiled_contract = node.get("contract")
        line_delta = actual_start - node["start_line"]
        node = dict(node)
        node["file"] = filename
        signature = old_id.rsplit("::", 1)[1]
        node["contract"] = original_contract
        node["id"] = f"{filename}::{original_contract or '<top-level>'}::{signature}"
        node["start_line"] = actual_start
        node["end_line"] = actual_end
        if compiled_contract:
            context_map[f"{compiled_file}::{compiled_contract}"] = f"{filename}::{original_contract}"
        if compiled_contract and original_contract != compiled_contract:
            for key in ("state_variables_read", "state_variables_written"):
                node[key] = [
                    variable.replace(
                        f"{compiled_contract}.", f"{original_contract}.", 1
                    )
                    for variable in node.get(key, [])
                ]
        node["finding_ids"] = [finding_id]
        id_map[old_id] = node["id"]
        locations[old_id] = (compiled_file, filename, line_delta)
        nodes.append(node)

    for node in nodes:
        node["storage_contexts"] = sorted({
            context_map[context] for context in node.get("storage_contexts", [])
            if context in context_map
        })

    edges: list[dict[str, Any]] = []
    for edge in fragment["edges"]:
        if edge["source"] not in id_map or edge["target"] not in id_map:
            continue
        old_source = edge["source"]
        edge = dict(edge)
        edge["source"] = id_map[edge["source"]]
        edge["target"] = id_map[edge["target"]]
        edge["finding_ids"] = [finding_id]
        edge["storage_contexts"] = sorted({
            context_map[context] for context in edge.get("storage_contexts", [])
            if context in context_map
        })
        sites = []
        for site in edge.get("call_sites", []):
            compiled_file, filename, line_delta = locations[old_source]
            if site["file"] == compiled_file:
                sites.append({"file": filename, "line": site["line"] + line_delta})
        if "call_sites" in edge:
            edge["call_sites"] = sites
        edges.append(edge)

    unresolved = []
    for call in fragment["unresolved_calls"]:
        if call["source"] not in id_map:
            continue
        old_source = call["source"]
        call = dict(call)
        call["source"] = id_map[call["source"]]
        call["finding_id"] = finding_id
        site = call.get("site")
        if site:
            compiled_file, filename, line_delta = locations[old_source]
            call["site"] = (
                {"file": filename, "line": site["line"] + line_delta}
                if site["file"] == compiled_file
                else None
            )
        unresolved.append(call)
    return {"nodes": nodes, "edges": edges, "unresolved_calls": unresolved}, ungrounded


def prepare_task(
    capsule: Path, finding_task: Path, source: Path, destination: Path
) -> None:
    destination.mkdir(parents=True)
    shutil.copytree(capsule, destination / "capsule", symlinks=True)
    for name in ("functionality", "security"):
        supporting_directory = finding_task / name
        if supporting_directory.is_dir():
            shutil.copytree(supporting_directory, destination / name, symlinks=True)
    for path in destination.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o200)
    (destination / "source").symlink_to(source, target_is_directory=True)

    # Capsule validators commonly delete their temporary build root on EXIT. The
    # copied script keeps it so its build-info remains available to Slither.
    for script in (destination / "capsule").rglob("*.sh"):
        lines = script.read_text(encoding="utf-8").splitlines()
        body = (
            "\n".join(
                ":" if line.lstrip().startswith("trap ") else line for line in lines
            )
            + "\n"
        )
        script.write_text(
            re.sub(
                r"mktemp -d /tmp/([^\s)\"']+)",
                r'mktemp -d "${TMPDIR}/\1"',
                body,
            ),
            encoding="utf-8",
        )


def compile_capsule(
    task: Path, generator: Path, project_id: str, finding_id: str, timeout: int
) -> tuple[list[tuple[dict[str, Any], Path]], str | None]:
    temporary_root = task / ".tmp"
    workspace = task / ".forge-workspace"
    temporary_root.mkdir()
    workspace.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "FOUNDRY_BUILD_INFO": "true",
            "FORGE_WORKSPACE": str(workspace),
            "HARDHAT_DISABLE_TELEMETRY": "1",
            "TMPDIR": str(temporary_root),
        }
    )
    try:
        compiled = subprocess.run(
            ["bash", "capsule/compile.sh"],
            cwd=task,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        )
    except subprocess.TimeoutExpired:
        return [], f"compile timed out after {timeout}s"
    if compiled.returncode != 0:
        details = (compiled.stderr or compiled.stdout).strip().splitlines()
        return [], "compile failed: " + " | ".join(details[-6:])

    build_infos = discover_build_info(task)
    if not build_infos:
        return [], "compile produced no build-info"
    fragments: list[tuple[dict[str, Any], Path]] = []
    for number, build_info in enumerate(build_infos):
        valid_artifacts = 0
        for artifact in build_info.glob("*.json"):
            loaded = json.loads(artifact.read_text(encoding="utf-8"))
            if isinstance(loaded.get("input"), dict) and isinstance(
                loaded.get("output"), dict
            ):
                valid_artifacts += 1
            else:
                artifact.rename(artifact.with_suffix(".ignored"))
        if not valid_artifacts:
            continue
        compiled_root = build_info.parent.parent
        materialize_build_info_sources(build_info, compiled_root)
        output = task / f"fragment-{number}.json"
        command = [
            sys.executable,
            str(generator),
            str(compiled_root),
            str(output),
            "--project",
            project_id,
            "--source-dir",
            ".",
            "--build-info",
            str(build_info),
        ]
        try:
            analyzed = subprocess.run(
                command,
                cwd=compiled_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=environment,
            )
        except subprocess.TimeoutExpired:
            return [], f"analysis timed out after {timeout}s"
        if analyzed.returncode != 0:
            details = (analyzed.stderr or analyzed.stdout).strip().splitlines()
            return [], "analysis failed: " + " | ".join(details[-6:])
        fragments.append((json.loads(output.read_text(encoding="utf-8")), compiled_root))
    if not fragments:
        return [], "compile produced no complete build-info"
    return fragments, None


def compile_independent_unit(
    project: Path,
    source: Path,
    task: Path,
    generator: Path,
    timeout: int,
    index: dict[str, list[str]],
    finding_ids: list[str],
) -> tuple[dict[str, Any] | None, str | None]:
    configuration = INDEPENDENT_UNITS.get(project.name)
    if configuration is None:
        return None, None
    work = task / "independent"
    work.mkdir()
    source_path = source / configuration["source"]
    destination = work / configuration["destination"]
    if configuration["destination"] == ".":
        shutil.copytree(source_path, work, dirs_exist_ok=True, symlinks=True)
    else:
        shutil.copytree(source_path, destination, symlinks=True)
    dependency = project.parent.parent / configuration["dependency"]
    dependency_destination = work / configuration["dependency_destination"]
    dependency_destination.parent.mkdir(parents=True, exist_ok=True)
    dependency_destination.symlink_to(dependency.resolve(), target_is_directory=True)

    remappings = "\n".join(
        f'  "{value}",' for value in configuration["remappings"]
    )
    (work / "foundry.toml").write_text(
        "\n".join(
            [
                "[profile.default]",
                f'src = "{configuration["source_dir"]}"',
                'out = "out"',
                'cache_path = "cache"',
                'libs = ["lib"]',
                f'solc_version = "{configuration["solc"]}"',
                "offline = true",
                "remappings = [",
                remappings,
                "]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    command = ["forge", "build"]
    if configuration["target"]:
        command.append(str(work / configuration["target"]))
    command.extend(
        [
            "--root",
            str(work),
            "--build-info",
            "--build-info-path",
            str(work / "out" / "build-info"),
            "--offline",
        ]
    )
    try:
        compiled = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return None, f"independent compilation timed out after {timeout}s"
    if compiled.returncode != 0:
        details = (compiled.stderr or compiled.stdout).strip().splitlines()
        return None, "independent compilation failed: " + " | ".join(details[-8:])

    fragment_path = task / "independent-fragment.json"
    analyzed = subprocess.run(
        [
            sys.executable,
            str(generator),
            str(work),
            str(fragment_path),
            "--project",
            project.name,
            "--source-dir",
            configuration["source_dir"],
            "--build-info",
            str(work / "out" / "build-info"),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if analyzed.returncode != 0:
        details = (analyzed.stderr or analyzed.stdout).strip().splitlines()
        return None, "independent analysis failed: " + " | ".join(details[-8:])
    fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
    preferred = []
    for finding_id in finding_ids:
        preferred.extend(
            preferred_source_files(
                project.parent.parent / "findings" / finding_id / "task"
            )
        )
    grounded, _ = ground_fragment(
        fragment, work, source, index, finding_ids[0], preferred
    )
    for node in grounded["nodes"]:
        node["finding_ids"] = finding_ids
    for edge in grounded["edges"]:
        edge["finding_ids"] = finding_ids
    return grounded, None


def merge_project(
    project_id: str,
    fragments: list[dict[str, Any]],
    capsule_count: int,
    successful_capsules: int,
    failed_capsules: list[dict[str, str]],
    source_file_count: int,
    ungrounded_nodes: int,
    independent_compilation: bool,
) -> dict[str, Any]:
    nodes_by_id: dict[str, dict[str, Any]] = {}
    for fragment in fragments:
        for node in fragment["nodes"]:
            existing = nodes_by_id.get(node["id"])
            if existing is None:
                nodes_by_id[node["id"]] = node
                continue
            existing["finding_ids"] = sorted(
                set(existing["finding_ids"]) | set(node["finding_ids"])
            )
            for key in ("state_variables_read", "state_variables_written", "storage_contexts"):
                existing[key] = sorted(set(existing.get(key, [])) | set(node.get(key, [])))

    call_edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    for fragment in fragments:
        for edge in fragment["edges"]:
            if edge["type"] != "CALL":
                continue
            key = (edge["source"], edge["target"], edge["call_type"])
            existing = call_edges.setdefault(
                key,
                {
                    "source": edge["source"],
                    "target": edge["target"],
                    "type": "CALL",
                    "call_type": edge["call_type"],
                    "call_sites": [],
                    "finding_ids": [],
                },
            )
            existing["call_sites"] = sorted(
                {tuple(site.items()) for site in existing["call_sites"] + edge.get("call_sites", [])}
            )
            existing["call_sites"] = [dict(site) for site in existing["call_sites"]]
            existing["finding_ids"] = sorted(
                set(existing["finding_ids"]) | set(edge["finding_ids"])
            )

    from generate_project_graph import state_dependencies

    state_edges, omitted_self = state_dependencies(
        {identifier: set(node.get("state_variables_read", [])) for identifier, node in nodes_by_id.items()},
        {identifier: set(node.get("state_variables_written", [])) for identifier, node in nodes_by_id.items()},
        {identifier: set(node.get("storage_contexts", [])) for identifier, node in nodes_by_id.items()},
    )
    for edge in state_edges.values():
        edge["state_variables"].sort()
        edge["storage_contexts"].sort()

    unresolved = [call for fragment in fragments for call in fragment["unresolved_calls"]]
    unresolved.sort(
        key=lambda call: (
            call["source"],
            call["call_type"],
            call.get("target") or "",
            (call.get("site") or {}).get("line", 0),
        )
    )
    edges = list(call_edges.values()) + list(state_edges.values())
    edges.sort(
        key=lambda edge: (
            edge["source"],
            edge["target"],
            edge["type"],
            edge.get("call_type", ""),
            edge.get("relation", ""),
        )
    )
    covered_files = {node["file"] for node in nodes_by_id.values()}
    state_relations = Counter(
        edge["relation"] for edge in state_edges.values()
    )
    unresolved_reasons = Counter(call["reason"] for call in unresolved)
    coverage_limitation = (
        "This graph combines source-grounded finding capsules with an independently compiled original-source unit; it may still not cover files outside that unit."
        if independent_compilation
        else "This graph is the source-grounded union of functions compiled by available READY finding capsules; it may not cover every project function."
    )
    limitations = [
        coverage_limitation,
        "Synthetic, stubbed, or modified capsule Solidity files are excluded unless the complete file exactly matches the vulnerable source snapshot.",
        "CALL edges not present in any capsule compilation unit may be absent.",
        "Low-level calls without a statically known function target are recorded but do not form CALL edges.",
        "High-level calls typed through interfaces are linked to interface declarations, not guessed implementations.",
        "Raw storage accesses through inline assembly sload/sstore are not converted to STATE_DEP edges.",
    ]
    if failed_capsules:
        limitations.append("Some finding capsules failed to compile or analyze; see generation.failed_capsules.")
    return {
        "schema_version": "1.2",
        "project": project_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "source_directories": ["."],
            "includes": "source-grounded functions and public state-variable getters compiled by finding capsules or independent source units",
            "excludes": sorted(EXCLUDED_PARTS),
            "state_access": "direct function body plus directly applied modifiers",
            "state_dependency": "same contract storage context, including inherited functions",
            "coverage_kind": (
                "capsule_and_independent_units"
                if independent_compilation
                else "finding_capsule_union"
            ),
        },
        "known_limitations": limitations,
        "generation": {
            "capsules": capsule_count,
            "successful_capsules": successful_capsules,
            "failed_capsules": failed_capsules,
            "source_solidity_files": source_file_count,
            "covered_source_files": len(covered_files),
            "ungrounded_capsule_nodes_excluded": ungrounded_nodes,
            "independent_compilation": independent_compilation,
        },
        "nodes": sorted(nodes_by_id.values(), key=lambda node: node["id"]),
        "edges": edges,
        "unresolved_calls": unresolved,
        "statistics": {
            "nodes": len(nodes_by_id),
            "call_edges": len(call_edges),
            "state_dep_edges": len(state_edges),
            "state_dep_edges_by_relation": dict(sorted(state_relations.items())),
            "edges": len(edges),
            "unresolved_or_out_of_scope_calls": len(unresolved),
            "unresolved_calls_by_reason": dict(sorted(unresolved_reasons.items())),
            "omitted_state_self_dependencies": omitted_self,
        },
    }


def run_project(
    project: Path, source: Path, output: Path, generator: Path, timeout: int
) -> tuple[str, int, int]:
    index, source_file_count = source_index(source)
    capsules = sorted((project / "sandbox").glob("*/capsule"))
    grounded_fragments: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    ungrounded_nodes = 0
    successful = 0
    independent_compilation = False
    with tempfile.TemporaryDirectory(prefix=f"forge-graph-{project.name}-") as temporary:
        temporary_root = Path(temporary)
        for capsule in capsules:
            finding_id = capsule.parent.name
            task = temporary_root / finding_id
            finding_task = project.parent.parent / "findings" / finding_id / "task"
            preferred = preferred_source_files(finding_task)
            prepare_task(capsule, finding_task, source, task)
            fragments, error = compile_capsule(
                task, generator, project.name, finding_id, timeout
            )
            if error is not None:
                failures.append({"finding_id": finding_id, "reason": error})
                continue
            successful += 1
            for fragment, compiled_root in fragments:
                grounded, excluded = ground_fragment(
                    fragment, compiled_root, source, index, finding_id, preferred
                )
                grounded_fragments.append(grounded)
                ungrounded_nodes += excluded

        independent, independent_error = compile_independent_unit(
            project,
            source,
            temporary_root,
            generator,
            timeout,
            index,
            [capsule.parent.name for capsule in capsules],
        )
        if independent is not None:
            grounded_fragments.append(independent)
            independent_compilation = True
        elif independent_error is not None:
            failures.append(
                {"finding_id": "<independent>", "reason": independent_error}
            )

    graph = merge_project(
        project.name,
        grounded_fragments,
        len(capsules),
        successful,
        failures,
        source_file_count,
        ungrounded_nodes,
        independent_compilation,
    )
    if not graph["nodes"]:
        detail = failures[0]["reason"] if failures else "all compiled nodes were synthetic or modified"
        raise RuntimeError(
            f"no source-grounded nodes ({successful}/{len(capsules)} capsules analyzed): {detail}"
        )
    atomic_write_json(output, graph)
    return project.name, len(graph["nodes"]), len(failures)


def main() -> int:
    args = parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    dataset_root = args.dataset_root.resolve()
    assets = dataset_root / "project_assets"
    output_dir = args.output_dir.resolve()
    generator = Path(__file__).with_name("generate_project_graph.py").resolve()
    requested = set(args.project)
    projects = sorted(
        path for path in assets.glob("forge_*") if (path / "sandbox").is_dir()
    )
    if requested:
        missing = requested - {path.name for path in projects}
        if missing:
            raise SystemExit(f"unknown forge projects: {', '.join(sorted(missing))}")
        projects = [path for path in projects if path.name in requested]

    jobs: list[tuple[Path, Path, Path]] = []
    failures: list[tuple[str, str]] = []
    for project in projects:
        output = output_dir / f"{project.name}.json"
        if output.exists() and not args.force:
            log(f"[skip] {project.name}: {output.name} exists")
            continue
        source_links = {
            (dataset_root / "findings" / capsule.parent.name / "task" / "source").resolve()
            for capsule in (project / "sandbox").glob("*/capsule")
        }
        if len(source_links) != 1 or not next(iter(source_links), Path()).is_dir():
            message = "capsules do not resolve to one source repository"
            failures.append((project.name, message))
            log(f"[failed] {project.name}: {message}")
            continue
        source = next(iter(source_links))
        jobs.append((project, source, output))
        if args.dry_run:
            count = sum(1 for _ in (project / "sandbox").glob("*/capsule"))
            log(f"[plan/{count} capsules] {project.name}: {source} -> {output}")

    if args.dry_run:
        log(f"planned={len(jobs)} failed={len(failures)}")
        return 1 if failures else 0

    generated = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(run_project, *job, generator, args.timeout): job[0].name
            for job in jobs
        }
        for future in as_completed(futures):
            project_id = futures[future]
            try:
                _, nodes, capsule_failures = future.result()
                generated += 1
                log(
                    f"[ok] {project_id}: nodes={nodes} capsule_failures={capsule_failures}"
                )
            except Exception as exc:
                failures.append((project_id, str(exc)))
                log(f"[failed] {project_id}: {exc}")
    log(f"generated={generated} failed={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
