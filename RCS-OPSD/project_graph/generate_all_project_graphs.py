#!/usr/bin/env python3
"""Generate one function-level graph per dataset project with bounded concurrency."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


CONFIG_NAMES = {
    "brownie-config.json",
    "brownie-config.yaml",
    "brownie-config.yml",
    "foundry.toml",
    "hardhat.config.cjs",
    "hardhat.config.cts",
    "hardhat.config.js",
    "hardhat.config.mjs",
    "hardhat.config.ts",
    "remappings.txt",
    "truffle-config.js",
    "truffle.js",
}
SKIPPED_DIRECTORIES = {
    ".git",
    "artifacts",
    "build",
    "cache",
    "dist",
    "lib",
    "node_modules",
    "out",
    "third_party",
    "vendor",
}
COPY_IGNORES = {".git", "artifacts", "cache", "out"}
PRINT_LOCK = Lock()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rcs_opsd.artifact_paths import GRAPH_BASE_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset_root"))
    parser.add_argument(
        "--output-dir", type=Path, default=GRAPH_BASE_ROOT
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--project", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--compile-sandbox",
        action="store_true",
        help="Run each project's dataset sandbox/compile.sh before analysis",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Compile source repositories directly instead of temporary copies",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def discover_source(dataset_root: Path, project_id: str) -> Path:
    findings_root = dataset_root / "findings"
    links = sorted(
        path / "task" / "source"
        for path in findings_root.glob(f"{project_id}__*")
        if (path / "task" / "source").is_symlink()
    )
    if not links:
        raise RuntimeError("no finding source link")
    resolved = {link.resolve() for link in links}
    if len(resolved) != 1:
        raise RuntimeError("findings resolve to multiple source repositories")
    source = next(iter(resolved))
    if not source.is_dir():
        raise RuntimeError(f"source directory does not exist: {source}")
    return source


def discover_configs(root: Path, max_depth: int = 4) -> list[Path]:
    found: list[Path] = []
    stack = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            entries = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError:
            continue
        for entry in entries:
            if entry.is_file() and entry.name in CONFIG_NAMES:
                found.append(entry)
            elif (
                entry.is_dir()
                and depth < max_depth
                and entry.name not in SKIPPED_DIRECTORIES
            ):
                stack.append((entry, depth + 1))
    return sorted(found)


def analysis_roots(project_root: Path) -> list[Path]:
    configs = discover_configs(project_root)
    root_configs = [path for path in configs if path.parent == project_root]
    if root_configs:
        return [project_root]
    roots = sorted({path.parent for path in configs})
    return roots or [project_root]


def source_directory(root: Path) -> str:
    foundry_config = root / "foundry.toml"
    if foundry_config.is_file():
        match = re.search(
            r"(?m)^\s*src\s*=\s*['\"]([^'\"]+)['\"]",
            foundry_config.read_text(encoding="utf-8"),
        )
        configured = match.group(1) if match else None
        if configured is not None and (root / configured).is_dir():
            return configured
    for candidate in ("src", "contracts"):
        directory = root / candidate
        if directory.is_dir() and any(directory.rglob("*.sol")):
            return candidate
    return "."


def prefix_fragment(fragment: dict[str, Any], prefix: str) -> dict[str, Any]:
    if not prefix:
        return fragment
    prefix_path = Path(prefix)
    node_id_map: dict[str, str] = {}
    for node in fragment["nodes"]:
        old_id = node["id"]
        node["file"] = (prefix_path / node["file"]).as_posix()
        node["id"] = f"{node['file']}::{old_id.split('::', 1)[1]}"
        node_id_map[old_id] = node["id"]
        node["storage_contexts"] = [
            f"{(prefix_path / context.split('::', 1)[0]).as_posix()}::{context.split('::', 1)[1]}"
            for context in node.get("storage_contexts", [])
        ]
    for edge in fragment["edges"]:
        edge["source"] = node_id_map[edge["source"]]
        edge["target"] = node_id_map[edge["target"]]
        edge["storage_contexts"] = [
            f"{(prefix_path / context.split('::', 1)[0]).as_posix()}::{context.split('::', 1)[1]}"
            for context in edge.get("storage_contexts", [])
        ]
        for site in edge.get("call_sites", []):
            site["file"] = (prefix_path / site["file"]).as_posix()
    for call in fragment["unresolved_calls"]:
        call["source"] = node_id_map[call["source"]]
        site = call.get("site")
        if site:
            site["file"] = (prefix_path / site["file"]).as_posix()
    return fragment


def merge_fragments(project_id: str, fragments: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = [node for fragment in fragments for node in fragment["nodes"]]
    edges = [edge for fragment in fragments for edge in fragment["edges"]]
    unresolved = [
        call for fragment in fragments for call in fragment["unresolved_calls"]
    ]
    if len({node["id"] for node in nodes}) != len(nodes):
        raise RuntimeError("duplicate node IDs across compilation roots")

    edges.sort(
        key=lambda edge: (
            edge["source"],
            edge["target"],
            edge["type"],
            edge.get("call_type", ""),
            edge.get("relation", ""),
        )
    )
    unresolved.sort(
        key=lambda call: (
            call["source"],
            call["call_type"],
            call.get("target") or "",
            (call.get("site") or {}).get("line", 0),
        )
    )
    state_relations = Counter(
        edge["relation"] for edge in edges if edge["type"] == "STATE_DEP"
    )
    unresolved_reasons = Counter(call["reason"] for call in unresolved)
    limitations = sorted(
        {
            limitation
            for fragment in fragments
            for limitation in fragment.get("known_limitations", [])
        }
    )
    source_directories = sorted(
        {
            directory
            for fragment in fragments
            for directory in fragment["scope"].get(
                "source_directories",
                [fragment["scope"].get("source_directory", ".")],
            )
        }
    )
    call_count = sum(edge["type"] == "CALL" for edge in edges)
    state_count = len(edges) - call_count
    return {
        "schema_version": "1.2",
        "project": project_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "source_directories": source_directories,
            "includes": "declared functions and public state-variable getters in project-owned Solidity files",
            "excludes": [
                "lib",
                "mock",
                "mocks",
                "test",
                "script",
                "node_modules",
            ],
            "state_access": "direct function body plus directly applied modifiers",
            "state_dependency": "same contract storage context, including inherited functions",
        },
        "known_limitations": limitations,
        "nodes": sorted(nodes, key=lambda node: node["id"]),
        "edges": edges,
        "unresolved_calls": unresolved,
        "statistics": {
            "nodes": len(nodes),
            "call_edges": call_count,
            "state_dep_edges": state_count,
            "state_dep_edges_by_relation": dict(sorted(state_relations.items())),
            "edges": len(edges),
            "unresolved_or_out_of_scope_calls": len(unresolved),
            "unresolved_calls_by_reason": dict(sorted(unresolved_reasons.items())),
            "omitted_state_self_dependencies": sum(
                fragment["statistics"]["omitted_state_self_dependencies"]
                for fragment in fragments
            ),
        },
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def copy_project(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        symlinks=True,
        ignore=shutil.ignore_patterns(*COPY_IGNORES),
    )


def classify_compile_script(script: Path) -> str:
    body = "\n".join(
        line
        for line in script.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    if "forge build" in body and "hardhat" in body:
        return "foundry+hardhat"
    if "forge build" in body:
        return "foundry"
    if "hardhat" in body or "workspace @equilibria/perennial-v2 compile" in body:
        return "hardhat"
    if "brownie" in body:
        return "brownie"
    if "waffle" in body:
        return "waffle"
    return "direct"


def discover_build_info(root: Path) -> list[Path]:
    directories: list[Path] = []
    for path in root.rglob("build-info"):
        relative_parts = path.relative_to(root).parts
        if (
            path.is_dir()
            and not any(part in SKIPPED_DIRECTORIES for part in relative_parts[:-2])
            and any(path.glob("*.json"))
        ):
            directories.append(path)
    return sorted(directories)


def materialize_build_info_sources(build_info: Path, root: Path) -> None:
    for artifact in build_info.glob("*.json"):
        loaded = json.loads(artifact.read_text(encoding="utf-8"))
        for filename, source in loaded.get("input", {}).get("sources", {}).items():
            destination = (root / filename).resolve()
            try:
                destination.relative_to(root.resolve())
            except ValueError:
                continue
            if not destination.exists() and "content" in source:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(source["content"], encoding="utf-8")


def run_project(
    project_id: str,
    source: Path,
    output: Path,
    generator: Path,
    timeout: int,
    in_place: bool,
    compile_script: Path | None,
) -> tuple[str, str]:
    if in_place:
        temporary_context = None
        work_root = source
    else:
        temporary_context = tempfile.TemporaryDirectory(prefix=f"project-graph-{project_id}-")
        work_root = Path(temporary_context.name) / "source"
        copy_project(source, work_root)

    try:
        if compile_script is not None:
            environment = os.environ.copy()
            environment["FOUNDRY_BUILD_INFO"] = "true"
            environment["HARDHAT_DISABLE_TELEMETRY"] = "1"
            try:
                compiled = subprocess.run(
                    ["bash", str(compile_script)],
                    cwd=work_root,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=environment,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"compilation timed out after {timeout}s") from exc
            if compiled.returncode != 0:
                details = (compiled.stderr or compiled.stdout).strip().splitlines()
                tail = "\n".join(details[-20:])
                raise RuntimeError(f"sandbox compilation failed:\n{tail}")

        build_infos = discover_build_info(work_root) if compile_script is not None else []
        if build_infos:
            roots_with_build_info = [
                (
                    build_info.parent.parent,
                    None
                    if (build_info.parent.parent / "foundry.toml").is_file()
                    else build_info,
                )
                for build_info in build_infos
            ]
            for root, build_info in roots_with_build_info:
                if build_info is not None:
                    materialize_build_info_sources(build_info, root)
        else:
            roots_with_build_info = [
                (root, None) for root in analysis_roots(work_root)
            ]
        fragments: list[dict[str, Any]] = []
        for index, (root, build_info) in enumerate(roots_with_build_info):
            prefix = root.relative_to(work_root).as_posix()
            if prefix == ".":
                prefix = ""
            source_dir = source_directory(root)
            fragment_path = Path(tempfile.gettempdir()) / (
                f"project-graph-{os.getpid()}-{project_id}-{index}.json"
            )
            command = [
                sys.executable,
                str(generator),
                str(root),
                str(fragment_path),
                "--project",
                project_id,
                "--source-dir",
                source_dir,
            ]
            if build_info is not None:
                command.extend(["--build-info", str(build_info)])
            elif compile_script is not None and any(
                (root / name).is_file()
                for name in (
                    "brownie-config.json",
                    "brownie-config.yaml",
                    "brownie-config.yml",
                    "truffle-config.js",
                    "truffle.js",
                    "waffle.json",
                )
            ) and (root / "build").is_dir():
                command.append("--ignore-compile")
            try:
                analysis_environment = os.environ.copy()
                analysis_environment["FOUNDRY_BUILD_INFO"] = "true"
                result = subprocess.run(
                    command,
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=analysis_environment,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"analysis timed out after {timeout}s") from exc
            if result.returncode != 0:
                details = (result.stderr or result.stdout).strip().splitlines()
                tail = "\n".join(details[-20:])
                raise RuntimeError(f"analysis root {prefix or '.'} failed:\n{tail}")
            try:
                fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
            finally:
                fragment_path.unlink(missing_ok=True)
            fragment["scope"]["source_directories"] = [
                (Path(prefix) / source_dir).as_posix() if prefix else source_dir
            ]
            fragments.append(prefix_fragment(fragment, prefix))

        merged = merge_fragments(project_id, fragments)
        if not merged["nodes"]:
            raise RuntimeError("analysis produced zero function nodes")
        atomic_write_json(output, merged)
        return project_id, "generated"
    finally:
        if temporary_context is not None:
            temporary_context.cleanup()


def main() -> int:
    args = parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    dataset_root = args.dataset_root.resolve()
    project_assets = dataset_root / "project_assets"
    if not project_assets.is_dir():
        raise SystemExit(f"missing project_assets: {project_assets}")
    generator = Path(__file__).with_name("generate_project_graph.py").resolve()
    output_dir = args.output_dir.resolve()
    requested = set(args.project)
    projects = sorted(path.name for path in project_assets.iterdir() if path.is_dir())
    if requested:
        missing = sorted(requested - set(projects))
        if missing:
            raise SystemExit(f"unknown projects: {', '.join(missing)}")
        projects = [project for project in projects if project in requested]

    jobs: list[tuple[str, Path, Path, Path | None]] = []
    failures: list[tuple[str, str]] = []
    for project_id in projects:
        output = output_dir / f"{project_id}.json"
        if output.exists() and not args.force:
            log(f"[skip] {project_id}: {output.name} exists")
            continue
        try:
            source = discover_source(dataset_root, project_id)
        except RuntimeError as exc:
            failures.append((project_id, str(exc)))
            log(f"[failed] {project_id}: {exc}")
            continue
        compile_script = None
        if args.compile_sandbox:
            compile_script = project_assets / project_id / "sandbox" / "compile.sh"
            if not compile_script.is_file():
                failures.append((project_id, "missing sandbox/compile.sh"))
                log(f"[failed] {project_id}: missing sandbox/compile.sh")
                continue
        jobs.append((project_id, source, output, compile_script))

    if args.dry_run:
        for project_id, source, output, compile_script in jobs:
            framework = classify_compile_script(compile_script) if compile_script else "auto"
            log(f"[plan/{framework}] {project_id}: {source} -> {output}")
        log(f"planned={len(jobs)} failed={len(failures)}")
        return 1 if failures else 0

    generated_count = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                run_project,
                project_id,
                source,
                output,
                generator,
                args.timeout,
                args.in_place,
                compile_script,
            ): project_id
            for project_id, source, output, compile_script in jobs
        }
        for future in as_completed(futures):
            project_id = futures[future]
            try:
                _, status = future.result()
                generated_count += 1
                log(f"[ok] {project_id}: {status}")
            except Exception as exc:  # keep independent projects running
                failures.append((project_id, str(exc)))
                log(f"[failed] {project_id}: {exc}")

    log(f"generated={generated_count} failed={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
