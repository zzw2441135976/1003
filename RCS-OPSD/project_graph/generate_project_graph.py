#!/usr/bin/env python3
"""Export a function-level Solidity project graph from Slither build info."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from slither.core.declarations.function import Function
from slither.core.variables.state_variable import StateVariable
from slither.slither import Slither


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Solidity project root")
    parser.add_argument("output", type=Path, help="Output graph JSON")
    parser.add_argument("--project", required=True, help="Project identifier")
    parser.add_argument(
        "--build-info",
        type=Path,
        help="Existing Hardhat-compatible build-info directory; omit to let Slither compile",
    )
    parser.add_argument(
        "--ignore-compile",
        action="store_true",
        help="Read existing framework artifacts without compiling again",
    )
    parser.add_argument("--source-dir", default="src")
    return parser.parse_args()


def source_file(item: Any, root: Path) -> str | None:
    mapping = getattr(item, "source_mapping", None)
    filename = getattr(mapping, "filename", None)
    absolute = getattr(filename, "absolute", "")
    if not absolute:
        return None
    try:
        relative = Path(absolute).resolve().relative_to(root).as_posix()
        solpp_prefix = "cache/solpp-generated-contracts/"
        if relative.startswith(solpp_prefix):
            return f"contracts/{relative[len(solpp_prefix):]}"
        return relative
    except ValueError:
        return None


def in_source_scope(filename: str, source_dir: str) -> bool:
    source_path = Path(source_dir)
    path = Path(filename)
    try:
        relative = path.relative_to(source_path)
    except ValueError:
        return False
    excluded_parts = {
        "hardhat-dependency-compiler",
        "lib",
        "mock",
        "mocks",
        "node_modules",
        "script",
        "scripts",
        "test",
        "test-contracts",
        "tests",
    }
    for index, part in enumerate(path.parts[:-1]):
        if part == "lib" and path.parts[index + 1] in {"openzeppelin-contracts", "openzeppelin-contracts-upgradeable", "v3-core", "v3-periphery", "forge-std"}:
            return False
        if part == "lib" and index > 0 and path.parts[index - 1] in {"src", "contracts"}:
            continue
        if part in excluded_parts or part.startswith("@"):
            return False
    return True


def function_contract(function: Function) -> str | None:
    contract = getattr(function, "contract_declarer", None)
    return contract.name if contract is not None else None


def function_id(function: Function, root: Path) -> str | None:
    filename = source_file(function, root)
    if filename is None:
        return None
    contract = function_contract(function) or "<top-level>"
    return f"{filename}::{contract}::{function.full_name}"


def special_kind(function: Function) -> str | None:
    if function.is_constructor:
        return "constructor"
    if function.is_fallback:
        return "fallback"
    if function.is_receive:
        return "receive"
    return None


def mutability(function: Function) -> str:
    if function.payable:
        return "payable"
    if function.pure:
        return "pure"
    if function.view:
        return "view"
    return "nonpayable"


def state_id(variable: Any) -> str:
    canonical_name = getattr(variable, "canonical_name", None)
    return canonical_name or str(variable)


def is_persistent_state(variable: Any) -> bool:
    return not variable.is_constant and not variable.is_immutable


def execution_units(function: Function) -> Iterable[Function]:
    """Project modifier effects onto the function because modifiers are not nodes."""
    yield function
    yield from function.modifiers


def storage_contexts(contracts: Iterable[Any], root: Path, state_getters: dict) -> dict:
    """Attach declared and inherited functions to their possible contract storage."""
    contexts: dict[str, set[str]] = defaultdict(set)
    for contract in contracts:
        if contract.contract_kind != "contract":
            continue
        context = f"{source_file(contract, root)}::{contract.name}"
        for function in contract.functions:
            identifier = function_id(function, root)
            if identifier is not None:
                contexts[identifier].add(context)
        for variable in contract.state_variables:
            getter = state_getters.get(state_id(variable))
            if getter is not None:
                contexts[getter].add(context)
    return contexts


def state_dependencies(reads: dict, writes: dict, contexts: dict) -> tuple[dict, int]:
    readers: dict[tuple[str, str], set[str]] = defaultdict(set)
    writers: dict[tuple[str, str], set[str]] = defaultdict(set)
    for identifier in reads.keys() | writes.keys():
        for context in contexts.get(identifier, ()):
            for variable in reads.get(identifier, ()):
                readers[(context, variable)].add(identifier)
            for variable in writes.get(identifier, ()):
                writers[(context, variable)].add(identifier)
    edges = {}
    omitted = set()
    for (context, variable), sources in writers.items():
        for source in sources:
            for relation, targets in (
                ("WRITE_READ", readers[(context, variable)]),
                ("WRITE_WRITE", sources),
            ):
                for target in targets:
                    if source == target:
                        omitted.add((source, variable))
                        continue
                    edge = edges.setdefault((source, target, relation), {
                        "source": source, "target": target, "type": "STATE_DEP",
                        "relation": relation, "state_variables": [], "storage_contexts": [],
                    })
                    if variable not in edge["state_variables"]:
                        edge["state_variables"].append(variable)
                    if context not in edge["storage_contexts"]:
                        edge["storage_contexts"].append(context)
    return edges, len(omitted)


def call_site(operation: Any, root: Path) -> dict[str, Any] | None:
    node = getattr(operation, "node", None)
    mapping = getattr(node, "source_mapping", None)
    if mapping is None:
        return None
    filename = source_file(node, root)
    lines = list(getattr(mapping, "lines", []))
    if filename is None or not lines:
        return None
    return {"file": filename, "line": min(lines)}


def main() -> None:
    args = parse_args()
    root = args.source.resolve()
    if args.build_info is None:
        slither = Slither(str(root), ignore_compile=args.ignore_compile)
    else:
        build_info = args.build_info.resolve()
        slither = Slither(
            str(root),
            compile_force_framework="Hardhat",
            hardhat_ignore_compile=True,
            hardhat_artifacts_directory=str(build_info.parent),
            hardhat_working_dir=str(root),
        )

    declared_functions: list[Function] = []
    for contract in slither.contracts:
        declared_functions.extend(contract.functions_declared)
    for compilation_unit in slither.compilation_units:
        declared_functions.extend(compilation_unit.functions_top_level)

    functions: list[Function] = []
    for function in declared_functions:
        filename = source_file(function, root)
        if (
            filename is not None
            and in_source_scope(filename, args.source_dir)
            and not function.is_constructor_variables
        ):
            functions.append(function)

    nodes: list[dict[str, Any]] = []
    node_ids: set[str] = set()
    functions_by_id: dict[str, Function] = {}
    state_getters: dict[str, str] = {}
    for function in functions:
        identifier = function_id(function, root)
        if identifier is None or identifier in node_ids:
            continue
        mapping = function.source_mapping
        lines = list(mapping.lines)
        contract = getattr(function, "contract_declarer", None)
        nodes.append(
            {
                "id": identifier,
                "file": source_file(function, root),
                "contract": function_contract(function),
                "contract_kind": getattr(contract, "contract_kind", "top-level"),
                "function": function.name,
                "signature": function.full_name,
                "visibility": function.visibility,
                "mutability": mutability(function),
                "implemented": bool(function.is_implemented),
                "special": special_kind(function),
                "start_line": min(lines) if lines else None,
                "end_line": max(lines) if lines else None,
            }
        )
        node_ids.add(identifier)
        functions_by_id[identifier] = function

    # Solidity exposes each public state variable as a generated view function.
    for contract in slither.contracts:
        for variable in contract.state_variables_declared:
            filename = source_file(variable, root)
            if (
                variable.visibility != "public"
                or filename is None
                or not in_source_scope(filename, args.source_dir)
            ):
                continue
            identifier = f"{filename}::{contract.name}::{variable.full_name}"
            if identifier in node_ids:
                continue
            lines = list(variable.source_mapping.lines)
            nodes.append(
                {
                    "id": identifier,
                    "file": filename,
                    "contract": contract.name,
                    "contract_kind": contract.contract_kind,
                    "function": variable.name,
                    "signature": variable.full_name,
                    "visibility": "external",
                    "mutability": "view",
                    "implemented": True,
                    "special": "state_variable_getter",
                    "start_line": min(lines) if lines else None,
                    "end_line": max(lines) if lines else None,
                }
            )
            node_ids.add(identifier)
            state_getters[state_id(variable)] = identifier

    call_edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    unresolved_calls: list[dict[str, Any]] = []

    def add_call(source: str, target: Any, call_type: str, operation: Any) -> None:
        if isinstance(target, StateVariable) and state_id(target) in state_getters:
            target_id = state_getters[state_id(target)]
            key = (source, target_id, call_type)
            edge = call_edges.setdefault(
                key,
                {
                    "source": source,
                    "target": target_id,
                    "type": "CALL",
                    "call_type": call_type,
                    "call_sites": [],
                },
            )
            site = call_site(operation, root)
            if site is not None and site not in edge["call_sites"]:
                edge["call_sites"].append(site)
            return
        if not isinstance(target, Function):
            unresolved_calls.append(
                {
                    "source": source,
                    "call_type": call_type,
                    "reason": "non_function_target",
                    "target": str(target) if target is not None else None,
                    "site": call_site(operation, root),
                }
            )
            return
        target_id = function_id(target, root)
        if target_id not in node_ids:
            unresolved_calls.append(
                {
                    "source": source,
                    "call_type": call_type,
                    "reason": "out_of_scope_target",
                    "target": target.canonical_name,
                    "site": call_site(operation, root),
                }
            )
            return
        key = (source, target_id, call_type)
        edge = call_edges.setdefault(
            key,
            {
                "source": source,
                "target": target_id,
                "type": "CALL",
                "call_type": call_type,
                "call_sites": [],
            },
        )
        site = call_site(operation, root)
        if site is not None and site not in edge["call_sites"]:
            edge["call_sites"].append(site)

    for source, function in functions_by_id.items():
        for unit in execution_units(function):
            library_operation_ids = {id(operation) for operation in unit.library_calls}
            for operation in unit.internal_calls:
                if not getattr(operation, "is_modifier_call", False):
                    add_call(source, operation.function, "internal", operation)
            for operation in unit.library_calls:
                add_call(source, operation.function, "library", operation)
            for _, operation in unit.high_level_calls:
                if id(operation) not in library_operation_ids:
                    add_call(source, operation.function, "external", operation)
            for operation in unit.low_level_calls:
                unresolved_calls.append(
                    {
                        "source": source,
                        "call_type": "low-level",
                        "reason": "unresolved_low_level_target",
                        "target": None,
                        "operation": str(operation.function_name),
                        "destination": str(operation.destination),
                        "site": call_site(operation, root),
                    }
                )

    reads: dict[str, set[str]] = defaultdict(set)
    writes: dict[str, set[str]] = defaultdict(set)
    for identifier, function in functions_by_id.items():
        for unit in execution_units(function):
            reads[identifier].update(
                state_id(var) for var in unit.state_variables_read if is_persistent_state(var)
            )
            writes[identifier].update(
                state_id(var) for var in unit.state_variables_written if is_persistent_state(var)
            )
    for variable, identifier in (
        (variable, state_getters[state_id(variable)])
        for contract in slither.contracts
        for variable in contract.state_variables_declared
        if state_id(variable) in state_getters
    ):
        if is_persistent_state(variable):
            reads[identifier].add(state_id(variable))

    nodes_by_id = {node["id"]: node for node in nodes}
    for identifier in node_ids:
        nodes_by_id[identifier]["state_variables_read"] = sorted(reads[identifier])
        nodes_by_id[identifier]["state_variables_written"] = sorted(
            writes[identifier]
        )

    contexts = storage_contexts(slither.contracts, root, state_getters)
    for node in nodes:
        node["storage_contexts"] = sorted(contexts.get(node["id"], ()))
    state_edges, omitted_self_dependencies = state_dependencies(reads, writes, contexts)

    nodes.sort(key=lambda item: item["id"])
    edges = list(call_edges.values()) + list(state_edges.values())
    for edge in edges:
        edge.get("call_sites", []).sort(key=lambda item: (item["file"], item["line"]))
        edge.get("state_variables", []).sort()
        edge.get("storage_contexts", []).sort()
    edges.sort(
        key=lambda item: (
            item["source"],
            item["target"],
            item["type"],
            item.get("call_type", ""),
            item.get("relation", ""),
        )
    )
    unresolved_calls.sort(
        key=lambda item: (
            item["source"],
            item["call_type"],
            item.get("target") or "",
            (item.get("site") or {}).get("line", 0),
        )
    )
    unresolved_by_reason: dict[str, int] = defaultdict(int)
    for call in unresolved_calls:
        unresolved_by_reason[call["reason"]] += 1
    state_edges_by_relation: dict[str, int] = defaultdict(int)
    for edge in state_edges.values():
        state_edges_by_relation[edge["relation"]] += 1

    graph = {
        "schema_version": "1.2",
        "project": args.project,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "source_directory": args.source_dir,
            "includes": "declared functions and public state-variable getters in project-owned Solidity files",
            "excludes": ["lib", "mock", "mocks", "test", "script", "node_modules"],
            "state_access": "direct function body plus directly applied modifiers",
            "state_dependency": "same contract storage context, including inherited functions",
        },
        "known_limitations": [
            "Low-level calls without a statically known function target are recorded but do not form CALL edges.",
            "Calls into excluded dependencies are recorded but their targets are not graph nodes.",
            "High-level calls typed through interfaces are linked to interface declarations, not guessed implementations.",
            "Raw storage accesses through inline assembly sload/sstore are not yet converted to STATE_DEP edges.",
        ],
        "nodes": nodes,
        "edges": edges,
        "unresolved_calls": unresolved_calls,
        "statistics": {
            "nodes": len(nodes),
            "call_edges": len(call_edges),
            "state_dep_edges": len(state_edges),
            "state_dep_edges_by_relation": dict(sorted(state_edges_by_relation.items())),
            "edges": len(edges),
            "unresolved_or_out_of_scope_calls": len(unresolved_calls),
            "unresolved_calls_by_reason": dict(sorted(unresolved_by_reason.items())),
            "omitted_state_self_dependencies": omitted_self_dependencies,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(graph, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
