#!/usr/bin/env python3
"""Compile original-source units for the 49 canonical S graph coverage gaps."""
import argparse
import copy
import csv
import hashlib
import json
import posixpath
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from generate_forge_project_graphs import source_file_excluded
from generate_all_project_graphs import prefix_fragment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rcs_opsd.artifact_paths import ARTIFACT_ROOT, GRAPH_BASE_ROOT, GRAPH_ROOT
from generate_project_graph import state_dependencies

REPORT = ARTIFACT_ROOT / "graph_repairs"
DEPS = ROOT / "data/graph_dependencies"
GRAPHS = GRAPH_BASE_ROOT
BASE_OZ = ROOT / "data/graph_dependencies/blackhole-openzeppelin"
IMPORTS = re.compile(r'import\s+(?:[^;]*?\sfrom\s+)?["\']([^"\']+)["\']\s*;')
VERSIONS = {
    "code4rena_2022-05-opensea-seaport": "0.8.13",
    "forge_2024.12.21_-_Final_-_Idle_Finance_Credit_Vaults_Audit_Report": "0.8.10",
    "forge_2025.01.31_-_Final_-_Perennial_V2_Update_4_Audit_Report": "0.8.24",
    "forge_2025.03.10_-_Final_-_Symmio_Staking_and_Vesting_Audit_Report": "0.8.28",
    "forge_2025.07.18_-_Final_-_Notional_Exponent_Audit_Report": "0.8.29",
    "forge_Across_Protocol_OFT_Integration_Differential_Audit": "0.8.23",
    "forge_Bridged_USDC_Support_Audit": "0.8.23",
    "forge_Cove_-_Zellic_Audit_Report": "0.8.28",
    "forge_Parallel_Protocol_-_Zenith_Audit_Report": "0.8.25",
    "forge_Paxos_Labs_-_Zenith_Audit_Report": "0.8.21",
    "forge_Periphery_Changes_Audit": "0.8.23",
    "forge_Sorella_-_Zenith_Audit_Report": "0.8.28",
    "forge_Zenith_Audit_Report_-_Stream_Protocol": "0.8.28",
    "forge_cantina_eco_pr28_pr38_aug2025": "0.8.28",
    "forge_cantina_euler_swap_april2025": "0.8.28",
    "forge_cantina_liminal_oct2025": "0.8.28",
    "forge_cantina_rocketpool_jun2025": "0.8.30",
    "forge_cantina_stryke_aug2025": "0.8.28",
    "forge_f_x_v2_Audit": "0.8.28",
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def dependencies(project, source, owner):
    old_oz = project in {
        "forge_2024.12.21_-_Final_-_Idle_Finance_Credit_Vaults_Audit_Report",
        "forge_2025.01.31_-_Final_-_Perennial_V2_Update_4_Audit_Report",
        "forge_Across_Protocol_OFT_Integration_Differential_Audit",
        "forge_Bridged_USDC_Support_Audit", "forge_Periphery_Changes_Audit",
    }
    result = {
        "@openzeppelin/contracts": BASE_OZ / "contracts" if old_oz else DEPS / "oz5",
        "@openzeppelin/contracts-upgradeable": BASE_OZ / "contracts-upgradeable" if old_oz else DEPS / "oz5-upgradeable",
        "@openzeppelin-upgradeable/contracts": DEPS / "oz5-upgradeable",
        "@uniswap/v2-periphery": DEPS / "uniswap-v2-periphery",
        "@uniswap/v3-periphery": DEPS / "uniswap-v3-periphery",
        "@uniswap/v3-core": DEPS / "uniswap-v3-core",
        "@equilibria/root": DEPS / "equilibria-root",
        "@pythnetwork/pyth-sdk-solidity": DEPS / "pyth",
        "@uma/core": DEPS / "uma-core",
        "@layerzerolabs/oft-evm": DEPS / "lz-oft",
        "@layerzerolabs/oapp-evm": DEPS / "lz-oapp",
        "@layerzerolabs/lz-evm-protocol-v2": DEPS / "lz-protocol",
        "@layerzerolabs/lz-evm-messagelib-v2": DEPS / "lz-messagelib",
        "solady": DEPS / "solady",
        "@solady": DEPS / "solady/src",
        "v4-core": DEPS / "v4-core",
        "evc": DEPS / "evc/src",
        "evk": DEPS / "evk/src",
        "ethereum-vault-connector": DEPS / "evc/src",
        "euler-price-oracle": DEPS / "euler-price-oracle",
        "token-plugins-upgradeable": DEPS / "token-plugins-upgradeable",
        "@1inch/solidity-utils": DEPS / "solidity-utils",
        "clones-with-immutable-args": DEPS / "clone-imm-args",
        "forge-std": DEPS / "forge-std/src",
        "solmate": DEPS / "solmate",
        "@uniswap/v4-core": DEPS / "v4-core",
        "v4-periphery": DEPS / "v4-periphery",
        "permit2": DEPS / "permit2",
        "base64-sol": DEPS / "base64-sol",
    }
    if project == "forge_2025.01.31_-_Final_-_Perennial_V2_Update_4_Audit_Report":
        result["@equilibria/root"] = DEPS / "equilibria-root-rc8"
    if project == "forge_cantina_eco_pr28_pr38_aug2025":
        result["@openzeppelin/contracts"] = DEPS / "oz5-4"
    if project == "forge_Cove_-_Zellic_Audit_Report":
        result.update({
            "token-plugins-upgradeable": DEPS / "cove-token-plugins",
            "@1inch/solidity-utils": DEPS / "cove-solidity-utils",
            "@1inch/farming": DEPS / "cove-farming",
            "euler-price-oracle": DEPS / "cove-oracle",
            "ethereum-vault-connector": DEPS / "cove-evc/src",
            "@solady": DEPS / "cove-solady/src",
            "clones-with-immutable-args": DEPS / "cove-clones/src",
        })
    if project == "forge_cantina_euler_swap_april2025":
        result["openzeppelin-contracts"] = DEPS / "oz5"
    if project == "forge_cantina_stryke_aug2025":
        result["openzeppelin-contracts"] = source / owner / "lib/openzeppelin-contracts"
        result["@uniswap/v3-core"] = source / owner / "lib/v3-core"
        result["v3-periphery"] = source / owner / "lib/v3-periphery/contracts"
        result["v3-core"] = source / owner / "lib/v3-core/contracts"
    return result


def compile_unit(project, source, scope, targets, work, version, targets_only, via_ir):
    owner = str(Path(scope).parent)
    unit_source = source / owner
    deps = dependencies(project, source, owner)
    remappings = {alias + "/": "node_modules/" + alias + "/" for alias in deps}
    remappings["@openzeppelin-upgradeable/contracts/"] = "node_modules/@openzeppelin/contracts-upgradeable/"
    if project == "forge_2025.07.18_-_Final_-_Notional_Exponent_Audit_Report":
        remappings["src/withdraws/ClonedCoolDownHolder.sol"] = "src/withdraws/ClonedCooldownHolder.sol"
    queue = [Path(p).relative_to(owner).as_posix() for p in targets] if targets_only else [
        p.relative_to(unit_source).as_posix() for p in (source / scope).rglob("*.sol")
        if not source_file_excluded(p.relative_to(source))
    ]
    sources = {}
    origins = {}
    sorted_remaps = sorted(remappings.items(), key=lambda item: len(item[0]), reverse=True)
    while queue:
        key = queue.pop()
        if key in sources:
            continue
        path = unit_source / key
        if key.startswith("node_modules/"):
            name = key[len("node_modules/"):]
            alias = next(a for a in sorted(deps, key=len, reverse=True) if name.startswith(a + "/"))
            path = deps[alias] / name[len(alias) + 1:]
        content = path.read_text()
        sources[key] = {"content": content}
        origins[key] = {"path": str(path.resolve()), "sha256": hashlib.sha256(content.encode()).hexdigest()}
        for imported in IMPORTS.findall(content):
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(key), imported)) if imported.startswith(".") else imported
            for before, after in sorted_remaps:
                if resolved.startswith(before):
                    resolved = after + resolved[len(before):]
                    break
            queue.append(resolved)
    work.mkdir(parents=True)
    for key, value in sources.items():
        path = work / "source" / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value["content"])
    request = {
        "language": "Solidity", "sources": sources,
        "settings": {
            "optimizer": {"enabled": True, "runs": 200},
            "remappings": [a + "=" + b for a, b in sorted_remaps],
            "outputSelection": {"*": {"*": ["abi", "evm.bytecode", "evm.deployedBytecode"], "": ["ast"]}},
        },
    }
    if via_ir:
        request["settings"]["viaIR"] = True
    compiler = Path.home() / ".solc-select/artifacts" / ("solc-" + version) / ("solc-" + version)
    write_json(work / "input.json", request)
    write_json(work / "source_manifest.json", origins)
    completed = subprocess.run([str(compiler), "--standard-json"], input=json.dumps(request), capture_output=True, text=True, timeout=1200)
    output = json.loads(completed.stdout)
    write_json(work / "compiler_output.json", output)
    errors = [e["formattedMessage"] for e in output.get("errors", []) if e["severity"] == "error"]
    if errors:
        raise RuntimeError("\n".join(errors))
    build_info = work / "source/artifacts/build-info"
    write_json(build_info / "original-source.json", {"solcVersion": version, "input": request, "output": output})
    command = [sys.executable, str(Path(__file__).with_name("generate_project_graph.py")), str(work / "source"), str(work / "fragment.json"), "--project", project, "--source-dir", Path(scope).name, "--build-info", str(build_info)]
    analyzed = subprocess.run(command, capture_output=True, text=True, timeout=1200)
    (work / "analysis.log").write_text(analyzed.stdout + analyzed.stderr)
    if analyzed.returncode:
        raise RuntimeError(analyzed.stderr[-6000:])
    fragment = json.loads((work / "fragment.json").read_text())
    fragment = prefix_fragment(fragment, owner if owner != "." else "")
    write_json(work / "project_fragment.json", fragment)
    return fragment, {
        "scope": scope, "solc": version, "targets_only": targets_only, "via_ir": via_ir,
        "compiled_original_files": sorted(posixpath.normpath(posixpath.join(owner, k)) for k in sources if not k.startswith("node_modules/")),
        "artifacts": str(work.relative_to(ROOT)),
    }


def merge(original, fragments):
    graph = copy.deepcopy(original)
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = {}
    for fragment in [original, *fragments]:
        for node in fragment["nodes"]:
            if "storage_contexts" not in node:
                raise ValueError("Regenerate graphs with storage_contexts before merging")
            previous = nodes.get(node["id"])
            merged = copy.deepcopy(node)
            if previous is not None:
                for field in ("state_variables_read", "state_variables_written", "storage_contexts"):
                    merged[field] = sorted(set(previous.get(field, [])) | set(node.get(field, [])))
            nodes[node["id"]] = merged
        for edge in fragment["edges"]:
            if edge["type"] == "STATE_DEP":
                continue
            key = (edge["source"], edge["target"], edge["type"], edge.get("relation"), edge.get("call_type"))
            if key not in edges:
                edges[key] = copy.deepcopy(edge)
            else:
                for field in ["state_variables", "storage_contexts", "call_sites", "finding_ids"]:
                    if field in edge:
                        values = {json.dumps(v, sort_keys=True): v for v in edges[key].get(field, []) + edge[field]}
                        edges[key][field] = [values[k] for k in sorted(values)]
    state_edges, omitted_self = state_dependencies(
        {key: set(node.get("state_variables_read", [])) for key, node in nodes.items()},
        {key: set(node.get("state_variables_written", [])) for key, node in nodes.items()},
        {key: set(node["storage_contexts"]) for key, node in nodes.items()},
    )
    for key, edge in state_edges.items():
        edge["state_variables"].sort()
        edge["storage_contexts"].sort()
        edges[(edge["source"], edge["target"], "STATE_DEP", key[2], None)] = edge
    graph["schema_version"] = "1.2"
    graph["nodes"] = sorted(nodes.values(), key=lambda n: n["id"])
    graph["edges"] = sorted(edges.values(), key=lambda e: json.dumps(e, sort_keys=True))
    calls = {json.dumps(c, sort_keys=True): c for f in [original, *fragments] for c in f["unresolved_calls"]}
    graph["unresolved_calls"] = [calls[k] for k in sorted(calls)]
    graph["statistics"] = {
        "nodes": len(nodes), "edges": len(edges),
        "call_edges": sum(e["type"] == "CALL" for e in edges.values()),
        "state_dep_edges": sum(e["type"] == "STATE_DEP" for e in edges.values()),
        "state_dep_edges_by_relation": dict(Counter(e["relation"] for e in edges.values() if e["type"] == "STATE_DEP")),
        "omitted_self_state_dependencies": omitted_self,
        "unresolved_or_out_of_scope_calls": len(calls),
        "unresolved_calls_by_reason": dict(Counter(c["reason"] for c in calls.values())),
    }
    graph["scope"]["coverage_kind"] = "original_plus_compiled_source_units"
    graph["scope"]["state_dependency"] = "same contract storage context, including inherited functions"
    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-csv", type=Path, required=True)
    parser.add_argument("--project", action="append", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--targets-only", action="store_true")
    parser.add_argument("--via-ir", action="store_true")
    args = parser.parse_args()
    mapping = {r["finding_id"]: r for r in json.loads((ROOT / "data/finding_vulnerable_files.json").read_text())}
    rows = [r for r in csv.DictReader(args.coverage_csv.open()) if r["coverage_status"] == "ALL_MISSING"]
    for project in args.project:
        selected = [r for r in rows if r["graph_file"] == project + ".json"]
        assert selected, project
        source = (ROOT / mapping[selected[0]["finding_id"]]["source_root"]).resolve()
        targets = sorted({f.split("::")[0] for r in selected for f in json.loads(r["missing_functions"])})
        groups = defaultdict(list)
        for target in targets:
            parts = Path(target).parts
            index = next(i for i, part in enumerate(parts) if part in {"src", "contracts"} and (i > 0 or project.startswith("code4rena_")))
            groups[Path(*parts[:index + 1]).as_posix()].append(target)
        original_path = GRAPHS / (project + ".json")
        original = json.loads(original_path.read_text())
        result = {"project": project, "original_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(), "units": [], "failures": []}
        fragments = []
        for i, (scope, unit_targets) in enumerate(sorted(groups.items())):
            work = REPORT / args.attempt / project / ("unit-" + str(i))
            assert not work.exists(), work
            version = "0.8.19" if scope.startswith("731e3f7") else VERSIONS[project]
            try:
                fragment, metadata = compile_unit(project, source, scope, unit_targets, work, version, args.targets_only, args.via_ir)
                fragments.append(fragment)
                result["units"].append(metadata)
                print("[unit-ok]", project, scope, len(fragment["nodes"]), len(fragment["edges"]), flush=True)
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                result["failures"].append({"scope": scope, "error": str(error)})
                print("[unit-failed]", project, scope, str(error)[:1800], flush=True)
        if fragments:
            previous = GRAPH_ROOT / (project + ".json")
            base = json.loads(previous.read_text()) if previous.exists() else original
            graph = merge(base, fragments)
            graph["repair"] = result
            write_json(previous, graph)
            ids = {n["id"] for n in graph["nodes"]}
            result["remaining_missing_functions"] = sorted({f for r in selected for f in json.loads(r["missing_functions"]) if f not in ids})
        write_json(REPORT / args.attempt / project / "result.json", result)


if __name__ == "__main__":
    main()
