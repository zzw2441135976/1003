#!/usr/bin/env python3
"""Map one frozen rollout's repository interactions to per-turn function focus."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from collections import Counter, defaultdict
import sys
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rcs_opsd.artifact_paths import GRAPH_ROOT, GRAPH_INDEX
FINAL_ROLLOUT = ROOT / "RCS-OPSD/Rollout_collector/final_rollout/findings"
DATASET_FINDINGS = ROOT / "dataset_root/findings"
AST_PARSER = Path(__file__).resolve().with_name("parse_solidity_locations.js")

TESTBED_PREFIXES = ("/testbed/repo/", "/testbed/")
FILE_LINE_RE = re.compile(r"^(.*?\.[A-Za-z0-9_+-]+):(\d+)(?::|$)")
TAIL_RE = re.compile(r"(?:^|\s)tail\s+-(\d+)\s+([^|;&]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("finding_id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rollout-root", type=Path, default=FINAL_ROLLOUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def result_text(call: dict[str, Any]) -> str:
    result = call.get("result") or {}
    return "\n".join(
        block.get("text", "")
        for block in result.get("content", [])
        if block.get("type") == "text"
    )


def normalize_path(raw_path: str, base: str | None = None) -> str | None:
    path = raw_path.strip().strip("'\"").replace("\\", "/")
    if not path:
        return None
    if not path.startswith("/") and base:
        path = f"{base.rstrip('/')}/{path}"
    for prefix in TESTBED_PREFIXES:
        if path.startswith(prefix):
            path = path[len(prefix) :]
            break
    if path.startswith("/"):
        return None
    normalized = PurePosixPath(path)
    if ".." in normalized.parts:
        return None
    value = normalized.as_posix()
    return None if value in {"", "."} else value.removeprefix("./")


def line_count(text: str) -> int:
    return len(text.splitlines())


def line_range_for_text(content: str, needle: str) -> tuple[int, int, int]:
    translation = str.maketrans({"\u2018": "'", "\u2019": "'"})
    searchable = content.translate(translation)
    normalized_needle = needle.translate(translation)
    start = searchable.find(normalized_needle)
    if start < 0 or searchable.find(normalized_needle, start + 1) >= 0:
        raise ValueError("edit oldText is missing or not unique in the current file")
    start_line = content.count("\n", 0, start) + 1
    end_line = start_line + needle.count("\n")
    return start, start_line, end_line


def index_nodes(nodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        if node.get("file") and node.get("start_line") and node.get("end_line"):
            by_file[node["file"]].append(node)
    for entries in by_file.values():
        entries.sort(key=lambda node: (node["start_line"], node["end_line"], node["id"]))
    return dict(by_file)


class Mapper:
    def __init__(
        self,
        finding_id: str,
        rollout: dict[str, Any],
        graph: dict[str, Any],
        distance_index: dict[str, Any] | None,
        source_root: Path,
    ) -> None:
        self.finding_id = finding_id
        self.rollout = rollout
        self.graph = graph
        self.distance_index = distance_index
        self.source_root = source_root
        self.causal_targets_available = distance_index is not None
        self.canonical_ids = {node["id"] for node in graph["nodes"]}
        self.graph_files = {node["file"] for node in graph["nodes"]}
        self.function_index = index_nodes(graph["nodes"])
        self.original_index = index_nodes(graph["nodes"])
        self.dirty_files: set[str] = set()
        self.contents: dict[str, str] = {}
        self.file_revisions: dict[str, int] = defaultdict(int)
        self.parsed_revisions: dict[str, int] = {}
        self.baseline_symbols: dict[str, dict[str, str]] = {}
        self.parse_count = 0
        self.parse_errors: list[dict[str, Any]] = []
        self.alignment_counts: Counter[str] = Counter()

    def current_content(self, file: str) -> str | None:
        if file in self.contents:
            return self.contents[file]
        path = self.source_root / file
        try:
            return path.read_text() if path.is_file() else None
        except (OSError, UnicodeDecodeError):
            return None

    def coordinate_system(self, file: str) -> str:
        return "runtime" if file in self.dirty_files else "baseline"

    def parse_locations(
        self, file: str, content: str, phase: str
    ) -> list[dict[str, Any]] | None:
        completed = subprocess.run(
            ["node", str(AST_PARSER)],
            input=content,
            capture_output=True,
            text=True,
        )
        self.parse_count += 1
        if completed.returncode != 0:
            self.parse_errors.append(
                {
                    "file": file,
                    "phase": phase,
                    "revision": self.file_revisions[file],
                    "stderr": completed.stderr[-4000:],
                }
            )
            return None
        return json.loads(completed.stdout)

    @staticmethod
    def descriptor_matches_node(
        descriptor: dict[str, Any], node: dict[str, Any]
    ) -> bool:
        if node.get("contract") != descriptor["contract"]:
            return False
        if node.get("function") != descriptor["name"]:
            return False
        if descriptor["kind"] == "state_variable_getter":
            return node.get("special") == "state_variable_getter"
        return node.get("special") != "state_variable_getter"

    def baseline_symbol_map(self, file: str) -> dict[str, str]:
        if file in self.baseline_symbols:
            return self.baseline_symbols[file]
        path = self.source_root / file
        if not path.is_file():
            self.baseline_symbols[file] = {}
            return {}
        descriptors = self.parse_locations(file, path.read_text(), "baseline") or []
        mapping: dict[str, str] = {}
        ambiguous: set[str] = set()
        for descriptor in descriptors:
            candidates = [
                node
                for node in self.original_index.get(file, [])
                if self.descriptor_matches_node(descriptor, node)
                and node["start_line"] <= descriptor["end_line"]
                and descriptor["start_line"] <= node["end_line"]
            ]
            if len(candidates) != 1:
                self.alignment_counts[
                    "ambiguous" if candidates else "unmatched_descriptor"
                ] += 1
                continue
            key = descriptor["symbol_key"]
            identifier = candidates[0]["id"]
            if key in mapping and mapping[key] != identifier:
                ambiguous.add(key)
            else:
                mapping[key] = identifier
                self.alignment_counts["matched"] += 1
        for key in ambiguous:
            mapping.pop(key, None)
            self.alignment_counts["duplicate_symbol_key"] += 1
        self.baseline_symbols[file] = mapping
        return mapping

    def reparse_file_if_needed(self, file: str) -> None:
        revision = self.file_revisions[file]
        if self.parsed_revisions.get(file) == revision:
            return
        self.parsed_revisions[file] = revision
        content = self.current_content(file)
        if content is None:
            self.function_index[file] = []
            return
        descriptors = self.parse_locations(file, content, "runtime")
        if descriptors is None:
            self.function_index[file] = []
            return
        symbols = self.baseline_symbol_map(file)
        nodes = []
        for descriptor in descriptors:
            identifier = symbols.get(descriptor["symbol_key"])
            if identifier is None:
                continue
            nodes.append(
                {
                    "id": identifier,
                    "start_line": descriptor["start_line"],
                    "end_line": descriptor["end_line"],
                }
            )
        self.function_index[file] = nodes

    def functions_for_range(self, file: str, start: int, end: int) -> list[str]:
        if file not in self.graph_files:
            return []
        if file in self.dirty_files:
            self.reparse_file_if_needed(file)
        return [
            node["id"]
            for node in self.function_index.get(file, [])
            if node["start_line"] <= end and start <= node["end_line"]
        ]

    def evidence(
        self,
        call: dict[str, Any],
        file: str,
        start: int | None = None,
        end: int | None = None,
        coordinate_system: str | None = None,
    ) -> dict[str, Any]:
        functions: list[str] = []
        if start is not None and end is not None:
            functions = self.functions_for_range(file, start, end)
        return {
            "tool": call["tool_name"],
            "tool_call_id": call["tool_call_id"],
            "scope": "line" if start is not None else "file",
            "file": file,
            "start_line": start,
            "end_line": end,
            "coordinate_system": coordinate_system or self.coordinate_system(file),
            "function_ids": functions,
        }

    def read_evidence(self, call: dict[str, Any]) -> list[dict[str, Any]]:
        args = call.get("args") or {}
        file = normalize_path(str(args.get("path") or args.get("file_path") or ""))
        if file is None:
            return []
        text = result_text(call)
        count = line_count(text)
        if count == 0:
            return [self.evidence(call, file)]
        start = int(args.get("offset", 1))
        returned_lines = int(args["limit"]) if args.get("limit") is not None else count
        return [self.evidence(call, file, start, start + returned_lines - 1)]

    def grep_evidence(self, call: dict[str, Any]) -> list[dict[str, Any]]:
        args = call.get("args") or {}
        raw_base = str(args.get("path") or ".")
        base_file = normalize_path(raw_base)
        output: list[dict[str, Any]] = []
        for line in result_text(call).splitlines():
            match = FILE_LINE_RE.match(line.strip())
            if match:
                file = normalize_path(match.group(1), raw_base)
                if file:
                    number = int(match.group(2))
                    output.append(self.evidence(call, file, number, number))
                continue
            number_only = re.match(r"^(\d+):", line.strip())
            if number_only and base_file:
                number = int(number_only.group(1))
                output.append(self.evidence(call, base_file, number, number))
        return output

    def find_evidence(self, call: dict[str, Any]) -> list[dict[str, Any]]:
        args = call.get("args") or {}
        raw_base = str(args.get("path") or ".")
        output: list[dict[str, Any]] = []
        for line in result_text(call).splitlines():
            file = normalize_path(line.rstrip("/"), raw_base)
            if file:
                output.append(self.evidence(call, file))
        return output

    def edit_evidence(self, call: dict[str, Any]) -> list[dict[str, Any]]:
        args = call.get("args") or {}
        file = normalize_path(str(args.get("path") or args.get("file_path") or ""))
        edits = args.get("edits") or []
        if file is None or not isinstance(edits, list) or call.get("is_error"):
            return []
        content = self.current_content(file)
        if content is None:
            return [self.evidence(call, file)]
        coordinates = self.coordinate_system(file)
        replacements: list[tuple[int, int, str]] = []
        output: list[dict[str, Any]] = []
        for edit in edits:
            old_text = edit.get("oldText")
            new_text = edit.get("newText")
            if not isinstance(old_text, str) or not isinstance(new_text, str):
                continue
            position, start, end = line_range_for_text(content, old_text)
            replacements.append((position, position + len(old_text), new_text))
            output.append(self.evidence(call, file, start, end, coordinates))
        for start, end, replacement in sorted(replacements, reverse=True):
            content = content[:start] + replacement + content[end:]
        self.contents[file] = content
        self.dirty_files.add(file)
        if file in self.graph_files:
            self.file_revisions[file] += 1
        return output

    def write_evidence(self, call: dict[str, Any]) -> list[dict[str, Any]]:
        args = call.get("args") or {}
        file = normalize_path(str(args.get("path") or args.get("file_path") or ""))
        content = args.get("content")
        if file is None or not isinstance(content, str) or call.get("is_error"):
            return []
        self.contents[file] = content
        self.dirty_files.add(file)
        if file in self.graph_files:
            self.file_revisions[file] += 1
        count = line_count(content)
        if count == 0:
            return [self.evidence(call, file)]
        return [self.evidence(call, file, 1, count, "runtime")]

    def restore_baseline_path(self, raw_path: str, base: str | None) -> list[str]:
        if raw_path in {".", "./"} and base in {"/testbed", "/testbed/repo"}:
            restored = list(self.contents)
            for path in restored:
                self.contents.pop(path, None)
                self.dirty_files.discard(path)
                self.function_index[path] = self.original_index.get(path, [])
                self.parsed_revisions.pop(path, None)
            return restored
        file = normalize_path(raw_path, base)
        if file is None:
            return []
        source = self.source_root / file
        if source.is_file():
            restored = [file]
        elif source.is_dir():
            prefix = file.rstrip("/") + "/"
            restored = [path for path in self.contents if path.startswith(prefix)]
        else:
            return []
        for path in restored:
            self.contents.pop(path, None)
            self.dirty_files.discard(path)
            self.function_index[path] = self.original_index.get(path, [])
            self.parsed_revisions.pop(path, None)
        return restored

    def replay_git_restore(self, command: str) -> list[str]:
        cwd: str | None = None
        restored: list[str] = []
        for segment in re.split(r"\s*(?:&&|;)\s*", command):
            try:
                tokens = shlex.split(segment)
            except ValueError:
                continue
            if not tokens:
                continue
            if tokens[0] == "cd" and len(tokens) >= 2:
                cwd = tokens[1]
                continue
            if tokens[0] != "git":
                continue
            command_cwd = cwd
            if "-C" in tokens:
                index = tokens.index("-C")
                if index + 1 < len(tokens):
                    command_cwd = tokens[index + 1]
            action_index = next(
                (i for i, token in enumerate(tokens) if token in {"checkout", "restore"}),
                None,
            )
            if action_index is None:
                continue
            for token in tokens[action_index + 1 :]:
                if token == "--" or token.startswith("-"):
                    continue
                if token in {"2>/dev/null", "2>&1"}:
                    break
                restored.extend(self.restore_baseline_path(token, command_cwd))
        return sorted(set(restored))

    @staticmethod
    def shell_segments(command: str) -> list[tuple[str | None, list[str]]]:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        lexer.commenters = ""
        try:
            command_tokens = list(lexer)
        except ValueError:
            return []
        cwd: str | None = None
        output: list[tuple[str | None, list[str]]] = []
        segments: list[list[str]] = [[]]
        for token in command_tokens:
            if token in {"&&", ";"}:
                segments.append([])
            else:
                segments[-1].append(token)
        for tokens in segments:
            if not tokens:
                continue
            if tokens[0] == "cd" and len(tokens) >= 2:
                cwd = tokens[1]
            else:
                output.append((cwd, tokens))
        return output

    def replay_sed_in_place(
        self, call: dict[str, Any], command: str
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for cwd, tokens in self.shell_segments(command):
            if tokens[0] != "sed" or not any(
                token == "-i" or token.startswith("-i") for token in tokens[1:]
            ):
                continue
            tokens = tokens[: next((i for i, token in enumerate(tokens) if token in {"|", ">", ">>"}), len(tokens))]
            files: list[tuple[str, str]] = []
            for token in tokens[1:]:
                if "\n" in token or len(token) > 1024:
                    continue
                file = normalize_path(token, cwd)
                if file and self.current_content(file) is not None:
                    files.append((token, file))
            if len(files) != 1:
                continue
            raw_file, file = files[0]
            sed_args = [
                token
                for token in tokens[1:]
                if token != raw_file and token != "-i" and not token.startswith("-i")
            ]
            before = self.current_content(file)
            if before is None:
                continue
            completed = subprocess.run(
                ["sed", *sed_args], input=before, capture_output=True, text=True
            )
            if completed.returncode != 0:
                continue
            before_lines = before.splitlines()
            after_lines = completed.stdout.splitlines()
            prefix = 0
            while (
                prefix < len(before_lines)
                and prefix < len(after_lines)
                and before_lines[prefix] == after_lines[prefix]
            ):
                prefix += 1
            suffix = 0
            while (
                suffix < len(before_lines) - prefix
                and suffix < len(after_lines) - prefix
                and before_lines[-suffix - 1] == after_lines[-suffix - 1]
            ):
                suffix += 1
            coordinates = self.coordinate_system(file)
            if before != completed.stdout:
                start = prefix + 1
                end = max(start, len(before_lines) - suffix)
                output.append(self.evidence(call, file, start, end, coordinates))
                self.contents[file] = completed.stdout
                self.dirty_files.add(file)
                if file in self.graph_files:
                    self.file_revisions[file] += 1
            else:
                output.append(self.evidence(call, file))
        return output

    def explicit_shell_files(self, command: str) -> list[str]:
        files: set[str] = set()
        for cwd, tokens in self.shell_segments(command):
            for token in tokens:
                if "\n" in token or len(token) > 1024:
                    continue
                file = normalize_path(token, cwd)
                if file and self.current_content(file) is not None:
                    files.add(file)
        return sorted(files)

    def reconcile_sed_output(
        self, file: str, start: int, end: int, observed: str
    ) -> None:
        content = self.current_content(file)
        if content is None:
            return
        lines = content.splitlines()
        observed_lines = observed.splitlines()
        lines[start - 1 : min(end, len(lines))] = observed_lines
        reconciled = "\n".join(lines) + ("\n" if content.endswith("\n") else "")
        if reconciled == content:
            return
        self.contents[file] = reconciled
        self.dirty_files.add(file)
        if file in self.graph_files:
            self.file_revisions[file] += 1

    def bash_evidence(self, call: dict[str, Any]) -> list[dict[str, Any]]:
        args = call.get("args") or {}
        command = str(args.get("command") or "")
        text = result_text(call)
        output = self.replay_sed_in_place(call, command)

        for file in self.replay_git_restore(command):
            output.append(self.evidence(call, file, coordinate_system="baseline"))

        explicit_files = self.explicit_shell_files(command)
        sed_print = re.search(r"\bsed\s+-n\s+['\"]?(\d+)(?:,(\d+))?p['\"]?", command)
        if sed_print and len(explicit_files) == 1 and "|" not in command:
            start = int(sed_print.group(1))
            end = int(sed_print.group(2) or start)
            self.reconcile_sed_output(explicit_files[0], start, end, text)
            output.append(self.evidence(call, explicit_files[0], start, end))

        if "grep" in command and "-n" in command and len(explicit_files) == 1:
            for line in text.splitlines():
                match = re.match(r"^(\d+)[:-]", line.strip())
                if match:
                    number = int(match.group(1))
                    output.append(self.evidence(call, explicit_files[0], number, number))

        tail = TAIL_RE.search(command)
        if tail:
            file = normalize_path(tail.group(2).strip())
            if file:
                content = self.current_content(file)
                if content is not None:
                    total = line_count(content)
                    start = max(1, total - int(tail.group(1)) + 1)
                    output.append(self.evidence(call, file, start, total))

        for line in text.splitlines():
            stripped = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
            stripped = stripped.removeprefix("--> ").strip()
            match = FILE_LINE_RE.match(stripped)
            if match:
                file = normalize_path(match.group(1))
                if file:
                    number = int(match.group(2))
                    output.append(self.evidence(call, file, number, number))
                continue
            if "find " in command and stripped:
                file = normalize_path(stripped)
                if file:
                    output.append(self.evidence(call, file))
        return output

    def call_evidence(self, call: dict[str, Any]) -> list[dict[str, Any]]:
        if not call.get("complete") or call.get("is_error"):
            return []
        handlers = {
            "read": self.read_evidence,
            "grep": self.grep_evidence,
            "find": self.find_evidence,
            "edit": self.edit_evidence,
            "write": self.write_evidence,
            "bash": self.bash_evidence,
        }
        handler = handlers.get(call.get("tool_name"))
        return handler(call) if handler else []

    def map_turn(self, turn: dict[str, Any]) -> dict[str, Any]:
        evidence = [
            item
            for call in turn.get("tool_calls", [])
            for item in self.call_evidence(call)
        ]
        file_focus = sorted({item["file"] for item in evidence})
        function_focus = sorted(
            {function for item in evidence for function in item["function_ids"]}
        )
        distances = (
            {
                function: self.distance_index["nodes"][function]["distance"]
                for function in function_focus
            }
            if self.causal_targets_available
            else {}
        )
        finite = [distance for distance in distances.values() if distance is not None]
        if not function_focus:
            status = "UNRESOLVED"
            distance = None
            reachability = None
        elif not self.causal_targets_available:
            status = "NO_CAUSAL_TARGET"
            distance = None
            reachability = None
        elif not finite:
            status = "UNREACHABLE"
            distance = None
            reachability = "unreachable"
        else:
            distance = min(finite)
            status = "LOCATED" if distance == 0 else "LOCALIZING"
            reachability = "reachable"
        return {
            "turn_id": turn["turn_id"],
            "turn_complete": turn.get("complete", False),
            "file_focus": file_focus,
            "function_focus": function_focus,
            "focus": function_focus,
            "evidence": evidence,
            "focus_status": status,
            "distance": distance,
            "reachability": reachability,
            "function_distances": distances,
        }

    def run(self) -> dict[str, Any]:
        completed = [turn for turn in self.rollout["turns"] if turn.get("complete")]
        skipped_partial = [
            turn["turn_id"] for turn in self.rollout["turns"] if not turn.get("complete")
        ]
        turns = [self.map_turn(turn) for turn in completed]
        return {
            "finding_id": self.finding_id,
            "repair_status": self.rollout["repair_status"],
            "capture_complete": self.rollout["capture_complete"],
            "inputs": {
                "rollout": str(
                    FINAL_ROLLOUT / self.finding_id / "rollout.json"
                ).removeprefix(str(ROOT) + "/"),
                "project_graph": str(
                    GRAPH_ROOT.relative_to(ROOT)
                    / f"{self.graph['project']}.json"
                ),
                "distance_index": (
                    str(GRAPH_INDEX / self.finding_id / "distance_index.json").removeprefix(
                        str(ROOT) + "/"
                    )
                    if self.causal_targets_available
                    else None
                ),
                "source": str(
                    DATASET_FINDINGS / self.finding_id / "task/source"
                ).removeprefix(str(ROOT) + "/"),
            },
            "method": {
                "turn_boundary": "PI turn_start/turn_end normalized as rollout.turns[]",
                "thinking_used_for_focus": False,
                "focus_accumulates_across_turns": False,
                "file_only_maps_to_function": False,
                "dirty_file_coordinates": (
                    "dependency-free single-file AST reparse, then baseline symbol-key match"
                ),
                "distance_rule": "minimum finite Distance Index value over function_focus",
                "empty_causal_functions": "NO_CAUSAL_TARGET when function_focus is non-empty",
                "partial_turns": "skip turns without turn_end",
            },
            "turns": turns,
            "summary": {
                "source_turns": len(self.rollout["turns"]),
                "turns": len(turns),
                "skipped_partial_turn_ids": skipped_partial,
                "status_counts": dict(Counter(turn["focus_status"] for turn in turns)),
                "evidence": sum(len(turn["evidence"]) for turn in turns),
                "solidity_ast_parses": self.parse_count,
                "solidity_ast_parse_errors": self.parse_errors,
                "baseline_alignment": dict(self.alignment_counts),
                "dirty_files": sorted(self.dirty_files),
            },
        }


def map_finding(
    finding_id: str, rollout_root: Path = FINAL_ROLLOUT,
    distance_index_path: Path | None = None,
) -> dict[str, Any]:
    rollout_path = rollout_root / finding_id / "rollout.json"
    distance_path = distance_index_path or GRAPH_INDEX / finding_id / "distance_index.json"
    source_root = (DATASET_FINDINGS / finding_id / "task/source").resolve()
    rollout = load_json(rollout_path)
    distance_index = load_json(distance_path) if distance_path.is_file() else None
    project = (
        distance_index["project"] if distance_index else finding_id.rsplit("__", 1)[0]
    )
    graph_path = GRAPH_ROOT / f"{project}.json"
    graph = load_json(graph_path)
    result = Mapper(
        finding_id,
        rollout,
        graph,
        distance_index,
        source_root,
    ).run()
    if distance_index_path is not None:
        result["inputs"]["distance_index"] = (
            str(distance_index_path.resolve().relative_to(ROOT)) if distance_index is not None else None
        )
    return result


def main() -> None:
    args = parse_args()
    finding_id = args.finding_id
    result = map_finding(finding_id, args.rollout_root)

    output = args.output or Path(__file__).resolve().parent / "runs" / finding_id / "focus_status.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(output)


if __name__ == "__main__":
    main()
