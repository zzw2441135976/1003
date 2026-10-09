"""Construct turn-specific localization guidance from a fixed causal site."""
from __future__ import annotations

import json
from collections import deque
import sys
from pathlib import Path
import time
from urllib.request import ProxyHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rcs_opsd.artifact_paths import GRAPH_ROOT
TEMPLATE = ROOT / "RCS-OPSD/S_Extractor/S_Constructor.md"
EDGE_TYPES = {"CALL", "STATE_DEP"}


def build_case(turn: dict, mechanism: str, site: dict, audit: str, rollout: dict) -> dict:
    graph = json.loads((GRAPH_ROOT / f"{site['project']}.json").read_text())
    adjacent: dict[str, list[tuple[str, str]]] = {node["id"]: [] for node in graph["nodes"]}
    for edge in graph["edges"]:
        if edge["type"] in EDGE_TYPES:
            adjacent[edge["source"]].append((edge["target"], edge["type"]))
            adjacent[edge["target"]].append((edge["source"], edge["type"]))
    core = site["core_functions"]
    distances = {node: 0 for node in core}
    queue = deque(core)
    while queue:
        current = queue.popleft()
        for neighbor, _ in adjacent[current]:
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    focus = sorted(set(turn["function_focus"]))
    reachable = [node for node in focus if node in distances]
    primary = min(reachable, key=lambda node: (distances[node], node)) if reachable else None
    relations = []
    if primary and distances[primary]:
        for edge in graph["edges"]:
            if edge["type"] not in EDGE_TYPES or primary not in (edge["source"], edge["target"]):
                continue
            neighbor = edge["target"] if edge["source"] == primary else edge["source"]
            if distances.get(neighbor) == distances[primary] - 1:
                relations.append({
                    "from": primary, "to": neighbor, "type": edge["type"],
                    "source": edge["source"], "target": edge["target"],
                    "relation": edge.get("relation"),
                    "state_variables": edge.get("state_variables", []),
                    "storage_contexts": edge.get("storage_contexts", []),
                })
        relations.sort(key=lambda edge: (edge["to"], edge["type"], edge["relation"] or ""))
    state = {
        "focus_turn_id": turn["focus_turn_id"],
        "focus_status": turn["state"],
        "completed_observations": [],
    }
    for previous in rollout["turns"]:
        if previous["turn_id"] == turn["turn_id"]:
            break
        if previous.get("complete"):
            state["completed_observations"].append({
                "turn_id": previous["turn_id"],
                "assistant": previous["assistant"]["content"],
                "tool_calls": [
                    {"tool": call["tool_name"], "args": call["args"], "result": call.get("result")}
                    for call in previous.get("tool_calls", [])
                ],
            })
    target = next((item for item in rollout["turns"] if item["turn_id"] == turn["turn_id"]), None)
    if target is not None and "student_request" in target:
        state = {
            "focus_turn_id": turn["focus_turn_id"],
            "focus_status": turn["state"],
            "messages": target["student_request"]["messages"],
        }
    return {
        "finding_id": turn["finding_id"],
        "turn_id": turn["turn_id"],
        "audit": audit,
        "mechanism": mechanism,
        "causal_core": core,
        "student_state": state,
        "current_focus": focus,
        "primary_focus": primary,
        "distance_to_core": distances.get(primary),
        "repository_relations": relations,
    }


def make_prompt(case: dict) -> str:
    values = {
        "AUDIT": case["audit"],
        "MECHANISM": case["mechanism"],
        "CAUSAL_CORE": case["causal_core"],
        "STUDENT_STATE": case["student_state"],
        "CURRENT_FOCUS": case["current_focus"],
        "DISTANCE_TO_CORE": case["distance_to_core"],
        "REPOSITORY_RELATIONS": case["repository_relations"],
    }
    prompt = TEMPLATE.read_text()
    for name, value in values.items():
        prompt = prompt.replace("{{" + name + "}}", value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
    return prompt


def parse_guidance(content: str) -> dict:
    text = content.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[len("```json\n") : -len("\n```")]
    return json.loads(text)


def valid_guidance(case: dict, guidance: dict) -> bool:
    if not isinstance(guidance, dict):
        return False
    distance = case["distance_to_core"]
    expected = (
        "insufficient" if distance is None
        else "causal_site_reached" if distance == 0 else "localizing"
    )
    if guidance.get("status") != expected:
        return False
    if not isinstance(guidance.get("current_focus_assessment"), str):
        return False
    if not isinstance(guidance.get("guidance"), str):
        return False
    if expected == "localizing":
        options = {
            (edge["to"], edge["type"]) for edge in case["repository_relations"]
        }
        if (guidance.get("next_target"), guidance.get("relation")) not in options:
            return False
        if not guidance["guidance"].strip():
            return False
    elif any(guidance.get(key) for key in ("next_target", "relation", "guidance")):
        return False
    output = (guidance["current_focus_assessment"] + " " + guidance["guidance"]).lower()
    if any(phrase in output for phrase in (
        "causal core", "causal site", "core vulnerability", "vulnerability site",
        "steps from", "steps away", "reference patch", "ground truth", "should be called",
        "should be triggered", "add a call", "fix by", "patch the",
    )):
        return False
    if distance is not None and distance > 1:
        for function in case["causal_core"]:
            file, _, name = function.partition("::")
            contract, _, signature = name.partition("::")
            function_name = signature.split("(")[0].lower()
            if function.lower() in output or (
                (Path(file).name.lower() in output or contract.lower() in output)
                and function_name in output
            ):
                return False
    return True


def run_one(case: dict, output: Path, endpoint: str, model: str, timeout: int, seed: int) -> dict:
    artifact = output / "turns" / f"{case['finding_id']}__turn_{case['turn_id']}"
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "input.json").write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n")
    prompt = make_prompt(case)
    (artifact / "prompt.md").write_text(prompt)
    duration = 0
    if case["distance_to_core"] == 0:
        guidance = {
            "status": "causal_site_reached",
            "current_focus_assessment": (
                f"Current observations include {case['primary_focus']}; check the reported unsafe condition against the current code."
            ),
            "next_target": None, "relation": None, "guidance": "",
        }
    elif not case["causal_core"] or not case["primary_focus"]:
        guidance = {
            "status": "insufficient",
            "current_focus_assessment": "No mapped causal function or connecting repository relation is available.",
            "next_target": None, "relation": None, "guidance": "",
        }
    else:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8192,
            "temperature": 0.6,
            "seed": seed + int(case["turn_id"]),
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": True},
        }
        started = time.monotonic()
        for attempt in range(3):
            if attempt:
                body["messages"][0]["content"] = prompt + "\n\nYour previous output broke the output rules. Return only one valid step; do not mention any privileged target, site, core, graph distance, or repair."
                body["seed"] += 1
            request = Request(
                endpoint.rstrip("/") + "/chat/completions",
                data=json.dumps(body, ensure_ascii=False).encode(),
                headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
                method="POST",
            )
            with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
                raw = json.load(response)
            (artifact / f"response-{attempt + 1}.json").write_text(
                json.dumps(raw, ensure_ascii=False, indent=2) + "\n"
            )
            content = raw["choices"][0]["message"].get("content")
            try:
                guidance = parse_guidance(content) if content else {}
            except json.JSONDecodeError:
                guidance = {}
            if valid_guidance(case, guidance):
                break
        else:
            result = {
                "status": "INVALID",
                "finding_id": case["finding_id"],
                "turn_id": case["turn_id"],
                "reason": "state_guidance_output_rule_violation",
            }
            (artifact / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            return result
        duration = round(time.monotonic() - started, 3)
    result = {
        "status": "GENERATED",
        "finding_id": case["finding_id"],
        "turn_id": case["turn_id"],
        "current_focus": case["primary_focus"],
        "state_conditioned_guidance": guidance,
        "privileged_information": json.dumps(guidance, ensure_ascii=False),
        "duration_seconds": duration,
    }
    (artifact / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result
