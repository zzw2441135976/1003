#!/usr/bin/env python3
"""Alternate current-student repair rollouts and PI-guided KL updates."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import ProxyHandler, Request, build_opener


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "RCS-OPSD/Focus_Mapper"))
sys.path.insert(0, str(ROOT / "RCS-OPSD/PI_Constructor"))
sys.path.insert(0, str(ROOT / "RCS-OPSD/S_Extractor"))
sys.path.insert(0, str(HERE))

from map_turn_focus import map_finding  # noqa: E402
from pi_inputs import REPAIR_TOOLS  # noqa: E402
import pi_inputs as pi_constructor  # noqa: E402
from build_single_sample import build_sample  # noqa: E402
from build_causal_site import build_distance_index, build_site  # noqa: E402
import state_guidance  # noqa: E402
from opsd.collator import load_tokenizer, serialize_sample  # noqa: E402


COLLECTOR = ROOT / "RCS-OPSD/Rollout_collector/run_rollout_repair.py"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def service_models(endpoint: str) -> set[str]:
    with build_opener(ProxyHandler({})).open(endpoint.rstrip("/") + "/models", timeout=30) as response:
        return {entry["id"] for entry in json.load(response)["data"]}


def load_adapter(endpoint: str, name: str, adapter: Path) -> None:
    body = json.dumps({"lora_name": name, "lora_path": str(adapter.resolve())}).encode()
    request = Request(
        endpoint.rstrip("/") + "/load_lora_adapter",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with build_opener(ProxyHandler({})).open(request, timeout=120):
        pass
    if name not in service_models(endpoint):
        raise RuntimeError(f"adapter {name} is absent from {endpoint}/models")


def unload_adapter(endpoint: str, name: str) -> None:
    body = json.dumps({"lora_name": name}).encode()
    request = Request(
        endpoint.rstrip("/") + "/unload_lora_adapter",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with build_opener(ProxyHandler({})).open(request, timeout=120):
        pass


def next_turn_selections(focus: dict, rollout: dict) -> list[dict]:
    turns = rollout["turns"]
    observations = {turn["turn_id"]: turn for turn in focus["turns"]}
    current = {"focus_status": "UNRESOLVED", "function_focus": [], "distance": None}
    focus_turn_id = None
    selected = []
    for index, target in enumerate(turns):
        # Only completed observations preceding this response may determine PI.
        if index and turns[index - 1].get("complete"):
            previous = observations.get(turns[index - 1]["turn_id"])
            if previous and previous["function_focus"]:
                current = previous
                focus_turn_id = previous["turn_id"]
        tools = [
            part["name"] for part in target["assistant"].get("content", [])
            if part.get("type") == "toolCall"
        ]
        if not target.get("complete") or not tools:
            continue
        if "student_request" in target:
            # Compaction may have removed the observations defining old focus.
            visible = {
                message["tool_call_id"] for message in target["student_request"]["messages"]
                if message["role"] == "tool"
            }
            current = {"focus_status": "UNRESOLVED", "function_focus": [], "distance": None}
            focus_turn_id = None
            for prior in turns[:index]:
                observation = observations.get(prior["turn_id"])
                if not prior.get("complete") or observation is None:
                    continue
                functions = sorted({
                    function for evidence in observation["evidence"]
                    if evidence["tool_call_id"] in visible
                    for function in evidence["function_ids"]
                })
                if not functions:
                    continue
                distances = observation["function_distances"]
                finite = [distances[function] for function in functions if distances.get(function) is not None]
                distance = min(finite) if finite else None
                status = ("LOCATED" if distance == 0 else "LOCALIZING") if finite else (
                    "UNREACHABLE" if distances else "NO_CAUSAL_TARGET"
                )
                current = {"focus_status": status, "function_focus": functions, "distance": distance}
                focus_turn_id = prior["turn_id"]
        selected.append({
            "finding_id": focus["finding_id"],
            "turn_id": target["turn_id"],
            "focus_turn_id": focus_turn_id,
            "state": current["focus_status"],
            "function_focus": current["function_focus"],
            "distance": current["distance"],
            "repair_action": any(tool in REPAIR_TOOLS for tool in tools),
            "selection_reasons": ["target_tool_action", "focus_from_prior_observations"],
        })
    return selected


def collect(args: argparse.Namespace, directory: Path, model_id: str, round_index: int) -> Path:
    rollout_root = directory / "rollouts"
    environment = {key: value for key, value in os.environ.items() if key.lower() not in {
        "http_proxy", "https_proxy", "all_proxy"
    }}

    def collect_one(finding_id: str) -> None:
        output = rollout_root / finding_id
        command = [
            sys.executable, str(COLLECTOR), "--finding", finding_id,
            "--seed", str(args.seed + round_index), "--thinking", args.thinking,
            "--endpoint", args.endpoint, "--model-id", model_id,
            "--pi-timeout", str(args.pi_timeout),
            "--output-dir", str(output),
        ]
        output.parent.mkdir(parents=True, exist_ok=True)
        log_path = output.parent / f"{finding_id}.collector.log"
        with log_path.open("w") as log:
            subprocess.run(command, check=True, env=environment, stdout=log, stderr=subprocess.STDOUT)
        rollout = json.loads((output / "rollout.json").read_text())
        if rollout["model"] != f"local-qwen/{model_id}":
            raise RuntimeError(f"rollout for {finding_id} used {rollout['model']}, expected {model_id}")

    with ThreadPoolExecutor(max_workers=args.rollout_workers) as executor:
        list(executor.map(collect_one, args.finding))
    return rollout_root


def prepare_samples(
    args: argparse.Namespace, directory: Path, rollout_root: Path, model_id: str
) -> Path:
    pi_root = directory / "pi_state_v2"
    site_root = directory.parent / "causal_sites"
    site_root.mkdir(exist_ok=True)
    sites = {}
    audits = {}
    for finding_id in args.finding:
        site_path = site_root / f"{finding_id}.json"
        index_path = site_root / f"{finding_id}.distance_index.json"
        sites[finding_id] = build_site(finding_id)
        write_json(site_path, sites[finding_id])
        write_json(index_path, build_distance_index(sites[finding_id]))
        audits[finding_id] = (
            ROOT / "dataset_root/findings" / finding_id / "task/full_prompt.md"
        ).read_text()
    selected = []
    for finding_id in args.finding:
        rollout = json.loads((rollout_root / finding_id / "rollout.json").read_text())
        if any(turn.get("complete") and "student_request" not in turn for turn in rollout["turns"]):
            raise RuntimeError(f"{finding_id}: recollect rollouts with captured student requests; reconstructed history is insufficient")
        focus = map_finding(
            finding_id, rollout_root, site_root / f"{finding_id}.distance_index.json"
        )
        selected.extend(next_turn_selections(focus, rollout))

    if not selected:
        raise RuntimeError("current rollouts produced no eligible next-turn samples")
    selected_path = directory / "selected_turns.jsonl"
    selected_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))

    m_records = pi_constructor.by_finding(pi_constructor.M_RESULTS)
    i_records = pi_constructor.by_finding(pi_constructor.I_RESULTS)
    samples_path = directory / "samples_state_v2.jsonl"
    tokenizer = load_tokenizer(args.base_model, args.chat_template)

    def generate(row: dict) -> dict:
        finding_id = row["finding_id"]
        rollout = json.loads((rollout_root / finding_id / "rollout.json").read_text())
        case = state_guidance.build_case(
            row, m_records[finding_id]["mechanism"], sites[finding_id],
            audits[finding_id], rollout,
        )
        result_path = pi_root / "turns" / f"{finding_id}__turn_{row['turn_id']}" / "result.json"
        input_path = result_path.with_name("input.json")
        prompt_path = result_path.with_name("prompt.md")
        if (
            result_path.exists() and json.loads(input_path.read_text()) == case
            and prompt_path.read_text() == state_guidance.make_prompt(case)
        ):
            cached = json.loads(result_path.read_text())
            if cached["status"] == "INVALID" or state_guidance.valid_guidance(
                case, cached["state_conditioned_guidance"]
            ):
                return cached
        return state_guidance.run_one(
            case, pi_root, args.pi_endpoint, args.pi_model, args.pi_timeout, args.seed
        )

    excluded_path = directory / "excluded_state_guidance_v2.jsonl"
    with ThreadPoolExecutor(max_workers=args.rollout_workers) as executor, samples_path.open("w") as stream, excluded_path.open("w") as excluded:
        for offset in range(0, len(selected), args.rollout_workers):
            batch = selected[offset : offset + args.rollout_workers]
            futures = [executor.submit(generate, row) for row in batch]
            for row, future in zip(batch, futures):
                finding_id = row["finding_id"]
                result = future.result()
                if result["status"] != "GENERATED":
                    excluded.write(json.dumps({
                        "finding_id": finding_id, "turn_id": row["turn_id"],
                        "reason": result["reason"],
                    }) + "\n")
                    continue
                sample, checks = build_sample(
                    finding_id, row["turn_id"], selected_path=selected_path,
                    rollout_root=rollout_root, prompt_root=rollout_root, pi_root=pi_root / "turns",
                )
                if not checks["all_pass"]:
                    excluded.write(json.dumps({
                        "finding_id": finding_id, "turn_id": row["turn_id"],
                        "reason": "sample_validation_failed", "checks": checks,
                    }) + "\n")
                    continue
                sample["metadata"]["policy_model_id"] = model_id
                sample["metadata"]["focus_turn_id"] = row["focus_turn_id"]
                sample["metadata"]["causal_site"] = str((site_root / f"{finding_id}.json").relative_to(ROOT))
                sample["audit_report"] = audits[finding_id]
                sample["repository"] = f"/testbed ({sites[finding_id]['project']})"
                sample["mechanism"] = m_records[finding_id]["mechanism"]
                sample["state_conditioned_guidance"] = result["state_conditioned_guidance"]
                sample["repair_invariant"] = i_records[finding_id]["invariant"]
                routes = serialize_sample(tokenizer, sample, thinking_mode="auto" if args.thinking == "medium" else "off")
                lengths = {name: len(routes[name]["input_ids"]) for name in ("student", "teacher")}
                if max(lengths.values()) > args.max_seq_len:
                    excluded.write(json.dumps({
                        "finding_id": finding_id, "turn_id": row["turn_id"],
                        "reason": "complete_context_exceeds_max_seq_len",
                        "lengths": lengths, "max_seq_len": args.max_seq_len,
                    }) + "\n")
                    continue
                stream.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
    return samples_path


def train(args: argparse.Namespace, directory: Path, samples: Path, adapter: Path | None) -> Path:
    output = directory / "training"
    with samples.open() as stream:
        sample_count = sum(1 for _ in stream)
    if not sample_count:
        raise RuntimeError("No eligible complete-context samples; inspect exclusions before training")
    command = [
        str(args.train_python), "-m", "opsd.train", "train",
        "--data", str(samples), "--output-dir", str(output),
        "--model", str(args.base_model), "--chat-template", str(args.chat_template),
        "--thinking-mode", "auto" if args.thinking == "medium" else "off",
        "--max-seq-len", str(args.max_seq_len), "--learning-rate", str(args.learning_rate),
        "--epochs", "1", "--gradient-accumulation", str(sample_count),
        "--gradient-checkpointing", "--save-every", "0", "--log-every", "1",
        "--seed", str(args.seed),
    ]
    if adapter:
        command.extend(("--resume-adapter", str(adapter)))
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.train_gpu)
    subprocess.run(command, check=True, cwd=HERE, env=environment)
    return output / "final-adapter"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finding", action="append", default=[])
    parser.add_argument("--findings-file", type=Path, help="JSONL with finding_id entries")
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--rollout-workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:9531/v1")
    parser.add_argument("--base-model-id", default="qwen3.5-9b")
    parser.add_argument("--pi-endpoint")
    parser.add_argument("--pi-model", default="qwen3.5-9b")
    parser.add_argument("--initial-adapter", type=Path)
    parser.add_argument("--resume-round-0-rollouts", action="store_true")
    parser.add_argument("--resume-round-1", action="store_true")
    parser.add_argument("--resume-round-1-pi", action="store_true")
    parser.add_argument("--base-model", type=Path, default=Path(os.environ.get("RCS_OPSD_MODEL_PATH", str(ROOT / "models/Qwen3.5-9B"))))
    parser.add_argument("--train-python", type=Path, default=HERE / ".venv-cu130/bin/python")
    parser.add_argument("--train-gpu", type=int, required=True)
    parser.add_argument("--chat-template", type=Path, default=ROOT / "templates/chat_template_codex.jinja")
    parser.add_argument("--thinking", choices=("medium", "off"), default="medium")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--pi-timeout", type=int, default=1800)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    args = parser.parse_args()
    args.pi_endpoint = args.pi_endpoint or args.endpoint
    if args.findings_file:
        with args.findings_file.open() as stream:
            args.finding.extend(json.loads(line)["finding_id"] for line in stream if line.strip())
    args.finding = list(dict.fromkeys(args.finding))
    if not args.finding:
        parser.error("provide --finding or --findings-file")
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.rollout_workers < 1:
        parser.error("--rollout-workers must be positive")
    output = args.output_dir.resolve()
    if not output.is_relative_to(HERE):
        parser.error("--output-dir must be inside OPSD_Trainer so the model service can read adapters")
    if output.exists() and any(output.iterdir()) and not (args.resume_round_0_rollouts or args.resume_round_1 or args.resume_round_1_pi):
        parser.error("--output-dir must be empty")
    output.mkdir(parents=True, exist_ok=True)
    adapter = args.initial_adapter.resolve() if args.initial_adapter else None
    if adapter and not adapter.is_relative_to(HERE):
        parser.error("--initial-adapter must be inside OPSD_Trainer so the model service can read it")
    if args.resume_round_1:
        first = output / "round-000"
        policy = json.loads((first / "policy.json").read_text())
        result = json.loads((first / "result.json").read_text())
        if policy["adapter"] != str(adapter) or policy["findings"] != args.finding:
            parser.error("round-000 policy does not match the requested adapter and findings")
        adapter = Path(result["updated_adapter"])
        if result["sample_policy"] != policy["model_id"] or not (adapter / "adapter_model.safetensors").is_file():
            parser.error("round-000 training result is incomplete")
        second = output / "round-001"
        if second.exists() and any(second.iterdir()):
            parser.error("round-001 must be empty to resume")
    if args.resume_round_1_pi:
        first = output / "round-000"
        policy = json.loads((first / "policy.json").read_text())
        result = json.loads((first / "result.json").read_text())
        second = output / "round-001"
        second_policy = json.loads((second / "policy.json").read_text())
        if policy["adapter"] != str(adapter) or policy["findings"] != args.finding:
            parser.error("round-000 policy does not match the requested adapter and findings")
        adapter = Path(result["updated_adapter"])
        if result["sample_policy"] != policy["model_id"] or not (adapter / "adapter_model.safetensors").is_file():
            parser.error("round-000 training result is incomplete")
        if second_policy["round"] != 1 or second_policy["findings"] != args.finding:
            parser.error("round-001 policy does not match the requested findings")
    if args.resume_round_0_rollouts:
        directory = output / "round-000"
        policy = json.loads((directory / "policy.json").read_text())
        expected_model = "opsd-student-000" if adapter else args.base_model_id
        if policy != {
            "round": 0, "model_id": expected_model,
            "adapter": str(adapter) if adapter else None,
            "findings": args.finding,
        }:
            parser.error("round-000/policy.json does not match the requested policy")
        if (output / "round-001").exists() or (directory / "result.json").exists():
            parser.error("round 0 has already advanced beyond rollout collection")
        rollout_root = directory / "rollouts"
        for finding_id in args.finding:
            rollout = json.loads((rollout_root / finding_id / "rollout.json").read_text())
            if rollout.get("model") != f"local-qwen/{expected_model}":
                parser.error(f"rollout for {finding_id} used a different model")

    for round_index in range(1 if (args.resume_round_1 or args.resume_round_1_pi) else 0, args.rounds):
        directory = output / f"round-{round_index:03d}"
        if round_index == 0 and args.resume_round_0_rollouts:
            model_id = expected_model
            rollout_root = directory / "rollouts"
        elif args.resume_round_1_pi and round_index == 1:
            directory.mkdir(exist_ok=True)
            model_id = "opsd-student-001"
            rollout_root = directory / "rollouts"
        else:
            directory.mkdir(exist_ok=args.resume_round_1 and round_index == 1)
            model_id = args.base_model_id
            if adapter:
                model_id = f"opsd-student-{round_index:03d}"
                load_adapter(args.endpoint, model_id, adapter)
            elif model_id not in service_models(args.endpoint):
                raise RuntimeError(f"base model {model_id} is absent from {args.endpoint}/models")
            write_json(directory / "policy.json", {
                "round": round_index, "model_id": model_id,
                "adapter": str(adapter) if adapter else None,
                "findings": args.finding,
            })
            rollout_root = collect(args, directory, model_id, round_index)
            if adapter:
                unload_adapter(args.endpoint, model_id)
        samples = prepare_samples(args, directory, rollout_root, model_id)
        updated = train(args, directory, samples, adapter)
        with samples.open() as stream:
            sample_count = sum(1 for _ in stream)
        write_json(directory / "result.json", {
            "sample_count": sample_count,
            "sample_policy": model_id,
            "updated_adapter": str(updated),
        })
        adapter = updated
    print(json.dumps({"final_adapter": str(adapter), "rounds": args.rounds}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
