import argparse
import json
import math
import random
from collections import defaultdict
import os
from pathlib import Path

from .collator import DEFAULT_CHAT_TEMPLATE, OPSDCollator, load_tokenizer, serialize_sample
from .dataset import IndexedDataset, OPSDJsonlDataset


TRAINER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(os.environ.get("RCS_OPSD_MODEL_PATH", str(TRAINER_ROOT.parents[1] / "models/Qwen3.5-9B")))
DEFAULT_DATA = TRAINER_ROOT.parents[1] / "artifacts/opsd-v2/samples.jsonl"


def _percentile(values, percentile):
    ordered = sorted(values)
    index = math.ceil((percentile / 100) * len(ordered)) - 1
    return ordered[max(0, index)]


def _summary(values):
    return {
        "min": min(values),
        "median": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values),
    }


def run_inspect(args):
    dataset = OPSDJsonlDataset(args.data)
    tokenizer = load_tokenizer(args.model, args.chat_template)
    item = serialize_sample(
        tokenizer,
        dataset[args.index],
        args.max_seq_len or None,
        args.thinking_mode,
    )
    target_ids = [
        token
        for token, label in zip(
            item["student"]["input_ids"], item["student"]["labels"]
        )
        if label != -100
    ]
    report = {
        "finding_id": item["finding_id"],
        "turn_id": item["turn_id"],
        "state": item["state"],
        "enable_thinking": item["enable_thinking"],
        "student_length": len(item["student"]["input_ids"]),
        "teacher_length": len(item["teacher"]["input_ids"]),
        "target_length": len(target_ids),
        "target_token_ids_equal": target_ids
        == [
            token
            for token, label in zip(
                item["teacher"]["input_ids"], item["teacher"]["labels"]
            )
            if label != -100
        ],
        "decoded_target": tokenizer.decode(target_ids, skip_special_tokens=False),
        "student_prefix_tail": item["student"]["rendered_prefix"][-2000:],
        "teacher_prefix_tail": item["teacher"]["rendered_prefix"][-2000:],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")


def run_stats(args):
    dataset = OPSDJsonlDataset(args.data)
    tokenizer = load_tokenizer(args.model, args.chat_template)
    lengths = defaultdict(list)
    states = defaultdict(int)
    for index in range(len(dataset)):
        item = serialize_sample(
            tokenizer, dataset[index], thinking_mode=args.thinking_mode
        )
        lengths["student"].append(len(item["student"]["input_ids"]))
        lengths["teacher"].append(len(item["teacher"]["input_ids"]))
        lengths["target"].append(item["student"]["target_length"])
        states[item["state"]] += 1
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(json.dumps({"processed": index + 1, "total": len(dataset)}))
    report = {
        "samples": len(dataset),
        "states": dict(sorted(states.items())),
        "lengths": {name: _summary(values) for name, values in lengths.items()},
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text + "\n")


def _load_model(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForCausalLM

    model = Qwen3_5ForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    if not hasattr(model, "disable_adapter"):
        raise RuntimeError("installed PEFT does not support model.disable_adapter()")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    model.to(args.device)
    return model


def _route_to_device(route, device):
    return {key: value.to(device) for key, value in route.items()}


def _dual_forward(model, batch, device):
    import torch

    from .loss import (
        forward_kl,
        shifted_target_ids,
        target_prediction_positions,
    )

    student = _route_to_device(batch["student"], device)
    teacher = _route_to_device(batch["teacher"], device)
    student_ids = shifted_target_ids(student["labels"])
    teacher_ids = shifted_target_ids(teacher["labels"])
    if not torch.equal(student_ids, teacher_ids):
        raise AssertionError("student and teacher shifted target IDs differ")

    student_positions = target_prediction_positions(student["labels"])
    teacher_positions = target_prediction_positions(teacher["labels"])

    model.eval()
    with torch.no_grad(), model.disable_adapter():
        teacher_logits = model(
            input_ids=teacher["input_ids"],
            attention_mask=teacher["attention_mask"],
            use_cache=False,
            logits_to_keep=teacher_positions,
        ).logits
        teacher_logits = teacher_logits.detach().cpu()
    torch.cuda.empty_cache()

    model.train()
    student_logits = model(
        input_ids=student["input_ids"],
        attention_mask=student["attention_mask"],
        use_cache=False,
        logits_to_keep=student_positions,
    ).logits
    if teacher_logits.shape[1] != len(teacher_ids):
        raise AssertionError("teacher target logit count does not match target token count")
    if student_logits.shape[1] != len(student_ids):
        raise AssertionError("student target logit count does not match target token count")
    loss = forward_kl(teacher_logits, student_logits)
    del teacher_logits
    return loss, student_logits


def _sft_forward(model, batch, device):
    import torch.nn.functional as F

    from .loss import shifted_target_ids, target_prediction_positions

    student = _route_to_device(batch["student"], device)
    target_ids = shifted_target_ids(student["labels"])
    positions = target_prediction_positions(student["labels"])
    model.train()
    logits = model(
        input_ids=student["input_ids"],
        attention_mask=student["attention_mask"],
        use_cache=False,
        logits_to_keep=positions,
    ).logits
    if logits.shape[1] != len(target_ids):
        raise AssertionError("student target logit count does not match target token count")
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1)), logits


def _grad_norm(parameters):
    import torch

    norms = [p.grad.detach().float().norm(2) for p in parameters if p.grad is not None]
    if not norms:
        return 0.0
    return torch.stack(norms).norm(2).item()


def _training_dataset(args):
    dataset = OPSDJsonlDataset(args.data)
    if args.indices:
        indices = [int(value) for value in args.indices.split(",")]
        return IndexedDataset(dataset, indices)
    if not args.limit or args.limit >= len(dataset):
        return dataset
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    return IndexedDataset(dataset, indices[: args.limit])


def run_training(args, single_sample=False):
    import torch
    from peft import load_peft_weights, set_peft_model_state_dict
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = load_tokenizer(args.model, args.chat_template)
    dataset = OPSDJsonlDataset(args.data) if single_sample else _training_dataset(args)
    if single_sample:
        dataset = IndexedDataset(dataset, [args.index])
    collator = OPSDCollator(
        tokenizer, args.max_seq_len or None, thinking_mode=args.thinking_mode
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=not single_sample,
        collate_fn=collator,
        num_workers=0,
    )

    model = _load_model(args)
    resume_state = None
    if args.resume_adapter:
        adapter_state = load_peft_weights(str(args.resume_adapter), device="cpu")
        set_peft_model_state_dict(model, adapter_state)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    if args.resume_state:
        resume_state = torch.load(args.resume_state, map_location="cpu")
        optimizer.load_state_dict(resume_state["optimizer"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "model": str(args.model),
        "data": str(args.data),
        "samples": len(dataset),
        "thinking_mode": args.thinking_mode,
        "seed": args.seed,
        "epochs": args.epochs,
        "gradient_accumulation": args.gradient_accumulation,
        "max_seq_len": args.max_seq_len,
        "learning_rate": args.learning_rate,
        "objective": args.objective,
        "repair_action_weight": args.repair_action_weight,
        "non_repair_action_weight": args.non_repair_action_weight,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n"
    )
    metrics_path = output_dir / "train_metrics.jsonl"

    resume_skip_steps = args.skip_steps
    global_step = args.skip_steps if not single_sample else 0
    if resume_state is not None:
        resume_skip_steps = resume_state["micro_step"]
        global_step = resume_state["global_step"]
    micro_step = 0
    window_loss = 0.0
    window_count = 0
    state_loss = defaultdict(float)
    state_count = defaultdict(int)
    first_logits = None

    optimizer.zero_grad(set_to_none=True)
    stop = False
    for epoch in range(args.epochs):
        for batch in loader:
            if not single_sample and micro_step < resume_skip_steps:
                micro_step += 1
                continue
            if args.objective == "sft":
                loss, student_logits = _sft_forward(model, batch, args.device)
            else:
                loss, student_logits = _dual_forward(model, batch, args.device)
            sample_weight = (
                args.repair_action_weight
                if batch["repair_action"][0]
                else args.non_repair_action_weight
            )
            loss = loss * sample_weight
            if not torch.isfinite(loss) or loss.item() <= 0:
                raise AssertionError(f"KL must be finite and positive, got {loss.item()}")
            if single_sample and first_logits is None:
                first_logits = student_logits.detach().clone()

            (loss / args.gradient_accumulation).backward()
            micro_step += 1
            loss_value = loss.item()
            window_loss += loss_value
            window_count += 1
            state = batch["state"][0]
            state_loss[state] += loss_value
            state_count[state] += 1

            should_step = micro_step % args.gradient_accumulation == 0
            if should_step:
                grad_norm = _grad_norm(trainable)
                if grad_norm == 0:
                    raise AssertionError("LoRA gradient norm is zero")
                if any(
                    parameter.grad is not None
                    for parameter in model.parameters()
                    if not parameter.requires_grad
                ):
                    raise AssertionError("a frozen base-model parameter received a gradient")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0 or single_sample:
                    metrics = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "finding_id": batch["finding_id"][0],
                        "turn_id": batch["turn_id"][0],
                        "state": batch["state"][0],
                        f"train/{args.objective}_loss": window_loss / window_count,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/grad_norm": grad_norm,
                    }
                    for name in sorted(state_count):
                        metrics[f"{args.objective}/{name}"] = (
                            state_loss[name] / state_count[name]
                        )
                    line = json.dumps(metrics, ensure_ascii=False)
                    print(line, flush=True)
                    with metrics_path.open("a") as handle:
                        handle.write(line + "\n")
                    window_loss = 0.0
                    window_count = 0
                    state_loss.clear()
                    state_count.clear()

                if args.save_every and global_step % args.save_every == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(checkpoint_dir)
                    torch.save(
                        {
                            "optimizer": optimizer.state_dict(),
                            "global_step": global_step,
                            "micro_step": micro_step,
                            "epoch": epoch + 1,
                        },
                        checkpoint_dir / "trainer_state.pt",
                    )
                if args.max_steps and global_step >= args.max_steps:
                    stop = True
                    break
            if single_sample:
                stop = True
                break
        if stop:
            break

    if micro_step % args.gradient_accumulation and not single_sample:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    if single_sample:
        batch = next(iter(loader))
        with torch.no_grad():
            if args.objective == "sft":
                _, updated_logits = _sft_forward(model, batch, args.device)
            else:
                _, updated_logits = _dual_forward(model, batch, args.device)
        changed = not torch.equal(first_logits, updated_logits)
        if not changed:
            raise AssertionError("student target logits did not change after optimizer step")
        report = {
            "status": "PASS",
            "finding_id": batch["finding_id"][0],
            "turn_id": batch["turn_id"][0],
            "student_target_tokens": batch["student"]["labels"].ne(-100).sum().item(),
            "teacher_target_tokens": batch["teacher"]["labels"].ne(-100).sum().item(),
            "student_logits_changed": changed,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        model.save_pretrained(output_dir / "final-adapter")
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "micro_step": micro_step,
            },
            output_dir / "trainer_state.pt",
        )


def run_evaluate(args):
    import torch
    from peft import load_peft_weights, set_peft_model_state_dict
    from torch.utils.data import DataLoader

    tokenizer = load_tokenizer(args.model, args.chat_template)
    dataset = OPSDJsonlDataset(args.data)
    collator = OPSDCollator(
        tokenizer, args.max_seq_len or None, thinking_mode=args.thinking_mode
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    model = _load_model(args)
    if args.adapter:
        adapter_state = load_peft_weights(str(args.adapter), device="cpu")
        set_peft_model_state_dict(model, adapter_state)

    total_loss = 0.0
    state_loss = defaultdict(float)
    state_count = defaultdict(int)
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            loss, _ = _dual_forward(model, batch, args.device)
            value = loss.item()
            total_loss += value
            state = batch["state"][0]
            state_loss[state] += value
            state_count[state] += 1
            if args.progress_every and index % args.progress_every == 0:
                print(json.dumps({"evaluated": index, "total": len(dataset)}), flush=True)

    report = {
        "samples": len(dataset),
        "adapter": str(args.adapter) if args.adapter else None,
        "thinking_mode": args.thinking_mode,
        "validation/kl_loss": total_loss / len(dataset),
        "validation/by_state": {
            state: state_loss[state] / state_count[state]
            for state in sorted(state_count)
        },
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description="Minimal OPSD KL trainer")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    common.add_argument("--data", type=Path, default=DEFAULT_DATA)
    common.add_argument("--chat-template", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    common.add_argument("--max-seq-len", type=int, default=0)
    common.add_argument("--thinking-mode", choices=("auto", "off"), default="auto")
    common.add_argument("--learning-rate", type=float, default=5e-6)

    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", parents=[common])
    inspect_parser.add_argument("--index", type=int, default=0)
    inspect_parser.add_argument("--output", type=Path)
    inspect_parser.set_defaults(function=run_inspect)

    stats_parser = subparsers.add_parser("stats", parents=[common])
    stats_parser.add_argument(
        "--output", type=Path, default=TRAINER_ROOT / "token_statistics.json"
    )
    stats_parser.add_argument("--progress-every", type=int, default=100)
    stats_parser.set_defaults(function=run_stats)

    evaluate_parser = subparsers.add_parser("evaluate", parents=[common])
    evaluate_parser.add_argument("--device", default="cuda")
    evaluate_parser.add_argument("--adapter", type=Path)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--progress-every", type=int, default=10)
    evaluate_parser.add_argument("--gradient-checkpointing", action="store_true")
    evaluate_parser.set_defaults(function=run_evaluate)

    for name, single_sample in (("single", True), ("train", False)):
        train_parser = subparsers.add_parser(name, parents=[common])
        train_parser.add_argument("--device", default="cuda")
        train_parser.add_argument("--output-dir", type=Path, required=True)
        train_parser.add_argument("--seed", type=int, default=1701)
        train_parser.add_argument("--epochs", type=int, default=1)
        train_parser.add_argument("--gradient-accumulation", type=int, default=1)
        train_parser.add_argument("--gradient-checkpointing", action="store_true")
        train_parser.add_argument("--log-every", type=int, default=10)
        train_parser.add_argument("--save-every", type=int, default=0)
        train_parser.add_argument("--max-steps", type=int, default=0)
        train_parser.add_argument("--skip-steps", type=int, default=0)
        train_parser.add_argument("--resume-adapter", type=Path)
        train_parser.add_argument("--resume-state", type=Path)
        train_parser.add_argument("--limit", type=int, default=0)
        train_parser.add_argument(
            "--indices",
            help="Comma-separated dataset indices for a fixed overfit subset",
        )
        train_parser.add_argument("--index", type=int, default=0)
        train_parser.add_argument("--repair-action-weight", type=float, default=1.0)
        train_parser.add_argument("--non-repair-action-weight", type=float, default=1.0)
        train_parser.add_argument("--objective", choices=("forward-kl", "sft"), default="forward-kl")
        train_parser.set_defaults(
            function=lambda args, single=single_sample: run_training(args, single)
        )
    return parser


def main():
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
