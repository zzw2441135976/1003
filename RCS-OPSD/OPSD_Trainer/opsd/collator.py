import json
from copy import deepcopy
from pathlib import Path


IGNORE_INDEX = -100
DEFAULT_CHAT_TEMPLATE = Path(__file__).resolve().parents[3] / "templates/chat_template_codex.jinja"

PI_TEMPLATE = """Privileged repair information:
{pi}

Continue from the preceding student conversation and the same response prefix.
Use this guidance silently to choose the next useful code inspection, repair, or
verification action. Check proposed edits against actual code and tool results;
do not assume the original vulnerability remains after student edits. Preserve
intended legitimate behavior and use the student's existing tool-call format.
Do not output this guidance or refer to privileged information or a reference patch."""

STATE_TEACHER_TEMPLATE = """You are an autonomous smart-contract vulnerability repair agent.
Continue the repair from the exact student state in the preceding conversation.

Repository:
{repository}

The preceding conversation is the student's on-policy trajectory immediately before this decision.
The audit, existing edits, and tool results are in that conversation. Do not start
a new repair trajectory or assume access to unseen tool results.
Mapped State: {state}
Current Program Focus:
{focus}

## Training-Only Privileged Information
Original Vulnerability Mechanism (check whether it still applies to current code):
{mechanism}

{stage_information}

Use this information silently to choose the next useful action. It may direct an
inspection of code the student has not yet read; inspect that code before relying
on its behavior for an edit. Ground concrete edits in the current code and tool
results, account for changes already made, and preserve intended legitimate behavior.
Do not copy an assumed reference implementation, output the privileged guidance,
or mention privileged information, a reference repair, or ground-truth locations.

Continue the same assistant response using the student's text and tool-call format.
Choose ordinary repository tools; include a brief evidence-based reason only when
useful. Do not output a separate analysis, a replacement conversation, or a claim
of successful repair without verification evidence."""

ROLLOUT_SYSTEM_PROMPT = """You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
- read: Read file contents
- bash: Execute bash commands (ls, grep, find, etc.)
- edit: Make precise file edits with exact text replacement, including multiple disjoint edits in one call
- write: Create or overwrite files
- grep: Search file contents for patterns (respects .gitignore)
- find: Find files by glob pattern (respects .gitignore)
- ls: List directory contents

In addition to the tools above, you may have access to other custom tools depending on the project.

Guidelines:
- Use read to examine files instead of cat or sed.
- You can inspect PI_* environment variables for current model and session details.
- Use edit for precise changes (edits[].oldText must match exactly)
- When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls
- Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.
- Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.
- Use write only for new files or complete rewrites.
- Be concise in your responses
- Show file paths clearly when working with files

Pi documentation (read only when the user asks about pi itself, its SDK, extensions, themes, skills, or TUI):
- Main documentation: /opt/pi/node_modules/@earendil-works/pi-coding-agent/README.md
- Additional docs: /opt/pi/node_modules/@earendil-works/pi-coding-agent/docs
- Examples: /opt/pi/node_modules/@earendil-works/pi-coding-agent/examples (extensions, custom tools, SDK)
- When reading pi docs or examples, resolve docs/... under Additional docs and examples/... under Examples, not the current working directory
- When asked about: extensions (docs/extensions.md, examples/extensions/), themes (docs/themes.md), skills (docs/skills.md), prompt templates (docs/prompt-templates.md), TUI components (docs/tui.md), keybindings (docs/keybindings.md), SDK integrations (docs/sdk.md), custom providers (docs/custom-provider.md), adding models (docs/models.md), pi packages (docs/packages.md), environment variables (docs/environment-variables.md)
- When working on pi topics, read the docs and examples, and follow .md cross-references before implementing
- Always read pi .md files completely and follow links to related docs (e.g., tui.md for TUI API details)
Current working directory: /testbed"""

ROLLOUT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
                    "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
                    "limit": {"type": "number", "description": "Maximum number of lines to read"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last 2000 lines or 50KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.",
            "parameters": {
                "type": "object",
                "required": ["path", "edits"],
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
                    "edits": {
                        "type": "array",
                        "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead.",
                        "items": {
                            "type": "object",
                            "required": ["oldText", "newText"],
                            "properties": {
                                "oldText": {"type": "string", "description": "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call."},
                                "newText": {"type": "string", "description": "Replacement text for this targeted edit."},
                            },
                        },
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.",
            "parameters": {
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
                    "content": {"type": "string", "description": "Content to write to the file"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents for a pattern. Returns matching lines with file paths and line numbers. Respects .gitignore. Output is truncated to 100 matches or 50KB (whichever is hit first). Long lines are truncated to 500 chars.",
            "parameters": {
                "type": "object",
                "required": ["pattern"],
                "properties": {
                    "pattern": {"type": "string", "description": "Search pattern (regex or literal string)"},
                    "path": {"type": "string", "description": "Directory or file to search (default: current directory)"},
                    "glob": {"type": "string", "description": "Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'"},
                    "ignoreCase": {"type": "boolean", "description": "Case-insensitive search (default: false)"},
                    "literal": {"type": "boolean", "description": "Treat pattern as literal string instead of regex (default: false)"},
                    "context": {"type": "number", "description": "Number of lines to show before and after each match (default: 0)"},
                    "limit": {"type": "number", "description": "Maximum number of matches to return (default: 100)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": "Search for files by glob pattern. Returns matching file paths relative to the search directory. Respects .gitignore. Output is truncated to 1000 results or 50KB (whichever is hit first).",
            "parameters": {
                "type": "object",
                "required": ["pattern"],
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern to match files, e.g. '*.ts', '**/*.json', or 'src/**/*.spec.ts'"},
                    "path": {"type": "string", "description": "Directory to search in (default: current directory)"},
                    "limit": {"type": "number", "description": "Maximum number of results (default: 1000)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ls",
            "description": "List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. Includes dotfiles. Output is truncated to 500 entries or 50KB (whichever is hit first).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to list (default: current directory)"},
                    "limit": {"type": "number", "description": "Maximum number of entries to return (default: 500)"},
                },
            },
        },
    },
]


def load_tokenizer(model_path, chat_template_path=DEFAULT_CHAT_TEMPLATE):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    tokenizer.chat_template = Path(chat_template_path).read_text()
    return tokenizer


def _assistant_message(raw_message, include_reasoning=True):
    reasoning = []
    text = []
    tool_calls = []
    for part in raw_message.get("content", []):
        part_type = part.get("type")
        if part_type == "thinking" and include_reasoning:
            reasoning.append(part.get("thinking", ""))
        elif part_type == "text":
            text.append(part.get("text", ""))
        elif part_type == "toolCall":
            tool_calls.append(
                {
                    "id": part.get("id"),
                    "type": "function",
                    "function": {
                        "name": part["name"],
                        "arguments": part.get("arguments", {}),
                    },
                }
            )
    message = {
        "role": "assistant",
        "content": "".join(text),
        "reasoning_content": "".join(reasoning),
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _tool_message(call):
    result = call["result"]
    content = result.get("content", []) if isinstance(result, dict) else result
    return {
        "role": "tool",
        "tool_call_id": call["tool_call_id"],
        "content": content,
    }


def _history_messages(sample, include_reasoning=True):
    if "student_request" in sample:
        messages = deepcopy(sample["student_request"]["messages"])
        # vLLM parses OpenAI JSON argument strings before applying this template.
        for message in messages:
            for call in message.get("tool_calls", []):
                function = call["function"]
                if isinstance(function["arguments"], str):
                    function["arguments"] = json.loads(function["arguments"])
        return messages
    messages = [{"role": "system", "content": ROLLOUT_SYSTEM_PROMPT}]
    for entry in sample["history"]:
        if entry.get("role") == "user":
            messages.append({"role": "user", "content": entry["content"]})
            continue
        messages.append(_assistant_message(entry["assistant"], include_reasoning))
        messages.extend(_tool_message(call) for call in entry.get("tool_calls", []))
    return messages


def _thinking_enabled(sample):
    messages = [entry.get("assistant", {}) for entry in sample["history"]]
    messages.append(sample["target"]["assistant_message"])
    return any(
        part.get("type") == "thinking"
        for message in messages
        for part in message.get("content", [])
    )


def _render(tokenizer, messages, add_generation_prompt, enable_thinking, tools=ROLLOUT_TOOLS):
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


def _serialize_route(tokenizer, prefix_text, target_text, action_span):
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    encoded = tokenizer(
        prefix_text + target_text, add_special_tokens=False, return_offsets_mapping=True
    )
    full_ids = encoded["input_ids"]
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise AssertionError("tokenized target is not a continuation of the prefix")
    target_ids = full_ids[len(prefix_ids) :]
    if not target_ids:
        raise AssertionError("empty target token sequence")

    action_start, action_end = (len(prefix_text) + offset for offset in action_span)
    target_labels = [
        token if action_start <= start < end <= action_end else IGNORE_INDEX
        for token, (start, end) in zip(
            target_ids, encoded["offset_mapping"][len(prefix_ids):]
        )
    ]

    return {
        "input_ids": full_ids,
        "labels": [IGNORE_INDEX] * len(prefix_ids) + target_labels,
        "prefix_length": len(prefix_ids),
        "target_length": len(target_ids),
        "action_length": sum(label != IGNORE_INDEX for label in target_labels),
        "rendered_prefix": prefix_text,
        "rendered_target": target_text,
    }


def serialize_sample(tokenizer, sample, max_seq_len=None, thinking_mode="auto"):
    if thinking_mode not in {"auto", "off"}:
        raise ValueError(f"unsupported thinking mode: {thinking_mode}")
    enable_thinking = thinking_mode == "auto" and _thinking_enabled(sample)
    request = sample.get("student_request")
    tools = request.get("tools", []) if request is not None else ROLLOUT_TOOLS
    if request is not None:
        enable_thinking = request["chat_template_kwargs"]["enable_thinking"]
    history = _history_messages(sample, include_reasoning=enable_thinking)
    target = _assistant_message(
        sample["target"]["assistant_message"], include_reasoning=enable_thinking
    )

    history_text = _render(tokenizer, history, False, enable_thinking, tools)
    student_prefix = _render(tokenizer, history, True, enable_thinking, tools)
    full_text = _render(tokenizer, history + [target], False, enable_thinking, tools)
    if not student_prefix.startswith(history_text):
        raise AssertionError("generation prompt is not a continuation of history")
    if not full_text.startswith(student_prefix):
        raise AssertionError("target is not a continuation of the generation prompt")

    generation_prompt = student_prefix[len(history_text) :]
    target_text = full_text[len(student_prefix) :]
    # Locate tool calls after the actual assistant prose/reasoning, so literal
    # tool tags in those fields cannot accidentally become supervised actions.
    prose = {key: value for key, value in target.items() if key != "tool_calls"}
    prose_text = _render(tokenizer, history + [prose], False, enable_thinking, tools)[len(student_prefix):]
    message_end = "<|im_end|>\n"
    prose_end = len(prose_text) - len(message_end)
    if target.get("tool_calls"):
        action_start = target_text.index("<tool_call>", prose_end)
        action_span = (action_start, len(target_text) - len(message_end))
    else:
        action_span = (0, 0)
    if "state_conditioned_guidance" in sample:
        if sample["state"] == "LOCATED":
            stage_information = (
                "Repair Invariant:\n" + sample["repair_invariant"]
                + "\n\nThe mapped region has been observed; this does not establish understanding "
                "or repair success. Use M and I to check the current code, identify any remaining "
                "violation, and choose a necessary edit or verification. Permit alternative "
                "implementations that satisfy the security condition."
            )
        else:
            stage_information = (
                "Localization Guidance:\n"
                + json.dumps(sample["state_conditioned_guidance"], ensure_ascii=False)
                + "\n\nUse M and this guidance to choose the next focused search or inspection "
                "and test the mechanism-related hypothesis. Missing graph connectivity is "
                "inconclusive. Verify call direction and storage context from code; structural "
                "relations alone do not prove runtime behavior."
            )
        pi_text = STATE_TEACHER_TEMPLATE.format(
            repository=sample["repository"],
            state=sample["state"],
            focus=json.dumps(sample["metadata"]["focus"], ensure_ascii=False),
            mechanism=sample["mechanism"],
            stage_information=stage_information,
        )
    else:
        pi_text = PI_TEMPLATE.format(pi=sample["privileged_information"]).strip()
    teacher_prefix = (
        history_text
        + f"<|im_start|>user\n{pi_text}<|im_end|>\n"
        + generation_prompt
    )

    student = _serialize_route(tokenizer, student_prefix, target_text, action_span)
    teacher = _serialize_route(tokenizer, teacher_prefix, target_text, action_span)
    history_ids = tokenizer.encode(history_text, add_special_tokens=False)
    for route in (student, teacher):
        if route["input_ids"][:len(history_ids)] != history_ids:
            raise AssertionError("route does not preserve the common history tokens")
    if max_seq_len and any(len(route["input_ids"]) > max_seq_len for route in (student, teacher)):
        raise ValueError("Complete on-policy context exceeds max_seq_len; exclude this sample instead of truncating")
    if student["input_ids"][student["prefix_length"]:] != teacher["input_ids"][teacher["prefix_length"]:]:
        raise AssertionError("student and teacher complete response token IDs differ")

    student_target = [
        token for token, label in zip(student["input_ids"], student["labels"])
        if label != IGNORE_INDEX
    ]
    teacher_target = [
        token for token, label in zip(teacher["input_ids"], teacher["labels"])
        if label != IGNORE_INDEX
    ]
    if student_target != teacher_target:
        raise AssertionError("student and teacher target token IDs differ")

    return {
        "finding_id": sample["finding_id"],
        "turn_id": sample["turn_id"],
        "state": sample["state"],
        "repair_action": bool(sample.get("metadata", {}).get("repair_action")),
        "enable_thinking": enable_thinking,
        "common_history_length": len(history_ids),
        "history_tokens_removed": 0,
        "student": student,
        "teacher": teacher,
    }


class OPSDCollator:
    def __init__(self, tokenizer, max_seq_len=None, thinking_mode="auto"):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.thinking_mode = thinking_mode

    def __call__(self, samples):
        import torch

        serialized = [
            serialize_sample(
                self.tokenizer, sample, self.max_seq_len, self.thinking_mode
            )
            for sample in samples
        ]
        if any(item["student"]["action_length"] == 0 for item in serialized):
            raise ValueError("OPSD training samples must contain tool-action tokens")
        batch = {
            "finding_id": [item["finding_id"] for item in serialized],
            "turn_id": [item["turn_id"] for item in serialized],
            "state": [item["state"] for item in serialized],
            "repair_action": [item["repair_action"] for item in serialized],
        }
        pad_id = self.tokenizer.pad_token_id
        for route in ("student", "teacher"):
            max_length = max(len(item[route]["input_ids"]) for item in serialized)
            input_ids = []
            labels = []
            attention_mask = []
            for item in serialized:
                ids = item[route]["input_ids"]
                item_labels = item[route]["labels"]
                padding = max_length - len(ids)
                input_ids.append(ids + [pad_id] * padding)
                labels.append(item_labels + [IGNORE_INDEX] * padding)
                attention_mask.append([1] * len(ids) + [0] * padding)
            batch[route] = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        return batch
