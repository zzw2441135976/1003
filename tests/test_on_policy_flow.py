from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "RCS-OPSD/OPSD_Trainer/on_policy.py"
spec = importlib.util.spec_from_file_location("on_policy", SCRIPT)
on_policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(on_policy)


class OnPolicySelectionTests(unittest.TestCase):
    def test_reference_patch_builds_one_hop_core(self):
        import build_causal_site
        with tempfile.TemporaryDirectory() as tmp:
            graph_root = Path(tmp)
            core = 'Vault.sol::Vault::withdraw()'
            neighbor = 'Vault.sol::Vault::debit()'
            far = 'Vault.sol::Vault::deposit()'
            graph = {'nodes': [
                {'id': name, 'file': 'Vault.sol', 'start_line': line, 'end_line': line + 2}
                for name, line in [(core, 4), (neighbor, 10), (far, 20)]
            ], 'edges': [
                {'source': core, 'target': neighbor, 'type': 'CALL'},
                {'source': neighbor, 'target': far, 'type': 'STATE_DEP'},
            ]}
            (graph_root / 'fixture.json').write_text(json.dumps(graph))
            task = graph_root / 'findings/fixture__H-1/task'; task.mkdir(parents=True)
            (task / 'reference_patch.diff').write_text('--- a/Vault.sol\n+++ b/Vault.sol\n@@ -4 +4 @@\n-old\n+new\n')
            with patch.object(build_causal_site, 'GRAPH_ROOT', graph_root), \
                 patch.object(build_causal_site, 'FINDINGS', graph_root / 'findings'):
                site = on_policy.build_site('fixture__H-1')
        self.assertEqual(site['patch_functions'], [core])
        self.assertEqual(set(site['core_functions']), {core, neighbor})
        self.assertNotIn(far, site['target_subgraph']['nodes'])

    def test_guidance_must_follow_graph_edge_without_leaking_core(self):
        case = {
            "distance_to_core": 2,
            "repository_relations": [{"to": "next-function", "type": "CALL"}],
            "causal_core": ["src/Vault.sol::Vault::withdraw()"],
        }
        guidance = {
            "status": "localizing",
            "current_focus_assessment": "The current balance check does not explain the withdrawal flow.",
            "next_target": "next-function", "relation": "CALL",
            "guidance": "Inspect the next function's balance reads.",
        }
        self.assertTrue(on_policy.state_guidance.valid_guidance(case, guidance))
        self.assertFalse(on_policy.state_guidance.valid_guidance(
            case, {**guidance, "next_target": "invented-function"}
        ))
        self.assertFalse(on_policy.state_guidance.valid_guidance(
            case, {**guidance, "guidance": "The causal core is in Vault.withdraw."}
        ))
        self.assertFalse(on_policy.state_guidance.valid_guidance(
            case, {key: value for key, value in guidance.items() if key != "guidance"}
        ))

    def test_reached_core_has_no_next_step_or_repair_guidance(self):
        case = {
            "finding_id": "finding-a", "turn_id": 1,
            "audit": "audit", "mechanism": "mechanism", "causal_core": ["Vault::withdraw"],
            "student_state": {}, "current_focus": ["Vault::withdraw"],
            "primary_focus": "Vault::withdraw", "distance_to_core": 0,
            "repository_relations": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            result = on_policy.state_guidance.run_one(
                case, Path(temporary), "http://invalid", "model", 1, 1701
            )
        guidance = result["state_conditioned_guidance"]
        self.assertEqual(guidance["status"], "causal_site_reached")
        self.assertIsNone(guidance["next_target"])
        self.assertEqual(guidance["guidance"], "")

    def test_fenced_json_guidance_is_parsed(self):
        parsed = on_policy.state_guidance.parse_guidance(
            '\n```json\n{"status":"localizing"}\n```\n'
        )
        self.assertEqual(parsed["status"], "localizing")

    def test_guidance_input_preserves_reverse_edge_and_state_evidence(self):
        graph = {
            "nodes": [{"id": "focus"}, {"id": "core"}],
            "edges": [
                {"source": "core", "target": "focus", "type": "CALL"},
                {"source": "core", "target": "focus", "type": "STATE_DEP",
                 "relation": "WRITE_READ", "state_variables": ["Vault.balances"]},
            ],
        }
        turn = {
            "finding_id": "finding-a", "turn_id": 1, "focus_turn_id": 0,
            "function_focus": ["focus"], "state": "LOCALIZING",
        }
        rollout = {"turns": [{"turn_id": 0, "assistant": {"content": []}, "tool_calls": []}]}
        with tempfile.TemporaryDirectory() as temporary:
            graph_root = Path(temporary)
            (graph_root / "project.json").write_text(json.dumps(graph))
            with patch.object(on_policy.state_guidance, "GRAPH_ROOT", graph_root):
                case = on_policy.state_guidance.build_case(
                    turn, "mechanism", {"project": "project", "core_functions": ["core"]},
                    "audit", rollout,
                )
        self.assertEqual(case["distance_to_core"], 1)
        call, dependency = case["repository_relations"]
        self.assertEqual((call["from"], call["to"]), ("focus", "core"))
        self.assertEqual((call["source"], call["target"]), ("core", "focus"))
        self.assertEqual(dependency["relation"], "WRITE_READ")
        self.assertEqual(dependency["state_variables"], ["Vault.balances"])

    def test_teacher_receives_stage_pi_and_student_does_not(self):
        from opsd.collator import serialize_sample

        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                rendered = "".join(
                    f"{message['role']}:{message.get('content', '')}\n" for message in messages
                )
                return rendered + ("assistant:" if kwargs["add_generation_prompt"] else "")

            def encode(self, text, **kwargs):
                return list(map(ord, text))

            def __call__(self, text, **kwargs):
                return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}

            def decode(self, tokens, **kwargs):
                return "".join(map(chr, tokens))

        sample = {
            "finding_id": "finding-a", "turn_id": 1, "state": "LOCALIZING",
            "history": [{"role": "user", "content": "audit task"}],
            "target": {"assistant_message": {"content": [{"type": "text", "text": "next action"}]}},
            "metadata": {"focus": ["Vault::read"]},
            "audit_report": "audit detail", "repository": "/testbed",
            "mechanism": "original M", "repair_invariant": "original I",
            "state_conditioned_guidance": {"status": "localizing", "guidance": "original S"},
        }
        for state in ("LOCALIZING", "UNREACHABLE", "LOCATED"):
            with self.subTest(state=state):
                routes = serialize_sample(Tokenizer(), {**sample, "state": state})
                teacher = routes["teacher"]["rendered_prefix"]
                student = routes["student"]["rendered_prefix"]
                self.assertIn("original M", teacher)
                self.assertIn("original I" if state == "LOCATED" else "original S", teacher)
                self.assertNotIn("original S" if state == "LOCATED" else "original I", teacher)
                for value in ("original M", "original S", "original I"):
                    self.assertNotIn(value, student)
                self.assertEqual(routes["student"]["rendered_target"], routes["teacher"]["rendered_target"])
                self.assertEqual(routes["student"]["target_length"], routes["teacher"]["target_length"])

    def test_focus_is_taken_from_prior_completed_turn(self):
        focus = {
            "finding_id": "finding-a",
            "turns": [
                {"turn_id": 0, "focus_status": "LOCALIZING", "function_focus": ["Vault::read"], "distance": 1},
                {"turn_id": 1, "focus_status": "LOCATED", "function_focus": ["Vault::fix"], "distance": 0},
            ],
        }
        rollout = {
            "turns": [
                {"turn_id": 0, "complete": True, "assistant": {"content": []}, "tool_calls": []},
                {"turn_id": 1, "complete": True, "assistant": {"content": [
                    {"type": "toolCall", "name": "edit"}
                ]}, "tool_calls": [
                    {"tool_name": "edit", "tool_call_id": "call-1", "complete": True, "is_error": False}
                ]},
                {"turn_id": 2, "complete": True, "assistant": {"content": [
                    {"type": "toolCall", "name": "read"}
                ]}, "tool_calls": []},
            ]
        }
        selected = on_policy.next_turn_selections(focus, rollout)
        self.assertEqual([(row["focus_turn_id"], row["turn_id"]) for row in selected], [(0, 1), (1, 2)])
        self.assertEqual([row["state"] for row in selected], ["LOCALIZING", "LOCATED"])
        self.assertEqual([row["repair_action"] for row in selected], [True, False])

    def test_incomplete_target_is_not_a_training_sample(self):
        focus = {
            "finding_id": "finding-a",
            "turns": [{"turn_id": 0, "focus_status": "LOCATED", "function_focus": ["Vault::fix"], "distance": 0}],
        }
        rollout = {"turns": [
            {"turn_id": 0, "complete": True, "assistant": {"content": []}, "tool_calls": []},
            {"turn_id": 1, "complete": False, "assistant": {"content": []}, "tool_calls": []},
        ]}
        self.assertEqual(on_policy.next_turn_selections(focus, rollout), [])

    def test_action_sampling_keeps_failed_attempts_and_prior_focus_without_leaking_target(self):
        focus = {"finding_id": "finding-a", "turns": [
            {"turn_id": 0, "focus_status": "UNRESOLVED", "function_focus": [], "distance": None},
            {"turn_id": 1, "focus_status": "LOCALIZING", "function_focus": ["old-focus"], "distance": 2},
            {"turn_id": 2, "focus_status": "UNRESOLVED", "function_focus": [], "distance": None},
            {"turn_id": 3, "focus_status": "LOCATED", "function_focus": ["future-core"], "distance": 0},
        ]}
        rollout = {"turns": [
            {"turn_id": 0, "complete": True, "assistant": {"content": [
                {"type": "toolCall", "name": "read"}
            ]}},
            {"turn_id": 1, "complete": True, "assistant": {"content": [
                {"type": "toolCall", "name": "read"}
            ]}},
            {"turn_id": 2, "complete": True, "assistant": {"content": [
                {"type": "text", "text": "explanation only"}
            ]}},
            {"turn_id": 3, "complete": True, "assistant": {"content": [
                {"type": "toolCall", "name": "edit"}
            ]}, "tool_calls": [{"tool_name": "edit", "complete": True, "is_error": True}]},
        ]}
        selected = on_policy.next_turn_selections(focus, rollout)
        self.assertEqual([row["turn_id"] for row in selected], [0, 1, 3])
        self.assertIsNone(selected[0]["focus_turn_id"])
        self.assertEqual(selected[-1]["function_focus"], ["old-focus"])
        self.assertEqual(selected[-1]["focus_turn_id"], 1)
        self.assertEqual(selected[-1]["state"], "LOCALIZING")
        self.assertTrue(selected[-1]["repair_action"])

    def test_guidance_contains_all_prior_completed_observations_and_no_target(self):
        graph = {"nodes": [{"id": "focus"}], "edges": []}
        rollout = {"turns": [
            {"turn_id": i, "complete": True, "assistant": {"content": [str(i)]}, "tool_calls": []}
            for i in range(4)
        ]}
        turn = {"finding_id": "finding-a", "turn_id": 3, "focus_turn_id": 1,
                "function_focus": ["focus"], "state": "LOCATED"}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "project.json").write_text(json.dumps(graph))
            with patch.object(on_policy.state_guidance, "GRAPH_ROOT", directory):
                case = on_policy.state_guidance.build_case(
                    turn, "mechanism", {"project": "project", "core_functions": ["focus"]}, "audit", rollout
                )
        self.assertEqual([row["turn_id"] for row in case["student_state"]["completed_observations"]], [0, 1, 2])

    def test_round_uses_one_update_from_the_collected_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            samples = directory / "samples.jsonl"
            samples.write_text("{}\n{}\n{}\n")
            adapter = directory / "previous-adapter"
            args = SimpleNamespace(
                train_python=Path("/tmp/train-python"), base_model=Path("/tmp/base"),
                chat_template=Path("/tmp/template"), thinking="medium", max_seq_len=32768,
                learning_rate=1e-6, seed=1701, train_gpu=1,
            )
            with patch.object(on_policy.subprocess, "run") as run:
                updated = on_policy.train(args, directory, samples, adapter)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--gradient-accumulation") + 1], "3")
            self.assertEqual(command[command.index("--resume-adapter") + 1], str(adapter))
            self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")
            self.assertEqual(updated, directory / "training/final-adapter")


if __name__ == "__main__":
    unittest.main()
