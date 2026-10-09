from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for folder in ['src', 'RCS-OPSD/project_graph', 'RCS-OPSD/Rollout_collector', 'RCS-OPSD/OPSD_Trainer']:
    sys.path.insert(0, str(ROOT / folder))
import on_policy
import state_guidance
from repair_missing_causal_graphs import merge
from consolidate_repaired_graphs import consolidate
from run_rollout_repair import collect_rollout
from rcs_opsd import artifact_paths


class GraphMergeTests(unittest.TestCase):
    def graph(self, nodes, edges=()):
        return {'project': 'fixture', 'nodes': nodes, 'edges': list(edges), 'unresolved_calls': [], 'scope': {}}

    def node(self, name, context, read=(), write=()):
        return {'id': name, 'storage_contexts': [context],
                'state_variables_read': list(read), 'state_variables_written': list(write)}

    def test_overlay_discards_false_old_state_edge_and_builds_missing_valid_edge(self):
        nodes = [self.node('left-write', 'Left', write=['Base.nonce']),
                 self.node('right-read', 'Right', read=['Base.nonce'])]
        wrong = {'source': 'left-write', 'target': 'right-read', 'type': 'STATE_DEP',
                 'relation': 'WRITE_READ', 'state_variables': ['Base.nonce']}
        call = {'source': 'left-write', 'target': 'right-read', 'type': 'CALL', 'call_type': 'external'}
        original = self.graph(nodes, [wrong, call])
        overlay = self.graph([self.node('left-read', 'Left', read=['Base.nonce'])])
        graph = merge(original, [overlay])
        state = [e for e in graph['edges'] if e['type'] == 'STATE_DEP']
        self.assertEqual([(e['source'], e['target']) for e in state], [('left-write', 'left-read')])
        self.assertEqual(state[0]['storage_contexts'], ['Left'])
        self.assertIn(call, graph['edges'])
        self.assertIn(wrong, original['edges'])  # The input artifact is not mutated.

    def test_old_graph_without_contexts_must_be_regenerated(self):
        with self.assertRaisesRegex(ValueError, 'Regenerate graphs'):
            merge(self.graph([{'id': 'old'}]), [])

    def test_consolidation_recomputes_even_base_only_graphs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); base = root / 'base'; base.mkdir()
            graph = self.graph([self.node('write', 'Vault', write=['Vault.balance']),
                                self.node('read', 'Vault', read=['Vault.balance'])])
            (base / 'fixture.json').write_text(json.dumps(graph))
            manifest = consolidate(base, root / 'current', root / 'fresh-repairs')
            result = json.loads(Path(manifest['fixture']).read_text())
            self.assertEqual(len(result['edges']), 1)
            self.assertEqual(result['schema_version'], '1.2')

    def test_consolidation_reads_current_repair_fragments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); base = root / 'base'; base.mkdir()
            graph = self.graph([self.node('write', 'Vault', write=['Vault.balance'])])
            (base / 'fixture.json').write_text(json.dumps(graph))
            unit = root / 'repairs/attempt-01/fixture/unit-0'; unit.mkdir(parents=True)
            overlay = self.graph([self.node('read', 'Vault', read=['Vault.balance'])])
            (unit / 'project_fragment.json').write_text(json.dumps(overlay))
            (unit.parent / 'result.json').write_text(json.dumps({'project': 'fixture', 'units': [
                {'scope': 'contracts', 'targets_only': False, 'artifacts': str(unit)}]}))
            manifest = consolidate(base, root / 'current', root / 'repairs')
            result = json.loads(Path(manifest['fixture']).read_text())
            self.assertEqual([(e['source'], e['target']) for e in result['edges']], [('write', 'read')])


class ContextTests(unittest.TestCase):
    def target(self, number, request=None):
        turn = {'turn_id': number, 'complete': True, 'assistant': {'content': [
            {'type': 'toolCall', 'id': f'call-{number}', 'name': 'read', 'arguments': {'path': 'Vault.sol'}}]}}
        if request is not None:
            turn['student_request'] = request
        return turn

    def test_compaction_recomputes_focus_from_only_visible_tool_evidence(self):
        observation = {'turn_id': 0, 'function_focus': ['core', 'near'], 'focus_status': 'LOCATED', 'distance': 0,
                       'function_distances': {'core': 0, 'near': 2},
                       'evidence': [{'tool_call_id': 'forgotten', 'function_ids': ['core']},
                                    {'tool_call_id': 'visible', 'function_ids': ['near']}]}
        request = {'messages': [{'role': 'tool', 'tool_call_id': 'visible', 'content': 'near code'}]}
        rollout = {'turns': [self.target(0), self.target(1, request),
                             self.target(2, {'messages': [{'role': 'user', 'content': 'summary'}]})]}
        rows = on_policy.next_turn_selections({'finding_id': 'finding-a', 'turns': [observation]}, rollout)
        self.assertEqual(rows[1]['function_focus'], ['near'])
        self.assertEqual(rows[1]['state'], 'LOCALIZING')
        self.assertEqual(rows[1]['distance'], 2)
        self.assertEqual(rows[2]['function_focus'], [])
        self.assertEqual(rows[2]['state'], 'UNRESOLVED')
        self.assertIsNone(rows[2]['focus_turn_id'])

    def test_pi_uses_actual_compacted_messages_without_forgotten_observations(self):
        messages = [{'role': 'system', 'content': 'actual'}, {'role': 'user', 'content': 'summary'}]
        prior = self.target(0); prior['tool_calls'] = []
        target = self.target(1, {'messages': messages})
        selection = {'finding_id': 'finding-a', 'turn_id': 1, 'focus_turn_id': None,
                     'state': 'UNRESOLVED', 'function_focus': []}
        with tempfile.TemporaryDirectory() as tmp:
            graph_root = Path(tmp)
            (graph_root / 'fixture.json').write_text(json.dumps({'nodes': [{'id': 'core'}], 'edges': []}))
            with patch.object(state_guidance, 'GRAPH_ROOT', graph_root):
                case = state_guidance.build_case(selection, 'M', {'project': 'fixture', 'core_functions': ['core']},
                                                 'audit', {'turns': [prior, target]})
        self.assertEqual(case['student_state']['messages'], messages)
        self.assertNotIn('completed_observations', case['student_state'])

    def test_collector_attaches_actual_request_to_matching_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'result.json').write_text(json.dumps({'finding_id': 'fixture', 'status': 'PASS'}))
            assistant = {'role': 'assistant', 'content': [{'type': 'text', 'text': 'done'}]}
            events = [{'type': 'turn_start'}, {'type': 'turn_end', 'message': assistant}, {'type': 'agent_settled'}]
            (out / 'pi-events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
            request = {'messages': [{'role': 'user', 'content': 'actual'}], 'tools': []}
            (out / 'provider-requests.jsonl').write_text(json.dumps({'turn_id': 0, 'request': request}) + '\n')
            summary = collect_rollout(out)
            rollout = json.loads((out / 'rollout.json').read_text())
            self.assertEqual(rollout['turns'][0]['student_request'], request)
            self.assertEqual(summary['turns_with_student_request'], 1)

    def test_training_consumers_share_current_artifact_paths(self):
        import build_causal_site
        import pi_inputs
        import map_turn_focus
        self.assertEqual(state_guidance.GRAPH_ROOT, artifact_paths.GRAPH_ROOT)
        self.assertEqual(build_causal_site.GRAPH_ROOT, artifact_paths.GRAPH_ROOT)
        self.assertEqual(map_turn_focus.GRAPH_ROOT, artifact_paths.GRAPH_ROOT)
        self.assertEqual(build_causal_site.S_RESULTS, artifact_paths.S_RESULTS)
        for name in ['M_RESULTS', 'S_RESULTS', 'I_RESULTS']:
            self.assertEqual(getattr(pi_inputs, name), getattr(artifact_paths, name))

    def test_sample_builder_checks_pi_against_captured_context(self):
        import build_single_sample
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); finding = 'fixture'; rollouts = root / 'rollouts'
            out = rollouts / finding; out.mkdir(parents=True)
            request = {'messages': [{'role': 'user', 'content': 'actual compacted state'}]}
            target = self.target(0, request)
            target['tool_calls'] = [{'tool_call_id': 'call-0', 'result': {'content': []}}]
            (out / 'rollout.json').write_text(json.dumps({'turns': [target]}))
            (out / 'prompt.md').write_text('Original audit')
            selected = root / 'selected.jsonl'
            selected.write_text(json.dumps({'finding_id': finding, 'turn_id': 0, 'state': 'UNRESOLVED',
                                           'function_focus': [], 'distance': None, 'repair_action': False,
                                           'selection_reasons': []}) + '\n')
            pi_root = root / 'pi'; pi_dir = pi_root / 'fixture__turn_0'; pi_dir.mkdir(parents=True)
            (pi_dir / 'result.json').write_text(json.dumps({'finding_id': finding, 'turn_id': 0,
                                                          'current_focus': None, 'privileged_information': 'M'}))
            input_path = pi_dir / 'input.json'
            input_path.write_text(json.dumps({'student_state': {'messages': request['messages']}}))
            with patch.object(build_single_sample, 'ROOT', root):
                sample, checks = build_single_sample.build_sample(finding, 0, selected, rollouts, rollouts, pi_root)
                self.assertEqual(sample['student_request'], request)
                self.assertTrue(checks['all_pass'])
                input_path.write_text(json.dumps({'student_state': {'messages': [{'role': 'user', 'content': 'wrong state'}]}}))
                _, checks = build_single_sample.build_sample(finding, 0, selected, rollouts, rollouts, pi_root)
                self.assertFalse(checks['pi_context_matches_student_request'])
                self.assertFalse(checks['all_pass'])


if __name__ == '__main__':
    unittest.main()
