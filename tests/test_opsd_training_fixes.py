from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'RCS-OPSD/OPSD_Trainer'))
sys.path.insert(0, str(ROOT / 'RCS-OPSD/S_Extractor'))
from opsd.collator import IGNORE_INDEX, load_tokenizer, serialize_sample
from build_causal_site import build_site, changed_lines, build_distance_index


def load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SerializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # These tests use only the tokenizer, without model weights or GPU work.
        os.environ['USE_TORCH'] = '0'
        os.environ['USE_TF'] = '0'
        model = Path(os.environ.get('RCS_OPSD_MODEL_PATH', str(ROOT / 'models/Qwen3.5-9B')))
        if not model.exists():
            raise unittest.SkipTest('Set RCS_OPSD_MODEL_PATH to a local Qwen3.5 tokenizer directory')
        cls.tokenizer = load_tokenizer(model, ROOT / 'templates/chat_template_codex.jinja')

    def sample(self):
        return {
            'finding_id': 'finding-a', 'turn_id': 2, 'state': 'LOCATED',
            'history': [
                {'role': 'user', 'content': 'Audit task: preserve legitimate withdrawals.'},
                {'assistant': {'content': [{'type': 'toolCall', 'id': 'old-call', 'name': 'read', 'arguments': {'path': 'Vault.sol'}}]},
                 'tool_calls': [{'tool_call_id': 'old-call', 'result': {'content': [{'type': 'text', 'text': 'old observation ' * 1000}]}}]},
            ],
            'target': {'assistant_message': {'content': [
                {'type': 'thinking', 'thinking': 'PRIVATE_REASONING <tool_call>example</tool_call>'},
                {'type': 'text', 'text': 'PROSE_ONLY <tool_call>example</tool_call>'},
                {'type': 'toolCall', 'id': 'call-1', 'name': 'edit', 'arguments': {
                    'path': 'Vault.sol', 'edits': [{'oldText': 'balance -= amount;', 'newText': '余额 -= amount;'}]}},
                {'type': 'toolCall', 'id': 'call-2', 'name': 'bash', 'arguments': {'command': 'forge test --match-test testWithdraw'}},
            ]}},
            'metadata': {'focus': ['Vault::withdraw'], 'repair_action': True},
            'repository': '/testbed', 'mechanism': 'M_SENTINEL',
            'repair_invariant': 'I_SENTINEL',
            'state_conditioned_guidance': {'status': 'causal_site_reached', 'guidance': ''},
        }

    def test_complete_context_is_preserved_or_rejected_without_truncation(self):
        sample = self.sample()
        full = serialize_sample(self.tokenizer, sample)
        common = full['common_history_length']
        self.assertEqual(full['history_tokens_removed'], 0)
        self.assertEqual(full['student']['input_ids'][:common], full['teacher']['input_ids'][:common])
        self.assertIn('Audit task:', full['student']['rendered_prefix'])
        self.assertIn('I_SENTINEL', full['teacher']['rendered_prefix'])
        limit = len(full['teacher']['input_ids'])
        self.assertEqual(serialize_sample(self.tokenizer, sample, max_seq_len=limit), full)
        with self.assertRaisesRegex(ValueError, 'exclude this sample instead of truncating'):
            serialize_sample(self.tokenizer, sample, max_seq_len=limit - 1)

    def test_captured_compacted_request_controls_history_tools_and_thinking(self):
        sample = self.sample()
        sample['student_request'] = {
            'messages': [
                {'role': 'system', 'content': 'ACTUAL_SYSTEM'},
                {'role': 'user', 'content': 'COMPACTED_SUMMARY'},
                {'role': 'assistant', 'content': None, 'tool_calls': [
                    {'id': 'recent', 'type': 'function', 'function': {
                        'name': 'read', 'arguments': '{"path":"Recent.sol"}'}}]},
                {'role': 'tool', 'tool_call_id': 'recent', 'content': 'VISIBLE_CODE'},
            ],
            'tools': [{'type': 'function', 'function': {'name': 'read', 'description': 'ACTUAL_TOOL',
                       'parameters': {'type': 'object', 'properties': {}}}}],
            'chat_template_kwargs': {'enable_thinking': False},
        }
        routes = serialize_sample(self.tokenizer, sample)
        expected_messages = json.loads(json.dumps(sample['student_request']['messages']))
        expected_messages[2]['tool_calls'][0]['function']['arguments'] = {'path': 'Recent.sol'}
        expected = self.tokenizer.apply_chat_template(
            expected_messages, tools=sample['student_request']['tools'], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        self.assertEqual(routes['student']['rendered_prefix'], expected)
        self.assertFalse(routes['enable_thinking'])
        self.assertNotIn('old observation', routes['teacher']['rendered_prefix'])
        self.assertNotIn('Audit task:', routes['teacher']['rendered_prefix'])
        self.assertIn('COMPACTED_SUMMARY', routes['teacher']['rendered_prefix'])
        self.assertEqual(sample['student_request']['messages'][2]['tool_calls'][0]['function']['arguments'],
                         '{"path":"Recent.sol"}')

    def test_only_actual_tool_calls_are_supervised_in_both_thinking_modes(self):
        for mode in ['auto', 'off']:
            with self.subTest(mode=mode):
                routes = serialize_sample(self.tokenizer, self.sample(), thinking_mode=mode)
                outputs = []
                for name in ['student', 'teacher']:
                    route = routes[name]
                    ids = [label for label in route['labels'] if label != IGNORE_INDEX]
                    text = self.tokenizer.decode(ids, skip_special_tokens=False)
                    self.assertNotIn('PROSE_ONLY', text)
                    self.assertNotIn('PRIVATE_REASONING', text)
                    self.assertNotIn('example', text)
                    self.assertIn('<function=edit>', text)
                    self.assertIn('余额', text)
                    self.assertIn('<function=bash>', text)
                    self.assertIn('forge test', text)
                    self.assertNotIn('<|im_end|>', text)
                    self.assertLess(len(ids), route['target_length'])
                    outputs.append(ids)
                self.assertEqual(*outputs)

    def test_prose_only_has_no_loss_tokens(self):
        sample = self.sample()
        sample['target']['assistant_message']['content'] = [{'type': 'text', 'text': 'No action: <tool_call>example</tool_call>'}]
        routes = serialize_sample(self.tokenizer, sample)
        self.assertEqual(routes['student']['action_length'], 0)
        self.assertTrue(all(label == IGNORE_INDEX for label in routes['student']['labels']))


class PatchAndGraphTests(unittest.TestCase):
    def test_structural_and_unmapped_patches_use_existing_causal_anchors(self):
        import build_causal_site as sites
        patches = [
            '--- a/Vault.sol\n+++ b/Vault.sol\n@@ -1 +1 @@\n-contract Vault {}\n+contract Vault is Guard {}\n',
            '--- a/Vault.sol\n+++ b/Vault.sol\n@@ -20,0 +21 @@\n+function newGuard() public {}\n',
            '--- a/Missing.sol\n+++ b/Missing.sol\n@@ -4 +4 @@\n-old\n+new\n',
        ]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'fixture.json').write_text(json.dumps({'nodes': [
                {'id': 'anchor', 'file': 'Vault.sol', 'start_line': 4, 'end_line': 6}], 'edges': []}))
            task = directory / 'findings/fixture__H-1/task'; task.mkdir(parents=True)
            static_s = directory / 'static-s.jsonl'
            static_s.write_text(json.dumps({'finding_id': 'fixture__H-1', 'selection': {'functions': ['anchor']}}) + '\n')
            for diff in patches:
                with self.subTest(diff=diff):
                    (task / 'reference_patch.diff').write_text(diff)
                    with patch.object(sites, 'GRAPH_ROOT', directory), \
                         patch.object(sites, 'FINDINGS', directory / 'findings'), \
                         patch.object(sites, 'S_RESULTS', static_s):
                        site = build_site('fixture__H-1')
                    self.assertEqual(site['patch_functions'], [])
                    self.assertEqual(site['anchor_functions'], ['anchor'])
                    self.assertEqual(site['anchor_source'], 'static_causal_site_without_patch_function_overlap')

    def test_capsule_merge_does_not_restore_cross_contract_state_edges(self):
        sys.path.insert(0, str(ROOT / 'RCS-OPSD/project_graph'))
        module = load_script('forge_graph_test', 'RCS-OPSD/project_graph/generate_forge_project_graphs.py')
        nodes = [
            {'id': name, 'file': 'Contracts.sol', 'contract': context,
             'finding_ids': ['finding-a'], 'storage_contexts': ['Contracts.sol::' + context],
             'state_variables_read': ['Base.nonce'], 'state_variables_written': writes}
            for name, context, writes in [('left-write', 'Left', ['Base.nonce']),
                                         ('left-read', 'Left', []), ('right-read', 'Right', [])]
        ]
        graph = module.merge_project('fixture', [
            {'nodes': nodes[:1], 'edges': [], 'unresolved_calls': []},
            {'nodes': nodes[1:], 'edges': [], 'unresolved_calls': []},
        ], 2, 2, [], 1, 0, False)
        pairs = {(edge['source'], edge['target']) for edge in graph['edges']}
        self.assertIn(('left-write', 'left-read'), pairs)
        self.assertNotIn(('left-write', 'right-read'), pairs)

    def test_subproject_prefix_preserves_storage_context_identity(self):
        sys.path.insert(0, str(ROOT / 'RCS-OPSD/project_graph'))
        module = load_script('all_graph_test', 'RCS-OPSD/project_graph/generate_all_project_graphs.py')
        fragment = {'nodes': [
            {'id': 'A.sol::A::write()', 'file': 'A.sol', 'storage_contexts': ['A.sol::A']},
            {'id': 'A.sol::A::read()', 'file': 'A.sol', 'storage_contexts': ['A.sol::A']},
        ], 'edges': [{'source': 'A.sol::A::write()', 'target': 'A.sol::A::read()',
                     'type': 'STATE_DEP', 'storage_contexts': ['A.sol::A']}], 'unresolved_calls': []}
        graph = module.prefix_fragment(fragment, 'subproject')
        self.assertEqual(graph['nodes'][0]['storage_contexts'], ['subproject/A.sol::A'])
        self.assertEqual(graph['edges'][0]['storage_contexts'], ['subproject/A.sol::A'])

    def test_multi_file_unified_diff_and_header_like_code(self):
        diff = ('--- a/A.sol\n+++ b/A.sol\n@@ -4,2 +4,2 @@\n'
                '--- old-code\n+++ new-code\n unchanged\n'
                '--- a/B.sol\n+++ b/B.sol\n@@ -8 +8 @@\n-old\n+new\n')
        self.assertEqual(changed_lines(diff), {'A.sol': {4, 5}, 'B.sol': {8, 9}})
        extractor = load_script('i_extractor_test', 'RCS-OPSD/I_Extractor/run_i_extractor.py')
        old, new = extractor.reconstruct_diff_versions(diff)
        self.assertIn('FILE: A.sol', old)
        self.assertIn('FILE: B.sol', new)
        self.assertIn('-- old-code', old)
        self.assertIn('++ new-code', new)
        self.assertNotIn('+++ b/B.sol', new)

    def test_core_distance_zero_for_one_hop_but_not_two_hops(self):
        import build_causal_site as sites
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            graph = {'nodes': [
                {'id': name, 'file': 'Vault.sol', 'start_line': line, 'end_line': line + 2}
                for name, line in [('patch', 4), ('neighbor', 10), ('far', 20)]
            ], 'edges': [
                {'source': 'patch', 'target': 'neighbor', 'type': 'CALL'},
                {'source': 'neighbor', 'target': 'far', 'type': 'STATE_DEP'},
            ]}
            (directory / 'project.json').write_text(json.dumps(graph))
            task = directory / 'findings/project__H-1/task'
            task.mkdir(parents=True)
            (task / 'reference_patch.diff').write_text('--- a/Vault.sol\n+++ b/Vault.sol\n@@ -4 +4 @@\n-old\n+new\n')
            with patch.object(sites, 'GRAPH_ROOT', directory), patch.object(sites, 'FINDINGS', directory / 'findings'):
                site = build_site('project__H-1')
                distances = build_distance_index(site)
            self.assertEqual(site['patch_functions'], ['patch'])
            self.assertEqual(site['core_functions'], ['neighbor', 'patch'])
            self.assertEqual(distances['nodes']['neighbor']['distance'], 0)
            self.assertEqual(distances['nodes']['far']['distance'], 1)

    def test_compiled_inheritance_has_shared_base_access_but_no_cross_contract_storage(self):
        source = '''pragma solidity ^0.8.20;
contract Base {
    uint256 public nonce;
    function baseWrite() public { nonce++; }
}
contract Left is Base {
    function leftWrite() public { nonce++; }
    function leftRead() public view returns (uint256) { return nonce; }
}
contract Right is Base {
    function rightWrite() public { nonce++; }
    function rightRead() public view returns (uint256) { return nonce; }
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'Test.sol').write_text(source)
            output = directory / 'graph.json'
            env = os.environ.copy()
            solc = os.environ.get('SOLC_0_8_20') or shutil.which('solc')
            if not solc:
                self.skipTest('Install solc 0.8.20 or set SOLC_0_8_20')
            (directory / 'solc').symlink_to(Path(solc).resolve())
            env['PATH'] = str(directory) + os.pathsep + env['PATH']
            subprocess.run([sys.executable, str(ROOT / 'RCS-OPSD/project_graph/generate_project_graph.py'),
                            str(directory / 'Test.sol'), str(output), '--project', 'fixture', '--source-dir', '.'],
                           env=env, check=True, capture_output=True, text=True)
            graph = json.loads(output.read_text())
        edges = [edge for edge in graph['edges'] if edge['type'] == 'STATE_DEP']
        self.assertTrue(edges)
        self.assertFalse(any('::Left::' in edge['source'] and '::Right::' in edge['target']
                            or '::Right::' in edge['source'] and '::Left::' in edge['target'] for edge in edges))
        self.assertTrue(any('::Base::baseWrite' in edge['source'] and '::Left::leftRead' in edge['target'] for edge in edges))
        self.assertTrue(any('::Base::baseWrite' in edge['source'] and '::Right::rightRead' in edge['target'] for edge in edges))
        self.assertTrue(any('::Left::leftWrite' in edge['source'] and '::Left::leftRead' in edge['target'] for edge in edges))
        self.assertTrue(all(edge['storage_contexts'] for edge in edges))


if __name__ == '__main__':
    unittest.main()
