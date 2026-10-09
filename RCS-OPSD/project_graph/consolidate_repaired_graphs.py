#!/usr/bin/env python3
"""Consolidate freshly generated base graphs and current repair fragments."""
import argparse
import json
from pathlib import Path

from repair_missing_causal_graphs import ROOT, REPORT, merge, write_json
from rcs_opsd.artifact_paths import GRAPH_BASE_ROOT, GRAPH_ROOT


def consolidate(base_root: Path, output: Path, repair_root: Path) -> dict:
    fragments = {}
    for path in sorted(repair_root.glob('attempt-*/*/result.json')):
        result = json.loads(path.read_text())
        project_units = fragments.setdefault(result['project'], {})
        for unit in result['units']:
            previous = project_units.get(unit['scope'])
            if previous is None or not unit['targets_only'] or previous['targets_only']:
                project_units[unit['scope']] = unit
    manifest = {}
    for path in sorted(base_root.glob('*.json')):
        original = json.loads(path.read_text())
        units = list(fragments.get(original['project'], {}).values())
        overlays = [json.loads((ROOT / unit['artifacts'] / 'project_fragment.json').read_text()) for unit in units]
        graph = merge(original, overlays)
        graph['repair'] = {'units': units}
        target = output / path.name
        write_json(target, graph)
        manifest[original['project']] = str(target)
    write_json(output / 'manifest.json', {'project_graphs': manifest})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, default=GRAPH_BASE_ROOT)
    parser.add_argument('--output-dir', type=Path, default=GRAPH_ROOT)
    parser.add_argument('--repair-dir', type=Path, default=REPORT)
    args = parser.parse_args()
    manifest = consolidate(args.base_dir, args.output_dir, args.repair_dir)
    print(f'Consolidated {len(manifest)} project graphs into {args.output_dir}')


if __name__ == '__main__':
    main()
