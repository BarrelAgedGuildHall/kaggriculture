"""Test unique travel destinations while preserving all on-tile jobs, without imitation.

Build only by default. Run sequentially with two workers after other arenas stop.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark_brain_dispatch_cost import load
from compare_paired import compare, read

ROOT = Path(__file__).resolve().parents[1]


def build(source):
    start = source.index('    def _assign(')
    end = source.index('    @staticmethod\n    def dispatch', start)
    block = source[start:end]
    marker = '        while available and pending:'
    assert block.count(marker) == 1
    block = block.replace(marker,
        '        claimed = {positions[idx] for idx in actions}\n'
        '        pending = [ti for ti in pending if tasks[ti][1] not in claimed]\n'+marker)
    old = '            pending.remove(ti)\n        if tasks:'
    assert block.count(old) == 1
    block = block.replace(old,
        '            claimed.add(target)\n'
        '            pending = [j for j in pending if tasks[j][1] != target]\n'
        '        if pending:')
    block = block.replace('targets = [task[1] for task in tasks]', 'targets = [tasks[ti][1] for ti in pending]')
    fallback_start = block.index('        if pending:')
    fallback_end = block.index('        return actions', fallback_start)
    block = block[:fallback_start]+block[fallback_end:]
    return {'unique': source[:start]+block+source[end:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='bots/brain/main.py')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    source = (ROOT/args.baseline).read_text(encoding='utf-8-sig')
    out = ROOT/'data/results/brain-unique-travel'
    arena = ROOT/'data/arena-unique-travel'
    out.mkdir(parents=True, exist_ok=True)
    variants = {'control': source, **build(source)}
    tags = {name: f'unique-travel-{name}-screen-16' for name in variants}
    if args.run:
        assert not any((ROOT/'data/results'/tag/'results.json').exists() for tag in tags.values()), 'Existing results; do not overwrite'
    replay = json.loads((ROOT/'data/results/brain-own-diagnostic/main_vs_main_seed51.json').read_text())
    states = [step[0]['observation'] for step in replay['steps'][:-1:6]]
    reports = []
    for name, text in variants.items():
        path = ROOT/'bots'/f'brain_unique_travel_{name}'/'main.py'
        compile(text, str(path), 'exec')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        module = load(path, f'unique_travel_{name}')
        for obs in states:
            action = module._policy(obs, time.perf_counter())
            assert isinstance(action, dict) and 'market' in action
        reports.append(dict(variant=name, policy_calls=len(states)))
    (out/'offline-checks.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps(reports), flush=True)
    if not args.run:
        return
    for name in variants:
        with (out/f'{name}.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/'scripts/evaluate.py'),
                            '--agent', str(ROOT/'bots'/f'brain_unique_travel_{name}'/'main.py'),
                            '--opponents', 'bots/v46/main.py', 'bots/tapeselect/main.py',
                            '--n', '16', '--seed-base', '380', '--workers', '2', '--tag', tags[name]],
                           cwd=ROOT, env=dict(os.environ, KAGGRI_ARENA_DIR=str(arena)),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        data = json.loads((ROOT/'data/results'/tags[name]/'results.json').read_text())
        print(name, data['overall'], flush=True)
    control, control_stats = read(arena, tags['control'])
    for name in ('unique',):
        candidate, stats = read(arena, tags[name])
        paired = dict(per_opponent=compare(candidate, control), candidate=stats, control=control_stats)
        (out/f'{name}-paired.json').write_text(json.dumps(paired, indent=2), encoding='utf-8')
        print(name, json.dumps(paired), flush=True)


if __name__ == '__main__':
    main()
