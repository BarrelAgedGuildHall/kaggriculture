"""Test modest mixed-herd reductions under the current original policy.

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
    total = 'MAX_ANIMALS = 18'
    targets = 'ANIMAL_TARGETS = {"COW": 7, "SHEEP": 11}'
    assert source.count(total) == source.count(targets) == 1
    return {f'herd{cow+sheep}': source.replace(total, f'MAX_ANIMALS = {cow+sheep}').replace(
            targets, f'ANIMAL_TARGETS = {{"COW": {cow}, "SHEEP": {sheep}}}')
            for cow, sheep in ((5, 8), (6, 9))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='bots/brain/main.py')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    source = (ROOT/args.baseline).read_text(encoding='utf-8-sig')
    out = ROOT/'data/results/brain-compact-herd'
    arena = ROOT/'data/arena-compact-herd'
    out.mkdir(parents=True, exist_ok=True)
    variants = {'control': source, **build(source)}
    tags = {name: f'compact-herd-{name}-screen-16' for name in variants}
    if args.run:
        assert not any((ROOT/'data/results'/tag/'results.json').exists() for tag in tags.values()), 'Existing results; do not overwrite'
    replay = json.loads((ROOT/'data/results/brain-own-diagnostic/main_vs_main_seed51.json').read_text())
    states = [step[0]['observation'] for step in replay['steps'][:-1:6]]
    reports = []
    for name, text in variants.items():
        path = ROOT/'bots'/f'brain_compact_herd_{name}'/'main.py'
        compile(text, str(path), 'exec')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        module = load(path, f'compact_herd_{name}')
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
                            '--agent', str(ROOT/'bots'/f'brain_compact_herd_{name}'/'main.py'),
                            '--opponents', 'bots/v46/main.py', 'bots/tapeselect/main.py',
                            '--n', '16', '--seed-base', '470', '--workers', '2', '--tag', tags[name]],
                           cwd=ROOT, env=dict(os.environ, KAGGRI_ARENA_DIR=str(arena)),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        data = json.loads((ROOT/'data/results'/tags[name]/'results.json').read_text())
        print(name, data['overall'], flush=True)
    control, control_stats = read(arena, tags['control'])
    for name in ('herd13', 'herd15'):
        candidate, stats = read(arena, tags[name])
        paired = dict(per_opponent=compare(candidate, control), candidate=stats, control=control_stats)
        (out/f'{name}-paired.json').write_text(json.dumps(paired, indent=2), encoding='utf-8')
        print(name, json.dumps(paired), flush=True)


if __name__ == '__main__':
    main()
