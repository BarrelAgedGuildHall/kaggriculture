"""Test a bounded route credit for multiple eligible jobs at one destination, without imitation.

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
    marker = '            for ti in pending:'
    assert block.count(marker) == 1
    extra = """            bundle_counts = {}
            for unit in available:
                inv = inventories[unit]
                for job in pending:
                    _, target, _, need, skip = tasks[job]
                    if need is not None and inv.get(need, 0) <= 0: continue
                    if skip is not None and inv.get(skip, 0) > 0: continue
                    key = (unit, target)
                    bundle_counts[key] = bundle_counts.get(key, 0) + 1
"""
    block = block.replace(marker, extra+marker)
    old = '                    choices.append((priority, cluster_cost, distance, idx, ti))'
    assert block.count(old) == 1
    variants = {}
    for credit in (1, 2):
        new = (f'                    route_cost = distance - min({credit}, max(0, bundle_counts.get((idx, target), 1)-1))\n'
               '                    choices.append((priority, cluster_cost, route_cost, idx, ti))')
        variants[f'credit{credit}'] = source[:start]+block.replace(old, new)+source[end:]
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='bots/brain/main.py')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    source = (ROOT/args.baseline).read_text(encoding='utf-8-sig')
    out = ROOT/'data/results/brain-bundle-credit'
    arena = ROOT/'data/arena-bundle-credit'
    out.mkdir(parents=True, exist_ok=True)
    variants = {'control': source, **build(source)}
    tags = {name: f'bundle-credit-{name}-screen-16' for name in variants}
    if args.run:
        assert not any((ROOT/'data/results'/tag/'results.json').exists() for tag in tags.values()), 'Existing results; do not overwrite'
    replay = json.loads((ROOT/'data/results/brain-own-diagnostic/main_vs_main_seed51.json').read_text())
    states = [step[0]['observation'] for step in replay['steps'][:-1:6]]
    reports = []
    for name, text in variants.items():
        path = ROOT/'bots'/f'brain_bundle_credit_{name}'/'main.py'
        compile(text, str(path), 'exec')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        module = load(path, f'bundle_credit_{name}')
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
                            '--agent', str(ROOT/'bots'/f'brain_bundle_credit_{name}'/'main.py'),
                            '--opponents', 'bots/v46/main.py', 'bots/tapeselect/main.py',
                            '--n', '16', '--seed-base', '500', '--workers', '2', '--tag', tags[name]],
                           cwd=ROOT, env=dict(os.environ, KAGGRI_ARENA_DIR=str(arena)),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        data = json.loads((ROOT/'data/results'/tags[name]/'results.json').read_text())
        print(name, data['overall'], flush=True)
    control, control_stats = read(arena, tags['control'])
    for name in ('credit1', 'credit2'):
        candidate, stats = read(arena, tags[name])
        paired = dict(per_opponent=compare(candidate, control), candidate=stats, control=control_stats)
        (out/f'{name}-paired.json').write_text(json.dumps(paired, indent=2), encoding='utf-8')
        print(name, json.dumps(paired), flush=True)


if __name__ == '__main__':
    main()
