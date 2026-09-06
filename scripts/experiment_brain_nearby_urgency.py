"""Test nearby urgent work before routine on-tile work, without imitation.

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
    marker = '                ti = min(exact, key=lambda j: (tasks[j][0], j))\n'
    assert block.count(marker) == 1
    variants = {}
    for radius in (1, 2):
        extra = ("                urgent_nearby = False\n"
                 "                if tasks[ti][0] >= 2:\n"
                 "                    for job in pending:\n"
                 "                        priority, target, _, need, skip = tasks[job]\n"
                 "                        distance = abs(positions[idx][0]-target[0])+abs(positions[idx][1]-target[1])\n"
                 f"                        if priority > 0 or not 0 < distance <= {radius}: continue\n"
                 "                        if need is not None and inventories[idx].get(need, 0) <= 0: continue\n"
                 "                        if skip is not None and inventories[idx].get(skip, 0) > 0: continue\n"
                 "                        covered = any(other != idx and positions[other] == target\n"
                 "                                      and (need is None or inventories[other].get(need, 0) > 0)\n"
                 "                                      and (skip is None or inventories[other].get(skip, 0) <= 0)\n"
                 "                                      for other in available)\n"
                 "                        if not covered:\n"
                 "                            urgent_nearby = True\n"
                 "                            break\n"
                 "                if urgent_nearby: continue\n")
        variants[f'near{radius}'] = source[:start]+block.replace(marker, marker+extra)+source[end:]
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='bots/brain/main.py')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    source = (ROOT/args.baseline).read_text(encoding='utf-8-sig')
    out = ROOT/'data/results/brain-nearby-urgency'
    arena = ROOT/'data/arena-nearby-urgency'
    out.mkdir(parents=True, exist_ok=True)
    variants = {'control': source, **build(source)}
    tags = {name: f'nearby-urgency-{name}-screen-16' for name in variants}
    if args.run:
        assert not any((ROOT/'data/results'/tag/'results.json').exists() for tag in tags.values()), 'Existing results; do not overwrite'
    replay = json.loads((ROOT/'data/results/brain-own-diagnostic/main_vs_main_seed51.json').read_text())
    states = [step[0]['observation'] for step in replay['steps'][:-1:6]]
    reports = []
    for name, text in variants.items():
        path = ROOT/'bots'/f'brain_nearby_urgency_{name}'/'main.py'
        compile(text, str(path), 'exec')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        module = load(path, f'nearby_urgency_{name}')
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
                            '--agent', str(ROOT/'bots'/f'brain_nearby_urgency_{name}'/'main.py'),
                            '--opponents', 'bots/v46/main.py', 'bots/tapeselect/main.py',
                            '--n', '16', '--seed-base', '410', '--workers', '2', '--tag', tags[name]],
                           cwd=ROOT, env=dict(os.environ, KAGGRI_ARENA_DIR=str(arena)),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        data = json.loads((ROOT/'data/results'/tags[name]/'results.json').read_text())
        print(name, data['overall'], flush=True)
    control, control_stats = read(arena, tags['control'])
    for name in ('near1', 'near2'):
        candidate, stats = read(arena, tags[name])
        paired = dict(per_opponent=compare(candidate, control), candidate=stats, control=control_stats)
        (out/f'{name}-paired.json').write_text(json.dumps(paired, indent=2), encoding='utf-8')
        print(name, json.dumps(paired), flush=True)


if __name__ == '__main__':
    main()
