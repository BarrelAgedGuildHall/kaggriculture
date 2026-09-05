"""Screen joint travel assignment with weaker territory costs, without imitation.

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


HELPER = '''def _minimum_cost_columns(cost):
    """Rectangular minimum-cost assignment; one distinct column per row."""
    n = len(cost)
    if not n:
        return []
    m = len(cost[0])
    assert n <= m and all(len(row) == m for row in cost)
    u, v, p, way = [0]*(n+1), [0]*(m+1), [0]*(m+1), [0]*(m+1)
    for i in range(1, n+1):
        p[0] = i
        j0 = 0
        slack, used = [float('inf')]*(m+1), [False]*(m+1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float('inf'), 0
            for j in range(1, m+1):
                if used[j]:
                    continue
                reduced = cost[i0-1][j-1]-u[i0]-v[j]
                if reduced < slack[j]:
                    slack[j], way[j] = reduced, j0
                if slack[j] < delta:
                    delta, j1 = slack[j], j
            for j in range(m+1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    slack[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if not j0:
                break
    result = [-1]*n
    for j in range(1, m+1):
        if p[j]:
            result[p[j]-1] = j-1
    return result


'''
REPLACEMENT = '''        for priority in sorted({tasks[ti][0] for ti in pending}):
            if not available:
                break
            group = [ti for ti in pending if tasks[ti][0] == priority]
            workers = sorted(available)
            costs = []
            for idx in workers:
                row = []
                for ti in group:
                    _priority, target, _action, need, skip = tasks[ti]
                    valid = (need is None or inventories[idx].get(need, 0) > 0)
                    valid = valid and (skip is None or inventories[idx].get(skip, 0) <= 0)
                    distance = abs(positions[idx][0]-target[0])+abs(positions[idx][1]-target[1])
                    owner = LaborB1._owner(target, len(positions))
                    row.append(distance+(0 if idx == owner else 1000) if valid else 1000000000)
                costs.append(row+[1000000]*len(workers))
            for ri, column in enumerate(_minimum_cost_columns(costs)):
                if column >= len(group) or costs[ri][column] >= 1000000:
                    continue
                idx, ti = workers[ri], group[column]
                actions[idx] = LaborB1._move(positions[idx], tasks[ti][1])
                available.remove(idx)
                pending.remove(ti)
'''

def build(source):
    assert source.count('class LaborB1:') == 1
    a = source.index('        while available and pending:')
    b = source.index('        if tasks:', a)
    changed = source[:a]+REPLACEMENT+source[b:]
    changed = changed.replace('class LaborB1:', HELPER+'class LaborB1:')
    assert changed.count('else 1000) if valid') == 1
    return {name: changed.replace('else 1000) if valid', f'else {penalty}) if valid')
            for name, penalty in [('free', 0), ('soft', 4)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='bots/brain/main.py')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    source = (ROOT/args.baseline).read_text(encoding='utf-8-sig')
    out = ROOT/'data/results/brain-joint-travel'
    arena = ROOT/'data/arena-joint-travel'
    out.mkdir(parents=True, exist_ok=True)
    variants = {'control': source, **build(source)}
    tags = {name: f'joint-travel-{name}-screen-16' for name in variants}
    if args.run:
        assert not any((ROOT/'data/results'/tag/'results.json').exists() for tag in tags.values()), 'Existing results; do not overwrite'
    replay = json.loads((ROOT/'data/results/brain-own-diagnostic/main_vs_main_seed51.json').read_text())
    states = [step[0]['observation'] for step in replay['steps'][:-1:6]]
    reports = []
    for name, text in variants.items():
        path = ROOT/'bots'/f'brain_joint_travel_{name}'/'main.py'
        compile(text, str(path), 'exec')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        module = load(path, f'joint_travel_{name}')
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
                            '--agent', str(ROOT/'bots'/f'brain_joint_travel_{name}'/'main.py'),
                            '--opponents', 'bots/v46/main.py', 'bots/tapeselect/main.py',
                            '--n', '16', '--seed-base', '300', '--workers', '2', '--tag', tags[name]],
                           cwd=ROOT, env=dict(os.environ, KAGGRI_ARENA_DIR=str(arena)),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        data = json.loads((ROOT/'data/results'/tags[name]/'results.json').read_text())
        print(name, data['overall'], flush=True)
    control, control_stats = read(arena, tags['control'])
    for name in ('free', 'soft'):
        candidate, stats = read(arena, tags[name])
        paired = dict(per_opponent=compare(candidate, control), candidate=stats, control=control_stats)
        (out/f'{name}-paired.json').write_text(json.dumps(paired, indent=2), encoding='utf-8')
        print(name, json.dumps(paired), flush=True)


if __name__ == '__main__':
    main()
