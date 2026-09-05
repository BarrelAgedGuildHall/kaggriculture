"""Screen fourth-quadrant expansion and a higher labor ceiling; no imitation."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark_brain_dispatch_cost import load
from compare_paired import compare, read

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/results/brain-expansion'
ARENA = ROOT/'data/arena-expansion'


def build(source):
    old = 'min(3, 1 + view.day // 4)'
    gate = '0 <= n_extra < 2 and len(unlocked)'
    assert source.count(old) == source.count(gate) == 1
    expanded = source.replace(old, 'min(4, 1 + view.day // 4)').replace(
        gate, '0 <= n_extra < 3 and (n_extra < 2 or day <= 18) and len(unlocked)')
    assert source.count('HIRES_PER_DAY = 12') == source.count('(10, 12))') == 1
    more_labor = expanded.replace('HIRES_PER_DAY = 12', 'HIRES_PER_DAY = 14').replace(
        '(10, 12))', '(10, 14))')
    return {'control': source, 'land': expanded, 'labor': more_labor}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='bots/brain/main.py')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    variants = build((ROOT/args.baseline).read_text(encoding='utf-8-sig'))
    tags = {name: f'expansion-{name}-screen-16' for name in variants}
    if args.run:
        assert not any((ROOT/'data/results'/tag/'results.json').exists() for tag in tags.values()), 'Existing results; do not overwrite'
    replay = json.loads((ROOT/'data/results/brain-own-diagnostic/main_vs_main_seed51.json').read_text())
    states = [step[0]['observation'] for step in replay['steps'][:-1:6]]
    checks = []
    for name, source in variants.items():
        path = ROOT/'bots'/f'brain_expansion_{name}'/'main.py'
        compile(source, str(path), 'exec')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding='utf-8')
        bot = load(path, f'expansion_{name}')
        for obs in states:
            assert isinstance(bot._policy(obs, time.perf_counter()), dict)
        boundaries = 0
        if name != 'control':
            view = bot.View.from_obs(states[0])
            for day in (0, 8, 12, 19):
                plan = bot.DefaultSeason.plan_day(replace(view, day=day, town_shops=()))
                assert plan.target_quadrants == min(4, 1+day//4)
                if day >= 12:
                    assert plan.target_hands == (14 if name == 'labor' else 12)
                boundaries += 1
        checks.append(dict(variant=name, policy_calls=len(states), boundary_checks=boundaries))
        if not args.run:
            continue
        with (OUT/f'{name}.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/'scripts/evaluate.py'),
                            '--agent', str(path), '--opponents', 'bots/v46/main.py',
                            'bots/tapeselect/main.py', '--n', '16', '--seed-base', '230',
                            '--workers', '2', '--tag', tags[name]], cwd=ROOT,
                           env=dict(os.environ, KAGGRI_ARENA_DIR=str(ARENA)),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((ROOT/'data/results'/tags[name]/'results.json').read_text())
        print(name, report['overall'], flush=True)
    (OUT/'offline-checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps(checks), flush=True)
    if args.run:
        control, control_stats = read(ARENA, tags['control'])
        for name in ('land', 'labor'):
            candidate, stats = read(ARENA, tags[name])
            result = dict(per_opponent=compare(candidate, control), candidate=stats, control=control_stats)
            (OUT/f'{name}-paired.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(name, json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
