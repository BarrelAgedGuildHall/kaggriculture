"""Screen small tomato allocations; current observations and engine rules only."""
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
OUT = ROOT/'data/results/brain-tomato'
ARENA = ROOT/'data/arena-tomato'


def build(source):
    old = 'CROP_SHARE = {"WHEAT": 0.55, "STRAWBERRY": 0.25, "MELON": 0.20}'
    cap = 'SEED_CAP = {"WHEAT": 50, "STRAWBERRY": 24, "MELON": 16}'
    assert source.count(old) == source.count(cap) == 1
    variants = {'control': source}
    for share in (10, 20):
        weights = {'WHEAT': (55-share)/100, 'STRAWBERRY': .25, 'MELON': .20, 'TOMATO': share/100}
        variants[f'tomato{share}'] = source.replace(old, 'CROP_SHARE = '+repr(weights)).replace(
            cap, 'SEED_CAP = {"WHEAT": 50, "STRAWBERRY": 24, "MELON": 16, "TOMATO": 24}')
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='bots/brain/main.py')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    variants = build((ROOT/args.baseline).read_text(encoding='utf-8-sig'))
    tags = {name: f'tomato-{name}-screen-16' for name in variants}
    if args.run:
        assert not any((ROOT/'data/results'/tag/'results.json').exists() for tag in tags.values()), 'Existing results; do not overwrite'
    replay = json.loads((ROOT/'data/results/brain-own-diagnostic/main_vs_main_seed51.json').read_text())
    states = [step[0]['observation'] for step in replay['steps'][:-1:6]]
    checks = []
    for name, source in variants.items():
        path = ROOT/'bots'/f'brain_tomato_{name}'/'main.py'
        compile(source, str(path), 'exec')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding='utf-8')
        bot = load(path, f'tomato_{name}')
        for obs in states:
            assert isinstance(bot._policy(obs, time.perf_counter()), dict)
        boundaries = 0
        if name != 'control':
            share = int(name.removeprefix('tomato'))/100
            view = bot.View.from_obs(states[0])
            for day in (0, 4, 5, 16):
                plan = bot.DefaultSeason.plan_day(replace(view, day=day, town_shops=()))
                weights = dict(plan.crop_weights)
                assert weights.get('TOMATO', 0) == (0 if day < 5 else share)
                assert weights['MELON'] == .20
                assert abs(sum(weights.values())-1) < 1e-9
                boundaries += 1
        checks.append(dict(variant=name, policy_calls=len(states), boundary_checks=boundaries))
        if not args.run:
            continue
        with (OUT/f'{name}.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/'scripts/evaluate.py'),
                            '--agent', str(path), '--opponents', 'bots/v46/main.py',
                            'bots/tapeselect/main.py', '--n', '16', '--seed-base', '220',
                            '--workers', '2', '--tag', tags[name]], cwd=ROOT,
                           env=dict(os.environ, KAGGRI_ARENA_DIR=str(ARENA)),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((ROOT/'data/results'/tags[name]/'results.json').read_text())
        print(name, report['overall'], flush=True)
    (OUT/'offline-checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps(checks), flush=True)
    if args.run:
        control, control_stats = read(ARENA, tags['control'])
        for name in ('tomato10', 'tomato20'):
            candidate, stats = read(ARENA, tags[name])
            result = dict(per_opponent=compare(candidate, control), candidate=stats, control=control_stats)
            (OUT/f'{name}-paired.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(name, json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
