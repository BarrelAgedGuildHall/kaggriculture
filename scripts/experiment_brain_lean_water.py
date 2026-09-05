"""Prepare survival/yield-driven watering; build only by default, no imitation."""
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
HELPER = '''def _fertilizer_water_tick(tile, spec, day):
    if not spec['ongoing'] or tile.get('fertilized_until_day', -1) < day:
        return False
    age = day-int(tile.get('planted_day', day))
    offset = age+1-spec['first_yield_day']
    return (offset >= 0 and offset % spec['interval'] == 0
            and offset // spec['interval'] < spec['max_yield']
            and int(tile.get('yield_units', 0)) < spec['max_yield'])


'''


def build(source):
    marker = 'def _policy(obs, t0):'
    condition = 'if not watered and viable and (unwatered >= 1 or gains or hour >= 16):'
    assert source.count(marker) == source.count(condition) == 1
    base = source.replace(marker, HELPER+marker)
    early = base.replace(condition,
        'if not watered and viable and (unwatered >= 1 or gains or _fertilizer_water_tick(t, cd, day) or (hour >= 16 and age >= cd["first_yield_day"])):')
    lean = base.replace(condition,
        'if not watered and viable and (unwatered >= 1 or gains or _fertilizer_water_tick(t, cd, day)):')
    return {'early': early, 'lean': lean}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='bots/brain/main.py')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    source = (ROOT/args.baseline).read_text(encoding='utf-8-sig')
    replay = json.loads((ROOT/'data/results/brain-own-diagnostic/main_vs_main_seed51.json').read_text())
    states = [step[0]['observation'] for step in replay['steps'][:-1:6]]
    reports = []
    for name, text in build(source).items():
        path = ROOT/'bots'/f'brain_lean_water_{name}'/'main.py'
        compile(text, str(path), 'exec')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        bot = load(path, 'lean_water_'+name)
        for obs in states:
            assert isinstance(bot._policy(obs, time.perf_counter()), dict)
        spec = bot.CROPS['STRAWBERRY']
        tile = dict(planted_day=0, fertilized_until_day=30, yield_units=0)
        for day, expected in [(8, False), (9, True), (10, False), (11, True), (15, True), (17, False)]:
            assert bot._fertilizer_water_tick(tile, spec, day) == expected
        assert not bot._fertilizer_water_tick(dict(tile, yield_units=4), spec, 9)
        assert not bot._fertilizer_water_tick(dict(tile, fertilized_until_day=8), spec, 9)
        assert not bot._fertilizer_water_tick(tile, bot.CROPS['WHEAT'], 9)
        reports.append(dict(variant=name, policy_calls=len(states), fertilizer_boundary_checks=9))
    out = ROOT/'data/results/brain-lean-water'
    out.mkdir(parents=True, exist_ok=True)
    (out/'offline-checks.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps(reports))
    if args.run:
        arena = ROOT/'data/arena-lean-water'
        tags = {name: f'lean-water-{name}-screen-16' for name in ('control', 'early', 'lean')}
        assert not any((ROOT/'data/results'/tag/'results.json').exists() for tag in tags.values()), 'Existing results; do not overwrite'
        control = ROOT/'bots/brain_lean_water_control/main.py'
        control.parent.mkdir(parents=True, exist_ok=True)
        control.write_text(source, encoding='utf-8')
        for name in tags:
            path = ROOT/'bots'/f'brain_lean_water_{name}'/'main.py'
            with (out/f'{name}.log').open('w', encoding='utf-8') as log:
                subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/'scripts/evaluate.py'),
                                '--agent', str(path), '--opponents', 'bots/v46/main.py',
                                'bots/tapeselect/main.py', '--n', '16', '--seed-base', '250',
                                '--workers', '2', '--tag', tags[name]], cwd=ROOT,
                               env=dict(os.environ, KAGGRI_ARENA_DIR=str(arena)),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((ROOT/'data/results'/tags[name]/'results.json').read_text())
            print(name, result['overall'], flush=True)
        baseline, control_stats = read(arena, tags['control'])
        for name in ('early', 'lean'):
            candidate, stats = read(arena, tags[name])
            paired = dict(per_opponent=compare(candidate, baseline), candidate=stats, control=control_stats)
            (out/f'{name}-paired.json').write_text(json.dumps(paired, indent=2), encoding='utf-8')
            print(name, json.dumps(paired), flush=True)


if __name__ == '__main__':
    main()
