"""Test duplicate-shop carrot demand using current town observations only."""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark_brain_dispatch_cost import load
from compare_paired import compare, read

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/results/brain-duplicate-shops'
ARENA = ROOT/'data/arena-duplicate-shops'


def build(source):
    markers = [('shops = set(view.town_shops)', 'shops = tuple(view.town_shops)'),
               ('+ 0.20\n', '+ 0.20 * {pet}\n'),
               ('+ 0.10\n', '+ 0.10 * {farm}\n')]
    variants = {'control': source}
    for name, cap in [('count', 8), ('cap2', 2)]:
        text = source
        for old, new in markers:
            assert text.count(old) == 1, old
            new = new.format(pet=f'min({cap}, shops.count("PET_CAFE"))',
                             farm=f'min({cap}, shops.count("FARMERS_MARKET"))')
            text = text.replace(old, new)
        variants[name] = text
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    variants = build((ROOT/'bots/brain_value_v5/main.py').read_text(encoding='utf-8-sig'))
    tags = {name: f'duplicate-shops-{name}-screen-16' for name in variants}
    if args.run:
        assert not any((ROOT/'data/results'/tag/'results.json').exists() for tag in tags.values()), 'Existing results; do not overwrite'
    OUT.mkdir(parents=True, exist_ok=True)
    states = [step[0]['observation'] for step in json.loads(
        (ROOT/'data/results/brain-v4-diagnostic/seat0.json').read_text(encoding='utf-8'))['steps'][:-1]]
    base = load(ROOT/'bots/brain_value_v5/main.py', 'duplicate_base')
    reference = [base._policy(obs, time.perf_counter()) for obs in states]
    counterfactual = []
    for obs in states[::6]:
        test = deepcopy(obs)
        test['town']['unlocked_shops'] = ['PET_CAFE', 'PET_CAFE', 'PET_CAFE', 'FARMERS_MARKET']
        counterfactual.append((test, base._policy(test, time.perf_counter())))
    reports = []
    for name, source in variants.items():
        path = ROOT/'bots'/f'brain_duplicate_shops_{name}'/'main.py'
        compile(source, str(path), 'exec')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding='utf-8')
        bot = load(path, f'duplicate_{name}')
        view = base.View.from_obs(states[0])
        checked = 0
        for day in (0, 4, 5, 20, 29):
            for pet in range(9):
                for farm in range(9-pet):
                    test = replace(view, day=day, town_shops=('PET_CAFE',)*pet+('FARMERS_MARKET',)*farm)
                    original = base.DefaultSeason.plan_day(test)
                    plan = bot.DefaultSeason.plan_day(test)
                    assert asdict(replace(plan, crop_weights=original.crop_weights)) == asdict(original)
                    weights = dict(plan.crop_weights)
                    cap = {'control': 1, 'count': 8, 'cap2': 2}[name]
                    assert abs(weights.get('CARROT', 0)-(.2*min(cap, pet)+.1*min(cap, farm))) < 1e-9
                    if pet <= 1 and farm <= 1:
                        assert asdict(plan) == asdict(original)
                    checked += 1
        changed = 0
        for obs, expected in zip(states, reference):
            action = bot._policy(obs, time.perf_counter())
            assert isinstance(action, dict) and 'market' in action
            changed += action != expected
        synthetic_changed = sum(bot._policy(obs, time.perf_counter()) != expected
                                for obs, expected in counterfactual)
        if name != 'control':
            assert synthetic_changed > 0
        reports.append(dict(variant=name, boundary_cases=checked, policy_calls=len(states), changed_states=changed,
                            counterfactual_calls=len(counterfactual), counterfactual_changed=synthetic_changed))
    (OUT/'policy-checks.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps(reports), flush=True)
    if not args.run:
        return
    for name in variants:
        with (OUT/f'{name}.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/'scripts/evaluate.py'),
                            '--agent', f'bots/brain_duplicate_shops_{name}/main.py',
                            '--opponents', 'bots/v46/main.py', 'bots/tapeselect/main.py',
                            '--n', '16', '--seed-base', '430', '--workers', '2', '--tag', tags[name]],
                           cwd=ROOT, env=dict(os.environ, KAGGRI_ARENA_DIR=str(ARENA)),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print(name, json.loads((ROOT/'data/results'/tags[name]/'results.json').read_text(encoding='utf-8'))['overall'], flush=True)
    control, control_stats = read(ARENA, tags['control'])
    for name in ('count', 'cap2'):
        candidate, stats = read(ARENA, tags[name])
        report = dict(per_opponent=compare(candidate, control), candidate=stats, control=control_stats)
        (OUT/f'{name}-paired.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(name, json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
