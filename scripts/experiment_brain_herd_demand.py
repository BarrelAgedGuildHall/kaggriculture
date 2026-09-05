"""Prepare observed-demand herd caps; run only after the active trial finishes.

Existing animals are retained. Sheep expansion follows revealed wool demand;
duplicate yarn shops count separately. No opponent actions enter the policy.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark_brain_dispatch_cost import load

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/results/brain-herd-demand'


def build(source):
    marker = '        animal_targets = {kind: 0 for kind in ANIMAL_TARGETS}'
    loop = '            for kind, cap in ANIMAL_TARGETS.items():'
    assert source.count(marker) == source.count(loop) == 1
    variants = {}
    for floor in (2, 5):
        caps = ('        demand_caps = dict(ANIMAL_TARGETS)\n'
                f'        demand_caps["SHEEP"] = min(ANIMAL_TARGETS["SHEEP"], {floor} + 4 * view.town_shops.count("YARN_STORE"))\n')
        variants[f'sheep{floor}'] = source.replace(marker, caps+marker).replace(
            loop, '            for kind, cap in demand_caps.items():')
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='bots/brain/main.py')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    source = (ROOT/args.baseline).read_text(encoding='utf-8-sig')
    replay = json.loads((ROOT/'data/results/brain-own-diagnostic/main_vs_main_seed51.json').read_text())
    # Functional coverage on our recorded observations, not imitation fitting.
    states = [step[0]['observation'] for step in replay['steps'][:-1:6]]
    variants = {'control': source, **build(source)}
    checks = []
    for name, text in variants.items():
        path = ROOT/'bots'/f'brain_herd_demand_{name}'/'main.py'
        compile(text, str(path), 'exec')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        module = load(path, f'herd_demand_{name}')
        for obs in states:
            action = module._policy(obs, time.perf_counter())
            assert isinstance(action, dict) and 'market' in action
        boundary_checks = 0
        if name != 'control':
            floor = int(name.removeprefix('sheep'))
            view = module.View.from_obs(states[0])
            for shops in range(4):
                test_view = replace(view, day=20, town_shops=('YARN_STORE',)*shops)
                targets = dict(module.DefaultSeason.plan_day(test_view).animal_targets)
                assert targets == {'COW': 7, 'SHEEP': min(11, floor+4*shops)}
                boundary_checks += 1
        checks.append(dict(variant=name, policy_calls=len(states), boundary_checks=boundary_checks))
        if not args.run:
            continue
        tag = f'herd-demand-{name}-screen-16'
        environment = dict(os.environ, KAGGRI_ARENA_DIR=str(ROOT/'data/arena-herd-demand'))
        with (OUT/f'{name}.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/'scripts/evaluate.py'),
                            '--agent', str(path), '--opponents', 'bots/v46/main.py',
                            'bots/tapeselect/main.py', '--n', '16', '--seed-base', '200',
                            '--workers', '2', '--tag', tag], cwd=ROOT, env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((ROOT/'data/results'/tag/'results.json').read_text())
        print(name, result['overall'], flush=True)
    (OUT/'offline-checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps(checks), flush=True)


if __name__ == '__main__':
    main()
