"""Build care/feeding variants that value shared-price effects on visible rivals.

This bounded hypothesis assumes some rival stock sells after our added unit.
It uses current public tiles and exact price arithmetic, never rival actions.
Build only by default; --run measures paired games after other jobs finish.
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
HELPER = '''def _rival_price_pressure(farm, day, inventory, weight):
    quantities = {a['product']: 0 for a in ANIMALS.values()}
    for row in farm.get('tiles', []):
        for tile in row:
            if not isinstance(tile, dict) or tile.get('animal') not in ANIMALS:
                continue
            spec = ANIMALS[tile['animal']]
            first = int(tile.get('placed_day', day)) + spec['first_yield_day']
            ticks = sum(d >= first and (d-first) % spec['interval'] == 0
                        for d in range(day+1, min(FINAL_DAY, day+3)+1))
            pending = int(tile.get('pending_care_bonus', 0)) if ticks else 0
            quantity = min(spec['max_held'], int(tile.get('yield_units', 0)) + ticks + pending)
            quantities[spec['product']] += quantity
    pressure = {}
    for product, quantity in quantities.items():
        inv = inventory.get(product, MARKET_I0) + 2
        # A floor-price sale does not increase market inventory in the engine.
        if quantity <= 0 or market_price(product, inv) <= PRICE_FLOOR:
            pressure[product] = 0.0
            continue
        quantity = min(24, quantity)
        before = predict_sale(product, inv, quantity)[0]
        after = predict_sale(product, inv+1, quantity)[0]
        pressure[product] = weight * max(0, before-after)
    return pressure


'''


def build(source, weight):
    marker = 'def _animal_work_value(a, tile, day, market_inventory):'
    inventory = '    mkt_inv = dict(_get(market, "inventory", {}) or {})'
    call = '_animal_work_value(a, t, day, mkt_inv)'
    quote = '    wheat_quote = market_price("WHEAT", market_inventory.get("WHEAT", MARKET_I0) - 1)'
    assert all(source.count(x) == 1 for x in (marker, inventory, call, quote))
    return source.replace(marker, HELPER+'def _animal_work_value(a, tile, day, market_inventory, pressure=None):').replace(
        quote, '    product_quote += (pressure or {}).get(a["product"], 0.0)\n'+quote).replace(
        inventory, inventory+f'\n    rival_pressure = _rival_price_pressure(farms[1-me], day, mkt_inv, {weight})').replace(
        call, '_animal_work_value(a, t, day, mkt_inv, rival_pressure)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='bots/brain/main.py')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    source = (ROOT/args.baseline).read_text(encoding='utf-8-sig')
    replay = json.loads((ROOT/'data/results/brain-own-diagnostic/main_vs_main_seed51.json').read_text())
    states = [step[0]['observation'] for step in replay['steps'][:-1:6]]
    reports = []
    for name, weight in [('half', .5), ('full', 1.0)]:
        path = ROOT/'bots'/f'brain_market_pressure_{name}'/'main.py'
        text = build(source, weight)
        compile(text, str(path), 'exec')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        bot = load(path, 'market_pressure_'+name)
        for obs in states:
            assert isinstance(bot._policy(obs, time.perf_counter()), dict)
        empty = bot._rival_price_pressure({'tiles': []}, 10, {}, weight)
        assert all(value == 0 for value in empty.values())
        farm = {'tiles': [[{'animal': 'COW', 'placed_day': 0, 'yield_units': 3}]]}
        pressure = bot._rival_price_pressure(farm, 29, {}, weight)['MILK']
        expected = weight*(bot.predict_sale('MILK', 10002, 3)[0]-bot.predict_sale('MILK', 10003, 3)[0])
        assert pressure == expected and pressure > 0
        floor = bot._rival_price_pressure(farm, 20, {'MILK': 20000}, weight)
        assert floor['MILK'] == 0
        tile = {'animal': 'COW', 'placed_day': 28, 'yield_units': 0}
        assert bot._rival_price_pressure({'tiles': [[tile]]}, 28, {}, weight)['MILK'] == 0
        reports.append(dict(variant=name, weight=weight, policy_calls=len(states), boundary_checks=4))
    out = ROOT/'data/results/brain-market-pressure'
    out.mkdir(parents=True, exist_ok=True)
    (out/'offline-checks.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps(reports))
    if args.run:
        arena = ROOT/'data/arena-market-pressure'
        tags = {name: f'market-pressure-{name}-screen-16' for name in ('control', 'half', 'full')}
        assert not any((ROOT/'data/results'/tag/'results.json').exists() for tag in tags.values()), 'Existing results; do not overwrite'
        control = ROOT/'bots/brain_market_pressure_control/main.py'
        control.parent.mkdir(parents=True, exist_ok=True)
        control.write_text(source, encoding='utf-8')
        for name in tags:
            path = ROOT/'bots'/f'brain_market_pressure_{name}'/'main.py'
            with (out/f'{name}.log').open('w', encoding='utf-8') as log:
                subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/'scripts/evaluate.py'),
                                '--agent', str(path), '--opponents', 'bots/v46/main.py',
                                'bots/tapeselect/main.py', '--n', '16', '--seed-base', '240',
                                '--workers', '2', '--tag', tags[name]], cwd=ROOT,
                               env=dict(os.environ, KAGGRI_ARENA_DIR=str(arena)),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads((ROOT/'data/results'/tags[name]/'results.json').read_text())
            print(name, report['overall'], flush=True)
        baseline, control_stats = read(arena, tags['control'])
        for name in ('half', 'full'):
            candidate, stats = read(arena, tags[name])
            paired = dict(per_opponent=compare(candidate, baseline), candidate=stats, control=control_stats)
            (out/f'{name}-paired.json').write_text(json.dumps(paired, indent=2), encoding='utf-8')
            print(name, json.dumps(paired), flush=True)


if __name__ == '__main__':
    main()
