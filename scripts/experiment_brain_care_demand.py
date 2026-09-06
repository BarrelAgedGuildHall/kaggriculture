"""Test revealed town demand in future care valuation, without imitation.

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


HELPER = '''def _revealed_product_drain(product, start, end, shops):
    if end <= start:
        return 0
    shop_events = (end-1)//4 - (start-1)//4
    center_events = (end-1)//24 - (start-1)//24
    each_event = sum((2 if len(SHOPS[shop]) == 1 else 1)
                     for shop in shops if shop in SHOPS and product in SHOPS[shop])
    return shop_events*each_event + (center_events if product != 'FERTILIZER' else 0)


'''

def build(source):
    signature = 'def _animal_work_value(a, tile, day, market_inventory):'
    marker = '    pending = int(tile.get("pending_care_bonus", 0))'
    condition = 'and product_quote > 0.5 * wheat_quote + 35)'
    call = '_animal_work_value(a, t, day, mkt_inv)'
    assert source.count(signature) == source.count(marker) == source.count(condition) == 1
    assert source.count(call) >= 1
    changed = source.replace(signature, HELPER+'def _animal_work_value(a, tile, day, market_inventory, hour=0, shops=()):')
    changed = changed.replace(call, '_animal_work_value(a, t, day, mkt_inv, hour, view.town_shops)')
    variants = {}
    for name, fraction in [('half', 0.5), ('full', 1.0)]:
        forecast = ('    drain = _revealed_product_drain(a["product"], day*24+hour, min(next_bonus_day, day+3)*24, shops)\n'
                    f'    care_quote = market_price(a["product"], market_inventory.get(a["product"], MARKET_I0)+2-int({fraction}*drain))\n')
        variants[name] = changed.replace(marker, forecast+marker).replace(condition, 'and care_quote > 0.5 * wheat_quote + 35)')
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='bots/brain/main.py')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    source = (ROOT/args.baseline).read_text(encoding='utf-8-sig')
    out = ROOT/'data/results/brain-care-demand'
    arena = ROOT/'data/arena-care-demand'
    out.mkdir(parents=True, exist_ok=True)
    variants = {'control': source, **build(source)}
    tags = {name: f'care-demand-{name}-screen-16' for name in variants}
    if args.run:
        assert not any((ROOT/'data/results'/tag/'results.json').exists() for tag in tags.values()), 'Existing results; do not overwrite'
    replay = json.loads((ROOT/'data/results/brain-own-diagnostic/main_vs_main_seed51.json').read_text())
    states = [step[0]['observation'] for step in replay['steps'][:-1:6]]
    reports = []
    for name, text in variants.items():
        path = ROOT/'bots'/f'brain_care_demand_{name}'/'main.py'
        compile(text, str(path), 'exec')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        module = load(path, f'care_demand_{name}')
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
                            '--agent', str(ROOT/'bots'/f'brain_care_demand_{name}'/'main.py'),
                            '--opponents', 'bots/v46/main.py', 'bots/tapeselect/main.py',
                            '--n', '16', '--seed-base', '320', '--workers', '2', '--tag', tags[name]],
                           cwd=ROOT, env=dict(os.environ, KAGGRI_ARENA_DIR=str(arena)),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        data = json.loads((ROOT/'data/results'/tags[name]/'results.json').read_text())
        print(name, data['overall'], flush=True)
    control, control_stats = read(arena, tags['control'])
    for name in ('half', 'full'):
        candidate, stats = read(arena, tags[name])
        paired = dict(per_opponent=compare(candidate, control), candidate=stats, control=control_stats)
        (out/f'{name}-paired.json').write_text(json.dumps(paired, indent=2), encoding='utf-8')
        print(name, json.dumps(paired), flush=True)


if __name__ == '__main__':
    main()
