"""Screen short daytime delivery for earlier market sales, without imitation.

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
    marker = '    # ---- market orders -----------------------------------------------------------'
    assert source.count(marker) == 1
    variants = {}
    for radius in (1, 3):
        patch = f"""
    # Short daytime delivery competes for earlier market prices.
    if not last_day and hour <= 18:
        delivery_room = shed_room
        for u in units:
            inv = u['inv']
            if inv.get('WHEAT', 0) or any(inv.get(a, 0) for a in ANIMALS):
                continue
            if fertilizer_reserve and inv.get('FERTILIZER', 0):
                continue
            quantity = sum(inv.values())
            if not quantity or quantity > delivery_room:
                continue
            tx, ty = min(_shed_tiles(n), key=lambda p: abs(p[0]-u['x'])+abs(p[1]-u['y']))
            distance = abs(tx-u['x'])+abs(ty-u['y'])
            if distance > {radius}:
                continue
            sale = _sell_orders(inv, mkt_inv, day, 0, MAX_MARKET_ORDERS)
            value = sum(predict_sale(item, mkt_inv.get(item, MARKET_I0), count)[0]
                        for _op, item, count in sale)
            if value >= 250:
                actions[u['i']] = _step_toward(u['x'], u['y'], tx, ty) or ['DROP']
                delivery_room -= quantity

"""
        variants[f'near{radius}'] = source.replace(marker, patch+marker)
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='bots/brain/main.py')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    source = (ROOT/args.baseline).read_text(encoding='utf-8-sig')
    out = ROOT/'data/results/brain-day-delivery'
    arena = ROOT/'data/arena-day-delivery'
    out.mkdir(parents=True, exist_ok=True)
    variants = {'control': source, **build(source)}
    tags = {name: f'day-delivery-{name}-screen-16' for name in variants}
    if args.run:
        assert not any((ROOT/'data/results'/tag/'results.json').exists() for tag in tags.values()), 'Existing results; do not overwrite'
    replay = json.loads((ROOT/'data/results/brain-own-diagnostic/main_vs_main_seed51.json').read_text())
    states = [step[0]['observation'] for step in replay['steps'][:-1:6]]
    reports = []
    for name, text in variants.items():
        path = ROOT/'bots'/f'brain_day_delivery_{name}'/'main.py'
        compile(text, str(path), 'exec')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        module = load(path, f'day_delivery_{name}')
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
                            '--agent', str(ROOT/'bots'/f'brain_day_delivery_{name}'/'main.py'),
                            '--opponents', 'bots/v46/main.py', 'bots/tapeselect/main.py',
                            '--n', '16', '--seed-base', '270', '--workers', '2', '--tag', tags[name]],
                           cwd=ROOT, env=dict(os.environ, KAGGRI_ARENA_DIR=str(arena)),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        data = json.loads((ROOT/'data/results'/tags[name]/'results.json').read_text())
        print(name, data['overall'], flush=True)
    control, control_stats = read(arena, tags['control'])
    for name in ('near1', 'near3'):
        candidate, stats = read(arena, tags[name])
        paired = dict(per_opponent=compare(candidate, control), candidate=stats, control=control_stats)
        (out/f'{name}-paired.json').write_text(json.dumps(paired, indent=2), encoding='utf-8')
        print(name, json.dumps(paired), flush=True)


if __name__ == '__main__':
    main()
