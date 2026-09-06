"""Measure own v6 travel and ready harvest backlog without opponent actions."""
import json
import time
from collections import Counter
from pathlib import Path

from benchmark_brain_dispatch_cost import load
from check_brain_collection_jobs import capture

ROOT = Path(__file__).resolve().parents[1]


def main():
    bot = load(ROOT/'bots/brain_value_v6/main.py', 'v6_labor_diagnostic')
    jobs = capture(bot)
    replay = json.loads((ROOT/'data/results/brain-v6-diagnostic/seat0.json').read_text(encoding='utf-8'))
    counts = Counter()
    backlogs = []
    previous = {}
    reverse = dict(NORTH='SOUTH', SOUTH='NORTH', EAST='WEST', WEST='EAST')
    for before, after in zip(replay['steps'], replay['steps'][1:]):
        obs = before[0]['observation']
        jobs.clear()
        action = bot._policy(obs, time.perf_counter())
        recorded = after[0]['action']
        assert all(action[k] == recorded[k] for k in ('farmer', 'hands', 'market'))
        counts['verified_own_actions'] += 1
        if obs['hour'] == 0:
            previous = {}
        units = [action['farmer']]+action['hands']
        for idx, unit in enumerate(units):
            op = unit[0]
            counts['moves'] += op in reverse
            counts['immediate_move_reversals'] += previous.get(idx) == reverse.get(op, 'INVALID')
            previous[idx] = op
        if obs['hour'] != 22:
            continue
        farm = obs['farms'][0]
        harvests = []
        for priority, (x, y), task, need, skip in jobs:
            if task[0] != 'HARVEST':
                continue
            tile = farm['tiles'][y][x]
            product = tile.get('crop') or bot.ANIMALS[tile['animal']]['product']
            units = tile['yield_units']
            inventory = obs['market']['inventory'].get(product, 10000)
            value = sum(bot.market_price(product, inventory+k+1) for k in range(units))
            harvests.append(dict(x=x, y=y, product=product, units=units, quoted_value=value))
        backlogs.append(dict(day=obs['day'], harvests=harvests))
    report = dict(counts=counts, late_day_ready_harvests=backlogs,
                  scope='One seed, own v6 seat0; quoted value is not realised revenue or lost profit. Repeated snapshots are not unique harvests.')
    out = ROOT/'data/results/brain-v6-diagnostic/labor.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(counts=counts, late_ready_jobs=sum(len(r['harvests']) for r in backlogs),
                          late_jobs_quote_at_least400=sum(h['quoted_value'] >= 400 for r in backlogs for h in r['harvests']))))


if __name__ == '__main__':
    main()
