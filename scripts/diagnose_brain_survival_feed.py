"""Count cheap-product survival feeding in our own saved observations.

This is a diagnostic of requested actions, not successful actions or profit.
"""
import json
from pathlib import Path
import time

from benchmark_brain_dispatch_cost import load

ROOT = Path(__file__).resolve().parents[1]


def main():
    bot = load(ROOT/'bots/brain_value_v5/main.py', 'survival_diagnostic')
    replay = json.loads((ROOT/'data/results/brain-v4-diagnostic/seat0.json').read_text(encoding='utf-8'))
    counts = {first: dict(eligible_snapshots=0, requested_feeds=0, tile_days=set()) for first in (20, 25)}
    total_feeds = 0
    for step in replay['steps'][:-1]:
        obs = step[0]['observation']
        farm = obs['farms'][obs.get('player', 0)]
        day = obs['day']
        inventory = obs['market']['inventory']
        action = bot._policy(obs, time.perf_counter())
        positions = [farm['farmer'], *farm.get('hands', [])]
        commands = [action['farmer'], *action.get('hands', [])]
        feeds = {tuple(pos) for pos, command in zip(positions, commands) if command[0] == 'FEED'}
        total_feeds += sum(command[0] == 'FEED' for command in commands)
        for y, row in enumerate(farm['tiles']):
            for x, tile in enumerate(row):
                if not isinstance(tile, dict) or tile.get('animal') not in bot.ANIMALS:
                    continue
                animal = bot.ANIMALS[tile['animal']]
                product = animal['product']
                _, feed_worth = bot._animal_work_value(animal, tile, day, inventory)
                eligible = (day < 29 and not tile.get('fed_today', False)
                            and tile.get('consecutive_unfed', 0) >= 1 and not feed_worth
                            and tile.get('yield_units', 0) == 0
                            and tile.get('pending_care_bonus', 0) == 0
                            and bot.market_price(product, inventory.get(product, bot.MARKET_I0)+1) <= 10
                            and bot.market_price('FERTILIZER', inventory.get('FERTILIZER', bot.MARKET_I0)+1) <= 10)
                for first, count in counts.items():
                    if eligible and day >= first:
                        count['eligible_snapshots'] += 1
                        count['requested_feeds'] += (x, y) in feeds
                        count['tile_days'].add((day, x, y))
    for count in counts.values():
        count['distinct_tile_days'] = len(count.pop('tile_days'))
    report = dict(states=len(replay['steps'])-1, total_requested_feeds=total_feeds,
                  thresholds=counts, scope='Frozen v5 on one own v4 seed/seat; repeated snapshots are not independent games. No policy change or profit estimate.')
    out = ROOT/'data/results/brain-v4-diagnostic/survival-feed.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
