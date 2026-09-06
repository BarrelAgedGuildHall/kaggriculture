"""Reconstruct crop deaths in one own saved game using exact engine operations."""
import copy
import json
from pathlib import Path

from _engine import import_engine
import_engine()
from kaggle_environments.envs.kaggriculture import kaggriculture as kg
from kpi import _apply_units, _market_orders

ROOT = Path(__file__).resolve().parents[1]


def plants(farm):
    return {(x, y): copy.deepcopy(tile) for y, row in enumerate(farm['tiles'])
            for x, tile in enumerate(row) if isinstance(tile, dict) and tile.get('kind') == 'PLANT'}


def main():
    replay = json.loads((ROOT/'data/results/brain-v4-diagnostic/seat0.json').read_text(encoding='utf-8'))
    losses = []
    checked = 0
    for before, after in zip(replay['steps'], replay['steps'][1:]):
        pre = [s['observation'] for s in before]
        post = after[0]['observation']
        actions = [s.get('action') or {} for s in after]
        day, hour = pre[0]['day'], pre[0]['hour']
        step = day*24+hour
        farms = copy.deepcopy(pre[0]['farms'])
        privates = [copy.deepcopy(s['private']) for s in pre]
        market = copy.deepcopy(pre[0]['market'])
        _apply_units(kg, farms, privates, actions, day, 24, 100)
        _market_orders(kg, actions, farms, privates, market, 10, 10, 1, 100)
        assert farms[0]['money'] == post['farms'][0]['money']
        for cause, operation in [('lifespan', lambda: kg._decay_plants(farms[0], step)),
                                 ('drought', lambda: kg._daily_refresh_plants(farms[0], day, 24) if hour == 23 else None)]:
            old = plants(farms[0])
            operation()
            new = plants(farms[0])
            for pos in old.keys()-new.keys():
                tile = old[pos]
                losses.append(dict(day=day, hour=hour, x=pos[0], y=pos[1], crop=tile['crop'],
                                   age=day-tile['planted_day'], held_units=tile['yield_units'], cause=cause))
        assert plants(farms[0]) == plants(post['farms'][0]), step
        checked += 1
    report = dict(seed=2400, seat=0, verified_cash_and_plant_transitions=checked, losses=losses,
                  scope='Own historical v4 game only; exact recorded-action accounting, no action fitting or new game. Drought and lifespan losses separated; no profit estimate.')
    out = ROOT/'data/results/brain-v4-diagnostic/crop-loss-causes.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
