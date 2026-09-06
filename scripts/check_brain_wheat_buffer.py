"""Check late wheat reserves, order bounds and unchanged unit actions."""
import copy
import json
from pathlib import Path
import time
from benchmark_brain_dispatch_cost import load

ROOT = Path(__file__).resolve().parents[1]


def main():
    states = [step[0]['observation'] for step in json.loads(
        (ROOT/'data/results/brain-v4-diagnostic/seat0.json').read_text(encoding='utf-8'))['steps'][:-1]]
    base = load(ROOT/'bots/brain_value_v6/main.py', 'buffer_base')
    reports = []
    for buffer in (12, 32):
        bot = load(ROOT/f'bots/brain_wheat_buffer_buffer{buffer}/main.py', f'buffer{buffer}')
        observed_reserves = []
        original = bot._sell_orders
        def sell(shed, inventory, day, reserve, slots):
            observed_reserves.append(reserve)
            orders = original(shed, inventory, day, reserve, slots)
            assert len(orders) <= slots
            for op, product, quantity in orders:
                assert 0 < quantity <= shed[product]
                if product == 'WHEAT':
                    assert quantity <= max(0, shed[product]-reserve)
            return orders
        bot._sell_orders = sell
        changed = preserved = checked = 0
        def check(obs):
            nonlocal changed, preserved, checked
            observed_reserves.clear()
            a, b = bot._policy(obs, time.perf_counter()), base._policy(obs, time.perf_counter())
            assert {k:v for k,v in a.items() if k != 'market'} == {k:v for k,v in b.items() if k != 'market'}
            day = obs['day']
            animals = sum(isinstance(t, dict) and t.get('animal') in bot.ANIMALS for row in obs['farms'][0]['tiles'] for t in row)
            expected = 0 if day >= bot.LIQUIDATE_DAY else animals+(4 if day < 14 else buffer)
            assert all(value == expected for value in observed_reserves)
            assert len(a['market']) <= bot.MAX_MARKET_ORDERS
            for order in a['market']:
                if order[:2] == ['BUY_PRODUCT', 'WHEAT']:
                    assert 0 < order[2] <= 14
            if day < 14 or day >= bot.LIQUIDATE_DAY:
                assert a == b
                preserved += 1
            changed += a != b
            checked += 1
        for obs in states:
            check(obs)
        actual = dict(states=checked, changed_actions=changed, preserved_outside_window=preserved)
        for day in (13, 14, 27, 28, 29):
            for stock in (0, 20, 60):
                obs = copy.deepcopy(states[14*24+6])
                obs['day'] = day
                obs['private']['shed']['WHEAT'] = stock
                check(obs)
        reports.append(dict(buffer=buffer, own_observations=actual, additional_boundaries=checked-actual['states']))
    (ROOT/'data/results/brain-wheat-buffer/policy-checks.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps(reports))


if __name__ == '__main__':
    main()
