"""Check bundle routing preserves job graphs, market orders, and on-tile work."""
import json
import time
from pathlib import Path
from benchmark_brain_dispatch_cost import load
from check_brain_collection_jobs import capture

ROOT = Path(__file__).resolve().parents[1]
MOVES = {'NORTH', 'SOUTH', 'EAST', 'WEST'}


def main():
    base = load(ROOT/'bots/brain_value_v6/main.py', 'bundle_base')
    base_jobs = capture(base)
    replay = json.loads((ROOT/'data/results/brain-v6-diagnostic/seat0.json').read_text(encoding='utf-8'))
    reports = []
    for credit in (1, 2):
        bot = load(ROOT/f'bots/brain_bundle_credit_credit{credit}/main.py', f'bundle_credit{credit}')
        jobs = capture(bot)
        counts = dict(credit=credit, observations=0, changed_actions=0, preserved_on_tile_assignments=0)
        for step in replay['steps'][:-1]:
            obs = step[0]['observation']
            base_jobs.clear()
            jobs.clear()
            reference = base._policy(obs, time.perf_counter())
            action = bot._policy(obs, time.perf_counter())
            assert jobs == base_jobs
            assert action['market'] == reference['market']
            farm = obs['farms'][0]
            positions = [tuple(farm['farmer'])]+[tuple(p) for p in farm['hands']]
            inv = obs['private']['inventories']
            frozen_jobs = list(base_jobs)
            before = base.LaborB1._assign(positions, inv, frozen_jobs)
            after = bot.LaborB1._assign(positions, inv, frozen_jobs)
            for idx, unit in before.items():
                if unit[0] not in MOVES:
                    assert after[idx] == unit
                    counts['preserved_on_tile_assignments'] += 1
            counts['observations'] += 1
            counts['changed_actions'] += action != reference
        task = lambda priority, pos, op, need=None, skip=None: (priority, pos, (op,), need, skip)
        a = task(0, (1, 0), 'HARVEST')
        b = task(0, (0, 1), 'HARVEST')
        c = task(3, (0, 1), 'CARE')
        assert bot.LaborB1._assign([(0, 0)], [{}], [a,b,c])[0] == ['SOUTH']
        # Ineligible extra work earns no routing credit.
        c_need = task(3, (0, 1), 'FEED', need='WHEAT')
        c_skip = task(3, (0, 1), 'PICKUP', skip='WHEAT')
        assert bot.LaborB1._assign([(0, 0)], [{}], [a,b,c_need])[0] == ['EAST']
        assert bot.LaborB1._assign([(0, 0)], [{'WHEAT':1}], [a,b,c_skip])[0] == ['EAST']
        urgent = task(-2, (1, 0), 'FEED')
        assert bot.LaborB1._assign([(0, 0)], [{}], [urgent,b,c])[0] == ['EAST']
        exact = task(3, (0, 0), 'CARE')
        assert bot.LaborB1._assign([(0, 0)], [{}], [exact,b,c])[0] == ['CARE']
        counts['assignment_fixtures'] = 5
        assert counts['changed_actions'] > 0
        reports.append(counts)
    (ROOT/'data/results/brain-bundle-credit/policy-checks.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps(reports))


if __name__ == '__main__':
    main()
