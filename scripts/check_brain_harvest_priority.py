"""Verify only qualifying harvest priorities change on complete own v6 states."""
import json
import time
from pathlib import Path
from benchmark_brain_dispatch_cost import load
from check_brain_collection_jobs import capture

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = load(ROOT/'bots/brain_value_v6/main.py', 'harvest_priority_base')
    base_jobs = capture(base)
    replay = json.loads((ROOT/'data/results/brain-v6-diagnostic/seat0.json').read_text(encoding='utf-8'))
    reports = []
    for threshold in (400, 800):
        bot = load(ROOT/f'bots/brain_harvest_priority_value{threshold}/main.py', f'priority_{threshold}')
        jobs = capture(bot)
        counts = dict(threshold=threshold, observations=0, changed_actions=0, changed_job_graphs=0, preserved_outside_window=0)
        for step in replay['steps'][:-1]:
            obs = step[0]['observation']
            base_jobs.clear()
            jobs.clear()
            reference = base._policy(obs, time.perf_counter())
            action = bot._policy(obs, time.perf_counter())
            expected = []
            active = obs['hour'] >= 12 and obs['day'] < 29
            for priority, pos, task, need, skip in base_jobs:
                if active and task[0] == 'HARVEST':
                    tile = obs['farms'][0]['tiles'][pos[1]][pos[0]]
                    product = tile.get('crop') or base.ANIMALS[tile['animal']]['product']
                    inv = obs['market']['inventory'].get(product, 10000)
                    value = sum(base.market_price(product, inv+k+1) for k in range(tile['yield_units']))
                    if value >= threshold:
                        priority = -1
                expected.append((priority, pos, task, need, skip))
            expected.sort(key=lambda j: (j[0], j[1][0], j[1][1]))
            # Equal-priority tasks on one tile retain coordinator insertion order;
            # the already sorted baseline cannot reconstruct that tie order.
            assert sorted(jobs, key=repr) == sorted(expected, key=repr), (threshold, obs['day'], obs['hour'])
            if not active:
                assert action == reference
                counts['preserved_outside_window'] += 1
            counts['observations'] += 1
            counts['changed_actions'] += action != reference
            counts['changed_job_graphs'] += jobs != base_jobs
        assert counts['changed_actions'] > 0
        reports.append(counts)
    (ROOT/'data/results/brain-harvest-priority/policy-checks.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps(reports))


if __name__ == '__main__':
    main()
