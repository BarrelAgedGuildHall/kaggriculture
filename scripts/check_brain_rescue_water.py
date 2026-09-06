"""Verify only drought-water priorities and their travel assignment change."""
import copy
import json
from pathlib import Path
import time

from benchmark_brain_dispatch_cost import load
from check_brain_collection_jobs import capture

ROOT = Path(__file__).resolve().parents[1]


def main():
    states = [step[0]['observation'] for step in json.loads(
        (ROOT/'data/results/brain-v4-diagnostic/seat0.json').read_text(encoding='utf-8'))['steps'][:-1]]
    reports = []
    for threshold in (12, 18):
        base = load(ROOT/'bots/brain_value_v6/main.py', f'rescue_base{threshold}')
        bot = load(ROOT/f'bots/brain_rescue_water_rescue{threshold}/main.py', f'rescue{threshold}')
        positions, inventories = ((9, 9), (5, 0)), ({}, {})
        emergency = (-3, (4, 0), ('WATER',), None, None)
        ordinary = (-1, (4, 0), ('WATER',), None, None)
        routine = (3, (8, 9), ('COLLECT_FERTILIZER',), None, None)
        assert bot.LaborB1._assign(positions, inventories, [emergency, routine])[1] == ['WEST']
        assert base.LaborB1._assign(positions, inventories, [emergency, routine])[1] != ['WEST']
        assert bot.LaborB1._assign(positions, inventories, [ordinary, routine]) == base.LaborB1._assign(positions, inventories, [ordinary, routine])
        nonwater = (-3, (4, 0), ('HARVEST',), None, None)
        assert bot.LaborB1._assign(positions, inventories, [nonwater, routine]) == base.LaborB1._assign(positions, inventories, [nonwater, routine])
        base_jobs, jobs = capture(base), capture(bot)
        counts = dict(states=0, changed_actions=0, changed_job_lists=0, preserved_outside_window=0)
        def check(obs):
            base_jobs.clear(); jobs.clear()
            reference = base._policy(obs, time.perf_counter())
            action = bot._policy(obs, time.perf_counter())
            expected = []
            for job in base_jobs:
                priority, (x, y), op, need, skip = job
                tile = obs['farms'][0]['tiles'][y][x]
                rescue = (op[0] == 'WATER' and obs['day'] < 29 and obs['hour'] >= threshold
                          and isinstance(tile, dict) and tile.get('consecutive_unwatered', 0) >= 1)
                expected.append((-3 if rescue else priority, (x, y), op, need, skip))
            expected.sort(key=lambda job: (job[0], job[1][0], job[1][1]))
            assert jobs == expected, (obs['day'], obs['hour'], [(a,b) for a,b in zip(jobs, expected) if a != b][:4], len(jobs),len(expected))
            if obs['day'] == 29 or obs['hour'] < threshold:
                assert action == reference
                counts['preserved_outside_window'] += 1
            counts['states'] += 1
            counts['changed_actions'] += action != reference
            counts['changed_job_lists'] += jobs != base_jobs
        for obs in states:
            check(obs)
        actual = counts.copy()
        for day in (13, 29):
            for hour in (11, 12, 17, 18, 23):
                obs = copy.deepcopy(states[13*24+10])
                obs['day'], obs['hour'] = day, hour
                check(obs)
        reports.append(dict(threshold=threshold, assignment_checks=4, own_observations=actual,
                            additional_boundary_calls=counts['states']-actual['states']))
    out = ROOT/'data/results/brain-rescue-water/job-checks.json'
    out.write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps(reports))


if __name__ == '__main__':
    main()
