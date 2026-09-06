"""Check collection gating against the original policy's complete job graph.

Uses only saved own observations; does not run games or inspect opponent actions.
"""
import copy
import json
from pathlib import Path
import time

from benchmark_brain_dispatch_cost import load

ROOT = Path(__file__).resolve().parents[1]


def capture(module):
    jobs = []
    original = module.LaborB1._assign

    def assign(positions, inventories, tasks):
        jobs[:] = tasks
        return original(positions, inventories, tasks)

    module.LaborB1._assign = staticmethod(assign)
    return jobs


def main():
    baseline = load(ROOT/'bots/brain_value_v4/main.py', 'collection_jobs_base')
    candidate = load(ROOT/'bots/brain_fertilizer_collection/main.py', 'collection_jobs_candidate')
    base_jobs, candidate_jobs = capture(baseline), capture(candidate)
    counts = dict(observations=0, filtered=0, crop_reserve_preserved=0, changed_actions=0)
    boundary_observation = None

    def check(obs):
        nonlocal boundary_observation
        base_jobs.clear()
        candidate_jobs.clear()
        reference = baseline._policy(obs, time.perf_counter())
        action = candidate._policy(obs, time.perf_counter())
        private = obs['private']
        stock = private['shed'].get('FERTILIZER', 0)+sum(inv.get('FERTILIZER', 0) for inv in private['inventories'])
        reserve = min(8, sum(job[2][0] == 'FERTILIZE' for job in base_jobs))
        buffer = 0 if obs['day'] >= 29 else max(2, reserve)
        quote = baseline.market_price('FERTILIZER', obs['market']['inventory'].get('FERTILIZER', 10000)+1)
        suppress = quote <= 30 and stock >= buffer
        expected = [job for job in base_jobs if not suppress or job[2][0] != 'COLLECT_FERTILIZER']
        assert candidate_jobs == expected, (obs['day'], obs['hour'], stock, reserve)
        if not suppress:
            assert action == reference
        if quote <= 30 and 2 <= stock < reserve:
            counts['crop_reserve_preserved'] += 1
        if reserve >= 3 and any(job[2][0] == 'COLLECT_FERTILIZER' for job in base_jobs):
            boundary_observation = copy.deepcopy(obs)
        counts['observations'] += 1
        counts['filtered'] += candidate_jobs != base_jobs
        counts['changed_actions'] += action != reference

    for seat in (0, 1):
        replay = json.loads((ROOT/f'data/results/brain-v4-diagnostic/seat{seat}.json').read_text())
        for step in replay['steps'][:-1]:
            check(step[seat]['observation'])
    observed = dict(counts)
    assert boundary_observation is not None, 'Need an own observation with profitable crop jobs and available collection'
    fixture = copy.deepcopy(boundary_observation)
    counts = dict(observations=0, filtered=0, crop_reserve_preserved=0, changed_actions=0)
    for inventory in (9900, 10000, 50000):
        for stock in (0, 1, 2, 3, 7, 8, 20):
            obs = copy.deepcopy(fixture)
            obs['market']['inventory']['FERTILIZER'] = inventory
            obs['private']['shed']['FERTILIZER'] = stock
            for inv in obs['private']['inventories']:
                inv.pop('FERTILIZER', None)
            check(obs)
    assert counts['filtered'] > 0 and counts['crop_reserve_preserved'] > 0
    report = dict(own_saved_observations=observed, synthetic_supply_boundaries=counts,
                  invariant='Only eligible COLLECT_FERTILIZER jobs removed; all other jobs and ordering identical.',
                  limitation='Functional checks do not establish realized crop supply or economic gains.')
    out = ROOT/'data/results/fertilizer-collection-confirm/job-checks.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
