"""Check herd targets and preserved early opening on own saved observations."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from benchmark_brain_dispatch_cost import load

ROOT = Path(__file__).resolve().parents[1]


def main():
    states = [step[0]['observation'] for step in json.loads(
        (ROOT/'data/results/brain-v4-diagnostic/seat0.json').read_text(encoding='utf-8'))['steps'][:-1]]
    base = load(ROOT/'bots/brain_value_v6/main.py', 'compact_base')
    view = base.View.from_obs(states[0])
    reports = []
    for cows, sheep in ((5, 8), (6, 9)):
        size = cows+sheep
        bot = load(ROOT/f'bots/brain_compact_herd_herd{size}/main.py', f'herd{size}')
        for day in range(30):
            test = replace(view, day=day)
            plan, original = bot.DefaultSeason.plan_day(test), base.DefaultSeason.plan_day(test)
            targets = dict(plan.animal_targets)
            assert sum(targets.values()) == min(size, 2+2*day)
            assert 0 <= targets['COW'] <= cows and 0 <= targets['SHEEP'] <= sheep
            assert asdict(replace(plan, animal_targets=original.animal_targets)) == asdict(original)
        changed = early = 0
        for obs in states:
            a, b = bot._policy(obs, time.perf_counter()), base._policy(obs, time.perf_counter())
            assert isinstance(a, dict) and 'market' in a
            changed += a != b
            if obs['day'] < 4:
                assert a == b
                early += 1
        reports.append(dict(herd=size, cows=cows, sheep=sheep, day_boundaries=30,
                            policy_calls=len(states), changed_actions=changed, preserved_opening_states=early))
    (ROOT/'data/results/brain-compact-herd/policy-checks.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps(reports))


if __name__ == '__main__':
    main()
