"""Check carrot allocation variants without changing other day-plan fields."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from benchmark_brain_dispatch_cost import load

ROOT = Path(__file__).resolve().parents[1]


def main():
    states = [step[0]['observation'] for step in json.loads(
        (ROOT/'data/results/brain-v4-diagnostic/seat0.json').read_text(encoding='utf-8'))['steps'][:-1]]
    base = load(ROOT/'bots/brain_value_v6/main.py', 'carrot_base')
    view = base.View.from_obs(states[0])
    reports = []
    for name, scale in [('half', .5), ('none', 0)]:
        bot = load(ROOT/f'bots/brain_carrot_boost_{name}/main.py', f'carrot_{name}')
        checked = 0
        for day in (0, 4, 5, 25, 29):
            for shops in ((), ('PET_CAFE',), ('FARMERS_MARKET',), ('PET_CAFE', 'FARMERS_MARKET'), ('PET_CAFE',)*3):
                test = replace(view, day=day, town_shops=shops)
                plan, original = bot.DefaultSeason.plan_day(test), base.DefaultSeason.plan_day(test)
                expected = dict(original.crop_weights)
                if scale == 0:
                    expected.pop('CARROT', None)
                elif 'CARROT' in expected:
                    expected['CARROT'] *= scale
                assert dict(plan.crop_weights) == expected
                assert asdict(replace(plan, crop_weights=original.crop_weights)) == asdict(original)
                checked += 1
        changed = preserved = 0
        for obs in states:
            a, b = bot._policy(obs, time.perf_counter()), base._policy(obs, time.perf_counter())
            changed += a != b
            if not set(obs['town']['unlocked_shops']) & {'PET_CAFE', 'FARMERS_MARKET'}:
                assert a == b
                preserved += 1
            if name == 'none':
                assert all(order[:2] != ['BUY_SEED', 'CARROT'] for order in a['market'])
                assert all(op[:2] != ['PLANT', 'CARROT'] for op in [a['farmer'], *a['hands']])
        reports.append(dict(variant=name, plan_boundaries=checked, policy_calls=len(states),
                            changed_actions=changed, no_carrot_shop_preserved=preserved))
    (ROOT/'data/results/brain-carrot-boost/policy-checks.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps(reports))


if __name__ == '__main__':
    main()
