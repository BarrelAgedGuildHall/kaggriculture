import json
from pathlib import Path
import sys
from types import SimpleNamespace

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'scripts'))
from _engine import import_engine
import_engine()
from kaggle_environments.envs.kaggriculture import kaggriculture as kg

def fixture(seed, occupied):
    farms = [kg._new_farm(10, 3000), kg._new_farm(10, 3000)]
    if occupied:
        farms[0]['tiles'][0][0] = {'kind': 'PASTURE'}
    town = kg._new_town()
    state = [SimpleNamespace(observation=SimpleNamespace(farms=farms, town=town,
             private=kg._new_private())) for _ in range(2)]
    kg._end_of_day(state, SimpleNamespace(configuration={}, info={'seed': seed}), 2)
    return town['unlocked_shops'][0]

rows = []
for seed in range(430, 438):
    control = fixture(seed, False)
    occupied = fixture(seed, True)
    assert fixture(seed, False) == control
    assert fixture(seed, True) == occupied
    rows.append(dict(seed=seed, empty=control, pasture=occupied, changed=control != occupied))
assert any(row['changed'] for row in rows)
report = dict(cases=rows, different_draws=sum(row['changed'] for row in rows),
              scope='Actual day2 engine refresh; only one empty tile replaced by an empty pasture. Same seed is reproducible for same state, not fixed town across policies. No game or policy improvement claimed.')
(root/'data/results/brain-duplicate-shops/rng-coupling.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report))
