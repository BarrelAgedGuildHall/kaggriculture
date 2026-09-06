"""Save current original v6 games and own economic metrics; no opponent fitting."""
import json
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'scripts'))
from _engine import import_engine, to_plain
from kpi import _analyze

ke=import_engine()
out=root/'data/results/brain-v6-diagnostic'
out.mkdir(parents=True,exist_ok=True)
assert not any((out/f'seat{seat}.json').exists() for seat in (0,1)), 'Preserve existing diagnostic'
reports=[]
for seat in (0,1):
    env=ke.make('kaggriculture',configuration={'episodeSteps':720,'seed':3000},debug=False)
    agents=[str(root/'bots/brain_value_v6/main.py'),str(root/'bots/v46/main.py')]
    if seat:agents.reverse()
    env.run(agents)
    replay=env.toJSON()
    (out/f'seat{seat}.json').write_text(json.dumps(replay),encoding='utf-8')
    row=_analyze(env,seat,3000)
    (out/f'seat{seat}-kpi.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    obs=to_plain(env.steps[-1][seat].observation)
    report=dict(seat=seat,status=row['status'],bank=row['final_bank'],opponent_bank=obs['farms'][1-seat]['money'],
                productive_fraction=row['productive_action_fraction'],wages=row['wages_paid'],
                sold_units=row['sold_units'],realised_dollars=row['realised_dollars'])
    reports.append(report)
    print(json.dumps(report),flush=True)
(out/'summary.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')

