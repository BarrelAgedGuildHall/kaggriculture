"""Build original value-aware husbandry policies; --run measures paired games.

Survival feeding is preserved. Care must have a payable production date, room
for its bonus, and enough current product value to cover feed and labor. No
opponent code, actions or identity enter these rules.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/results/brain-husbandry-value'

HELPER = '''def _animal_work_value(a, tile, day, market_inventory):
    product_quote = market_price(a["product"], market_inventory.get(a["product"], MARKET_I0) + 2)
    wheat_quote = market_price("WHEAT", market_inventory.get("WHEAT", MARKET_I0) - 1)
    first = int(tile.get("placed_day", day)) + a["first_yield_day"]
    next_bonus_day = first
    # Today's care is banked AFTER tonight's production, so cannot pay tonight.
    while next_bonus_day <= day + 1:
        next_bonus_day += a["interval"]
    pending = int(tile.get("pending_care_bonus", 0))
    due = day + 1 >= first and (day + 1 - first) % a["interval"] == 0
    pending_after_production = 0 if due else pending
    care_worth = (next_bonus_day <= FINAL_DAY and pending_after_production < a["max_held"] - 1
                  and product_quote > 0.5 * wheat_quote + 35)
    usable_bonus = min(pending, max(0, a["max_held"] - int(tile.get("yield_units", 0)) - 1))
    feed_worth = care_worth or (due and usable_bonus * product_quote > wheat_quote)
    return care_worth, feed_worth


'''


def build_variants():
    source = (ROOT/'bots/brain_late_labor_keep12/main.py').read_text(encoding='utf-8')
    marker = 'def _policy(obs, t0):'
    animal_marker = '        # Day 29 never runs an end-of-day refresh, so feeding and caring are worthless then.'
    feed = 'if not last_day and not fed:'
    care = 'if not last_day and not cared:'
    assert all(source.count(x) == 1 for x in (marker, animal_marker, feed, care))
    base = source.replace(marker, HELPER+marker).replace(
        animal_marker, '        care_worth, feed_worth = _animal_work_value(a, t, day, mkt_inv)\n'+animal_marker)
    base = base.replace(care, 'if not last_day and not cared and care_worth:')
    maintenance = base.replace(feed, 'if not last_day and not fed and (unfed >= 1 or feed_worth):')
    urgent = maintenance.replace('add(-2 if unfed >= 1 else 3, x, y, ["FEED"], need="WHEAT")',
                                'add(-2 if unfed >= 1 else (-1 if hour >= 12 else 3), x, y, ["FEED"], need="WHEAT")')
    urgent = urgent.replace('add(3, x, y, ["CARE"])',
                            'add(-1 if hour >= 16 else 3, x, y, ["CARE"])')
    return {'care_value': base, 'maintenance': maintenance, 'value_deadline': urgent}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Run only when other experiments have finished')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, KAGGRI_ARENA_DIR=str(ROOT/'data/arena-husbandry-value'))
    rows = []
    for name, text in build_variants().items():
        path = ROOT/'bots'/f'brain_husbandry_value_{name}'/'main.py'
        compile(text, str(path), 'exec')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        print('Built', name, flush=True)
        if not args.run:
            continue
        tag = f'husbandry-value-{name}-screen-16'
        with (OUT/f'{name}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(
                [sys.executable, '-X', 'utf8', str(ROOT/'scripts/evaluate.py'), '--agent', str(path),
                 '--opponents', 'bots/v46/main.py', 'bots/tapeselect/main.py', '--n', '16',
                 '--seed-base', '190', '--workers', '2', '--tag', tag],
                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        data = json.loads((ROOT/'data/results'/tag/'results.json').read_text())
        rows.append(dict(name=name, path=str(path), run_id=data['run_id'],
                         per_opponent=data['per_opponent'], returncode=result.returncode))
        (OUT/'results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(name, data['overall'], flush=True)


if __name__ == '__main__':
    main()
