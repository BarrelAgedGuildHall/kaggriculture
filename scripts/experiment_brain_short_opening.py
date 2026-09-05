"""Measure faster opening cash cycles with engine-derived wheat/carrot mixes.

The existing opening locks 20% of crop allocation in long-maturing melons while
cash limits hiring and expansion. Carrots reach their full yield sooner than
wheat and cost less than melons. Test this hypothesis without changing later
crop allocation, animal rules, dispatch or selling, against two references.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/results/brain-short-opening'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = (ROOT/'bots/brain_late_labor_keep12/main.py').read_text(encoding='utf-8')
    old = '{"WHEAT": 0.80, "MELON": 0.20}'
    assert source.count(old) == 1
    choices = {
        'control': None,
        'wheat': {'WHEAT': 1.0},
        'mixed': {'WHEAT': 0.5, 'CARROT': 0.5},
        'carrot': {'CARROT': 1.0},
    }
    environment = dict(os.environ)
    environment['KAGGRI_ARENA_DIR'] = str(ROOT/'data/arena-short-opening')
    rows = []
    for name, weights in choices.items():
        text = source if weights is None else source.replace(old, repr(weights))
        path = ROOT/'bots'/f'brain_short_opening_{name}'/'main.py'
        compile(text, str(path), 'exec')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        tag = f'short-opening-{name}-screen-20'
        with (OUT/f'{name}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(
                [sys.executable, str(ROOT/'scripts/evaluate.py'), '--agent', str(path),
                 '--opponents', 'bots/v46/main.py', 'bots/tapeselect/main.py',
                 '--n', '20', '--seed-base', '180', '--workers', '6', '--tag', tag],
                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        report = ROOT/'data/results'/tag/'results.json'
        if not report.exists():
            raise RuntimeError(f'No measured result for {name}; inspect its log')
        data = json.loads(report.read_text())
        row = dict(name=name, path=str(path), run_id=data['run_id'],
                   per_opponent=data['per_opponent'], returncode=result.returncode)
        rows.append(row)
        (OUT/'results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(name, [(r['win_rate'], r['money_us_mean']-r['money_them_mean'])
                     for r in data['per_opponent']], flush=True)


if __name__ == '__main__':
    main()
