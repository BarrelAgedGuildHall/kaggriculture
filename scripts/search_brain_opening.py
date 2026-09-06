"""Bounded original-policy crop allocation search; no replay-derived parameters.

Tune on seeds 20-29 in both seats. Winners require a separate unseen-seed run.
Every variant remains a standalone observation-driven bot. Arena registration is
sequential because the shared JSON registry does not support concurrent writers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/brain-opening-search"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = (ROOT / "bots/brain_fertilizer/main.py").read_text(encoding="utf-8")
    original_opening = '{"WHEAT": 0.80, "MELON": 0.20}'
    original_share = 'CROP_SHARE = {"WHEAT": 0.55, "STRAWBERRY": 0.25, "MELON": 0.20}'
    assert source.count(original_opening) == source.count(original_share) == 1
    env = dict(os.environ)
    env['KAGGRI_ARENA_DIR'] = str(ROOT / 'data/arena-opening-search')
    cases = [('control', None, None)]
    cases.extend((f'w{int(w*100)}-s{int(s*100)}', w, s)
                 for w in (0.50, 0.65, 0.80) for s in (0.25, 0.50, 0.70))
    rows = []
    for name, wheat, strawberry in cases:
        text = source
        if wheat is not None:
            text = text.replace(original_opening, repr({'WHEAT': wheat, 'MELON': 1-wheat}))
            text = text.replace(original_share,
                                'CROP_SHARE = ' + repr({'WHEAT': 1-strawberry, 'STRAWBERRY': strawberry}))
        path = ROOT / 'bots' / f'brain_opening_{name}' / 'main.py'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        compile(text, str(path), 'exec')
        tag = f'opening-screen-{name}'
        command = [sys.executable, str(ROOT / 'scripts/evaluate.py'), '--agent', str(path),
                   '--opponents', 'bots/v46/main.py', '--n', '20', '--seed-base', '20',
                   '--workers', '10', '--tag', tag]
        with (OUT / f'{name}.log').open('w', encoding='utf-8') as log:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        result_path = ROOT / 'data/results' / tag / 'results.json'
        if not result_path.exists():
            raise RuntimeError(f'{name}: no measurement, see {OUT / (name + ".log")}')
        summary = json.loads(result_path.read_text())
        result = summary['per_opponent'][0]
        row = dict(name=name, path=str(path), bootstrap_wheat=wheat,
                   later_strawberry=strawberry, returncode=process.returncode, **result)
        row['margin'] = result['money_us_mean'] - result['money_them_mean']
        rows.append(row)
        (OUT / 'results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(f"{name}: wins={result['win_rate']:.1%} margin={row['margin']:.0f} "
              f"crashes={result['crashes']} timeouts={result['timeouts']}", flush=True)
    valid = [r for r in rows if not r['crashes'] and not r['timeouts']]
    best = max(valid, key=lambda r: (r['win_rate'], r['margin']))
    (OUT / 'best.json').write_text(json.dumps(best, indent=2), encoding='utf-8')
    print('SCREEN WINNER (unconfirmed):', best['name'], flush=True)


if __name__ == '__main__':
    main()
