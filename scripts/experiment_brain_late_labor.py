"""Measure whether a fixed late-season labor cut loses harvest revenue."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/results/brain-late-labor-screen'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = (ROOT/'bots/brain_fertilizer/main.py').read_text(encoding='utf-8')
    old = 'PHASE_CAPS = ((0, 6), (3, 8), (6, 10), (10, 12), (25, 10), (28, 8))'
    assert source.count(old) == 1
    variants = {
        'control': source,
        'keep12': source.replace(old, 'PHASE_CAPS = ((0, 6), (3, 8), (6, 10), (10, 12))'),
        'cap14': source.replace(old, 'PHASE_CAPS = ((0, 6), (3, 8), (6, 10), (10, 14))')
                       .replace('HIRES_PER_DAY = 12', 'HIRES_PER_DAY = 14'),
    }
    env = dict(os.environ)
    env['KAGGRI_ARENA_DIR'] = str(ROOT/'data/arena-late-labor-screen')
    rows = []
    for name, text in variants.items():
        path = ROOT/'bots'/f'brain_late_labor_{name}'/'main.py'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        compile(text, str(path), 'exec')
        tag = f'late-labor-{name}-screen-20'
        with (OUT/f'{name}.log').open('w', encoding='utf-8') as log:
            run = subprocess.run([sys.executable, str(ROOT/'scripts/evaluate.py'), '--agent', str(path),
                                  '--opponents', 'bots/v46/main.py', '--n', '20', '--seed-base', '90',
                                  '--workers', '8', '--tag', tag], cwd=ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
        result_path = ROOT/'data/results'/tag/'results.json'
        if not result_path.exists():
            raise RuntimeError(f'No game measurement for {name}; inspect log')
        summary = json.loads(result_path.read_text())
        row = dict(name=name, path=str(path), run_id=summary['run_id'], returncode=run.returncode,
                   **summary['per_opponent'][0])
        row['margin'] = row['money_us_mean'] - row['money_them_mean']
        rows.append(row)
        (OUT/'results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(f"{name}: wins={row['win_rate']:.1%} margin={row['margin']:.0f} "
              f"crash={row['crashes']} timeout={row['timeouts']}", flush=True)


if __name__ == '__main__':
    main()
