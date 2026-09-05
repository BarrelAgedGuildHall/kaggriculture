"""Test speculative pasture capacity and land purchase ordering independently."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/results/brain-capital-screen'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = (ROOT/'bots/brain_fertilizer/main.py').read_text(encoding='utf-8')
    old = 'need_coops = target_coops - len(coop_pos)'
    replacement = '''need_coops = min(target_coops - len(coop_pos),
                     max(0, sum(int(shed.get(a, 0)) + carried.get(a, 0)
                                for a in ANIMAL_TARGETS) + 2 - len(empty_coops)))'''
    assert source.count(old) == 1
    land_old = 'buys.append(["BUY_LAND"])'
    assert source.count(land_old) == 1
    # Unit actions precede market orders: this preserves the rest of the planner.
    # Earlier purchase ordering stops discretionary seeds/animals consuming funds
    # before an already-approved expansion order can execute.
    variants = {
        'control': source,
        'pastures': source.replace(old, replacement),
        'land': source.replace(land_old, 'buys.insert(0, ["BUY_LAND"])'),
        'both': source.replace(old, replacement).replace(land_old, 'buys.insert(0, ["BUY_LAND"])'),
    }
    env = dict(os.environ)
    env['KAGGRI_ARENA_DIR'] = str(ROOT/'data/arena-capital-screen')
    rows = []
    for name, text in variants.items():
        path = ROOT/'bots'/f'brain_capital_{name}'/'main.py'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        compile(text, str(path), 'exec')
        tag = f'capital-{name}-screen-20'
        with (OUT/f'{name}.log').open('w', encoding='utf-8') as log:
            run = subprocess.run([sys.executable, str(ROOT/'scripts/evaluate.py'), '--agent', str(path),
                                  '--opponents', 'bots/v46/main.py', '--n', '20', '--seed-base', '70',
                                  '--workers', '10', '--tag', tag], cwd=ROOT, env=env,
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
