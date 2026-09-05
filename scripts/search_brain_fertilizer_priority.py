"""Compare fertilizer dispatch priority against V46 on fresh paired seeds.

This tests whether investment work displaces survival work. It changes one task
priority only; all crop forecasts, reserves, and other actions stay fixed.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/results/brain-fertilizer-priority'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = (ROOT / 'bots/brain_fertilizer/main.py').read_text(encoding='utf-8')
    needle = "add(-1, x, y, ['FERTILIZE'], need='FERTILIZER')"
    assert source.count(needle) == 1
    environment = dict(os.environ)
    environment['KAGGRI_ARENA_DIR'] = str(ROOT / 'data/arena-fertilizer-priority')
    rows = []
    for priority in (-1, 0, 1):
        name = 'control' if priority == -1 else f'priority{priority}'
        path = ROOT / 'bots' / f'brain_fertilizer_{name}' / 'main.py'
        path.parent.mkdir(parents=True, exist_ok=True)
        candidate = source.replace(needle, needle.replace('add(-1,', f'add({priority},'))
        path.write_text(candidate, encoding='utf-8')
        compile(candidate, str(path), 'exec')
        tag = f'fertilizer-{name}-1300-n100'
        command = [sys.executable, str(ROOT / 'scripts/evaluate.py'), '--agent', str(path),
                   '--opponents', 'bots/v46/main.py', '--n', '100', '--seed-base', '1300',
                   '--workers', '10', '--tag', tag]
        with (OUT / f'{name}.log').open('w', encoding='utf-8') as log:
            run = subprocess.run(command, cwd=ROOT, env=environment,
                                 stdout=log, stderr=subprocess.STDOUT)
        result_path = ROOT / 'data/results' / tag / 'results.json'
        if not result_path.exists():
            raise RuntimeError(f'No measurement for {name}; inspect its log')
        summary = json.loads(result_path.read_text())
        result = summary['per_opponent'][0]
        row = dict(name=name, priority=priority, path=str(path), tag=tag,
                   run_id=summary['run_id'], returncode=run.returncode, **result)
        row['margin'] = row['money_us_mean'] - row['money_them_mean']
        rows.append(row)
        (OUT / 'results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(f"{name}: wins={row['win_rate']:.1%} margin={row['margin']:.0f} "
              f"crashes={row['crashes']} timeouts={row['timeouts']}", flush=True)
    print('Finished; calculate paired seed confidence intervals before promotion.', flush=True)


if __name__ == '__main__':
    main()
