"""Test feed distribution independently of herd size and crop economics.

Only workers carrying wheat can feed animals. Concentrated pickup saves shed
visits but may force long feeding routes; distributed pickup enables more local
workers. Compare both mechanisms and an earlier pickup priority against frozen
Brain v3, on paired seats against two black-box reference opponents.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/results/brain-feed-delivery'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = (ROOT/'bots/brain_late_labor_keep12/main.py').read_text(encoding='utf-8')
    old = 'carriers = min(3, max(1, (deficit + 4) // 5))'
    pickup = 'add(2, sx, sy, ["PICKUP", "WHEAT", per], skip="WHEAT")'
    assert source.count(old) == source.count(pickup) == 1
    variants = {
        'control': source,
        'concentrated': source.replace(old, 'carriers = 1'),
        'distributed': source.replace(old, 'carriers = min(8, max(1, (deficit + 1) // 2))'),
        'distributed_early': source.replace(old, 'carriers = min(8, max(1, (deficit + 1) // 2))')
            .replace(pickup, pickup.replace('add(2,', 'add(1,')),
    }
    environment = dict(os.environ)
    environment['KAGGRI_ARENA_DIR'] = str(ROOT/'data/arena-feed-delivery')
    rows = []
    for name, text in variants.items():
        path = ROOT/'bots'/f'brain_feed_delivery_{name}'/'main.py'
        compile(text, str(path), 'exec')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        tag = f'feed-delivery-{name}-screen-16'
        with (OUT/f'{name}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(
                [sys.executable, '-X', 'utf8', str(ROOT/'scripts/evaluate.py'), '--agent', str(path),
                 '--opponents', 'bots/v46/main.py', 'bots/tapeselect/main.py',
                 '--n', '16', '--seed-base', '190', '--workers', '2', '--tag', tag],
                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        report = ROOT/'data/results'/tag/'results.json'
        if not report.exists():
            raise RuntimeError(f'No measured result for {name}; inspect its log')
        data = json.loads(report.read_text())
        rows.append(dict(name=name, path=str(path), run_id=data['run_id'],
                         per_opponent=data['per_opponent'], returncode=result.returncode))
        (OUT/'results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(name, [(r['win_rate'], r['money_us_mean']-r['money_them_mean'])
                     for r in data['per_opponent']], flush=True)


if __name__ == '__main__':
    main()
