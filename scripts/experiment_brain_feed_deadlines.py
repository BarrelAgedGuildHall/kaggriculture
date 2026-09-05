"""Measure daily husbandry deadlines, using only current observations.

The completed feed-delivery control on seeds190-197 is the unchanged v3 control.
Reuse its paired games; keep independent arena registration for these variants.
Daily feed/care supports production bonuses, even when survival is not urgent.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/results/brain-feed-deadlines'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = (ROOT/'bots/brain_late_labor_keep12/main.py').read_text(encoding='utf-8')
    feed = 'add(-2 if unfed >= 1 else 3, x, y, ["FEED"], need="WHEAT")'
    care = 'add(3, x, y, ["CARE"])'
    assert source.count(feed) == source.count(care) == 1
    control = ROOT/'data/results/feed-delivery-control-screen-16/results.json'
    assert control.exists(), 'Complete the matched v3 control first'
    assert (ROOT/'bots/brain_feed_delivery_control/main.py').read_text(encoding='utf-8') == source
    environment = dict(os.environ)
    environment['KAGGRI_ARENA_DIR'] = str(ROOT/'data/arena-feed-deadlines')
    rows = []
    for name, feed_hour, care_hour in [('feed12', 12, None), ('both12', 12, 16), ('both6', 6, 12)]:
        text = source.replace(feed, feed.replace('else 3', f'else (-1 if hour >= {feed_hour} else 3)'))
        if care_hour is not None:
            text = text.replace(care, care.replace('add(3,', f'add(-1 if hour >= {care_hour} else 3,'))
        path = ROOT/'bots'/f'brain_feed_deadline_{name}'/'main.py'
        compile(text, str(path), 'exec')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        tag = f'feed-deadline-{name}-screen-16'
        with (OUT/f'{name}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(
                [sys.executable, '-X', 'utf8', str(ROOT/'scripts/evaluate.py'), '--agent', str(path),
                 '--opponents', 'bots/v46/main.py', 'bots/tapeselect/main.py',
                 '--n', '16', '--seed-base', '190', '--workers', '2', '--tag', tag],
                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        report = ROOT/'data/results'/tag/'results.json'
        if not report.exists():
            raise RuntimeError(f'No measured result for {name}; inspect log')
        data = json.loads(report.read_text())
        rows.append(dict(name=name, path=str(path), run_id=data['run_id'],
                         per_opponent=data['per_opponent'], returncode=result.returncode))
        (OUT/'results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(name, [(r['win_rate'], r['money_us_mean']-r['money_them_mean'])
                     for r in data['per_opponent']], flush=True)


if __name__ == '__main__':
    main()
