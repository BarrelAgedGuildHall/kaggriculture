"""Confirm the original short daytime delivery policy, then measure its mechanism.

Sequential two-worker runs; fresh paired seeds; no automatic promotion/submission.
Component games run only if both reference margins improve and at least one
paired interval excludes zero, with no crash/timeout flags.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

from compare_paired import compare, read

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/results/day-delivery-resumed-confirm'
ARENA = ROOT/'data/arena-day-delivery-resumed-confirm'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, KAGGRI_ARENA_DIR=str(ARENA))
    policies = [('control', 'bots/brain_value_v4/main.py'),
                ('candidate', 'bots/brain_day_delivery/main.py')]
    assert not any((ROOT/'data/results'/f'day-delivery-resumed-{name}-2200-n100'/'results.json').exists()
                   for name, _ in policies), 'Existing confirmation results; do not overwrite'
    for name, path in policies:
        tag = f'day-delivery-resumed-{name}-2200-n100'
        with (OUT/f'{name}.log').open('w', encoding='utf-8') as log:
            subprocess.run(
                [sys.executable, '-X', 'utf8', str(ROOT/'scripts/evaluate.py'), '--agent', path,
                 '--opponents', 'bots/v46/main.py', 'bots/tapeselect/main.py',
                 '--n', '100', '--seed-base', '2200', '--workers', '2', '--tag', tag],
                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = ROOT/'data/results'/tag/'results.json'
        if not result.exists():
            raise RuntimeError(f'No complete result for {tag}')
        data = json.loads(result.read_text())
        print(name, data['overall'], flush=True)
    candidate, candidate_stats = read(ARENA, 'day-delivery-resumed-candidate-2200-n100')
    control, control_stats = read(ARENA, 'day-delivery-resumed-control-2200-n100')
    pairs = compare(candidate, control)
    clean = all(s['crashes'] == s['timeouts'] == 0 for s in [candidate_stats, control_stats])
    measure_components = clean and all(p['margin_delta'] > 0 for p in pairs) and any(
        p['seed_pair_ci95'][0] > 0 for p in pairs)
    report = dict(per_opponent=pairs, candidate=candidate_stats, control=control_stats,
                  measure_components=measure_components, decision='Review required; no promotion or submission.')
    (OUT/'paired.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Paired results:', pairs, flush=True)
    if measure_components:
        for name, path in policies:
            with (OUT/f'{name}-kpi.log').open('w', encoding='utf-8') as log:
                result = subprocess.run(
                    [sys.executable, '-X', 'utf8', str(ROOT/'scripts/kpi.py'), '--agent', path,
                     '--opponent', 'bots/v46/main.py', '--n', '20', '--seed-base', '2200',
                     '--workers', '2', '--tag', f'day-delivery-resumed-{name}-kpi-40'],
                    cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Component run failed: {name}; inspect log')
            print('Component run completed:', name, flush=True)
    print('CONFIRMATION COMPLETE; inspect paired.json and any component reports.', flush=True)


if __name__ == '__main__':
    main()


