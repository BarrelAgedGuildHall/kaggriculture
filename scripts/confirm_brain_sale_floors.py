"""Confirm non-wheat sale floors on fresh paired seeds; no automatic promotion."""
import json
import os
from pathlib import Path
import subprocess
import sys

from compare_paired import compare, read
from experiment_brain_sale_floors import build

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/results/sale-floors-confirm'
ARENA = ROOT/'data/arena-sale-floors-confirm'


def main():
    policies = [('control', 'bots/brain_value_v5/main.py'),
                ('candidate', 'bots/brain_value_v6/main.py')]
    source = (ROOT/policies[0][1]).read_text(encoding='utf-8-sig')
    assert (ROOT/policies[1][1]).read_text(encoding='utf-8-sig') == build(source)['nonwheat']
    tags = {name: f'sale-floors-{name}-2700-n100' for name, _ in policies}
    assert not any((ROOT/'data/results'/tag/'results.json').exists() for tag in tags.values()), 'Existing results; do not overwrite'
    OUT.mkdir(parents=True, exist_ok=True)
    for name, path in policies:
        with (OUT/f'{name}.log').open('w', encoding='utf-8') as log:
            subprocess.run(
                [sys.executable, '-X', 'utf8', str(ROOT/'scripts/evaluate.py'),
                 '--agent', path, '--opponents', 'bots/v46/main.py', 'bots/tapeselect/main.py',
                 '--n', '100', '--seed-base', '2700', '--workers', '2', '--tag', tags[name]],
                cwd=ROOT, env=dict(os.environ, KAGGRI_ARENA_DIR=str(ARENA)),
                stdout=log, stderr=subprocess.STDOUT, check=True)
        print(name, json.loads((ROOT/'data/results'/tags[name]/'results.json').read_text(encoding='utf-8'))['overall'], flush=True)
    candidate, candidate_stats = read(ARENA, tags['candidate'])
    control, control_stats = read(ARENA, tags['control'])
    report = dict(per_opponent=compare(candidate, control), candidate=candidate_stats,
                  control=control_stats, decision='Review required; no promotion or submission.')
    (OUT/'paired.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/'scripts/compare_paired_outcomes.py'),
                    '--arena', str(ARENA), '--candidate-tag', tags['candidate'],
                    '--control-tag', tags['control'], '--out', str(OUT/'paired-outcomes.json')],
                   cwd=ROOT, check=True)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
