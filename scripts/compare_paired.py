"""Compare complete matched seed/seat runs with whole-seed bootstrap intervals.

Faulted games remain in the measurement and their counts are reported. Missing,
duplicate or unpaired games are errors, never silently dropped.
"""
import argparse
import json
from pathlib import Path
import random
import sqlite3
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(arena, tag):
    report = json.loads((ROOT/'data/results'/tag/'results.json').read_text())
    database = Path(arena)/'arena.sqlite'
    result = {}
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for row in connection.execute('select * from games where run_id=?', (report['run_id'],)):
            seat = 0 if row['p0'] == report['agent'] else 1
            assert row[f'p{seat}'] == report['agent'], 'Unexpected agent in run'
            key = (row[f'p{1-seat}'], row['seed'], seat)
            assert key not in result, 'Duplicate opponent/seed/seat'
            result[key] = (row[f'money{seat}']-row[f'money{1-seat}'], row[f'money{seat}'])
    assert len(result) == report['overall']['n'], 'Incomplete database run'
    assert result, 'Empty run'
    return result, report['overall']


def compare(candidate, control):
    assert candidate.keys() == control.keys(), 'Runs do not have identical opponent/seed/seat sets'
    output = []
    for opponent in sorted({op for op, _, _ in candidate}):
        seeds = sorted({seed for op, seed, _ in candidate if op == opponent})
        for seed in seeds:
            assert all((opponent, seed, seat) in candidate for seat in (0, 1)), 'Unpaired seed'
        margin = [statistics.mean(candidate[opponent, seed, seat][0]-control[opponent, seed, seat][0]
                                  for seat in (0, 1)) for seed in seeds]
        bank = [statistics.mean(candidate[opponent, seed, seat][1]-control[opponent, seed, seat][1]
                                for seat in (0, 1)) for seed in seeds]
        rng = random.Random(42)
        bootstrap = sorted(statistics.mean(rng.choices(margin, k=len(margin))) for _ in range(10000))
        output.append(dict(opponent=opponent, seeds=len(seeds), games=2*len(seeds),
                           margin_delta=statistics.mean(margin), bank_delta=statistics.mean(bank),
                           seed_pair_ci95=[bootstrap[250], bootstrap[9750]]))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-arena', required=True)
    parser.add_argument('--control-arena', required=True)
    parser.add_argument('--candidate-tag', required=True)
    parser.add_argument('--control-tag', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    candidate, candidate_summary = read(args.candidate_arena, args.candidate_tag)
    control, control_summary = read(args.control_arena, args.control_tag)
    result = dict(candidate=args.candidate_tag, control=args.control_tag,
                  candidate_faults={k: candidate_summary[k] for k in ('crashes', 'timeouts')},
                  control_faults={k: control_summary[k] for k in ('crashes', 'timeouts')},
                  per_opponent=compare(candidate, control))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
