"""Read actual arena outcomes and compare matched seed/seat win scores.

Ties count half. Faulted games remain included; complete matched runs required.
"""
import argparse
import json
from pathlib import Path
import sqlite3

from compare_paired import ROOT, compare, read


def read_outcomes(arena, tag):
    money, summary = read(arena, tag)
    report = json.loads((ROOT/'data/results'/tag/'results.json').read_text())
    outcomes = {}
    database = (Path(arena)/'arena.sqlite').resolve()
    with sqlite3.connect(database.as_uri()+'?mode=ro', uri=True) as connection:
        for seed, p0, p1, result0 in connection.execute(
                'select seed,p0,p1,result0 from games where run_id=?', (report['run_id'],)):
            seat = 0 if p0 == report['agent'] else 1
            key = (p1 if seat == 0 else p0, seed, seat)
            score = float(result0) if seat == 0 else 1-float(result0)
            assert score in (0, .5, 1), 'Unexpected game outcome'
            assert key not in outcomes, 'Duplicate outcome'
            outcomes[key] = (score, money[key][1])
    assert outcomes.keys() == money.keys()
    assert round(sum(row[0] for row in outcomes.values())/len(outcomes), 4) == summary['win_rate']
    return outcomes, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arena', type=Path, required=True)
    parser.add_argument('--candidate-tag', required=True)
    parser.add_argument('--control-tag', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    candidate, candidate_stats = read_outcomes(args.arena, args.candidate_tag)
    control, control_stats = read_outcomes(args.arena, args.control_tag)
    rows = compare(candidate, control)
    for row in rows:
        opponent = row['opponent']
        pairs = [(candidate[key][0], control[key][0]) for key in candidate if key[0] == opponent]
        row['score_delta'] = row.pop('margin_delta')
        row['candidate_wins'] = sum(a == 1 for a, b in pairs)
        row['control_wins'] = sum(b == 1 for a, b in pairs)
        row['candidate_ties'] = sum(a == .5 for a, b in pairs)
        row['control_ties'] = sum(b == .5 for a, b in pairs)
        row['gained_wins'] = sum(a == 1 and b < 1 for a, b in pairs)
        row['lost_wins'] = sum(b == 1 and a < 1 for a, b in pairs)
    result = dict(per_opponent=rows, candidate=candidate_stats, control=control_stats,
                  method='Actual game outcomes; ties half; 10000 whole-seed bootstrap samples; no fault exclusions.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
