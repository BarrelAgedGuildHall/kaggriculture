"""Compare realized town demand in completed paired component measurements."""
import argparse
import json
from pathlib import Path
import statistics


def read(path):
    report = json.loads(path.read_text(encoding='utf-8'))
    rows = {(row['seed'], row['seat']): row for row in report['games']}
    assert len(rows) == len(report['games']), 'Duplicate seed/seat'
    assert rows and all(row['status'] == 'DONE' for row in rows.values())
    for seed, seat in rows:
        assert (seed, 1-seat) in rows, 'Missing paired seat'
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    candidate, control = read(args.candidate), read(args.control)
    assert candidate.keys() == control.keys()
    changed = [key for key in candidate if candidate[key]['town_demand'] != control[key]['town_demand']]
    products = sorted(next(iter(candidate.values()))['town_demand'])
    metrics = {}
    for product in products:
        for row in [*candidate.values(), *control.values()]:
            assert product in row['town_demand']
        a = statistics.mean(row['town_demand'][product] for row in candidate.values())
        b = statistics.mean(row['town_demand'][product] for row in control.values())
        metrics[product] = dict(control=b, candidate=a, delta=a-b)
    report = dict(games_per_arm=len(candidate), changed_demand_games=len(changed),
                  changed_demand_seeds=len({seed for seed, seat in changed}), per_product=metrics,
                  limitation='Demand totals can coincide despite different timing. Descriptive component comparison, not causal separation or new win evidence.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
