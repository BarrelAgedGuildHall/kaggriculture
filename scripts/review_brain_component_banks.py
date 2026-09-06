"""Reconcile paired component banks with arena outcomes and summarize economics."""
import argparse
import json
from pathlib import Path
import statistics

from compare_paired import compare, read as read_arena
from review_brain_town_demand import read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--arena', type=Path, required=True)
    parser.add_argument('--candidate-tag', required=True)
    parser.add_argument('--control-tag', required=True)
    parser.add_argument('--opponent', default='main@h1b8c525dd4')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    arms = {'candidate': read(args.candidate), 'control': read(args.control)}
    assert arms['candidate'].keys() == arms['control'].keys()
    for name, tag in [('candidate', args.candidate_tag), ('control', args.control_tag)]:
        arena, _ = read_arena(args.arena, tag)
        for (seed, seat), row in arms[name].items():
            assert row['final_bank'] == arena[args.opponent, seed, seat][1], (name, seed, seat)
    metrics = {}
    getters = {
        'final_bank': lambda r: r['final_bank'],
        'wages': lambda r: r['wages_paid'],
        'sales_revenue': lambda r: sum(r['realised_dollars'].values()),
        'units_sold': lambda r: sum(r['sold_units'].values()),
        'terminal_unsold_value': lambda r: r['unsold_value_at_719'],
    }
    for product in sorted(next(iter(arms['control'].values()))['realised_dollars']):
        getters[f'{product}_revenue'] = lambda r, p=product: r['realised_dollars'][p]
    for label, getter in getters.items():
        mapped = {name: {(args.opponent, seed, seat): (getter(row), row['final_bank'])
                         for (seed, seat), row in rows.items()} for name, rows in arms.items()}
        delta = compare(mapped['candidate'], mapped['control'])[0]
        metrics[label] = dict(control=statistics.mean(getter(row) for row in arms['control'].values()),
                             candidate=statistics.mean(getter(row) for row in arms['candidate'].values()),
                             delta=delta['margin_delta'], seed_pair_ci95=delta['seed_pair_ci95'])
    report = dict(games_per_arm=len(arms['control']), all_banks_match=True, metrics=metrics,
                  limitation='Confirmation subset, not independent win evidence. Product intervals are descriptive; realized town demand can differ across policies.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
