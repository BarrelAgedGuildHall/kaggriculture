"""Compare identical fixed-reference game sets, clustering by shared engine seed."""
import argparse
import json
from pathlib import Path
import random
import statistics


def compare_reports(candidate, control):
    assert candidate['engine'] == control['engine'], 'Engine mismatch'
    def index(report):
        rows = {(r['episode_id'], r['ghost_seat']): r for r in report['rows']}
        assert len(rows) == len(report['rows']) == report['games'], 'Duplicate or missing games'
        return rows
    a, b = index(candidate), index(control)
    assert a and a.keys() == b.keys(), 'Empty or different fixed-reference sets'
    grouped = {}
    margin_deltas = []
    bank_deltas = []
    gained = lost = 0
    scores = {'win': 1., 'tie': .5, 'loss': 0.}
    for key in sorted(a):
        left, right = a[key], b[key]
        for field in ('seed', 'agent_seat', 'team_id', 'recorded_ghost_bank'):
            assert left[field] == right[field], f'Mismatched {field}'
        margin = left['margin']-right['margin']
        bank = left['agent_bank']-right['agent_bank']
        score = scores[left['result']]-scores[right['result']]
        grouped.setdefault(left['seed'], []).append((margin, bank, score))
        margin_deltas.append(margin)
        bank_deltas.append(bank)
        gained += left['result'] == 'win' and right['result'] != 'win'
        lost += right['result'] == 'win' and left['result'] != 'win'
    metrics = {}
    for col, name in enumerate(('margin', 'bank', 'win_score')):
        values = [statistics.mean(row[col] for row in grouped[seed]) for seed in sorted(grouped)]
        rng = random.Random(42)
        samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000))
        metrics[name] = dict(seed_weighted_delta=statistics.mean(values), seed_ci95=[samples[250], samples[9750]])
    return dict(games=len(a), seeds=len(grouped), game_weighted_margin_delta=statistics.mean(margin_deltas),
                game_weighted_bank_delta=statistics.mean(bank_deltas), gained_wins=gained, lost_wins=lost,
                candidate_wins=candidate['wins'], control_wins=control['wins'], metrics=metrics,
                faults={name: {k: report[k] for k in ('crashes', 'timeouts', 'over_budget_turns')}
                        for name, report in (('candidate', candidate), ('control', control))},
                limitation='Fixed historical opponents; not independent live ladder evidence. No fault exclusions.')


def compare(candidate, control):
    """Keep the existing margin fields alongside the additional outcome metrics."""
    result = compare_reports(candidate, control)
    result.update(
        distinct_seeds=result['seeds'],
        mean_game_margin_delta=result['game_weighted_margin_delta'],
        mean_seed_margin_delta=result['metrics']['margin']['seed_weighted_delta'],
        seed_bootstrap_ci95=result['metrics']['margin']['seed_ci95'],
        candidate={k: candidate[k] for k in ('wins', 'crashes', 'timeouts', 'over_budget_turns')},
        control={k: control[k] for k in ('wins', 'crashes', 'timeouts', 'over_budget_turns')})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.candidate.read_text(encoding='utf-8')),
                             json.loads(args.control.read_text(encoding='utf-8')))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
