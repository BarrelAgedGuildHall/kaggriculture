"""Compare complete, identical historical opponent sets, clustered by seed."""
import argparse
import json
from pathlib import Path
import random
import statistics


def compare(candidate, control):
    assert candidate['engine'] == control['engine'], 'Different engines'
    def index(report):
        rows = {(r['episode_id'], r['ghost_seat'], r['seed']): r for r in report['rows']}
        assert len(rows) == len(report['rows']) == report['games'], 'Incomplete or duplicate rows'
        return rows
    a, b = index(candidate), index(control)
    assert a and a.keys() == b.keys(), 'Different opponent/seed/seat sets'
    deltas = {key: a[key]['margin']-b[key]['margin'] for key in a}
    seeds = sorted({key[2] for key in a})
    means = [statistics.mean(value for key, value in deltas.items() if key[2] == seed) for seed in seeds]
    rng = random.Random(42)
    bootstrap = sorted(statistics.mean(rng.choices(means, k=len(means))) for _ in range(10000))
    return dict(games=len(a), distinct_seeds=len(seeds),
                mean_game_margin_delta=statistics.mean(deltas.values()),
                mean_seed_margin_delta=statistics.mean(means),
                seed_bootstrap_ci95=[bootstrap[250], bootstrap[9750]],
                candidate={k: candidate[k] for k in ('wins', 'crashes', 'timeouts', 'over_budget_turns')},
                control={k: control[k] for k in ('wins', 'crashes', 'timeouts', 'over_budget_turns')})


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
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
