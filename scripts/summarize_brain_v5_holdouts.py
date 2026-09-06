"""Aggregate completed v5 held-out cohorts without repeating seed/seat games."""
import json
from pathlib import Path

from compare_paired_outcomes import read_outcomes

ROOT = Path(__file__).resolve().parents[1]
COHORTS = [
    ('data/arena-fertilizer-collection-confirm', 'fertilizer-collection-candidate-2500-n100'),
    ('data/arena-v5-active-reference', 'v5-active-reference-2600-n100'),
    ('data/arena-sale-floors-confirm', 'sale-floors-control-2700-n100'),
]


def main():
    source = (ROOT/'bots/brain_value_v5/main.py').read_bytes()
    assert source == (ROOT/'bots/brain/main.py').read_bytes()
    assert source == (ROOT/'bots/brain_fertilizer_collection/main.py').read_bytes()
    combined = {}
    summaries = []
    for arena, tag in COHORTS:
        rows, summary = read_outcomes(ROOT/arena, tag)
        assert not combined.keys() & rows.keys(), 'Repeated opponent/seed/seat'
        for opponent, seed, seat in rows:
            assert (opponent, seed, 1-seat) in rows, 'Missing paired seat'
        combined.update(rows)
        summaries.append(dict(tag=tag, games=len(rows), crashes=summary['crashes'], timeouts=summary['timeouts']))
    opponents = []
    for opponent in sorted({key[0] for key in combined}):
        rows = [value for key, value in combined.items() if key[0] == opponent]
        wins = sum(score == 1 for score, bank in rows)
        ties = sum(score == .5 for score, bank in rows)
        opponents.append(dict(opponent=opponent, games=len(rows), wins=wins, ties=ties,
                              losses=len(rows)-wins-ties, win_rate=wins/len(rows)))
    report = dict(cohorts=summaries, per_opponent=opponents, games=len(combined),
                  scope='Completed fresh v5 cohorts only; excludes tuning screens, v4, historical ghosts, and candidate arms.')
    out = ROOT/'data/results/brain-v5-heldout-summary.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
