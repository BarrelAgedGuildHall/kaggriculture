# Runbook — ladder records and the Pit-vs-ladder calibration read

Owner: **Ops**. Standing work: fill the ladder columns of `workshop/notes/submissions.md`
after every submission, and run the weekly calibration read. Manual — scheduling is
disabled by founder rule, so these are run on request, never on a cron.

Tool: `scripts/ladder_record.py`. Prior read that established the method and the current
numbers: `workshop/deliverables/2026-08-22-pit-ladder-correlation/read.md`.

## The one rule you must not lose

**Kaggle's upload validation is a SELF-PLAY episode.** Both seats carry our own submission
id and our own team id, and it is the episode that stamps the initial 600 (`rating_before`
0 → `rating_after` 600 on **both** seats). Counted, it:

- adds two spurious rows to the record (+1 W and +1 L by the bank comparison, +2 W by the
  ladder's rating movement),
- reports `rating_start` as 0 instead of 600,
- and produces one bogus "rating movement disagrees with the bank comparison" row per agent.

It is excluded by `split_validation()`, which re-checks the exclusion against a **second,
independent signature** — the validation episode is the only episode a submission ever plays
with `rating_before == 0` — so a leak is caught even if the team-id filter is later weakened.
`tests/test_ladder_record.py` pins all of this. Do not "simplify" it away.

## Fill the ledger after a submission

Wait at least a couple of hours: a new submission plays ~20–25 games in its first two hours
and ~35/day thereafter while it is in the active pair. Anything under 10 games is refused as
too thin to read.

```
uv run python scripts/ladder_record.py record --fetch \
    --as-of "$(date -u +%Y-%m-%dT%H:%MZ)" --ledger-cells
```

This prints the exact markdown for the ledger's "ladder record … as of" column; paste it in.
Add the new submission to `KNOWN_SUBMISSIONS` at the top of the script in the same change
(it is maintained by hand alongside the ledger).

Costs: one `ListEpisodes` call per submission, metadata only. **Never** download replays or
the `kaggle/kaggriculture-episodes-*` daily datasets (~20 GB/day) for this.

## Weekly calibration read

```
uv run python scripts/ladder_record.py all --fetch \
    --as-of "$(date -u +%Y-%m-%dT%H:%MZ)" --json data/ladder/records.json
```

Three sections come out:

- **record** — W-L-T, rating trajectory, opponent-adjusted Elo per submission.
- **correlate** — Pit BT / win rate / hub game counts against the ladder, with Spearman and
  Kendall rank agreement. Read the per-pair rating gaps, not the concordance count: the live
  ladder is path-dependent (byte-identical agents diverged 1700 vs 3000 in the community's
  own experiment) and a gap under ~120 Elo is a coin flip at our sample sizes.
- **calibrate** — the Pit-BT → ladder-Elo scale factor and a per-pair consistency test.

For a snapshot that must be reproducible later, save the hub captures and pass them back in
instead of hitting the live hub:

```
curl -s $HUB/api/leaderboard                    > data/ladder/pit.json
curl -s "$HUB/api/leaderboard?include_retired=1" > data/ladder/pit-retired.json
uv run python scripts/ladder_record.py all --from data/ladder \
    --pit-json data/ladder/pit.json --pit-retired-json data/ladder/pit-retired.json
```

`?include_retired=1` is the only way to get a BT number for a retired agent, but it refits
BT over a different pool — keep it in its own column, never mixed with canonical numbers.

## What to escalate

- **A `NO` in the calibrate table** — that is a *pair-specific* divergence: the Pit is wrong
  about one particular agent rather than merely mis-scaled. Escalate to the CEO immediately.
- **The scale factor alone is not an escalation.** As of 2026-08-22 it is 0.137 ladder-Elo
  per Pit-BT point (a ~7× inflation); that is a known, accepted finding under adjudication.
  Report the number, do not re-raise it as new.
- **A rating/bank disagreement warning** — either a validation row leaked in or Kaggle
  changed how the ladder scores. Both are stop-and-check. `--strict` makes it exit 1 so a job
  turns red.
- **A new champion failing to beat its predecessor's opponent-adjusted Elo by
  `slope × ΔBT` after 30+ games** — our promotions are not translating into ladder strength.

## Resolving power (do not plan around the ladder)

Games needed **each** to resolve a gap between two of our agents, at the observed ~35/day:

| gap | games each | days |
|---|---|---|
| 330 Elo | ~9 | ~0.3 |
| 200 Elo | ~23 | ~0.7 |
| 100 Elo | ~93 | ~2.7 |
| 50 Elo | ~371 | ~10.6 |

The live ladder catches big Pit errors fast and can never settle a fine ranking on this
competition's timescale. That is the division of labour the Pit exists for.
