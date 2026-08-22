# Team A — Ensemble Behaviour Cloning

Adaptation of the Ensemble-BC method used in the Pokémon TCG AI Challenge to Kaggriculture.
The shipped agent is the **rebuild** (exp-007): three members trained on three disjoint chunks
of a corpus drawn from Kaggle's official daily Top Episodes datasets, merged by averaging
action probabilities.

- Method extraction from the Pokémon source, the Kaggriculture translation and every forced
  deviation: `workshop/deliverables/2026-08-20-team-a-bc-rebuild/method.md`.
- Corpus manifest (teacher identity, both ratings, provenance, split, chunk):
  `.../2026-08-20-team-a-bc-rebuild/corpus-manifest.csv`.
- The earlier single-teacher line (exp-003..006) is superseded; its corpus was disqualified.
  Historical rationale: `workshop/deliverables/2026-08-19-team-a-ensemble-bc/plan.md`.

## Pipeline (rebuild)

```bash
# 1. index every episode of the official daily Top Episodes datasets (metadata only --
#    replays are ~30 MB each). NB: scripts/episode_index.py truncates at 200 ids per day;
#    artisans/modeler/bench/rebuild/index_days.py follows the page tokens properly.
python artisans/modeler/bench/rebuild/index_days.py --days 2026-08-15 ... --out index/

# 2. select the corpus: winning seats, top N per day by the winner's rating, engine window
python artisans/modeler/bench/rebuild/select_corpus.py \
    --days 2026-08-15 2026-08-16 2026-08-17 2026-08-18 2026-08-19 --per-day-top 112

# 3. download it, verifying module_version + winner seat on arrival
python artisans/modeler/bench/rebuild/fetch_corpus.py

# 4+5. extract 3 chunks + shared validation, then train the 3 members (fixed seeds)
bash artisans/modeler/bench/rebuild/build_members.sh

# 6. re-tune the market thresholds on the ENSEMBLE's averaged probabilities
uv run python agents/team_a/bc/calibrate_ensemble.py \
    --rows "$GUILDHALL_WAREHOUSE/datasets/kaggriculture/bc-rows/rebuild-chunk0.npz"

# 7. evaluate in the Pit (the gate -- offline accuracy is only a filter)
uv run python -m arena register agents/team_a/bc --name TeamA-EnsembleBC-rebuild-v2 --team A
uv run python -m arena match TeamA-EnsembleBC-rebuild-v2 TeamA-Farmer-v0 --mode confirm

uv run pytest tests/test_bc_parity.py -q
```

Single-teacher training (`extract.py --teacher <name>` → `train.py`) still works and is what
the older experiments used.

## Files

| file | role |
|---|---|
| `features.py` | the single feature path, imported by training **and** by `main.py` |
| `extract.py` | replay → training rows; owns the replay off-by-one and the no-op relabelling |
| `net.py` | numpy MLP + Adam + losses (training and inference share it) |
| `train.py` | trains the unit net and the market net, tunes per-order thresholds, writes a model card |
| `policy.py` | inference, ensembling, safety layers, sanitiser |
| `main.py` | Kaggle entry point (`__file__` is not defined when the framework execs it) |
| `weights/*.npz` | one file per ensemble member; every `.npz` in the folder is loaded |

## Ensemble contract

Members are merged by **averaging their action probability vectors** over the same legal set,
then taking the argmax — the Pokémon `LogProbabilityEnsemble` formula
(`logsumexp_k log p_k − log K` is the same thing in log space). Not an average of logits, not
weighted. `BCPolicy` refuses members whose tensor schema differs from member 0.

Market orders fire on per-order thresholds. Those must be calibrated on the **ensemble's**
averaged probabilities (`calibrate_ensemble.py`, stored as `m_ens_thresholds` in member 0),
not per member: averaging pulls extremes toward the middle, and the observed per-member spread
reached 0.625 — enough to starve orders like `BUY_ANIMAL` that start the animal economy.

## Two traps that cost real time

1. **Replay off-by-one.** A kaggle-environments replay stores at index *t* the action that
   *produced* state *t*. Pair `steps[t].observation` with `steps[t+1].action`. Pairing them
   at the same index makes every `HARVEST` land on an already-harvested tile (22.3% of ops
   look like silent no-ops; `HARVEST` never once succeeds) and silently destroys the
   dataset while all the accuracy numbers still look plausible.
2. **Rare orders decide the game.** `BUY_ANIMAL` is 0.7% of the teacher's turns and starts
   the entire animal economy. An autoregressive market head with an `END` token (56% of
   rows) never fires it under argmax — the resulting clone banked 478. Per-order sigmoids
   with validation-tuned thresholds fire it at F1 0.93+.

## Safety levels (`BC_SAFETY`, default 3)

| level | adds | screen vs v0 | vs `starter` | our bank vs v0 |
|---|---|---|---|---|
| 0 | pure clone | 0.0% | 3.1% | 478 |
| 2 | + endgame drop/liquidate, keep-alive feed/water/wheat | 0.0% | 48.4% | 3,923 |
| 3 | + rule-based selling | 0.0% | 98.4% | 13,481 |

Levels are kept separable so each layer's contribution stays measurable. The clone is still
well below the v0 heuristic (39,579) — see the plan's Results section.
