# Team A — Ensemble Behaviour Cloning

Adaptation of the Ensemble-BC method used in the Pokémon TCG AI Challenge to Kaggriculture.
Design rationale, what carries over and what does not:
`workshop/deliverables/2026-08-19-team-a-ensemble-bc/plan.md`.

## Pipeline

```bash
# 1. index the ladder cheaply (metadata only -- replays are ~30 MB each)
uv run python scripts/episode_index.py --daily 2026-08-18
uv run python scripts/episode_index.py --submission 55574890

# 2. fetch a teacher's winning episodes (artisans/modeler/bench/fetch_teachers.py)
#    -> warehouse/datasets/kaggriculture/replays/team-a/1.32.7/<episode_id>.json

# 3. extract (obs, action) rows from the teacher's winning seat
uv run python agents/team_a/bc/extract.py --teacher tetsuya

# 4. train (pure numpy, ~30 s, 0.74 MB of weights)
uv run python agents/team_a/bc/train.py \
    --rows "$GUILDHALL_WAREHOUSE/datasets/kaggriculture/bc-rows/tetsuya-fold0.npz" \
    --out agents/team_a/bc/weights/tetsuya-a.npz

# 5. evaluate (the gate -- offline accuracy is only a filter)
uv run python scripts/evaluate.py --agent agents/team_a/bc/main.py \
    --opponents agent/main.py starter --n 64 --tag exp-005-screen

uv run pytest tests/test_bc_parity.py -q
```

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
