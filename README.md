# kaggriculture

Kaggler's entry for the Kaggle **Kaggriculture** simulation competition
(https://www.kaggle.com/competitions/kaggriculture): a two-player farming game, 720 turns,
1 s per turn, `main.py` with `agent(obs)`, ranked by ladder rating.

## Layout

| Path | What |
|---|---|
| `agent/main.py` | The submission: `agent(obs)`. Single-file today; helpers may be added next to it (packaged as tar.gz). |
| `scripts/run_episode.py` | Play one local episode A vs B with a fixed seed; prints money + per-turn timing; saves the replay. |
| `scripts/evaluate.py` | The gauntlet: paired seeds + side swap vs several opponents, win rate with Wilson CI, timing gates. |
| `scripts/replay_check.py` | Re-step an official Kaggle replay through the local engine and diff every step (engine-parity proof). |
| `scripts/fetch_replays.py` | Download official replays / the episodes index with the Kaggle CLI. |
| `scripts/make_submission.py` | Build `dist/main.py` (or `dist/submission.tar.gz`) and verify it plays standalone in a fresh venv. |
| `docs/engine/` | Copies of the engine's `README.md` (rules), `AGENTS.md` (API), `kaggriculture.json` (config) and `VERSION.md` (pinned version + sha256 + parity evidence). |
| `tests/` | `pytest` smoke tests (engine pin, determinism, replay checker, packaging). |
| `data/` | Git-ignored: replays (`data/replays/official`, `data/replays/local`), index, anything large. |

## Setup (Windows / Linux / WSL / remote node)

Requirements: [`uv`](https://docs.astral.sh/uv/) on PATH (it fetches Python 3.11 itself). Do not
use the system Python.

```bash
git clone https://github.com/BarrelAgedGuildHall/kaggriculture.git
cd kaggriculture
uv sync --group dev            # creates .venv (Python 3.11) from uv.lock: kaggle-environments==1.32.7, kaggle>=2.2, numpy, pytest
```

`uv sync` is exact and reproducible (`uv.lock`). `kaggle-environments` drags in a large
dependency set (jax, transformers, open-spiel, litellm ...) -- that is upstream's packaging,
expect ~1-3 min the first time; later syncs hit uv's cache.

Kaggle CLI auth (only needed for downloading replays / submitting): export the token in
the shell, never write it to a file.

```bash
export KAGGLE_API_TOKEN=...            # PowerShell: $env:KAGGLE_API_TOKEN = '...'
export KAGGLE_CONFIG_DIR=/tmp/nokaggle # optional: ignore a stale ~/.kaggle/kaggle.json
uv run kaggle competitions list -s kaggriculture
```

## Run

One local episode (our agent as player 0 vs the built-in `random`), seed 0, 720 turns:

```bash
uv run python scripts/run_episode.py --seed 0
# other agents: built-ins pass|random|starter, a .py path, or module:function
uv run python scripts/run_episode.py --agent-a agent/main.py --agent-b starter --seed 3
```

Output: final money and reward/status per player, per-turn wall-time stats of each agent
(n / mean / p50 / p99 / max ms, from the framework's own timer -- the budget is 1000 ms),
and the replay path under `data/replays/local/`. A machine-readable `SUMMARY {...}` line is
printed last. Note: `seed` fixes the engine RNG (weeds, shop draws); the built-in `random`
agent is itself unseeded, so pair deterministic agents for exact repeatability.

Tests:

```bash
uv run pytest -q
```

## Evaluate (the score that decides things)

The ladder counts win / loss / tie only, so the KPI is **win rate**, never bank margin.
`scripts/evaluate.py` plays every seed twice with the seats swapped, which removes most of
the seed variance (dominated by the town's shop draw) and the small player-0 advantage.

```bash
uv run python scripts/evaluate.py --n 64 --tag exp-001-screen                      # screen
uv run python scripts/evaluate.py --n 400 --seed-base 1000 --tag exp-001-confirm   # confirm
uv run python scripts/evaluate.py --opponents random starter path/to/v0.py --n 64  # add a generation
```

Screen on seeds 0..199, confirm on held-out 1000..1199, never quote a number from the
tuning seeds. The run exits non-zero (`RESULT GATES_FAIL`) on any agent exception, any
ERROR/TIMEOUT status, or a p99 turn time over 150 ms -- a voided run has no score. Results
land in git-ignored `data/results/<tag>/` (`results.json` + `episodes.csv`).

**Bank against a passive opponent is a misleading metric.** Both farms share one market, so
extra production beyond the town's appetite crashes the price for both players: a change
that adds $2k against `random` can be worth nothing head-to-head. Always compare against
the previous generation, not against `random`.

## Engine parity (is local tuning trustworthy?)

```bash
# download a few official replays (needs KAGGLE_API_TOKEN)
uv run python scripts/fetch_replays.py --daily 2026-08-16 --n 3
uv run python scripts/fetch_replays.py --episodes 93459920
# re-step them through the local engine
uv run python scripts/replay_check.py "data/replays/official/*.json"
```

`REPRODUCES ... mismatched_steps=0` with identical rewards means the local engine is
step-for-step identical to the one that produced the replay. Replays record the
`kaggle-environments` version that produced them (`module_version`); replays from a
different engine version generally do **not** reproduce (the economy changed in 1.32.7).
See `docs/engine/VERSION.md` for the evidence and the upgrade procedure.

## Submission

```bash
uv run python scripts/make_submission.py --verify   # builds dist/, plays an episode from a fresh venv
# then, deliberately and only after the Auditor's review:
uv run kaggle competitions submit kaggriculture -f dist/main.py -m "<experiment id>"
```

## Conventions

- Branch `feature/<topic>` / `fix/<topic>`, PR to `main`, one concern per PR.
- Never commit data, replays, weights, or credentials (`.gitignore` covers `data/`, `.venv/`, `dist/`).
- Pin = `pyproject.toml` + `uv.lock`; bump deliberately and re-run the replay check.
