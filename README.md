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
| `scripts/engine_watch.py` | Exit 1 if PyPI has a newer `kaggle-environments` than our pin (daily cron / CI). |
| `nav/` | **The shared economic substrate (S-1)** — balance sheet, runtime-fitted price curves, demand book, opponent book, λ. Both teams import it; it decides nothing. Runtime half is stdlib-only and vendored into submissions. See `docs/nav.md`. |
| `scripts/nav_tripwire.py` | One command: do nav's runtime-fitted curves still match the engine? Run it after every engine bump. |
| `arena/` | **The internal arena** — registry, paired-seed scheduler, SQLite store, BT/Elo leaderboard, gates, remote runs. See "Arena" below. |
| `tools/` | **Shared measurement tooling** — both teams read the same books. `pnl_census.py` (per-line income statement + terminal balance sheet, `docs/tools/pnl-census.md`), `agent_spec.py` (one agent-spec grammar). |
| `docs/engine/` | Copies of the engine's `README.md` (rules), `AGENTS.md` (API), `kaggriculture.json` (config) and `VERSION.md` (pinned version + sha256 + parity evidence). |
| `tests/` | `pytest` smoke tests (engine pin, determinism, replay checker, packaging) + `test_arena.py`. |
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

## The Pit (internal arena)

The Pit (`arena/`, run as `uv run python -m arena ...` **from the repo root**) is the org's single
source of truth for which agent is better. Both teams evaluate through it; nothing else counts.
Frozen generations live in `agents/<team>/` (see `agents/README.md`).
All state is git-ignored under `data/arena/`: `registry.json` (agents + champions), `agents/`
(immutable snapshots of every registered agent), `arena.sqlite` (every game), `replays/`
(kept only for crashes/timeouts, or losses with `--keep-replays loss`), `leaderboard.md/json`,
`runs/` (per-run deltas from remote nodes).

```bash
uv run python -m arena info                                   # paths, engine, default workers
uv run python -m arena register agent/main.py --name v0 --team A --notes "exp-001 baseline"
uv run python -m arena register-ghost 93459920 --seat 0       # replay-ghost from data/replays/official/
uv run python -m arena list
uv run python -m arena match v0 random starter --mode screen   # 64 eps per opponent, seeds 0..31 x both seats
uv run python -m arena match v1 v0 --mode confirm              # 400 eps, held-out seeds 1000..1199
uv run python -m arena match --mode round-robin --n 16         # every registered agent vs every other
uv run python -m arena match v1 v0 --mode confirm --node barrelaged@barrelagedworkflows-1.tail547ecd.ts.net
uv run python -m arena serve                                  # dashboard + JSON API on http://127.0.0.1:8765
uv run python -m arena serve --hub --host 0.0.0.0             # + the continuous ladder (the hub service; see docs/pit.md)
uv run python -m arena lease acquire --owner engineer-b --purpose "expB-303 gates" --ttl 3600
                                                              # pause the ladder and hold it: while a lease is live
                                                              # resume is refused (--hub URL to lease a remote hub);
                                                              # release with `arena lease release <id> --resume`
uv run python -m arena bench v0a                              # out of the ladder (still rated); --undo re-activates
uv run python -m arena leaderboard                            # markdown/json export -> data/arena/leaderboard.md (+ workshop/notes/leaderboard-internal.md)
uv run python -m arena champion set v0 --team A               # teams: A | B | org
uv run python -m arena gate v1 --team A                       # PASS/FAIL vs team A's champion on the confirm band (1000-1199)
uv run python -m arena gate v1 --team A --band 3000-3199      # ... on a second, disjoint band: pool >=2 before promoting
uv run python -m arena verify --superseded 20                 # integrity: does every run still own the games it played?
uv run python -m arena retire v0                              # drop from the rating fit (rows kept); --undo
uv run python -m arena relabel old@4c859c9 new@12a5398        # rename an id across the registry AND its game
                                                              # rows in one transaction (game ids are recomputed);
                                                              # --dry-run first. Use when the LABEL was wrong --
                                                              # re-registering instead would orphan the rows.
uv run python -m arena nav --mode ladder --calibrate           # S-3: per-episode NAV trajectories the ladder
                                                              # records (off for gates/screens; --nav-telemetry
                                                              # on `match` to record a run). --trajectory prints
                                                              # every mark, --json dumps them for analysis.
uv run python -m arena push v1 --hub http://HOST:8765               # upload ONE agent snapshot to a hub:
                                                              # content-hash verified end to end, ONE registry
                                                              # key merged, champions/roster untouched, arrives
                                                              # benched. Never ship the registry (issue #15).
uv run python -m arena export --out data/arena/games.csv
```

How a match works:
- **Agents** are `name@sha` (git short sha of the file if committed & clean, else a content hash);
  registering snapshots the bytes, so the arena always plays what was registered. Plain `name`
  resolves to the most recently registered `name@...`. Built-ins `random`, `starter`, `pass` are
  auto-registered residents.
- **Paired seeds with side swap**: each seed is played twice (A as player 0, then as player 1).
  `--n` is episodes per opponent pair (so `--n 64` = 32 seed pairs). Seed hygiene: `screen` uses
  0..; `confirm` uses held-out 1000..; `--seeds 5-9` / `--seeds 1,4,9` for explicit sets.
- **Parallel & isolated**: one worker process per game (`--workers`, default cpu_count-2); a worker
  that dies hard is replaced and its game recorded as a harness crash; `--game-timeout` kills a hung
  game. Each file agent is exec'd into its own namespace by the framework, exceptions become status
  ERROR.
- **Per game** the store keeps: seed, seats, final money both, result for p0 (1/0.5/0), per-agent
  p50/p99/max turn time (the framework's timer), crash/timeout flags, engine version, arena git sha,
  host, run id. A crash or a timeout is scored as a **loss** for that agent regardless of bank
  (the strategy's rule: a crash is a certain loss; note the local engine itself lets a crashed agent
  keep its bank and stop acting). "Timeout" = framework TIMEOUT status **or any turn over 1000 ms**
  (locally the framework only times out after the 60 s overage bank is spent).
- Re-running the same pairing on the same seeds **replaces** those rows (game id = p0|p1|seed|engine);
  results from other machines merge by id (`arena import file.sqlite`).

Dashboard + hub (`arena serve [--hub]`, stdlib `http.server`, vanilla JS, no build step):
leaderboard with BT + Elo, win% by opponent class (builtin / team A / team B / ghost),
crash/timeout counts, p99/max, champions (★), head-to-head matrix with Wilson CIs, per-agent
detail pages (click a name), recent games (flagged in red), the live ladder round and any running
match with a progress bar, recent runs, engine version; header controls for pause/resume, worker
count, bench/retire/re-activate/champion per agent, and a register form (git ref or path).
Auto-refreshes every 3 s. With `--hub` the same process runs the **continuous ladder**: the
least-played active agent anchors a star match vs its most rating-similar active opponents on
fresh paired seeds, forever; without it the dashboard just watches CLI runs. JSON API for the
teams/Registrar: `/api/leaderboard`, `/api/games?limit=50&agent=<id>&flagged=1`, `/api/agents`,
`/api/runs`, `/api/status`, `/api/hub`, `/api/leases`, `/api/nav`, `/api/leaderboard.md`, and
`POST /api/control/*` (pause/resume/lease/config/bench/retire/champion/register/**push** — register
only takes `git_ref` + a `path` under `agents/`, `push` uploads ONE content-verified snapshot and
merges ONE registry key, and both start the agent benched; see docs/pit.md "Security").
**NAV telemetry (S-3)**: the ladder records per-episode NAV trajectories for a sample of its games
(`hub.json: nav_telemetry`, percent) — off for gates and screens, and computed after the episode
so it cannot move a measured turn time. `arena nav [--trajectory|--calibrate|--json]`, docs/pit.md
"NAV telemetry". **Pause leases**: a manual run holds the ladder under a named lease (owner, purpose,
TTL, explicit renew) and every resume path refuses while it is live — `docs/pit.md` "Pause
leases", and `docs/pit-host-windows.md` for who runs when on the shared host. Service deployment
on workflows-box + tailnet URL: **docs/pit.md**.

Leaderboard export: **BT** (Bradley-Terry MM fit over all stored games — the final Kaggle tournament is BT;
shown as 1000 + 400·log10(strength), 100 pts ≈ 64 % win odds), **Elo** (sequential, diagnostic),
head-to-head win % with **Wilson 95 % CI**, crash/timeout counts, worst p99/max. Written to
`data/arena/leaderboard.md` + `.json` and, when the workshop exists next to the clone, to
`workshop/notes/leaderboard-internal.md` (`--publish PATH` to send it elsewhere).

Gates (`arena gate <agent> --team T`): over games against the team's champion on the confirm seeds
(1000-1199, ≥ 400 games by default): 0 crashes, 0 timeouts, p99 ≤ 150 ms, max ≤ 400 ms, and
Wilson lower bound > 50 %. Exit 0 = PASS. The first champion of a team is set by decision.

Replay ghosts (`register-ghost`): a resident that replays one seat's recorded actions from an
official episode. **Only meaningful on its replay's seed and engine** — the scheduler forces the
seed (so a match vs one ghost is 2 games) and refuses other engine versions; ghosts are open-loop
(they do not react to you) and were validated against their recorded partner on their own seat
(bank must reproduce within ±15 %; with the pinned engine it reproduces exactly). Seat swap is
allowed but approximate. Ghosts are excluded from round-robin and the hub ladder unless named.

Remote (`--node host`): ships the working tree + registry + agent snapshots as a tar.gz over
scp (the node has no access to the private repo), bootstraps `uv` + `uv sync --frozen` in
`~/kaggri-arena/repo`, runs the same match with `--db data/arena/runs/<run>.sqlite`, copies that
file back and imports it. Crash replays stay on the node. `--remote-workers` overrides the node's
default (cpu_count-2; workflows-box has 16 vCPU).

Engine watch: `uv run python scripts/engine_watch.py` exits 1 when PyPI has a newer
`kaggle-environments`. Any bump voids tuning results: re-pin, re-run `replay_check.py`, re-run the
round-robin, and filter the leaderboard with `--engine <version>`.

Prior art: `docs/arena-prior-art.md`.

## The balance sheet (`nav`)

One shared library that answers *what is this farm worth* and *what is exactly inferable*,
for both seats. It contains no policy: every judgement stays in the agent. Full guide:
[`docs/nav.md`](docs/nav.md).

```python
import nav
n = nav.Nav()
n.observe(obs)                      # every turn (p50 ~0.03 ms)
sheet = n.sheet()                   # cash + marked lines (p50 ~0.3 ms); variant and lambda explicit
n.phi_rel()                         # (nav-cash)_us - (nav-cash)_them, from public state
```

Price curves are **fitted at runtime** from observed `(inventory, price)` pairs — nothing is
hard-coded — so an engine rebalance changes the numbers, not the code. That is only credible
because it is checked:

```bash
uv run python scripts/nav_tripwire.py     # exit 1 => do not trust a balance-sheet number
```

## Submission

```bash
uv run python scripts/make_submission.py --verify   # builds dist/, plays an episode from a fresh venv
uv run python scripts/make_submission.py --verify --vendor nav --agent-dir agents/team_b/routed
# then, deliberately and only after the Auditor's review:
uv run kaggle competitions submit kaggriculture -f dist/main.py -m "<experiment id>"
```

`--vendor nav` copies the library into the bundle (the engine's loader only puts the agent's
own directory on `sys.path`). **A nav bump re-ids every agent that vendors it and invalidates
its gates — never bump it mid-experiment or during a promotion window** (`docs/nav.md`).

## Conventions

- Branch `feature/<topic>` / `fix/<topic>`, PR to `main`, one concern per PR.
- Never commit data, replays, weights, or credentials (`.gitignore` covers `data/`, `.venv/`, `dist/`).
- Pin = `pyproject.toml` + `uv.lock`; bump deliberately and re-run the replay check.
