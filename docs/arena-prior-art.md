# Arena — prior art consulted (2026-08-19)

The founder's PTCG (Pokémon) internal arena at `D:\PokemonAIChallenge\arena` (not a git repo; ~250 KB
of loose Python: `arena.py`, `arena_worker.py`, `agent_host.py`, `exploitability.py`, `state/`, four
docs) was read before finalising `arena/`. What was taken, what was deliberately not, and why.

## Taken
| Idea | Where it lives here | Notes |
|---|---|---|
| **Bradley-Terry by MM iteration with a small anchor prior** (`bt_ratings`) — order-independent, finite for undefeated/winless agents, ties as half wins | `arena/ratings.py::bradley_terry` | Same algorithm (Hunter 2004 MM); we use a virtual reference opponent with 0.5 prior games and centre log-strength at 0. |
| **Display scale `centre + 400·log10(strength)`** so BT reads like Elo (100 pts ≈ 64 %) | `ratings.bt_display`, leaderboard column **BT** | Their centre 600 (Kaggle's PTCG mu0); ours 1000. |
| **Keep sequential Elo alongside BT as a diagnostic**, never as the ranking | `ratings.elo`, leaderboard column **Elo** | Their README's argument (path/intransitivity artefacts) is why BT is primary. |
| **Append-only results log is the source of truth; ratings are derived and recomputed over the full set, once per run, not per game** | `data/arena/arena.sqlite` (rows never edited, only upserted by game id); `arena leaderboard` recomputes | We chose SQLite over `results.jsonl` because remote merge needs idempotent upsert by game id (`INSERT OR REPLACE`) and the gate needs indexed queries (games between two agents on a seed range). CSV export exists for anything else. |
| **"Benched" agents excluded from the rating fit without deleting their games** | `arena retire <agent>` / `--include-retired` | Keeps old generations from distorting the current field. |
| **Newest/least-played-first star matchmaking** for continuous rating (their 24/7 worker: the least-played bot anchors a match vs the field) | `arena match --mode ladder` | A new agent gets rated fast; balanced coverage. |
| **Per-run leaderboard snapshot / run metadata travelling with the data** | `runs` table + "Recent runs" section | |
| **Process-parallel games, parent-only writes** | `arena/scheduler.py::Pool` | Their `ProcessPoolExecutor`; ours is a hand-rolled pool because a hard-dead worker must not hang or kill the run (their note: "safe to interrupt; results saved after every game" — we do the same: every row is upserted as it lands). |
| **Docs shape**: a README a teammate can follow end-to-end (commands, outputs, limits, parallelism advice) | README "Arena" section | |

## Noted for later (not built)
- `exploitability.py` — Nash/fictitious-play over the matchup matrix → "robustness" score per agent
  (win % vs the least-exploitable mixed field). Worth adding once we have ≥ 6 real agents; it answers
  "which champion is least exploitable", which matters for a BT tournament against unknown opponents.
- `agent_host.py` — **one subprocess per agent per game** with a newline-JSON protocol and fd-level
  stdout isolation. Their engine had native global state, so this was mandatory. Our engine is pure
  Python and stateless, and the framework already exec's each file agent into its own namespace and
  catches its exceptions, so v1 isolates at the **game** level (one worker process per game; a hard
  crash kills only that worker and is recorded). If an agent ever ships native deps or leaks module
  state, port the host pattern (it is ~80 lines).
- Kaggle-style mu/sigma online rating (`kaggle_ratings`) — Kaggriculture's final standing is BT and the
  strategy explicitly says not to chase the live rating number, so it is omitted.
- Their "cap per contributor / FIFO drop" policy — not needed with two teams.

## Deliberately not ported
`dashboard.py`, `arena_api.py`/`arena_client.py` (hub networking), `llm_correspondence.py`,
`mod_kit.py`, the inbox/promotion worker. v1 is a CLI + SQLite + Markdown leaderboard; a local
dashboard or API can come later if the teams ask for it.
