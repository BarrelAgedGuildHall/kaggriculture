# The Pit — runbook

The Pit is the org's internal arena: the continuous ladder + dashboard that is the **single
source of truth for which agent is better**. Code in `arena/`; commands in README section
"The Pit (internal arena)". This is the operator's guide.

## Where state lives

Everything persistent is under the repo's git-ignored `data/arena/` on the machine running the
service (the node's fast NVMe disk — never a network share):

| Path | What |
|---|---|
| `registry.json` | agents (`name@sha`), active/benched/retired flags, champion pointers |
| `agents/` | immutable snapshots of every registered agent (the bytes that play) |
| `arena.sqlite` | every game ever played (rows keyed by game id; merged from remotes) |
| `replays/` | replays kept for crashes/timeouts (and losses with `--keep-replays loss`) |
| `progress/` | one JSON per run, live progress for the dashboard |
| `runs/` | per-run sqlite deltas pulled back from remote `--node` matches |
| `paused` | flag file: the hub ladder idles while it exists (survives restarts) |
| `leases.json` | live **pause leases** — while any exists, resume is refused (see "Pause leases") |
| `lease-log.jsonl` | append-only audit of every acquire / renew / release / expiry / force-break |
| `hub.json` | hub config: `workers`, `games_per_pairing`, `max_opponents` |
| `leaderboard.md` / `.json` | the last `arena leaderboard` export |

Periodic export for the workshop: `uv run python -m arena leaderboard` (writes
`workshop/notes/leaderboard-internal.md` when run on the desktop next to the workshop) and
`uv run python -m arena export --out data/arena/games.csv`. Ops may cron these later.

## The hub service (dashboard + continuous ladder)

```bash
uv run python -m arena serve --hub --host 0.0.0.0 --port 8765
```

- The **hub** keeps pairing *active* agents, PTCG-style: each round the active agent with the
  fewest ladder games anchors a star match vs its `max_opponents` most rating-similar active
  opponents, `games_per_pairing` episodes per pairing, paired fresh seeds (band 2000+) with side
  swap. Results are appended to the sqlite as each game finishes; BT/Elo on the board update live.
- Ghosts and benched/retired agents never enter the ladder. Rounds are small, so pause and config
  changes take effect within ~1–3 minutes.
- Dashboard-only (no games played): omit `--hub`. Local dev: omit `--host` (binds 127.0.0.1).

### Controls (UI header, or POST — examples with curl)

```bash
curl -X POST http://HOST:8765/api/control/pause              # durable (flag file)
curl -X POST http://HOST:8765/api/control/resume
curl -X POST http://HOST:8765/api/control/config -d '{"workers":14,"games_per_pairing":4,"max_opponents":5}'
curl -X POST http://HOST:8765/api/control/bench    -d '{"agent":"v0a"}'          # out of the ladder, still rated
curl -X POST http://HOST:8765/api/control/bench    -d '{"agent":"v0a","undo":true}'
curl -X POST http://HOST:8765/api/control/retire   -d '{"agent":"v0a"}'          # out of ladder AND rating fit
curl -X POST http://HOST:8765/api/control/champion -d '{"team":"A","agent":"v1"}'
curl -X POST http://HOST:8765/api/control/register -d '{"name":"v1","team":"A","git_ref":"feature/exp-004","path":"agents/team_a/v1.py"}'
```

### Pause leases (mutual exclusion for manual runs)

**Pausing is not enough on a shared host.** `paused` is a flag file and `resume` deletes it, so
on 2026-08-21 the ladder was resumed by another actor in the middle of a live timing battery and
contaminated its arms. A **lease** makes the pause owned: while any lease is live, every resume
path (API, dashboard button, CLI, an import of `arena.hub`) refuses with the lease list.

**A lease is exclusive, and it also holds off other operatives' runs** (2026-08-22). The first
version did neither, and that cost a battery: on 2026-08-21 21:08–22:00 MDT `engineer-b` held a
lease for the expB-304a gate battery, `modeler` took a *second* one 15 minutes later and started
its own battery on the same 16 cores. One arm ran at load 36.67 and recorded 16 spurious
timeouts, p99 68.7 ms and a 2647 ms worst turn, for an agent that measures p99 7.2 ms quiesced —
and the arena scores an over-budget turn as a **loss** regardless of bank. So now:

- a second `acquire` is refused with **409**, naming the holder, their purpose and the release
  command (`--shared` opts two non-measuring holds into coexistence; a shared lease still cannot
  be taken beside an exclusive one, nor an exclusive beside a shared one);
- `arena match` and `arena gate` **refuse to start** while somebody else holds a lease
  (exit code 3, `REFUSED: …`). Pass `--lease <id>` — or export `KAGGRI_LEASE=<id>` once for the
  whole battery — so your own lease is not read as foreign. `--ignore-lease` is the deliberate
  override and is written to `lease-log.jsonl` as `guard-ignored`;
- **and a run that declares no lease now takes one** (added 2026-08-22 after the Auditor's L-1).
  Refusing to start *under* someone else's lease only fixes the incident in the form it happened.
  In the mirror form — a leaseless measured run, then somebody acquires and starts a battery
  under it — both sides were still silent, and that was the *easy* default because nothing ever
  prompted for a lease. `arena match` / `arena gate` now resolve it in this order:

  | situation | what happens |
  |---|---|
  | somebody else's lease is live | refuse, exit 3 (unchanged) |
  | `--ignore-lease` | start anyway, take nothing, `guard-ignored` on the log |
  | your lease is declared (`--lease` / `$KAGGRI_LEASE`) | use it; take nothing, change nothing |
  | `--no-lease` | start with no lease; loud warning + `run-unleased` on the log |
  | otherwise | **take an exclusive lease**, released however the run ends |

  The lease a run takes for itself names `--lease-owner` (or `$KAGGRI_LEASE_OWNER`, else
  user@host), lives for `--lease-ttl` seconds (default 1 h — **size it to the battery**, a run
  that outlives its lease loses the protection), and puts the ladder back the way it found it:
  `match` pauses for the duration and resumes on the way out *if it was running when it started*;
  `gate` plays no games, so it leases without pausing. A ladder that was already paused stays
  paused — it is not ours to resume.

  Scripted callers are unaffected either way: a battery that exports `KAGGRI_LEASE` once takes
  the third row and nothing changes for it; a script that genuinely wants the old leaseless
  behaviour says `--no-lease` once.

```bash
# before a timing arm / gate battery / any measured manual run on the Pit host
uv run python -m arena lease acquire --owner engineer-b --purpose "expB-303 gate battery" --ttl 3600
# ... from a desktop, against the hub over the tailnet:
uv run python -m arena lease acquire --owner modeler --purpose "exp-N1 traces" \
    --hub http://barrelagedworkflows-1.tail547ecd.ts.net:8765
uv run python -m arena lease list
uv run python -m arena lease renew  <lease-id> --ttl 3600     # nothing auto-renews
uv run python -m arena lease release <lease-id> --resume      # --resume restarts the ladder

# then run everything under it (the guard sees the declaration and lets you through)
export KAGGRI_LEASE=<lease-id>
uv run python -m arena match cand champ --mode confirm
uv run python -m arena gate cand --team B
```

#### Long-running scripts: release on EXIT, not on the last line

A script that acquires on line 3 and releases on line 60 releases **nothing** when it dies on
line 40 — a failed command, a Ctrl-C, an ssh drop. The lease then sits there for up to its whole
TTL: every other operative's `arena match` is refused, and the ladder stays paused because a live
lease re-asserts the pause. That has now cost three separate evenings, so the standing convention
is: **install the release in the same breath as the acquire.**

```bash
#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lease.sh"                  # scripts/lease.sh, in this repo
kaggri_lease_acquire engineer-b "expB-303 gate battery" 7200
#  ^ acquires (exclusive, pauses the ladder), exports $KAGGRI_LEASE, and installs
#    trap … EXIT INT TERM. Nothing below can leak the lease; no release line at the bottom.

uv run python -m arena match cand champ --mode confirm
kaggri_lease_renew 7200                            # nothing auto-renews, by design
uv run python -m arena gate cand --team B
```

In Python, the same thing is a context manager:

```python
from arena import leases
with leases.held("engineer-b", "expB-303 gate battery", ttl_s=7200) as lease:
    os.environ[leases.LEASE_ENV] = lease["id"]     # child `arena match`es inherit the declaration
    ...
```

and in PowerShell, `try { … } finally { arena lease release $id --resume }`. `arena lease acquire
--print-id` prints only the id on stdout (the report goes to stderr) so a script can capture it.
If you do nothing at all, `arena match` and `arena gate` take and release a lease per run, which
is the floor, not the goal: one lease for the whole battery is still the right shape, because it
covers the gaps *between* the runs.

or over HTTP:

```bash
curl -X POST $PIT/api/control/lease -d '{"op":"acquire","owner":"engineer-b","purpose":"expB-303 gates","ttl_s":3600}'
curl      $PIT/api/leases
curl -X POST $PIT/api/control/lease -d '{"op":"renew","lease_id":"..."}'
curl -X POST $PIT/api/control/lease -d '{"op":"release","lease_id":"...","resume":true}'
```

Rules that make it work:

- **Acquire pauses**; the ladder still finishes the round it is in, so wait for
  `/api/hub` → `phase: "paused"` before you start measuring. Re-check it at the **end** of every
  arm too (standing rule: an arm across a pause flip is discarded).
- **TTL, not forever.** Default 1 h, max 6 h; a crashed run's lease lapses instead of wedging the
  ladder. Renew explicitly while you are still measuring — nothing renews itself.
- **Release when done** (`--resume` to restart the ladder in the same step). Only shared leases
  can coexist; when several are live the ladder stays down until every one of them is released.
- **The guard is not a substitute for the lease.** It only knows about leases in the arena dir it
  runs against (plus `--lease-hub URL` / `KAGGRI_PIT_HUB`, which fails closed if the hub cannot be
  read). Take the lease on the host the games actually run on: `arena match --node` forwards
  `--lease`/`--ignore-lease` so the node-side guard checks the *hub's* lease file, which is the
  one that matters. The hub ladder itself is exempt — it is held off by the pause flag, which
  `hub.reconcile_pause()` re-asserts whenever a lease is live.
- **Overrides are possible but named and logged**: `POST /api/control/resume
  {"force":true,"actor":"ceo"}` breaks (releases) the leases it overrides and writes a
  `force-break` row to `lease-log.jsonl`. Use it only for a stale lease whose owner is gone —
  and tell the owner, because their run's timing is now suspect.
- The dashboard shows live leases read-only (header pill + ladder panel). Releasing someone
  else's lease from a button would be the same footgun again, so it is CLI/API only.
- Who runs when is a scheduling convention, not code: see `docs/pit-host-windows.md`.

Two things that caught the first real users (both now handled in code, kept here because the
first instinct is to distrust the lease, not the reading):

- **`renew`/`release` take `lease_id`.** `acquire` returns the record with the id under `id`, so
  echoing the record straight back used to fail. Both keys are accepted now; the CLI takes it
  positionally (`arena lease release <lease-id>`), which never had the problem.
- **A `leases: []` reading is not proof that the ladder is free.** It is proof about *that*
  field, at *that* level, from *that* process — and a hub that has not been restarted since the
  lease code was deployed does not have the field at all, so a client's `?? []` default invents
  an empty list. **The refusal is the authority**: if `POST /api/control/resume` answers `409`
  with a lease list, a lease is held, whatever any status field says. `/api/status` now reports
  `leases` and a `lease_held` boolean at the top level *and* under `hub`, so reading either level
  works; `GET /api/leases` is the direct answer, and `lease-log.jsonl` is the record after the
  fact. If the two ever disagree, suspect a stale hub process (`systemctl --user restart pit`)
  or two arena dirs (`/api/status` → `arena_dir` vs the `ARENA_DIR` your CLI prints from
  `arena info`) before suspecting the lease.

More, from the Auditor's reviews of the lease PR (2026-08-22) — all fixed in code, recorded
here because each one is a way the ladder could look free when it was not:

- **`op` is required and bodies are key-checked.** `POST /api/control/lease` used to default `op`
  to `"list"`, so a body with the op under the wrong key (or missing) answered **200 with an empty
  lease list** — which reads exactly like "no lease held" — instead of doing what was asked. A
  missing op and any unknown key are now **400** and name themselves. The op key is `op`:
  `{"op":"acquire",...}`.
- **Unknown lease state is held state.** A truncated or corrupt `leases.json` used to be swallowed
  (`except Exception: state = {}`) and reported as *no leases*; truncating the file under a live
  lease flipped resume from 409 to 200. Now every read of an unreadable file raises, `lease_held`
  reads **true**, `/api/status` carries `lease_state_error`, and the dashboard shows a red pill.
  The only way through is a deliberate force-resume with an actor, which **quarantines** the file
  (renames it `leases.json.corrupt-<ts>`, never deletes it) and logs `state-quarantined`.
- **A lease implies a paused ladder.** The hub loop checked the pause flag only, so a lease held
  over an un-paused ladder (a `--no-pause` acquire, a flag deleted by hand) had nothing
  reconciling the two. It now re-asserts the pause and writes a `reconciled-pause` row.
- **Concurrent lease writes are serialised** in-process (the hub is a threading server) *and*
  across processes, by an `O_EXCL` lock file around every mutation — two shells that each read
  "no leases" and each wrote themselves in was the cross-process race, and it is closed
  (`tests/test_arena.py::test_two_process_race_leaves_exactly_one_holder`). The runbook's "verify
  your lease is listed after acquiring" step still stands, because it costs nothing and catches
  the case where you are talking to a different arena dir than you think.

And two more from the Auditor's second pass (2026-08-22), both about the lock file itself:

- **The lock file carries its holder's identity.** A critical section that outlives
  `LOCK_STALE_S` (20 s) has its lock broken as stale so a crashed holder cannot wedge the arena
  forever — and the old code then let the slow holder delete its *successor's* lock on the way
  out, so a third process could enter while the second was still mutating (Auditor L-2,
  reproduced: `A-in, B-in, A-out, B-out`). Now each holder writes a unique token, the break is a
  **rename** (so exactly one breaker wins it), a holder only removes the lock that still carries
  its own token, and — the part that actually protects the file — **every write re-checks the
  token first and fails closed** (`LockLost`) rather than landing on top of the new holder's
  state. `lock-broken` and `lock-lost` in `lease-log.jsonl` are the evidence; a `lock-lost` row
  means a process stalled >20 s inside a small JSON read-modify-write, so any timing measured on
  that host at that moment is suspect.
- **Probes must point themselves at a scratch arena.** A probe whose docstring said
  "set `KAGGRI_ARENA_DIR`" but which did not set it itself was run the obvious way and wrote
  `leases.json`, `lease-log.jsonl` and a `paused` flag into a real arena dir (Auditor L-3). On the
  Pit host that invocation **pauses the live ladder, and nothing resumes it.** Any script that
  exercises control state now calls `arena.paths.require_scratch_arena(...)` first, which refuses
  unless `KAGGRI_ARENA_DIR` was deliberately pointed at a directory holding no live arena state.

**Is the deploy live?** `arena_sha` in `/api/status` may be a *declared* value: on the Pit host the
hub runs from an untarred tree with no `.git`, so it comes from `KAGGRI_ARENA_SHA` in the systemd
unit — written once at deploy time and updated by nothing (it read `12a5398` for days while the
running code was well past it, and it is stamped on every game row). `/api/status` therefore also
carries `arena_sha_source` (`env` = declared, `git`, `none`) and **`arena_code_hash`** — a
line-ending-normalised sha256 over `arena/*.py` computed from disk. Compare it with your clone:

```bash
uv run python -m arena info | grep 'arena code'          # local
curl -s $PIT/api/status | python -c "import json,sys; print(json.load(sys.stdin)['arena_code_hash'])"
```

Equal hashes mean the host is running your code, whatever the shas say. When you ship, update
`KAGGRI_ARENA_SHA` in the unit too, so stored rows stay attributable.

`register` requires `git_ref` + a `path` under the repo's `agents/` tree (the hub fetches the
ref from origin — needs git access on the hub machine) and the new agent starts **benched**:
someone must explicitly activate it before the ladder will play it. Teams' flow: commit the
frozen generation under `agents/team_a|team_b/`, push, register by ref, review, activate.

### Security (read before widening anything)

Registering an agent **is scheduling code for execution on this host**: `kaggle_environments`
loads a file agent by exec'ing the whole file as the `pit` service user — no sandbox, no
network block (this host does not replicate the Kaggle sandbox). That is why the API only
accepts committed files under `agents/` at an origin ref, and why new registrations are benched.
Residual risk the founder has accepted: anyone with tailnet access who can also push to the
repo (or who activates a malicious committed agent) runs code on workflows-box; the tailnet is
the auth boundary and the commit + explicit-activation steps are the review gate. Raw local
paths / inline `code_b64` are OFF by default; `"allow_unsafe_register": true` in
`data/arena/hub.json` re-enables them for local dev only — never set it on the node. The rest
of `/api/control/*` mutates arena state only (pause/bench/champion), which is the accepted
tailnet-trust level.

## Running it on workflows-box (16 vCPU)

Deploy = ship the working tree (`arena match --node` uses the same bundle) or, once Ops' clone at
`~/kaggler/repo` has pull access, plain `git pull`. The service lives in `~/kaggri-arena/repo`
until then. `uv sync --frozen --no-dev` builds the venv (uv fetches Python 3.11 itself).

**systemd user unit** (lingering is enabled on the node, so it survives logout and reboots):

```ini
# ~/.config/systemd/user/pit.service
[Unit]
Description=The Pit (Kaggriculture internal arena hub)
After=network-online.target

[Service]
WorkingDirectory=%h/kaggri-arena/repo
Environment=PATH=%h/.local/bin:/usr/bin:/bin
ExecStart=%h/.local/bin/uv run python -m arena serve --hub --host 0.0.0.0 --port 8765
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now pit          # start + start at boot
systemctl --user status pit                # health
journalctl --user -u pit -n 50             # logs
systemctl --user stop pit                  # stop  (disable: systemctl --user disable pit)
```

**nohup fallback** (no systemd):

```bash
cd ~/kaggri-arena/repo
nohup ~/.local/bin/uv run python -m arena serve --hub --host 0.0.0.0 --port 8765 > ~/pit.log 2>&1 &
pkill -f "python -m arena serve"           # stop
```

### Reaching it on the tailnet

Plain HTTP works with no extra setup because the service binds 0.0.0.0:

> **http://barrelagedworkflows-1.tail547ecd.ts.net:8765/**

For HTTPS via `tailscale serve` the node needs a one-time root command (our SSH user has no
non-interactive sudo): `sudo tailscale set --operator=barrelaged`, then
`tailscale serve --bg --https=8765 http://127.0.0.1:8765` →
`https://barrelagedworkflows-1.tail547ecd.ts.net:8765/`. Do **not** reconfigure the desktop's
tailscale serve (it carries the Guildhall share).

### Updating the code on the node

The bundle route (works today, no repo credentials on the node):
on the desktop, from the repo root: `uv run python -m arena match random pass --n 2 --node barrelaged@<node>`
re-ships the tree as a side effect, or ship explicitly with scp + `tar xzf` per `arena/remote.py`.
Then `systemctl --user restart pit`. Once the node has read access to the GitHub repo (Ops),
replace with `git -C ~/kaggler/repo pull && systemctl --user restart pit` and point the unit's
WorkingDirectory there. **Note:** `register --git-ref` on the hub also needs that git access;
until then register from the desktop (the registry + snapshot can be shipped the same way) or
use the API's `code_b64`/local-path form.

## Ladder discipline

- **Engine bumps void the board.** `scripts/engine_watch.py` (cron) exits 1 on a new PyPI
  release: pause the hub, re-pin, re-run parity, restart; filter old games with
  `leaderboard?engine=`.
- **The hub is the only ladder.** `arena match --mode ladder` was removed (it was a second,
  divergent implementation); the hub owns the ladder seed band `LADDER_SEED_BASE = 2000+`
  (`arena/paths.py`), strictly above screen (0–199) and confirm (1000–1199), so held-out seeds
  are never touched. The gate additionally counts only `mode='confirm'` rows and reports any
  other games it excluded on those seeds. Screen/confirm/gate remain CLI commands
  (`arena match --mode screen|confirm|round-robin`, `arena gate`). Promotion is still gate-based:
  0 crashes, 0 timeouts, p99 ≤ 150 ms, Wilson lower bound > 50 % vs the team champion over the
  gate's band (default `confirm` = seeds 1000–1199, 400 games).
- **Seed bands are declared, and every gate states the one it used.** `paths.SEED_BANDS` names
  them (`screen` 0–199, `confirm` 1000–1199, `ladder` 2000+); `arena gate --band` takes a name or
  an explicit `lo-hi`, `arena match --seeds <band>` takes the same names, and
  `KAGGRI_SEED_BANDS="confirm=3000-3199,confirmB=4000-4199"` rotates or adds bands without a code
  change. This exists because a single hardcoded confirm band is a **shared blind spot**: on
  2026-08-21 a +17.2pp read on 1000–1199 pooled to level (53.0 % [44.3–61.7]) over three disjoint
  bands, and every gate in the org's history had inherited the same 200 seeds. Pool ≥ 2 disjoint
  bands before comparing across families and report the between-band spread beside the CI.
- **`arena verify` — do the runs still own their games?** `game_id` is
  `sha1(p0|p1|seed|engine)` and rows upsert, which is what lets a clean node run supersede a noisy
  local one and remote deltas merge. The cost is that re-running a pairing takes the earlier run's
  rows: on 2026-08-21 run `20260821-080254-e57458` recorded `n_games=400` with **zero** games left
  in the store. Replaced rows from a *different* run are now copied into the `superseded` ledger
  first, and `arena verify` (exit 1 on failure, `--superseded N` to print the lost measurements)
  reports every run whose games no longer add up plus any game whose `run_id` has no run row.
  Run it before quoting a stored run, and after importing a remote delta.

  **Give each store a baseline first, or the exit code says nothing** (Auditor P21-3). The raw
  check exits 1 on every store in the org — 2 incomplete runs in engineer's, 10 in modeler's, 4 in
  engineer-b's on 2026-08-22 — and most of them are the *intended* supersession, with pre-ledger
  rows reading `taken by: unknown (pre-ledger)`. So:

  ```bash
  uv run python -m arena verify --baseline record --note "why these are accepted"
  uv run python -m arena verify            # only what is NEW or has got worse; exit 0 if nothing
  uv run python -m arena verify --baseline ignore     # the raw, unfiltered check
  ```

  The baseline is one JSON file per store, written beside it as
  `<db>.verify-baseline.json` (git-ignored with the rest of `data/arena/`), because it describes
  that store's history and must not follow the code between clones. A finding is new when the
  baseline has never seen its `run_id`, or when that run has lost *more* games since — accepting a
  supersession never blesses the next one. Re-record deliberately, with a `--note` saying why.
- Seat note (Auditor, 2026-08-19): the seat asymmetry the side-swap cancels comes from
  `_end_of_day` drawing weed-spawn randomness sequentially per farm index from one shared RNG —
  not from action resolution order.
