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

`register` accepts `git_ref` + `path` (the hub fetches the ref from origin — needs git access on
the hub machine), a local `path`, or inline `code_b64`. Teams' normal flow: commit the frozen
generation under `agents/team_a|team_b/`, push, then register it by git ref from the UI form or
the API. No auth — the service trusts the tailnet; keep it off public interfaces.

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
- Screen/confirm/gate remain CLI commands (`arena match --mode screen|confirm|round-robin`,
  `arena gate`); the hub only runs the ladder. Promotion is still gate-based:
  0 crashes, 0 timeouts, p99 ≤ 150 ms, Wilson lower bound > 50 % vs the team champion over the
  confirm set (seeds 1000–1199, 400 games).
- Seat note (Auditor, 2026-08-19): the seat asymmetry the side-swap cancels comes from
  `_end_of_day` drawing weed-spawn randomness sequentially per farm index from one shared RNG —
  not from action resolution order.
