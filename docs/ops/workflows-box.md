# Runbook — `workflows-box` (Pit hub node)

Owner split: **Ops** owns the node environment (this doc); **Engineer** owns the Pit hub
service itself (`arena/`, the web UI, `tailscale serve`). Reach the node via `mcp:nodes`
(`run_remote node="workflows-box"`) or `ssh user@workflows-box.example`.

## Hardware / OS (probed 2026-08-19)

- CPU: AMD Ryzen 7 7700X, 16 threads (`nproc` = 16)
- RAM: 29 GiB total, ~7-25 GiB free depending on load (ComfyUI/Ollama share the box)
- GPU: RTX 4070 SUPER, 12 GiB VRAM — not needed for Kaggriculture (no neural policy planned)
- Disk: `/` (nvme0n1p2) 1.8 TiB, 839 GiB free as of 2026-08-19 — plenty of headroom, but this
  disk is shared with ComfyUI (`~/comfyui` = 181 GB) and other projects; do not assume it is
  dedicated to Kaggler
- Tailscale 1.102.2, node reachable at `workflows-box.example`
  (100.x.y.z); a second, stale tailnet entry `workflows-box-old` (100.x.y.z) has
  been offline 43+ days — ignore it, use the `-1` hostname

## What Ops set up (2026-08-19)

- **uv 0.12.5** installed to `~/.local/bin` (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
  Not on `$PATH` in non-interactive `run_remote` shells by default — prefix commands with
  `export PATH="$HOME/.local/bin:$PATH"` (or ssh in and check `~/.bashrc` sourcing).
- **Python 3.11.16** via `uv python install 3.11` (uv-managed, not system Python — system
  `python3` here is 3.13.7 and `python3.11` is not on PATH otherwise).
- **Repo clone** at `~/kaggler/repo` (`BarrelAgedGuildHall/kaggriculture`, `main` branch).
  Cloned once with a short-lived installation token in the URL, then
  `git remote set-url origin https://github.com/BarrelAgedGuildHall/kaggriculture.git`
  (token-less) — verify with `grep -i token .git/config` before trusting the remote is clean.
  **Note:** this box's *global* `~/.gitconfig` independently carries an
  `url."https://x-access-token:<PAT>@github.com/".insteadOf = https://github.com/` rewrite
  (pre-existing, not set up by Ops/this repo) — `git remote -v` will therefore always print a
  token-looking URL for *any* github.com remote on this box regardless of what `.git/config`
  actually stores; that's a global rewrite doing display-time substitution, not a leaked
  secret in this repo's config. Left untouched — it predates this work and may be relied on
  by other projects on the box.
- `~/kaggler/repo` is **separate from `~/kaggri-arena/repo`** (the Engineer's own clone for
  the Pit hub, present as of 2026-08-19 13:59). Do not touch `~/kaggri-arena/` or `arena/`
  from Ops jobs — that's the Engineer's territory.
- `uv sync --group dev` — completes in well under a second once the package cache is warm
  (uv resolves from the same lock file as the fast-tier bench; no separate resolution needed).

## Verified run (2026-08-19)

```
$ uv run python scripts/run_episode.py --seed 0
engine      : kaggle-environments 1.32.7  seed=0  steps=720  wall=2.3s
player 0 (A): agent/main.py  money=39386  status=DONE
player 1 (B): random          money=220    status=DONE
A turn time : mean 0.185ms  p50 0.163ms  p99 0.657ms  max 4.519ms   (budget 1000ms/turn)
```
Full 720-step episode wall time 2.3s (vs ~1.1s reported locally on the bench in the Engineer's
2026-08-19 env-setup report — this box is under shared load from ComfyUI/Ollama, so expect
noisier wall-clock than a dedicated box; per-turn agent time is what matters for the 1s
gate and it has comfortable headroom either way).

## Tailscale serve (reported, not configured — Engineer's call)

- `tailscale serve status` → `No serve config` (nothing published yet).
- User `barrelaged` is in the `sudo` group; `tailscale serve` itself does not require sudo
  when `tailscaled` already runs with the right permissions (it does here — `tailscale
  status` works unprivileged). No blocker observed to the Engineer standing up the Pit's web
  UI here whenever they're ready; Ops did not run `tailscale serve` (not our call — the
  Guildhall wrapper is already published on this tailnet via the desktop node, so a second
  `serve` here is the Engineer's decision on how the Pit UI gets exposed).

## Running Kaggler jobs here

```bash
ssh user@workflows-box.example   # or run_remote via mcp:nodes
export PATH="$HOME/.local/bin:$PATH"
cd ~/kaggler/repo
git pull                              # keep the node's clone current before a job
uv sync --group dev
uv run python scripts/run_episode.py --seed 0
uv run pytest -q
```

For heavy/parallel jobs (round-robins, tuning sweeps — Engineer/Modeler territory), use all
16 threads; this box also runs ComfyUI/Ollama for other subsidiaries, so don't assume 100% of
CPU/RAM is free — check `nproc`/`free -h` before launching a big batch.

## Known gaps / follow-ups

- No `nohup`-launched long-running service here yet (that's the Pit hub, Engineer's PR).
- Node has no dedicated Kaggler cron yet — see the ingestion/engine-watch cron proposal in
  `workshop/notes/decisions.md` (2026-08-19, Ops) pending founder approval for the
  `scheduling` clearance.
