# Host windows on workflows-box — who runs when

`workflows-box` (16 vCPU, RTX 4070 SUPER) runs the Pit hub ladder **and** both teams' manual
work. Timing numbers are the only host-sensitive quantity we produce (banks are bit-identical
across hosts at fixed seeds), and they are ruined by foreign load: 16 oversubscribed workers
push single turns past 400 ms for *any* agent, and `arena gate` takes p99/max as the **maximum
over all stored games**, so one contaminated arm poisons an agent's timing clause permanently in
the hub DB. This page is the scheduling convention that keeps two teams out of each other's
measurements. The mechanism that enforces it is the **pause lease** (`docs/pit.md` → "Pause
leases"); this page only decides who wins a conflict.

## The two rules

1. **Every measured run takes a lease.** Timing arms, gate batteries, A/B confirms — anything
   whose numbers will be quoted. No lease, no quotable timing. The lease is required in your own
   window too: it is what stops the ladder, and it is the evidence trail if an arm is disputed.
2. **A window decides who yields, not whether you lease.** Outside your window you may run
   whenever nothing is leased; if the window's owner shows up, you release at the end of your
   current arm.

## Windows (America/Denver — the founder's clock; MDT until 2026-11-02)

| Window | Owner | Typical work |
|---|---|---|
| 04:00–07:00 | **Ops** | daily episode/replay ingestion, CI, engine watch, exports (I/O heavy, not timing-sensitive) |
| 09:00–12:00 | **Team A** (`modeler`, `architect`) | measured arms, PPO screens that will be quoted, gate batteries |
| 13:00–16:00 | **Team B** (`engineer-b`, `architect-b`) | measured arms, gate batteries |
| 16:00–09:00 | open | first-come; lease TTL ≤ 60 min so the next window's owner is never blocked long |
| any time | **`engineer` / `auditor`** | pre-submission gate runs and hub maintenance **pre-empt everything** (plan §7.3) |

Unmeasured background compute (PPO rollout training, corpus builds, ingestion) needs **no
lease** — but it must be **stopped** while someone holds one, because it is exactly the foreign
load that inflates turn times. Whoever starts training is responsible for stopping it when a
lease appears; check `curl $PIT/api/leases` before launching a long job.

## Before a measured arm (checklist)

```bash
uv run python -m arena lease acquire --owner <you> --purpose "<exp id + what>" --ttl 3600
export KAGGRI_LEASE=<lease-id>   # every match/gate in this shell runs under your own lease
curl -s $PIT/api/hub | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['phase'], d['paused'])"
uptime                      # 1-min loadavg must be < ~0.5 before the first game
```

Since 2026-08-22 the lease is a real lock, not a warning light: a second `acquire` is refused
(409), and `arena match` / `arena gate` refuse to start (exit 3) while somebody else holds one.
You no longer have to *notice* a foreign lease — but you do have to declare your own with
`--lease <id>` or `KAGGRI_LEASE`, or the guard will read it as somebody else's.

A run that declares no lease **takes one for itself** and releases it when it ends, so two
leaseless runs can no longer collide either. That is a floor, not a substitute for this
checklist: a per-run lease does not cover the gaps between your runs, and it does not wait for
`phase: paused` before the games start. Take the arm's lease yourself, wait for the ladder to be
quiet, and export `KAGGRI_LEASE`.

If the arm is driven by a script, take the lease through `scripts/lease.sh` so a Ctrl-C or a
failed command still releases it:

```bash
source scripts/lease.sh
kaggri_lease_acquire <you> "<exp id + what>" 3600     # + trap … EXIT INT TERM
```

- Wait for `phase: paused` — the ladder finishes the round it is already in.
- **Gate on loadavg, never on `pgrep -f <pattern>`.** A waiter written as
  `while pgrep -f ppo_train.py; do sleep 30; done` matches its own command line and every other
  watcher containing that string; on 2026-08-21 five such loops deadlocked permanently on this
  host. If you must match a process, use the bracket trick (`pgrep -f "ppo_screen[.]py"`), and
  never `pkill -f "<something in your own command line>"`.
- Record `load_before` for every arm. `load_after` is meaningless (your own workers drive it).
- Re-check `paused` **at the end** of each arm. An arm that spans a pause flip is discarded —
  with a lease held, a flip means someone forced it, and `data/arena/lease-log.jsonl` says who.
- Contended runs belong in a side DB (`arena match --db ...`) so contention can never reach the
  promotion store's timing rows.
- **"Money is bit-identical under load" is NOT true of every agent** — it is a property of the
  *agent*, not of the engine, and it only holds for a policy with no wall-clock branch. Two
  families in this repo have one, and under load they play **different moves** and bank different
  money on an identical seed (observed by the Modeler on melonrace, 2026-08-21):
  - `agents/team_b/{millrace,routed}/chassis.py` — the M0 chassis throws the whole planned action
    away and emits the precomputed fallback when a turn exceeds its own 250 ms budget
    (`deadline_s=0.25`). All-or-nothing, so one slow turn changes everything after it.
  - `agents/team_a/{v0,v0a,v0b}.py` — bails out of the greedy assignment loop at
    `PLANNING_BUDGET_S = 0.30` and emits the partial plan.

  Team A's EnsembleBC line has no such branch, which is why every check of the claim happened to
  pass. **Before quoting a contended run's money, check the agent for a self-imposed deadline.**
  This is a live competition risk as well as a measurement one: Kaggle's sandbox is 1.6 vCPU, so
  a self-imposed budget can fire there even when nothing is wrong.

## When you are finished

```bash
uv run python -m arena lease release <lease-id> --resume
```

Renew (`arena lease renew <id>`) rather than letting a lease lapse mid-battery: an expired lease
stops blocking the ladder silently, and the next arm is contaminated without anyone doing
anything wrong.

And release it from a **trap**, not from the last line of the script. Three separate evenings
have now ended with a lease held by a process that died — `scripts/lease.sh` (shell),
`with arena.leases.held(...)` (Python), `try/finally` (PowerShell).

## Conflicts

- Lease held, you need the host: ask the owner (Slack/A2A), or wait for the TTL.
- Owner unreachable and the lease looks stale (their run is done, load is 0): escalate to the
  **CEO**; force-resume is `POST /api/control/resume {"force":true,"actor":"<who>"}`, it releases
  the lease it breaks and logs a `force-break` row. Tell the owner afterwards — from their side
  their arm's timing is now suspect.
- Long batteries (> 6 h) do not get a single lease: 6 h is the TTL ceiling. Split them into arms
  and renew, so an abandoned run always frees the ladder.
