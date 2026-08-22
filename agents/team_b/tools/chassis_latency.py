"""Chassis timing instrument — ruling-04 C-4 (verification) and C-5/C-6 (histogram).

Two modes.

`verify` — prove the two-argument agent signature actually receives the engine's
`configuration`, on BOTH loader paths, by running with a deliberately
non-default `actTimeout` and asserting the chassis read that value rather than
its vendored fallback:
  * the real `kaggle_environments` path (`make()` + the framework's own agent
    wrapper, which slices the call on `co_argcount` at agent.py:152/171);
  * the Pit child harness (`arena/agent_child.py` behind
    `arena.isolation.SubprocessAgent`), read back through the chassis's
    `MILLRACE_DIAG` dump because only the action crosses that pipe.

`histogram` — play N episodes of an agent against an opponent and report per-turn
planning time AS A FRACTION OF `actTimeout` (never in bare milliseconds, which
is unreadable at an unknown host speed):
  * counts above {0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00} x actTimeout;
  * p50/p90/p99/p99.9/max;
  * total overage consumed, `sum_t max(0, d_t - actTimeout)`, against the 60 s
    per-episode bank;
  * min `remainingOverageTime` the agent ever observed;
  * peak RSS of the agent children (`RUSAGE_CHILDREN`, POSIX only);
  * `plan_exceptions` and the PANIC latch state per seat-season.
Exposure at any host slowdown factor k is then read straight off the histogram:
the mass at k is the mass above `threshold / k`.

Two clocks are reported and they are not the same number:
  `engine`  — the framework's own `duration`, i.e. what actually gets billed
              against the bank.  Under subprocess isolation it includes JSON IPC.
  `chassis` — the agent's internal `.timings`, i.e. planning only.
The engine clock is the one the pre-registered C-6 numbers are judged on.

Usage (from the repo root):
  uv run python agents/team_b/tools/chassis_latency.py verify
  uv run python agents/team_b/tools/chassis_latency.py histogram \
      --agent TeamB-Millrace-s32m6@hddf03d4580 --opponent TeamB-Millrace-pricer@hdd12354335 \
      --seeds 0-63 --out runs/expB-306c/hist.json

Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from arena import engine as eng           # noqa: E402
from arena.isolation import SubprocessAgent  # noqa: E402
from arena.paths import ENV_NAME           # noqa: E402
from arena import registry as areg         # noqa: E402

FRACTIONS = (0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00)
MILLRACE_ENTRY = REPO_ROOT / "agents" / "team_b" / "millrace" / "main.py"


# --------------------------------------------------------------- agent resolution
def resolve_entry(spec: str) -> Path:
    """An arena agent id, or a path to a main.py / an agent directory."""
    p = Path(spec)
    if p.is_dir():
        return p / "main.py"
    if p.suffix == ".py" and p.exists():
        return p
    entry = areg.resolve(spec)
    if entry.get("kind") != "file":
        raise SystemExit(f"agent {spec!r} is kind={entry.get('kind')!r}; this tool needs a file agent")
    return areg.entry_path(entry)


# ------------------------------------------------------------------ one episode
def play(entry0: Path, entry1: Path, seed: int, steps: int, cfg_extra: dict | None = None):
    ke = eng.import_engine()
    cfg = dict(cfg_extra or {})
    cfg.update({"episodeSteps": int(steps), "seed": int(seed)})
    env = ke.make(ENV_NAME, configuration=cfg, debug=False)
    agents = [SubprocessAgent(entry0, name="p0"), SubprocessAgent(entry1, name="p1")]
    try:
        env.run(agents)
    finally:
        for a in agents:
            a.close()
    final = env.steps[-1]
    durations = [[], []]
    for lg in env.logs:
        for i in (0, 1):
            if len(lg) > i and isinstance(lg[i], dict) and "duration" in lg[i]:
                durations[i].append(float(lg[i]["duration"]))
    banks = []
    for i in (0, 1):
        b = []
        for s in env.steps:
            try:
                b.append(float(s[i].observation["remainingOverageTime"]))
            except Exception:
                pass
        banks.append(b)
    return {
        "seed": seed,
        "money": [float(final[i].observation.farms[i]["money"]) for i in (0, 1)],
        "status": [str(final[i].status) for i in (0, 1)],
        "durations": durations,
        "overage_obs": banks,
    }


# ------------------------------------------------------------------ verify mode
def _load_like_kaggle(entry: Path):
    """Exactly `kaggle_environments.agent.get_last_callable`: exec the file with
    its directory APPENDED to sys.path, take the last callable defined."""
    sys.path.append(str(entry.parent))
    env: dict = {}
    exec(compile(entry.read_text(encoding="utf-8"), str(entry), "exec"), env)  # noqa: S102
    callables = [v for v in env.values() if callable(v)]
    return callables[-1], env


def cmd_verify(args) -> int:
    probe_timeout = float(args.act_timeout)
    if probe_timeout == 1.0:
        raise SystemExit("--act-timeout must differ from the engine default (1) or the check proves nothing")
    ok = True

    # ---- path 1: the real kaggle_environments loader + a real episode
    fn, glb = _load_like_kaggle(MILLRACE_ENTRY)
    argc = fn.__code__.co_argcount
    print(f"[loader] last callable {fn.__name__!r} co_argcount={argc} (need 2)")
    ok &= (argc == 2)

    ke = eng.import_engine()
    env = ke.make(ENV_NAME, configuration={"episodeSteps": args.steps, "seed": 0,
                                           "actTimeout": probe_timeout}, debug=False)
    env.run([fn, "starter"])
    diag = dict(glb["_AGENT"].diag)
    print(f"[loader] chassis read act_timeout={diag['act_timeout']} deadline_s={diag['deadline_s']} "
          f"turns={diag['turns']} plan_exceptions={diag['plan_exceptions']} "
          f"bank_start={diag['overage_bank_start']} bank_min={diag['overage_bank_min']}")
    ok &= (diag["act_timeout"] == probe_timeout)
    ok &= (diag["overage_bank_start"] is not None)
    ok &= (diag["plan_exceptions"] == 0)

    # ---- path 2: the Pit child harness, read back through MILLRACE_DIAG
    tmp = tempfile.mkdtemp(prefix="millrace-diag-")
    try:
        os.environ["MILLRACE_DIAG"] = tmp
        res = play(MILLRACE_ENTRY, MILLRACE_ENTRY, seed=0, steps=args.steps,
                   cfg_extra={"actTimeout": probe_timeout})
        dumps = sorted(Path(tmp).glob("millrace-diag-*.json"))
        print(f"[pit] {len(dumps)} diag dumps, statuses={res['status']}")
        ok &= (len(dumps) == 2)
        for d in dumps:
            j = json.loads(d.read_text(encoding="utf-8"))
            print(f"[pit] {d.name}: act_timeout={j['act_timeout']} deadline_s={j['deadline_s']} "
                  f"turns={j['turns']} plan_exceptions={j['plan_exceptions']} "
                  f"bank_start={j['overage_bank_start']} bank_min={j['overage_bank_min']}")
            ok &= (j["act_timeout"] == probe_timeout)
            ok &= (j["plan_exceptions"] == 0)
            ok &= (j["overage_bank_start"] is not None)
    finally:
        os.environ.pop("MILLRACE_DIAG", None)
        shutil.rmtree(tmp, ignore_errors=True)

    print("VERIFY", "OK" if ok else "FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------- histogram mode
def _pct(xs, q):
    if not xs:
        return 0.0
    i = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
    return xs[i]


def summarise(durations, act_timeout):
    ds = sorted(durations)
    n = len(ds)
    above = {}
    for f in FRACTIONS:
        thr = f * act_timeout
        above[str(f)] = sum(1 for d in ds if d > thr)
    return {
        "n_turns": n,
        "act_timeout_s": act_timeout,
        "frac_p50": _pct(ds, 0.50) / act_timeout,
        "frac_p90": _pct(ds, 0.90) / act_timeout,
        "frac_p99": _pct(ds, 0.99) / act_timeout,
        "frac_p999": _pct(ds, 0.999) / act_timeout,
        "frac_max": (ds[-1] / act_timeout) if ds else 0.0,
        "ms_p50": round(1000 * _pct(ds, 0.50), 3),
        "ms_p90": round(1000 * _pct(ds, 0.90), 3),
        "ms_p99": round(1000 * _pct(ds, 0.99), 3),
        "ms_p999": round(1000 * _pct(ds, 0.999), 3),
        "ms_max": round(1000 * ds[-1], 3) if ds else 0.0,
        "ms_mean": round(1000 * statistics.fmean(ds), 3) if ds else 0.0,
        "counts_above_frac": above,
        "overage_consumed_s": round(sum(max(0.0, d - act_timeout) for d in ds), 6),
    }


def parse_seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def cmd_histogram(args) -> int:
    e_agent = resolve_entry(args.agent)
    e_opp = resolve_entry(args.opponent)
    seeds = parse_seeds(args.seeds)
    act_timeout = float(args.act_timeout_report)

    diag_dir = tempfile.mkdtemp(prefix="millrace-diag-")
    os.environ["MILLRACE_DIAG"] = diag_dir
    engine_durations, chassis_timings = [], []
    per_game, banks_min, statuses = [], [], []
    exceptions_total, panics = 0, 0
    try:
        for k, seed in enumerate(seeds):
            # alternate seats so both sides of the board are measured
            swap = bool(k % 2)
            e0, e1 = (e_opp, e_agent) if swap else (e_agent, e_opp)
            mine = 1 if swap else 0
            r = play(e0, e1, seed, args.steps)
            d = r["durations"][mine]
            engine_durations.extend(d)
            statuses.append(r["status"][mine])
            obs_bank = r["overage_obs"][mine]
            if obs_bank:
                banks_min.append(min(obs_bank))
            per_game.append({"seed": seed, "seat": mine, "status": r["status"][mine],
                             "money": r["money"][mine],
                             "max_ms": round(1000 * max(d), 3) if d else 0.0,
                             "overage_s": round(sum(max(0.0, x - act_timeout) for x in d), 6)})
            print(f"  seed {seed} seat {mine}: max {per_game[-1]['max_ms']:.1f} ms  "
                  f"money {r['money'][mine]:.0f}  {r['status'][mine]}", flush=True)
        for f in sorted(Path(diag_dir).glob("millrace-diag-*.json")):
            j = json.loads(f.read_text(encoding="utf-8"))
            chassis_timings.extend(j.get("timings", []))
            exceptions_total += int(j.get("plan_exceptions", 0))
            panics += int(bool(j.get("panic")))
            if j.get("first_exc"):
                print("PLAN EXCEPTION:", j["first_exc"])
    finally:
        os.environ.pop("MILLRACE_DIAG", None)

    peak_rss_mib = None
    try:
        import resource
        peak_rss_mib = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0, 1)
    except Exception:
        pass

    report = {
        "agent": args.agent, "opponent": args.opponent, "n_games": len(seeds),
        "steps": args.steps, "engine": eng.engine_version(),
        "host": os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME", "?"),
        "engine_clock": summarise(engine_durations, act_timeout),
        "chassis_clock": summarise(chassis_timings, act_timeout) if chassis_timings else None,
        "min_remaining_overage_observed": min(banks_min) if banks_min else None,
        "plan_exceptions_total": exceptions_total,
        "panic_latches": panics,
        "non_done_statuses": [s for s in statuses if s != "DONE"],
        "peak_rss_children_mib": peak_rss_mib,
        "per_game": per_game,
    }
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)

    ec = report["engine_clock"]
    print(f"\nSUMMARY engine-clock: p99 {ec['ms_p99']} ms ({ec['frac_p99']:.5f} x actTimeout), "
          f"max {ec['ms_max']} ms ({ec['frac_max']:.5f} x actTimeout), "
          f"overage consumed {ec['overage_consumed_s']} s, "
          f"plan_exceptions {exceptions_total}, panics {panics}, peak RSS {peak_rss_mib} MiB")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="prove the 2-arg signature receives configuration on both loaders")
    v.add_argument("--act-timeout", type=float, default=7.0, help="non-default probe value")
    v.add_argument("--steps", type=int, default=720)
    v.set_defaults(func=cmd_verify)

    h = sub.add_parser("histogram", help="C-5 latency histogram in units of actTimeout")
    h.add_argument("--agent", required=True)
    h.add_argument("--opponent", required=True)
    h.add_argument("--seeds", default="0-63")
    h.add_argument("--steps", type=int, default=720)
    h.add_argument("--act-timeout-report", type=float, default=1.0,
                   help="the actTimeout the histogram is expressed in (competition value: 1)")
    h.add_argument("--out")
    h.set_defaults(func=cmd_histogram)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
