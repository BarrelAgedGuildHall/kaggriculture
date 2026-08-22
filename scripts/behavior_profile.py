"""Phase-resolved behavioural profile of one Kaggriculture seat.

Built for the founder's granular review of the competition's #1 team (Ryo Hasegawa, team
16644287, submission 55614463) against Team A's shipped clone. One profiler, two front ends,
so the two sides are measured by literally the same code:

    --replay  re-step a recorded Kaggle replay through the pinned engine with the recorded
              actions (`scripts/replay_check.py`'s recipe: `info={"seed": ...}` + `env.step`),
              and profile BOTH seats.
    --agent   run a local policy on the SAME seed and profile it.

Because the recorded replay carries `info.seed`, "comparable seeds" can be exact: our agent
plays the #1 team's own boards, with the same shop draw and the same weed rolls.

INSTRUMENTATION, identical in both modes (the `tools/pnl_census.py` / `scripts/d1_probe.py`
discipline):
  * `env.interpreter` wrapped -> the farm-object -> player map is rebuilt at the top of every
    step (the farm dicts are rebuilt per step, so a `_process_market` patch would mis-attribute).
  * `_commit_unit` wrapped -> exact per-unit market commits with the realised quote. This is
    the only way to get realised $/unit: the price re-quotes inside a single order, so an order
    of n units does not clear at the posted price.
  * `_apply_unit_action` wrapped with a state fingerprint -> exact executed/no-op per unit
    action (every failure path in that function returns before mutating).
  * `_do_hire` / `_do_buy_land` wrapped -> money delta (both fail silently on no cash).

Nothing takes a decision. In `--replay` mode the recorded actions are fed verbatim.

Usage:
    python scripts/behavior_profile.py --replay <a.json.gz> [...] --out-dir data/results/rank1
    python scripts/behavior_profile.py --agent base --seeds-from data/results/rank1/index.json \
        --out-dir data/results/ours
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL",
            "FERTILIZER"]
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
MOVES = ("NORTH", "SOUTH", "EAST", "WEST")
# Six 5-day phases; day 29 is broken out separately everywhere it matters.
NPH = 6


def phase(day):
    return min(NPH - 1, int(day) // 5)


def _open(p):
    p = str(p)
    if p.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, encoding="utf-8")


class Profiler:
    """Accumulates one profile per seat over a running episode."""

    def __init__(self, K, seats=(0, 1)):
        self.K = K
        self.seats = tuple(seats)
        self.fmap = {}
        self.step = -1
        self.day = 0
        self.hour = 0
        self.acts = {s: Counter() for s in self.seats}        # (phase, op) -> n
        self.acts_noop = {s: Counter() for s in self.seats}
        self.sell = {s: Counter() for s in self.seats}        # (phase, item) -> units
        self.sell_rev = {s: Counter() for s in self.seats}
        self.sell_floored = {s: Counter() for s in self.seats}
        self.sell_hour = {s: Counter() for s in self.seats}
        self.sell_day = {s: Counter() for s in self.seats}    # (day, item) -> units
        self.sell_rev_day = {s: Counter() for s in self.seats}
        self.buy = {s: Counter() for s in self.seats}         # (phase, op, item) -> units
        self.buy_cost = {s: Counter() for s in self.seats}
        self.orders = {s: Counter() for s in self.seats}      # (phase, type) -> n orders
        self.order_size = {s: defaultdict(list) for s in self.seats}
        self.daily = {s: [] for s in self.seats}              # hour-0 snapshots
        self.midday = {s: [] for s in self.seats}             # hour-12 snapshots
        self.market_daily = []
        self.town_daily = []
        self.fails = {s: Counter() for s in self.seats}

    # -- engine hooks -----------------------------------------------------
    def install(self):
        K = self.K
        self._orig = (K._apply_unit_action, K._commit_unit, K._do_hire, K._do_buy_land)

        def apply_unit(farm, private, idx, action, board_size, day, turns_per_day, cap=100):
            p = self.fmap.get(id(farm), -1)
            if p not in self.seats:
                return self._orig[0](farm, private, idx, action, board_size, day,
                                     turns_per_day, cap)
            before = _fp(farm, private, idx)
            out = self._orig[0](farm, private, idx, action, board_size, day, turns_per_day, cap)
            after = _fp(farm, private, idx)
            name = action[0] if isinstance(action, list) and action else "??"
            key = "MOVE" if name in MOVES else name
            ph = phase(self.day)
            if before != after:
                self.acts[p][(ph, key)] += 1
            else:
                self.acts[p][(ph, key)] += 1
                if key != "PASS":
                    self.acts_noop[p][(ph, key)] += 1
            return out

        def commit(op, item, price, farm, private, market, cap=100):
            p = self.fmap.get(id(farm), -1)
            ok = self._orig[1](op, item, price, farm, private, market, cap)
            if p in self.seats:
                ph = phase(self.day)
                if ok:
                    if op == "SELL":
                        self.sell[p][(ph, item)] += 1
                        self.sell_rev[p][(ph, item)] += float(price)
                        # A sale at $1 is the price floor: `_commit_unit` does not even add
                        # the unit to market inventory. Floored share is the sharpest
                        # racing-vs-metering statistic there is.
                        if float(price) <= 1.0:
                            self.sell_floored[p][(ph, item)] += 1
                        self.sell_hour[p][self.hour] += 1
                        self.sell_day[p][(self.day, item)] += 1
                        self.sell_rev_day[p][(self.day, item)] += float(price)
                    else:
                        self.buy[p][(ph, op, item)] += 1
                        self.buy_cost[p][(ph, op, item)] += float(price)
                else:
                    self.fails[p][(op, item)] += 1
            return ok

        def do_hire(farm, private, board_size, mult=K.FARM_HAND_COST_MULT):
            p = self.fmap.get(id(farm), -1)
            b = float(farm["money"])
            out = self._orig[2](farm, private, board_size, mult)
            if p in self.seats:
                spent = b - float(farm["money"])
                ph = phase(self.day)
                if spent > 0:
                    self.buy[p][(ph, "HIRE", "-")] += 1
                    self.buy_cost[p][(ph, "HIRE", "-")] += spent
                else:
                    self.fails[p][("HIRE", "-")] += 1
            return out

        def do_land(farm, board_size):
            p = self.fmap.get(id(farm), -1)
            b = float(farm["money"])
            out = self._orig[3](farm, board_size)
            if p in self.seats:
                spent = b - float(farm["money"])
                ph = phase(self.day)
                if spent > 0:
                    self.buy[p][(ph, "BUY_LAND", "-")] += 1
                    self.buy_cost[p][(ph, "BUY_LAND", "-")] += spent
                else:
                    self.fails[p][("BUY_LAND", "-")] += 1
            return out

        K._apply_unit_action = apply_unit
        K._commit_unit = commit
        K._do_hire = do_hire
        K._do_buy_land = do_land

    def restore(self):
        K = self.K
        (K._apply_unit_action, K._commit_unit, K._do_hire, K._do_buy_land) = self._orig

    def wrap(self, env):
        inner = env.interpreter

        def interpreter(st, env_):
            obs0 = st[0].observation
            if "farms" in obs0 and obs0["farms"]:
                self.fmap.clear()
                for i, f in enumerate(obs0["farms"]):
                    self.fmap[id(f)] = i
                self.step = int(obs0.get("step", self.step + 1))
                self.day = int(obs0.get("day", self.step // 24))
                self.hour = int(obs0.get("hour", self.step % 24))
                # Hour 0 is the clean daily state -- but `_end_of_day` clears the hand list
                # every night, so hands at hour 0 are always zero and "hire depth" is
                # invisible there. Hour 12 is the second snapshot: it is past `_keep_alive`'s
                # trigger and past the morning hiring, so it is where the labour force,
                # the shed and the carried inventory actually sit.
                if self.hour in (0, 12):
                    self._snapshot(st)
                # emitted market orders, before the engine truncates
                for i, s in enumerate(st):
                    if i not in self.seats:
                        continue
                    a = s.action if isinstance(s.action, dict) else {}
                    ph = phase(self.day)
                    for o in (a.get("market") or []):
                        if not isinstance(o, list) or not o:
                            continue
                        self.orders[i][(ph, o[0])] += 1
                        if o[0] in ("SELL", "BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL") \
                                and len(o) >= 3:
                            try:
                                self.order_size[i][(o[0], o[1])].append(int(o[2]))
                            except (TypeError, ValueError):
                                pass
            return inner(st, env_)

        env.interpreter = interpreter

    def _snapshot(self, st):
        obs0 = st[0].observation
        mk = obs0["market"]
        if self.hour == 0:
            self.market_daily.append({
                "day": self.day,
                "price": {p: int(mk["prices"][p]) for p in PRODUCTS},
                "inv": {p: int(mk["inventory"][p]) for p in PRODUCTS},
            })
            town = obs0.get("town") or {}
            self.town_daily.append({"day": self.day,
                                    "shops": sorted(list(town.get("unlocked_shops") or []))})
        for s in self.seats:
            farm = obs0["farms"][s]
            priv = st[s].observation["private"]
            tiles = Counter()
            for row in farm["tiles"]:
                for t in row:
                    if t == "LOCKED":
                        tiles["LOCKED"] += 1
                    elif t is None:
                        tiles["EMPTY"] += 1
                    elif isinstance(t, dict) and t.get("animal"):
                        tiles["animal_" + t["animal"]] += 1
                    elif isinstance(t, dict) and t.get("kind") == "PLANT":
                        tiles["plant_" + t.get("crop", "?")] += 1
                        if t.get("yield_units", 0) > 0:
                            tiles["ripe_" + t.get("crop", "?")] += 1
                    elif isinstance(t, dict):
                        tiles["struct_" + str(t.get("kind"))] += 1
            (self.daily[s] if self.hour == 0 else self.midday[s]).append({
                "day": self.day,
                "money": float(farm["money"]),
                "hands": len(farm["hands"]),
                "quadrants": len(farm.get("unlocked_quadrants") or ["NW"]),
                "tiles": dict(tiles),
                "shed": {k: int(v) for k, v in priv["shed"].items() if v},
                "seeds": {k: int(v) for k, v in priv["seeds"].items() if v},
                "carried": int(sum(sum(i.values()) for i in priv["inventories"])),
            })

    def dump(self, seat):
        def f(c):
            return {"|".join(map(str, k)): v for k, v in c.items()}
        return {
            "acts": f(self.acts[seat]), "acts_noop": f(self.acts_noop[seat]),
            "sell": f(self.sell[seat]), "sell_rev": f(self.sell_rev[seat]),
            "sell_floored": f(self.sell_floored[seat]),
            "sell_hour": {str(k): v for k, v in self.sell_hour[seat].items()},
            "sell_day": f(self.sell_day[seat]), "sell_rev_day": f(self.sell_rev_day[seat]),
            "buy": f(self.buy[seat]), "buy_cost": f(self.buy_cost[seat]),
            "orders": f(self.orders[seat]), "fails": f(self.fails[seat]),
            "order_size": {"|".join(k): [len(v), float(sum(v)) / len(v), max(v)]
                           for k, v in self.order_size[seat].items() if v},
            "daily": self.daily[seat],
            "midday": self.midday[seat],
            "market_daily": self.market_daily,
            "town_daily": self.town_daily,
        }


def _fp(farm, private, idx):
    pos = tuple(farm["farmer"]) if idx == 0 else tuple(farm["hands"][idx - 1])
    invs = private["inventories"]
    inv = invs[idx] if idx < len(invs) else {}
    x, y = pos
    return (pos, repr(sorted(inv.items())), repr(sorted(private["shed"].items())),
            repr(sorted(private["seeds"].items())), repr(farm["tiles"][y][x]),
            float(farm["money"]), len(farm["hands"]))


# --------------------------------------------------------------------------- front ends
def _replay_job(job):
    from arena import engine as arena_engine
    ke = arena_engine.import_engine()
    from kaggle_environments.envs.kaggriculture import kaggriculture as K
    path = Path(job["path"])
    replay = json.load(_open(path))
    steps = replay["steps"]
    cfg = dict(replay["configuration"])
    info = replay.get("info") or {}
    seed = info.get("seed")
    if seed is None:
        return {"path": str(path), "error": "no info.seed"}
    prof = Profiler(K, seats=(0, 1))
    prof.install()
    try:
        env = ke.make("kaggriculture", configuration=cfg, info={"seed": seed}, debug=False)
        env.reset(len(steps[0]))
        prof.wrap(env)
        for t in range(1, len(steps)):
            if env.done:
                break
            env.step([steps[t][i].get("action") for i in range(len(steps[t]))])
    finally:
        prof.restore()
    local = [float(s.reward) for s in env.state]
    rec = replay.get("rewards")
    out = {"path": str(path), "episode_id": info.get("EpisodeId"), "seed": seed,
           "engine": replay.get("module_version"),
           "recorded_rewards": rec, "local_rewards": local,
           "reproduced": bool(rec and all(abs(a - b) < 1e-6 for a, b in zip(local, rec))),
           "names": info.get("TeamNames"),
           "seats": {str(s): prof.dump(s) for s in (0, 1)}}
    return out


def _agent_job(job):
    from arena import engine as arena_engine
    ke = arena_engine.import_engine()
    from kaggle_environments.envs.kaggriculture import kaggriculture as K
    sys.path.insert(0, str(REPO_ROOT / "agents" / "team_a" / "bc"))
    import policy as P
    members = job["members"]
    prof = Profiler(K, seats=(0, 1))
    prof.install()
    try:
        env = ke.make("kaggriculture", configuration=dict(job["cfg"]),
                      info={"seed": job["seed"]}, debug=False)
        env.reset(2)
        prof.wrap(env)
        a = P.BCPolicy(members, safety=job.get("safety", 3))
        b = P.BCPolicy(members, safety=job.get("safety", 3))
        env.run([lambda o, c=None: a.act(o), lambda o, c=None: b.act(o)])
    finally:
        prof.restore()
    return {"path": "", "episode_id": job.get("episode_id"), "seed": job["seed"],
            "engine": None, "recorded_rewards": None,
            "local_rewards": [float(s.reward) for s in env.state], "reproduced": True,
            "names": ["ours", "ours"],
            "seats": {str(s): prof.dump(s) for s in (0, 1)}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays", nargs="*", default=[])
    ap.add_argument("--agent", default="")
    ap.add_argument("--seeds-from", default="", help="a directory of replay profiles")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.replays:
        jobs = [{"path": p} for p in args.replays]
        fn = _replay_job
    else:
        members = sorted(str(p) for p in
                         (REPO_ROOT / "agents" / "team_a" / "bc" / "weights").glob("*.npz"))
        src = json.loads(Path(args.seeds_from).read_text(encoding="utf-8"))
        jobs = [{"seed": r["seed"], "cfg": r["cfg"], "members": members,
                 "episode_id": r["episode_id"]} for r in src]
        fn = _agent_job

    ctx = mp.get_context("spawn")
    done = 0
    if args.workers <= 1:
        it = (fn(j) for j in jobs)
        for r in it:
            done += 1
            _write(outdir, r, done, len(jobs))
    else:
        with ctx.Pool(processes=args.workers) as pool:
            for r in pool.imap_unordered(fn, jobs, chunksize=1):
                done += 1
                _write(outdir, r, done, len(jobs))
    return 0


def _write(outdir, r, i, n):
    if "error" in r:
        print(f"  [{i}/{n}] ERROR {r['path']}: {r['error']}", flush=True)
        return
    name = f"{r['episode_id'] or r['seed']}.json"
    (outdir / name).write_text(json.dumps(r), encoding="utf-8")
    print(f"  [{i}/{n}] {name} rewards={r['local_rewards']} reproduced={r['reproduced']}",
          flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
