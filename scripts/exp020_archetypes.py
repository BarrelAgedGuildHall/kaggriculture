"""exp-020 -- is the BC corpus a mixture of behaviourally incompatible archetypes?

Ruling: `workshop/deliverables/2026-08-22-bc-representation-review/review.md` section 7, row
exp-020, testing defect R4: jun177's load-bearing corpus filter was 60-card deck-*exact*
compatibility; the rebuild (method.md D1) replaced it with a rating *quality* gate and asserted
the role transfers. If the 548 teacher seats are two or more incompatible strategies, the
per-turn argmax of the blend is an opening no teacher played.

DESIGN

  1. signature   one behavioural vector per teacher seat, read straight off the replay:
                 units SELL-ordered per product (the product mix), seeds bought per crop,
                 animals bought, hires, land, final bank. Clustering runs on the Hellinger
                 (sqrt) transform of the sell-mix simplex, which is the review's "product-mix
                 cluster".
  2. subsample   the same pass extracts the trajectory's rows with the SHIPPED extractor
                 (`agents/team_a/bc/extract.py::extract_episode`, off-by-one and no-op
                 relabelling included) and keeps a fixed random subsample per trajectory, so
                 clusters can be compared on identical machinery.
  3. hold-out    20% of trajectories in EVERY group are withheld before training; their unit
                 rows form the MATCHED STATE POOL -- one common question set that every
                 cluster-conditioned policy is asked.
  4. train       one policy per group, on an equal row budget (capped at the smallest group),
                 equal steps, equal seed, identical architecture to the shipped unit net.
  5. compare     pairwise argmax disagreement and Jensen-Shannon divergence on the matched
                 pool, overall and restricted to value-bearing states.

  6. PLACEBO, and it is the whole experiment. Two nets trained on different random halves of
     ONE distribution also disagree -- finite data, different minima. So the same pipeline is
     run on a random partition with the same group sizes and the same seed, and the reported
     effect is (clustered disagreement - placebo disagreement). Quoting the raw number would
     confirm R4 on any corpus at all.

Pre-registered reading (review section 7): >=2 clusters each >=20% of trajectories with argmax
disagreement >=15% of matched states => R4 confirmed, exp-027 opens.

Usage:
    python scripts/exp020_archetypes.py pool  --manifest <corpus-manifest.csv> \
        --replays <dir> --out data/results/exp-020/pool.npz --workers 6
    python scripts/exp020_archetypes.py analyse --pool data/results/exp-020/pool.npz \
        --out data/results/exp-020/result.json
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
import types
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BC_DIR = REPO_ROOT / "agents" / "team_a" / "bc"
if str(BC_DIR) not in sys.path:
    sys.path.insert(0, str(BC_DIR))

S_UNIT = 1200          # unit rows kept per trajectory
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL",
            "FERTILIZER"]
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
SIG_NAMES = (["sell_" + p for p in PRODUCTS] + ["seed_" + c for c in CROPS]
             + ["animal_" + a for a in ANIMALS] + ["hires", "land", "bank", "plant_ops"]
             + ["plant_" + c for c in CROPS])


def _signature(replay, seat):
    """Behaviour signature from the emitted actions -- what the teacher ASKED for. Emitted
    rather than executed on purpose: this is a probe of the demonstrated *policy*, which is
    what BC clones, not of the market's response to it."""
    sig = Counter()
    steps = replay["steps"]
    for t in range(1, len(steps)):
        act = steps[t][seat].get("action")
        if not isinstance(act, dict):
            continue
        for o in (act.get("market") or []):
            if not isinstance(o, list) or not o:
                continue
            if o[0] == "HIRE":
                sig["hires"] += 1
            elif o[0] == "BUY_LAND":
                sig["land"] += 1
            elif len(o) >= 3:
                try:
                    n = int(o[2])
                except (TypeError, ValueError):
                    continue
                if o[0] == "SELL" and o[1] in PRODUCTS:
                    sig["sell_" + o[1]] += n
                elif o[0] == "BUY_SEED" and o[1] in CROPS:
                    sig["seed_" + o[1]] += n
                elif o[0] == "BUY_ANIMAL" and o[1] in ANIMALS:
                    sig["animal_" + o[1]] += n
        for a in [act.get("farmer")] + list(act.get("hands") or []):
            if isinstance(a, list) and a and a[0] == "PLANT" and len(a) >= 2:
                sig["plant_ops"] += 1
                if a[1] in CROPS:
                    sig["plant_" + a[1]] += 1
    last = steps[-1][0].get("observation") or {}
    farms = last.get("farms") or []
    if len(farms) > seat:
        sig["bank"] = float(farms[seat].get("money", 0.0))
    return np.array([float(sig.get(k, 0.0)) for k in SIG_NAMES], dtype=np.float32)


def _pool_one(job):
    import collections as _c
    import extract as E           # noqa: E402  (agents/team_a/bc on sys.path)
    path = Path(job["path"])
    seat = int(job["seat"])
    try:
        replay = json.load(open(path, encoding="utf-8"))
    except Exception as exc:
        return {"episode": job["episode"], "error": f"load: {exc}"}
    stats = _c.Counter()
    rows = E.extract_episode(path, seat, stats)
    if rows is None:
        return {"episode": job["episode"], "error": "extract returned None"}
    sig = _signature(replay, seat)
    rng = np.random.default_rng(int(job["episode"]))
    n = len(rows["unit_x"])
    idx = rng.permutation(n)[:S_UNIT]
    idx.sort()
    return {
        "episode": job["episode"], "seat": seat, "team": job["team"], "split": job["split"],
        "rating": job["rating"], "sig": sig,
        "unit_x": rows["unit_x"][idx].astype(np.float16),
        "unit_op": rows["unit_op"][idx],
        "unit_crop": rows["unit_crop"][idx],
        "unit_item": rows["unit_item"][idx],
        "unit_qty": rows["unit_qty"][idx],
        "unit_mask": rows["unit_mask"][idx],
        "n_rows_total": n,
    }


def cmd_pool(args):
    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    reps = Path(args.replays)
    jobs = []
    for r in rows:
        p = reps / f"{r['episode_id']}.json"
        if not p.exists():
            continue
        jobs.append({"path": str(p), "seat": int(r["seat"]), "episode": int(r["episode_id"]),
                     "team": r["team_name"], "split": r.get("split", ""),
                     "rating": float(r.get("rating_before") or 0)})
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"{len(jobs)} trajectories on disk (manifest {len(rows)})", flush=True)
    t0 = time.perf_counter()
    if args.workers <= 1:
        out = [_pool_one(j) for j in jobs]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            out = []
            for i, r in enumerate(pool.imap_unordered(_pool_one, jobs, chunksize=2), 1):
                out.append(r)
                if i % 50 == 0:
                    print(f"  [{i}/{len(jobs)}] {time.perf_counter()-t0:.0f}s", flush=True)
    bad = [o for o in out if "error" in o]
    good = [o for o in out if "error" not in o]
    print(f"ok={len(good)} failed={len(bad)} {bad[:3]}", flush=True)
    good.sort(key=lambda o: o["episode"])
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        episode=np.array([o["episode"] for o in good]),
        seat=np.array([o["seat"] for o in good]),
        team=np.array([o["team"] for o in good]),
        split=np.array([o["split"] for o in good]),
        rating=np.array([o["rating"] for o in good]),
        sig=np.stack([o["sig"] for o in good]),
        sig_names=np.array(SIG_NAMES),
        traj_len=np.array([len(o["unit_x"]) for o in good]),
        unit_x=np.concatenate([o["unit_x"] for o in good]),
        unit_op=np.concatenate([o["unit_op"] for o in good]),
        unit_crop=np.concatenate([o["unit_crop"] for o in good]),
        unit_item=np.concatenate([o["unit_item"] for o in good]),
        unit_qty=np.concatenate([o["unit_qty"] for o in good]),
        unit_mask=np.concatenate([o["unit_mask"] for o in good]),
    )
    print(f"wrote {p} ({p.stat().st_size/1e6:.1f} MB) in {time.perf_counter()-t0:.0f}s")
    return 0


# --------------------------------------------------------------------------- clustering
def kmeans(x, k, seed=0, restarts=25, iters=200):
    best = None
    for r in range(restarts):
        rng = np.random.default_rng(seed * 1000 + r)
        c = x[rng.choice(len(x), k, replace=False)].copy()
        lab = None
        for _ in range(iters):
            d = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
            nl = d.argmin(1)
            if lab is not None and (nl == lab).all():
                break
            lab = nl
            for j in range(k):
                m = lab == j
                if m.any():
                    c[j] = x[m].mean(0)
        inertia = float(((x - c[lab]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, lab.copy(), c.copy())
    return best[1], best[2], best[0]


def silhouette(x, lab):
    k = lab.max() + 1
    if k < 2:
        return float("nan")
    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    s = np.zeros(len(x))
    for i in range(len(x)):
        own = lab == lab[i]
        own[i] = False
        a = d[i, own].mean() if own.any() else 0.0
        b = min((d[i, lab == j].mean() for j in range(k) if j != lab[i]), default=0.0)
        s[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
    return float(s.mean())


def js(p, q, eps=1e-12):
    m = 0.5 * (p + q)
    def kl(a, b):
        return np.where(a > eps, a * (np.log(a + eps) - np.log(b + eps)), 0.0).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# --------------------------------------------------------------------------- training
def train_group(x, y, m, xva, yva, mva, heads, seed, hidden, epochs, batch, lr, wd, quiet=True):
    import train as T
    args = types.SimpleNamespace(hidden=hidden, epochs=epochs, batch=batch, lr=lr, wd=wd)
    if quiet:
        buf, sys.stdout = sys.stdout, open(__import__("os").devnull, "w")
    try:
        net, mu, sd, ep = T.train_net("g", x, y, m, xva, yva, mva, heads, T.UNIT_W, args, seed)
    finally:
        if quiet:
            sys.stdout.close()
            sys.stdout = buf
    return net, mu, sd, ep


def probs_of(net, mu, sd, x, mask, batch=16384):
    from net import masked_softmax
    out = []
    for i in range(0, len(x), batch):
        o = net.forward(((x[i:i + batch] - mu) / sd))
        out.append(masked_softmax(o["op"], mask[i:i + batch]))
    return np.concatenate(out)


def run_partition(name, groups, pool, hold, args):
    """groups: list of arrays of trajectory indices (train side). `hold` indexes the matched
    state pool. Returns per-pair disagreement / JS on the matched pool."""
    import features as F
    starts = np.concatenate([[0], np.cumsum(pool["traj_len"])])

    def rows_of(traj_idx):
        sl = [np.arange(starts[t], starts[t + 1]) for t in traj_idx]
        return np.concatenate(sl)

    hold_rows = rows_of(hold)
    hx = pool["unit_x"][hold_rows].astype(np.float32)
    hm = pool["unit_mask"][hold_rows].astype(np.float32)
    hy = pool["unit_op"][hold_rows].astype(np.int32)
    hy_all = {k: pool["unit_" + k][hold_rows].astype(np.int32)
              for k in ("op", "crop", "item", "qty")}

    cap = min(len(rows_of(g)) for g in groups)
    cap = min(cap, args.cap)
    heads = {"op": F.N_OPS, "crop": F.N_CROPS, "item": F.N_ITEMS, "qty": F.N_UQTY}
    P = []
    for gi, g in enumerate(groups):
        r = rows_of(g)
        rng = np.random.default_rng(args.seed + gi)
        r = rng.permutation(r)[:cap]
        r.sort()
        x = pool["unit_x"][r].astype(np.float32)
        y = {k: pool["unit_" + k][r].astype(np.int32) for k in ("op", "crop", "item", "qty")}
        m = {"op": pool["unit_mask"][r].astype(np.float32)}
        t0 = time.perf_counter()
        net, mu, sd, ep = train_group(x, y, m, hx, hy_all, {"op": hm}, heads,
                                      args.seed, args.hidden, args.epochs, args.batch,
                                      args.lr, args.wd)
        P.append(probs_of(net, mu, sd, hx, hm))
        print(f"    [{name}] group {gi}: {len(g)} traj, {cap:,} rows, best epoch {ep}, "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

    # value-bearing = the state's own teacher label is not PASS/move (the exp-012b corrective:
    # accuracy is dominated by routing ties worth $0)
    trivial = set([F.OP_IX["PASS"]] + [F.OP_IX[o] for o in ("NORTH", "SOUTH", "EAST", "WEST")])
    vb = ~np.isin(hy, list(trivial))
    res = {"n_states": int(len(hx)), "n_value_bearing": int(vb.sum()), "cap_rows": int(cap),
           "pairs": []}
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            ai, aj = P[i].argmax(1), P[j].argmax(1)
            dis = ai != aj
            d = js(P[i], P[j])
            res["pairs"].append({
                "a": i, "b": j,
                "disagree": float(dis.mean()),
                "disagree_vb": float(dis[vb].mean()) if vb.any() else float("nan"),
                "js": float(d.mean()),
                "js_vb": float(d[vb].mean()) if vb.any() else float("nan"),
                "acc_a": float((ai == hy).mean()), "acc_b": float((aj == hy).mean()),
            })
    res["mean_disagree"] = float(np.mean([p["disagree"] for p in res["pairs"]]))
    res["mean_disagree_vb"] = float(np.mean([p["disagree_vb"] for p in res["pairs"]]))
    res["mean_js"] = float(np.mean([p["js"] for p in res["pairs"]]))
    res["mean_js_vb"] = float(np.mean([p["js_vb"] for p in res["pairs"]]))
    return res


def cmd_analyse(args):
    d = np.load(args.pool, allow_pickle=True)
    pool = {k: d[k] for k in d.files}
    sig = pool["sig"].astype(np.float64)
    names = list(pool["sig_names"])
    team = pool["team"]
    n = len(sig)
    print(f"{n} trajectories, {pool['unit_x'].shape[0]:,} subsampled unit rows, "
          f"dim {pool['unit_x'].shape[1]}")

    # product mix on the simplex, Hellinger transform
    si = [names.index("sell_" + p) for p in PRODUCTS]
    mix = sig[:, si]
    tot = mix.sum(1, keepdims=True)
    tot[tot == 0] = 1.0
    mix = mix / tot
    X = np.sqrt(mix)

    out = {"n_traj": n, "products": PRODUCTS,
           "mix_mean": mix.mean(0).round(4).tolist(),
           "teams": {t: int(c) for t, c in Counter(team.tolist()).most_common()}}

    print("\ncorpus mean sell-mix: " +
          "  ".join(f"{p}={m:.3f}" for p, m in zip(PRODUCTS, mix.mean(0))))
    print(f"teams: {len(out['teams'])}; top: "
          + ", ".join(f"{t} {c}" for t, c in list(out['teams'].items())[:6]))

    ks = {}
    for k in range(2, 6):
        lab, cen, inertia = kmeans(X, k, seed=args.seed)
        sizes = np.bincount(lab, minlength=k)
        sil = silhouette(X, lab)
        ks[k] = {"labels": lab, "sizes": sizes.tolist(), "silhouette": sil,
                 "inertia": inertia,
                 "centres_mix": (cen ** 2 / np.maximum(1e-12, (cen ** 2).sum(1, keepdims=True))
                                 ).round(3).tolist()}
        print(f"\nk={k}  sizes={sizes.tolist()}  silhouette={sil:.3f}  inertia={inertia:.2f}")
        for j in range(k):
            m = mix[lab == j].mean(0)
            top = sorted(zip(PRODUCTS, m), key=lambda t: -t[1])[:4]
            tt = Counter(team[lab == j].tolist()).most_common(3)
            print(f"   c{j}: {sizes[j]:3d} traj ({100*sizes[j]/n:4.1f}%)  "
                  + " ".join(f"{p}={v:.2f}" for p, v in top)
                  + "   teams: " + ", ".join(f"{a}({b})" for a, b in tt))
    out["kmeans"] = {str(k): {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                              for kk, vv in v.items()} for k, v in ks.items()}

    # ---- the argmax test -------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    k = args.k
    lab = ks[k]["labels"]
    hold_frac = args.hold
    groups, hold = [], []
    for j in range(k):
        idx = np.where(lab == j)[0]
        idx = rng.permutation(idx)
        nh = max(2, int(round(hold_frac * len(idx))))
        hold.append(idx[:nh])
        groups.append(np.sort(idx[nh:]))
    hold = np.sort(np.concatenate(hold))
    print(f"\nmatched pool: {len(hold)} held-out trajectories from all {k} clusters")
    print(f"[clustered k={k}]")
    res_c = run_partition("clustered", groups, pool, hold, args)

    # placebo: same group sizes, random trajectory partition, same hold-out trajectories
    train_idx = np.array(sorted(set(range(n)) - set(hold.tolist())))
    perm = rng.permutation(train_idx)
    sizes = [len(g) for g in groups]
    pgroups, o = [], 0
    for s in sizes:
        pgroups.append(np.sort(perm[o:o + s]))
        o += s
    print(f"[placebo   k={k}]  (same sizes, random partition, same matched pool)")
    res_p = run_partition("placebo", pgroups, pool, hold, args)

    out["matched"] = {"k": k, "hold_traj": len(hold), "clustered": res_c, "placebo": res_p}

    eff = res_c["mean_disagree"] - res_p["mean_disagree"]
    eff_vb = res_c["mean_disagree_vb"] - res_p["mean_disagree_vb"]
    big = [j for j in range(k) if ks[k]["sizes"][j] >= 0.20 * n]
    verdict = (len(big) >= 2 and res_c["mean_disagree"] >= 0.15)
    out["verdict"] = {
        "clusters_ge_20pct": len(big),
        "mean_disagree": res_c["mean_disagree"],
        "placebo_disagree": res_p["mean_disagree"],
        "excess_disagree": eff,
        "mean_disagree_value_bearing": res_c["mean_disagree_vb"],
        "placebo_value_bearing": res_p["mean_disagree_vb"],
        "excess_value_bearing": eff_vb,
        "mean_js": res_c["mean_js"], "placebo_js": res_p["mean_js"],
        "preregistered_R4_confirmed": bool(verdict),
    }
    print(f"\n{'='*74}\nPRE-REGISTERED READING")
    print(f"  clusters >=20% of trajectories : {len(big)} of {k}  (need >=2)")
    print(f"  argmax disagreement, matched   : {100*res_c['mean_disagree']:.2f}%  (need >=15%)")
    print(f"  placebo (same sizes, random)   : {100*res_p['mean_disagree']:.2f}%")
    print(f"  EXCESS over placebo            : {100*eff:+.2f} pp")
    print(f"  value-bearing states           : {100*res_c['mean_disagree_vb']:.2f}%  "
          f"placebo {100*res_p['mean_disagree_vb']:.2f}%  excess {100*eff_vb:+.2f} pp")
    print(f"  mean JS (op)                   : {res_c['mean_js']:.4f}  "
          f"placebo {res_p['mean_js']:.4f}")
    print(f"  => R4 confirmed by the letter of the gate: {verdict}")
    print("=" * 74)

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"wrote {p}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pool")
    a.add_argument("--manifest", required=True)
    a.add_argument("--replays", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--workers", type=int, default=6)
    a.add_argument("--limit", type=int, default=0)
    a.set_defaults(fn=cmd_pool)
    b = sub.add_parser("analyse")
    b.add_argument("--pool", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--k", type=int, default=3)
    b.add_argument("--hold", type=float, default=0.20)
    b.add_argument("--cap", type=int, default=200_000)
    b.add_argument("--hidden", type=int, default=256)
    b.add_argument("--epochs", type=int, default=6)
    b.add_argument("--batch", type=int, default=512)
    b.add_argument("--lr", type=float, default=1e-3)
    b.add_argument("--wd", type=float, default=1e-5)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(fn=cmd_analyse)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
