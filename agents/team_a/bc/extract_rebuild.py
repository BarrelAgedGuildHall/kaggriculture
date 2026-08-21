"""Corpus manifest -> three disjoint training chunks + one shared validation set.

The Team A BC rebuild follows the Pokemon (jun177) ensemble method: three members trained
independently, merged at inference by averaging their action *probabilities*. jun177 got member
diversity from three independent prior training runs; having no such pool, we get it from three
disjoint, distribution-matched data chunks plus three seeds. See
`workshop/deliverables/2026-08-20-team-a-bc-rebuild/method.md`.

This script is deliberately thin: it reuses `extract.extract_episode` unchanged, so training
rows come from exactly the same feature path as before -- including the replay off-by-one and
the no-op relabelling, which are bug fixes, not method choices.

Chunk/split assignment is read from the manifest (`select_corpus.py` owns that decision), never
recomputed here.

    uv run python agents/team_a/bc/extract_rebuild.py \
        --manifest artisans/modeler/bench/rebuild/manifest.csv \
        --replays "$GUILDHALL_WAREHOUSE/datasets/kaggriculture/replays/team-a-rebuild/1.32.7"
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
from extract import ENGINE, extract_episode  # noqa: E402

OUT_DIR = Path("D:/GuildhallData/kaggler/datasets/kaggriculture/bc-rows")

ROW_KEYS = (
    "unit_x", "unit_op", "unit_crop", "unit_item", "unit_qty", "unit_mask",
    "mkt_x", "mkt_multi", "mkt_qty", "mkt_hire", "mkt_mask",
)


def _one(job):
    """Worker: extract a single (episode, seat). Returns (key, arrays, stats, episode_id)."""
    path, seat, key, eid = job
    stats = collections.Counter()
    try:
        got = extract_episode(Path(path), int(seat), stats)
    except Exception as error:  # never let one bad replay kill the run
        stats["extract_exception"] += 1
        stats["exc_" + type(error).__name__] += 1
        got = None
    return key, got, stats, eid


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--replays", required=True)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--prefix", default="rebuild")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    replays = Path(args.replays)
    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    jobs = []
    missing = 0
    for r in rows:
        path = replays / f"{r['episode_id']}.json"
        if not path.exists():
            missing += 1
            continue
        key = "val" if r["split"] == "validation" else f"chunk{r['chunk']}"
        jobs.append((str(path), r["seat"], key, int(r["episode_id"])))
    if args.limit:
        jobs = jobs[: args.limit]
    if not jobs:
        raise SystemExit("no replays found -- has the corpus finished downloading?")

    print(f"{len(jobs)} trajectories to extract ({missing} manifest rows not yet downloaded)",
          flush=True)

    # The full corpus is ~3.9M unit rows (~2.7 GB just for unit_x), and concatenating it in
    # one go peaks at roughly double that. Extract one group at a time, stage it to an
    # uncompressed temp file and free it; assemble the final files in a second pass once the
    # corpus-wide slot statistics are known.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = out_dir / f".{args.prefix}-staging"
    staging.mkdir(parents=True, exist_ok=True)

    episodes: dict[str, list] = collections.defaultdict(list)
    stats = collections.Counter()
    failures = 0
    t0 = time.perf_counter()
    keys = sorted({job[2] for job in jobs})
    for key in keys:
        group_jobs = [j for j in jobs if j[2] == key]
        parts = []
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for _, got, st, eid in pool.map(_one, group_jobs, chunksize=1):
                stats.update(st)
                done += 1
                if got is None:
                    failures += 1
                    print(f"  {key} [{done}/{len(group_jobs)}] {eid} EXTRACT FAILED", flush=True)
                    continue
                parts.append(got)
                episodes[key].append(eid)
                if done % 25 == 0:
                    print(f"  {key} [{done}/{len(group_jobs)}] "
                          f"{time.perf_counter() - t0:.0f}s", flush=True)
        if not parts:
            continue
        stacked = {k: np.concatenate([c[k] for c in parts]) for k in ROW_KEYS}
        del parts
        np.savez(staging / f"{key}.npz", **stacked)
        print(f"  {key}: staged unit={len(stacked['unit_op']):,} "
              f"mkt={len(stacked['mkt_x']):,}", flush=True)
        del stacked

    # Mean emission slot per order type, pooled over the whole corpus: the order in which the
    # teacher places market orders is a real decision (each order spends what the previous
    # ones left), so the policy replays the teacher's ordering rather than inventing one.
    slot_order = []
    for oi in range(F.N_ORDERS):
        n = stats.get("slotn_%d" % oi, 0)
        slot_order.append(stats.get("slotsum_%d" % oi, 0) / n if n else 99.0)
    nh = stats.get("slotn_hire", 0)
    hire_slot = stats.get("slotsum_hire", 0) / nh if nh else 99.0
    clean = {k: v for k, v in stats.items() if not k.startswith(("slotsum_", "slotn_"))}

    val_path = staging / "val.npz"
    if not val_path.exists():
        raise SystemExit("no validation rows extracted")
    with np.load(val_path) as handle:
        val = {k: handle[k] for k in ROW_KEYS}

    written = []
    for key in keys:
        if not key.startswith("chunk"):
            continue
        with np.load(staging / f"{key}.npz") as handle:
            train = {k: handle[k] for k in ROW_KEYS}
        payload = {f"train_{k}": v for k, v in train.items()}
        payload.update({f"val_{k}": v for k, v in val.items()})
        payload["meta"] = np.asarray([json.dumps({
            "teacher": f"rebuild-{key}",  # train.py keys its model card off this
            "corpus": "team-a-bc-rebuild", "engine": ENGINE, "chunk": key,
            "unit_dim": F.UNIT_DIM, "market_turn_dim": F.MARKET_TURN_DIM,
            "episodes_train": sorted(episodes[key]),
            "episodes_val": sorted(episodes["val"]),
            "slot_order": slot_order, "hire_slot": hire_slot,
            "stats": clean,
        })])
        path = out_dir / f"{args.prefix}-{key}.npz"
        np.savez_compressed(path, **payload)
        written.append(path)
        print(f"{path.name}: train unit={len(train['unit_op']):,} mkt={len(train['mkt_x']):,} "
              f"| val unit={len(val['unit_op']):,} mkt={len(val['mkt_x']):,} "
              f"({path.stat().st_size / 1e6:.1f} MB)", flush=True)
        del train, payload

    for leftover in staging.glob("*.npz"):
        leftover.unlink()
    staging.rmdir()

    print(f"\ntrajectories: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(episodes.items())))
    print(f"extraction failures: {failures}")
    print("stats:", dict(sorted(clean.items())))
    print(f"wrote {len(written)} chunk files in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
