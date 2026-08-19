"""Re-step an official Kaggle replay through the local engine and diff it step by step.

This is the proof that local tuning is trustworthy: if the pinned kaggle-environments
engine, fed the recorded seed and the recorded actions, regenerates every recorded
observation and reward, then the simulator on Kaggle's servers and ours are the same.

Usage:
    uv run python scripts/replay_check.py data/replays/official/episode-93459920-replay.json
    uv run python scripts/replay_check.py data/replays/official/*.json --max-diffs 5

Exit code 0 = every replay reproduced exactly; 1 = at least one mismatch or error.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _engine import ENV_NAME, engine_version, import_engine, to_plain  # noqa: E402

# Fields that legitimately differ between the server run and a local re-step.
IGNORED_OBS_KEYS = {"remainingOverageTime"}


def _diff(a, b, path="", out=None, limit=20):
    """Collect up to `limit` leaf differences between two plain JSON values."""
    if out is None:
        out = []
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}/{k}: missing locally (replay={b[k]!r})")
            elif k not in b:
                out.append(f"{path}/{k}: extra locally (local={a[k]!r})")
            else:
                _diff(a[k], b[k], f"{path}/{k}", out, limit)
            if len(out) >= limit:
                break
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length local={len(a)} replay={len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", out, limit)
            if len(out) >= limit:
                break
    else:
        # Kaggle's JSON may carry 3000 vs 3000.0; treat numerically-equal as equal.
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
            if a != b:
                out.append(f"{path}: local={a!r} replay={b!r}")
        elif a != b:
            out.append(f"{path}: local={a!r} replay={b!r}")
    return out


def _obs_view(state_entry):
    obs = {k: v for k, v in state_entry["observation"].items() if k not in IGNORED_OBS_KEYS}
    return {"observation": obs, "reward": state_entry.get("reward"), "status": state_entry.get("status")}


def _action_for_step(recorded, ke):
    """Recorded action -> what to feed env.step(). Recorded exceptions are not in the
    JSON, only the resulting status; rebuild a matching exception so statuses line up."""
    act, status = recorded.get("action"), recorded.get("status")
    if act is None or status in ("TIMEOUT", "ERROR", "INVALID"):
        if status == "TIMEOUT":
            return ke.errors.DeadlineExceeded("replayed timeout")
        if status == "ERROR":
            return RuntimeError("replayed agent error")
        if status == "INVALID":
            return "this-is-not-a-valid-action"
    return act


def check_replay(path: Path, max_diffs: int, verbose: bool) -> dict:
    ke = import_engine()
    replay = json.load(open(path, encoding="utf-8"))
    steps = replay["steps"]
    cfg = dict(replay["configuration"])
    info = replay.get("info") or {}
    seed = info.get("seed")
    result = {
        "file": str(path),
        "episode_id": info.get("EpisodeId"),
        "replay_module_version": replay.get("module_version"),
        "local_module_version": engine_version(),
        "seed": seed,
        "recorded_steps": len(steps),
        "recorded_rewards": replay.get("rewards"),
    }
    if seed is None:
        result.update(ok=False, error="replay has no info.seed -- cannot re-step deterministically")
        return result

    env = ke.make(ENV_NAME, configuration=cfg, info={"seed": seed}, debug=False)
    env.reset(len(steps[0]))

    mismatched_steps = 0
    first_mismatch = None
    sample_diffs = []

    def compare(t):
        nonlocal mismatched_steps, first_mismatch
        local = to_plain(env.state)
        diffs = []
        for i in range(len(steps[t])):
            diffs += _diff(_obs_view(local[i]), _obs_view(steps[t][i]), path=f"p{i}", limit=max_diffs)
        if diffs:
            mismatched_steps += 1
            if first_mismatch is None:
                first_mismatch = t
                sample_diffs.extend(diffs[:max_diffs])
        return not diffs

    compare(0)
    for t in range(1, len(steps)):
        if env.done:
            result.update(ok=False, error=f"local env finished early at step {t}")
            break
        actions = [_action_for_step(steps[t][i], ke) for i in range(len(steps[t]))]
        env.step(actions)
        ok = compare(t)
        if verbose and not ok and mismatched_steps <= 3:
            print(f"  step {t}: mismatch", file=sys.stderr)

    local_rewards = [s.reward for s in env.state]
    result.update(
        local_steps=len(env.steps),
        local_rewards=local_rewards,
        local_done=env.done,
        mismatched_steps=mismatched_steps,
        first_mismatch_step=first_mismatch,
        sample_diffs=sample_diffs,
    )
    result["ok"] = (
        mismatched_steps == 0
        and len(env.steps) == len(steps)
        and local_rewards == replay.get("rewards")
        and "error" not in result
    )
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("replays", nargs="+", help="replay JSON file(s) or globs")
    p.add_argument("--max-diffs", type=int, default=10, help="sample diffs to print per replay")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    files = []
    for pat in args.replays:
        matches = glob.glob(pat)
        files += [Path(m) for m in matches] if matches else [Path(pat)]

    all_ok = True
    for f in files:
        if not f.exists():
            print(f"MISSING {f}")
            all_ok = False
            continue
        r = check_replay(f, args.max_diffs, args.verbose)
        all_ok &= bool(r.get("ok"))
        verdict = "REPRODUCES" if r.get("ok") else "MISMATCH"
        print(
            f"{verdict}  episode={r.get('episode_id')}  seed={r.get('seed')}  "
            f"steps={r.get('recorded_steps')}/{r.get('local_steps')}  "
            f"rewards replay={r.get('recorded_rewards')} local={r.get('local_rewards')}  "
            f"mismatched_steps={r.get('mismatched_steps')}  "
            f"engine replay={r.get('replay_module_version')} local={r.get('local_module_version')}"
        )
        if r.get("error"):
            print(f"  error: {r['error']}")
        for d in r.get("sample_diffs", []):
            print(f"  first mismatch at step {r['first_mismatch_step']}: {d}")
    print("RESULT", "ALL_REPRODUCE" if all_ok else "SOME_MISMATCH")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
