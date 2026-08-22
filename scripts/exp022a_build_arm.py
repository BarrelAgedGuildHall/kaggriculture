"""exp-022a -- materialise the `sells_first` arm as a self-contained agent directory.

The flag cannot travel in the environment: `arena/isolation.py` runs each agent in its own
child process, but BOTH children inherit the parent's environment, so `BC_LIQUIDATE_ORDER=1`
would flip the opponent as well and the A/B would measure nothing. The arm therefore carries
the flag as a literal in its OWN `main.py`, which is what the arena snapshots.

Everything else -- policy.py, features.py, net.py and the three .npz members -- is copied
byte-for-byte from `agents/team_a/bc`, so the registered arm differs from the frozen
3-member BC base in exactly one token.

Usage:
    python scripts/exp022a_build_arm.py --order sells_first \
        --out data/results/exp-022a/TeamA-EnsembleBC-022a-sellsfirst
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "agents" / "team_a" / "bc"
NEEDLE = '_LIQ = "buys_first"'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--order", required=True, choices=("buys_first", "sells_first"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SRC, out, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))

    m = out / "main.py"
    s = m.read_text(encoding="utf-8")
    if s.count(NEEDLE) != 1:
        print(f"FATAL: expected exactly one {NEEDLE!r} in main.py, found {s.count(NEEDLE)}",
              file=sys.stderr)
        return 2
    s = s.replace(NEEDLE, f'_LIQ = "{args.order}"')
    m.write_text(s, encoding="utf-8")

    # every other file must be byte-identical to the source tree
    drift = []
    for p in sorted(out.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(out)
        src = SRC / rel
        if not src.exists():
            drift.append(f"extra {rel}")
        elif rel.as_posix() != "main.py" and sha(p) != sha(src):
            drift.append(f"differs {rel}")
    if drift:
        print("FATAL: arm drifted from the source tree: " + "; ".join(drift), file=sys.stderr)
        return 2

    print(f"built {out}  _LIQ={args.order}  "
          f"files={sum(1 for p in out.rglob('*') if p.is_file())}  "
          f"(only main.py differs from {SRC.relative_to(REPO_ROOT).as_posix()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
