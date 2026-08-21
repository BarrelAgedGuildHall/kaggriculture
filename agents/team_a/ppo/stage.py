"""Build a self-contained agent directory from `agents/team_a/bc` + a chosen weight set.

Why staging instead of a second checked-in agent directory: the Pit registers *directories*
and snapshots them, and a PPO run produces a new candidate every 25 iterations. Copying the
current runtime files each time means an evaluated candidate can never be a stale fork of the
policy code -- there is one `policy.py` in the repo and every staged agent is a copy of it
taken at stage time.

Only the four runtime modules plus the weights are copied; `train.py`, `extract*.py`,
`audit.py` and `calibrate_ensemble.py` stay out, which is also the NB-1 packaging recipe, so
a staged directory is what `make_submission.py` would ship.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .bcpath import BC_DIR, WEIGHTS_DIR

RUNTIME_FILES = ("main.py", "policy.py", "features.py", "net.py", "fallback.py")
STAGE_ROOT = Path(__file__).resolve().parents[3] / "data" / "stage"


def stage(name: str, members, root: Path | None = None) -> Path:
    """Copy the runtime modules + `members` (.npz paths) into data/stage/<name>/."""
    out = (root or STAGE_ROOT) / name
    if out.exists():
        shutil.rmtree(out)
    (out / "weights").mkdir(parents=True)
    for f in RUNTIME_FILES:
        shutil.copy2(BC_DIR / f, out / f)
    for m in members:
        shutil.copy2(Path(m), out / "weights" / Path(m).name)
    return out


def stage_pi0(name="teamA-pi0") -> Path:
    """exp-011's stand-in actor: ensemble member 0 alone (spec section 8)."""
    return stage(name, [WEIGHTS_DIR / "rebuild-m0.npz"])


def stage_base(name="teamA-bc-base") -> Path:
    """The shipped 3-member champion, staged so it is compared like for like."""
    return stage(name, sorted(WEIGHTS_DIR.glob("*.npz")))


def stage_mirror(src_members, name) -> Path:
    """A behaviourally identical twin under a different id, for the mirror self-check.

    The arena identifies a directory agent by a content hash, and `_pair_jobs` collapses
    self-play (one game per seed, no swap) when both ids are equal -- so a byte-identical
    copy cannot be played against itself. A single extra comment line changes the id and
    nothing else. The check it enables is not ceremony: two identical agents must score
    exactly 50.0% with banks equal to the unit, and an *unexpected* exact 50.0% elsewhere is
    the tell for an accidental mirror in the harness.
    """
    out = stage(name, src_members)
    main = out / "main.py"
    main.write_text(main.read_text(encoding="utf-8") +
                    "\n# mirror twin: identical behaviour, different arena id (harness self-check)\n",
                    encoding="utf-8")
    return out
