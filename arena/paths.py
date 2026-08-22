"""Repo-relative locations used by the arena. All state is under the git-ignored data/arena/."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OFFICIAL_REPLAYS = DATA_DIR / "replays" / "official"

# ARENA_DIR can be relocated (tests, remote runs) with the KAGGRI_ARENA_DIR env var.
ARENA_DIR = Path(os.environ.get("KAGGRI_ARENA_DIR") or (DATA_DIR / "arena"))
REGISTRY_PATH = ARENA_DIR / "registry.json"
AGENTS_DIR = ARENA_DIR / "agents"
DB_PATH = ARENA_DIR / "arena.sqlite"
REPLAYS_DIR = ARENA_DIR / "replays"
RUNS_DIR = ARENA_DIR / "runs"           # per-run sqlite deltas (remote runs land here)
PROGRESS_DIR = ARENA_DIR / "progress"   # one json per live run, read by `arena serve`
LEADERBOARD_MD = ARENA_DIR / "leaderboard.md"
LEADERBOARD_JSON = ARENA_DIR / "leaderboard.json"
WORKSHOP_LEADERBOARD = REPO_ROOT.parents[3] / "workshop" / "notes" / "leaderboard-internal.md"

ENV_NAME = "kaggriculture"
BUILTIN_AGENTS = ("random", "starter", "pass")

# --------------------------------------------------------------------------- seed bands
# Seed hygiene (strategy section 3): tune/screen on 0..199, confirm on held-out 1000..1199, the
# hub's continuous ladder strictly above both on 2000+.
#
# These are DECLARED, not buried: every gate, screen and pooled comparison names the band it ran
# on, and a band can be rotated or split without editing code. That matters because a single fixed
# confirm band is a shared blind spot -- a result that is a band artefact looks like a result on
# every gate that inherits the constant (2026-08-21: a +17.2pp read that pooled to level over three
# disjoint bands). The doctrine is: cross-family comparisons pool >= 2 disjoint bands and report
# the between-band spread; this table is what makes that cheap to say and to do.
#
# A band is an inclusive (lo, hi) seed range; hi = None means open-ended.
# Override or add bands without touching code:
#     KAGGRI_SEED_BANDS="confirm=3000-3199,confirmB=4000-4199"
DEFAULT_SEED_BANDS: dict[str, tuple[int, int | None]] = {
    "screen": (0, 199),
    "confirm": (1000, 1199),
    "ladder": (2000, None),
}


def _parse_band_spec(text: str) -> tuple[int, int | None]:
    """`a-b` (inclusive) or `a-` / `a+` (open-ended)."""
    s = text.strip().replace("+", "-")
    if "-" not in s:
        raise ValueError(f"seed band must be 'lo-hi' or 'lo-', got {text!r}")
    lo, hi = s.split("-", 1)
    return int(lo), (int(hi) if hi.strip() else None)


def _bands_from_env(base: dict[str, tuple[int, int | None]]) -> dict[str, tuple[int, int | None]]:
    out = dict(base)
    for item in (os.environ.get("KAGGRI_SEED_BANDS") or "").split(","):
        if not item.strip():
            continue
        name, _, spec = item.partition("=")
        if not spec:
            raise ValueError(f"KAGGRI_SEED_BANDS entry must be name=lo-hi, got {item!r}")
        out[name.strip()] = _parse_band_spec(spec)
    return out


SEED_BANDS = _bands_from_env(DEFAULT_SEED_BANDS)

SCREEN_SEED_BASE = SEED_BANDS["screen"][0]
CONFIRM_SEED_BASE = SEED_BANDS["confirm"][0]
LADDER_SEED_BASE = SEED_BANDS["ladder"][0]
SCREEN_N = 64       # episodes in a screen (each seed is played on both seats)
CONFIRM_N = 400     # episodes in a confirm
DEFAULT_GATE_BAND = "confirm"   # `arena gate --band` default; state it, never assume it


def resolve_band(spec: str) -> tuple[int, int | None]:
    """A band name from SEED_BANDS, or an explicit `lo-hi` / `lo-` range."""
    if spec in SEED_BANDS:
        return SEED_BANDS[spec]
    try:
        return _parse_band_spec(spec)
    except ValueError:
        raise ValueError(f"unknown seed band {spec!r}: name one of "
                         f"{', '.join(sorted(SEED_BANDS))} or give an explicit lo-hi range") from None


def band_seeds(spec: str, count: int) -> list[int]:
    """The first `count` seeds of a band, refusing to run off its end (silently borrowing seeds
    from the next band is how a 'held-out' set stops being held out)."""
    lo, hi = resolve_band(spec)
    if hi is not None and lo + count - 1 > hi:
        raise ValueError(f"band {spec!r} holds {hi - lo + 1} seeds ({lo}-{hi}); {count} were asked for")
    return list(range(lo, lo + count))


def band_of(seed: int) -> str | None:
    """Which declared band a seed falls in (None if it falls in no band)."""
    for name, (lo, hi) in SEED_BANDS.items():
        if seed >= lo and (hi is None or seed <= hi):
            return name
    return None


def bands_of(seeds) -> list[str]:
    """The declared bands a seed collection touches, in band order. Report this with any pooled
    number: '53.0% over 704 paired seeds on three disjoint bands' is a claim about this list."""
    seen = {band_of(s) for s in seeds}
    return [n for n in SEED_BANDS if n in seen] + (["unbanded"] if None in seen else [])


def ensure_dirs() -> None:
    for d in (ARENA_DIR, AGENTS_DIR, REPLAYS_DIR, RUNS_DIR, PROGRESS_DIR):
        d.mkdir(parents=True, exist_ok=True)
