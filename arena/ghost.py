"""Replay ghosts: a resident that replays one seat's recorded action sequence from an official
Kaggle episode (data/replays/official/<episode>.json).

Why this is legitimate: the two farms are independent except through the shared market, so a
recorded top-50 trajectory stays largely legal whoever the opponent is (the market prices it
sees differ, so its SELL/BUY orders fill at other prices -- that is the approximation).

LIMITATIONS (read before trusting a ghost number):
* A ghost is open-loop. It cannot react to you; it replays what it did. It is a strength
  proxy for "a top agent's production schedule", not a real opponent.
* A ghost is only meaningful on the **exact seed and engine version of its replay**: the town's
  shop draw and weeds are seeded, and the recorded plan assumed that draw. The scheduler
  therefore forces the replay's seed for any match involving a ghost (so a "screen" vs one
  ghost is 2 games: one per seat). It refuses to run a ghost on another engine version.
* Seat swap is allowed but approximate: the engine's seat asymmetry is that `_end_of_day` draws
  weed-spawn randomness sequentially per farm index from one shared RNG (Auditor finding,
  2026-08-19), so swapping seats changes each farm's weed rolls. The ghost was validated on its
  recorded seat only.
* Validation at registration: the ghost is replayed on its seed against the ghost of its
  recorded partner; its final bank must be within +-15 % of the recorded bank (with the pinned
  engine it reproduces exactly, see scripts/replay_check.py). Ghosts that drift more are refused.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import engine as eng
from . import registry
from .paths import AGENTS_DIR, ENV_NAME, OFFICIAL_REPLAYS, ensure_dirs

PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}
MAX_DRIFT = 0.15


class GhostAgent:
    """Callable agent: returns the recorded action for the current observation step."""

    def __init__(self, actions: list, seat: int):
        self.actions = actions
        self.seat = seat

    def __call__(self, obs, configuration=None):  # framework passes both for non-function callables
        step = int(getattr(obs, "step", 0) or 0)
        # steps[t][seat].action is the action that *produced* state t, i.e. the one taken
        # when the observation's step was t-1.
        idx = step + 1
        if 0 <= idx < len(self.actions) and self.actions[idx] is not None:
            return self.actions[idx]
        return PASS_ACTION


def _load_replay(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ghost_spec_from_replay(replay: dict, seat: int, episode: str) -> dict:
    steps = replay["steps"]
    actions = []
    recorded_statuses = []
    for t in range(len(steps)):
        entry = steps[t][seat]
        act, status = entry.get("action"), entry.get("status")
        recorded_statuses.append(status)
        actions.append(act if status not in ("TIMEOUT", "ERROR", "INVALID") else None)
    info = replay.get("info") or {}
    final = steps[-1]
    banks = [float(final[i]["observation"]["farms"][i]["money"]) for i in range(len(final))]
    return {
        "episode": episode,
        "seat": seat,
        "seed": info.get("seed"),
        "configuration": dict(replay.get("configuration") or {}),
        "replay_engine": replay.get("module_version"),
        "recorded_bank": banks[seat],
        "partner_bank": banks[1 - seat],
        "recorded_steps": len(steps),
        "bad_steps": sum(1 for s in recorded_statuses if s in ("TIMEOUT", "ERROR", "INVALID")),
        "actions": actions,
        "partner_actions": [
            (steps[t][1 - seat].get("action") if steps[t][1 - seat].get("status") not in ("TIMEOUT", "ERROR", "INVALID") else None)
            for t in range(len(steps))
        ],
    }


def load_ghost(agent: dict) -> GhostAgent:
    spec = json.loads((AGENTS_DIR / agent["entry"]).read_text(encoding="utf-8"))
    return GhostAgent(spec["actions"], spec["seat"])


def ghost_seed(agent: dict) -> int:
    return int(agent["ghost"]["seed"])


def validate_ghost(spec: dict) -> dict:
    """Replay ghost vs its recorded partner (also a ghost) on the recorded seed; compare banks."""
    ke = eng.import_engine()
    cfg = dict(spec["configuration"])
    cfg["seed"] = spec["seed"]
    env = ke.make(ENV_NAME, configuration=cfg, debug=False)
    me = GhostAgent(spec["actions"], spec["seat"])
    partner = GhostAgent(spec["partner_actions"], 1 - spec["seat"])
    order = [me, partner] if spec["seat"] == 0 else [partner, me]
    env.run(order)
    final = env.steps[-1]
    bank = float(final[spec["seat"]].observation.farms[spec["seat"]]["money"])
    rec = spec["recorded_bank"]
    drift = abs(bank - rec) / max(1.0, abs(rec))
    return {"replayed_bank": bank, "recorded_bank": rec, "drift": round(drift, 4),
            "ok": drift <= MAX_DRIFT, "steps": len(env.steps), "engine": eng.engine_version()}


def register_ghost(replay_path: str | Path, seat: int, name: str | None = None, notes: str = "",
                   skip_validation: bool = False) -> dict:
    p = Path(replay_path)
    if not p.exists():
        cand = OFFICIAL_REPLAYS / p.name
        if not cand.exists():
            cand = OFFICIAL_REPLAYS / f"episode-{p.name}-replay.json"
        if not cand.exists():
            raise FileNotFoundError(f"replay not found: {replay_path}")
        p = cand
    replay = _load_replay(p)
    episode = str((replay.get("info") or {}).get("EpisodeId") or p.stem.replace("episode-", "").replace("-replay", ""))
    if seat not in (0, 1):
        raise ValueError("seat must be 0 or 1")
    spec = _ghost_spec_from_replay(replay, seat, episode)
    if spec["seed"] is None:
        raise ValueError("replay has no info.seed -- a ghost needs the episode seed to be meaningful")
    if spec["replay_engine"] and spec["replay_engine"] != eng.engine_version():
        raise ValueError(f"replay was produced by engine {spec['replay_engine']}, local is {eng.engine_version()}: "
                         "ghosts are only valid on the engine that recorded them")
    validation = None if skip_validation else validate_ghost(spec)
    if validation and not validation["ok"]:
        raise ValueError(f"ghost drifts {100 * validation['drift']:.1f}% from its recorded bank "
                         f"({validation['replayed_bank']:.0f} vs {validation['recorded_bank']:.0f}) -- refused")
    name = name or f"ghost-{episode}-p{seat}"
    ensure_dirs()
    body = json.dumps(spec, sort_keys=True).encode()
    import hashlib
    sha = "g" + hashlib.sha256(body).hexdigest()[:9]
    agent_id = f"{name}@{sha}"
    entry_file = AGENTS_DIR / f"{agent_id}.ghost.json"
    entry_file.write_bytes(body)
    meta = {k: v for k, v in spec.items() if k not in ("actions", "partner_actions", "configuration")}
    meta["validation"] = validation
    meta["limitation"] = (f"valid only on seed {spec['seed']} / engine {eng.engine_version()}; open-loop replay of the "
                          f"recorded p{seat}; validated on seat {seat} only")
    entry = {
        "id": agent_id, "name": name, "sha": sha, "team": "resident", "kind": "ghost",
        "entry": entry_file.name, "source": str(p),
        "engine": eng.engine_version(), "engine_sha": eng.engine_sha(),
        "registered_at": registry._now(), "notes": notes, "ghost": meta, "active": False,
    }
    return registry.register_entry(entry)
