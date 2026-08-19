"""Package the agent exactly as Kaggle expects and verify it runs standalone.

Kaggle accepts (docs/engine/AGENTS.md): a single `main.py` exposing `agent(obs)`, or a
`.tar.gz` with `main.py` at the root plus helper modules / weights. We package whatever is
in `agent/`: if it is only `main.py`, the artefact is `dist/main.py`; otherwise it is
`dist/submission.tar.gz` (flat, `main.py` at root; `__pycache__` and tests excluded).

`--verify` unpacks the artefact into a temporary directory and plays one full episode
(vs the built-in `random` agent) from a FRESH uv virtualenv containing only the pinned
kaggle-environments -- i.e. no access to this repo's modules -- which is what Kaggle's
runner sees. Pass `--no-fresh-venv` for a quick check with the current interpreter.

Usage:
    uv run python scripts/make_submission.py                 # build only
    uv run python scripts/make_submission.py --verify        # build + standalone episode in a fresh venv
    uv run python scripts/make_submission.py --verify --no-fresh-venv

By default this script never submits. `--submit` is a deliberate, separate, guarded path:

    uv run python scripts/make_submission.py --submit --message "<message>" --team teamA

Only the CEO submits to Kaggle (org policy, workshop/notes/team-charter.md). `--submit`
refuses unless the environment variable `KAGGLER_SUBMITTER=ceo` is set (never hardcode or
prompt for this -- it comes from the CEO's own session), always forces `--verify` first
(a broken artefact is never worth burning one of the 5 daily submission slots on) and, on a
successful `kaggle competitions submit`, appends a row to `workshop/notes/submissions.md`
(the ledger). Equivalent to running by hand:
    kaggle competitions submit kaggriculture -f dist/main.py -m "<message>"
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _engine import ENV_NAME, REPO_ROOT, engine_version  # noqa: E402

COMPETITION = "kaggriculture"
LEDGER_COLUMNS = [
    "date (UTC)", "team", "agent@sha", "artefact sha256", "Pit gate", "Auditor",
    "Kaggle sub id", "slot (active?)", "validation result",
    "ladder record (W-L-T, rating) as of", "notes",
]

AGENT_DIR = REPO_ROOT / "agent"
DIST_DIR = REPO_ROOT / "dist"
EXCLUDE_DIRS = {"__pycache__", "tests", ".pytest_cache"}
EXCLUDE_SUFFIXES = {".pyc", ".md"}

RUNNER = r'''
import json, os, sys, statistics
# Run with `python -I` (isolated: no PYTHONPATH, no script dir on sys.path) from a temp
# cwd, so the only importable code is the submission itself plus site-packages --
# the framework appends main.py's directory to sys.path while it execs the file.
os.chdir(sys.argv[1])
from kaggle_environments import make
env = make(sys.argv[2], configuration={"episodeSteps": int(sys.argv[3]), "seed": int(sys.argv[4])}, debug=True)
env.run([os.path.join(sys.argv[1], "main.py"), "random"])
final = env.steps[-1]
durs = [lg[0]["duration"] for lg in env.logs if lg and isinstance(lg[0], dict) and "duration" in lg[0]]
print("VERIFY " + json.dumps({
    "statuses": [s.status for s in final],
    "rewards": [s.reward for s in final],
    "money": [s.observation.farms[i]["money"] for i, s in enumerate(final)],
    "steps": len(env.steps),
    "turn_ms_mean": round(1000 * statistics.fmean(durs), 3) if durs else None,
    "turn_ms_max": round(1000 * max(durs), 3) if durs else None,
}))
'''


def collect_files() -> list[Path]:
    files = []
    for p in sorted(AGENT_DIR.rglob("*")):
        if p.is_dir():
            continue
        if any(part in EXCLUDE_DIRS for part in p.relative_to(AGENT_DIR).parts):
            continue
        if p.suffix in EXCLUDE_SUFFIXES:
            continue
        files.append(p)
    if not (AGENT_DIR / "main.py").exists():
        raise SystemExit(f"{AGENT_DIR / 'main.py'} is missing -- Kaggle requires main.py at the root")
    return files


def build(files: list[Path]) -> Path:
    DIST_DIR.mkdir(exist_ok=True)
    for old in DIST_DIR.glob("*"):
        old.unlink() if old.is_file() else shutil.rmtree(old)
    if len(files) == 1 and files[0].name == "main.py":
        out = DIST_DIR / "main.py"
        shutil.copyfile(files[0], out)
    else:
        out = DIST_DIR / "submission.tar.gz"
        with tarfile.open(out, "w:gz") as tar:
            for f in files:
                tar.add(f, arcname=str(f.relative_to(AGENT_DIR).as_posix()))
    return out


def unpack(artefact: Path, into: Path) -> None:
    if artefact.suffix == ".py":
        shutil.copyfile(artefact, into / "main.py")
    else:
        with tarfile.open(artefact, "r:gz") as tar:
            tar.extractall(into)
    if not (into / "main.py").exists():
        raise SystemExit("artefact does not contain main.py at the root")


def fresh_venv_python(work: Path, version: str) -> str:
    venv = work / "venv"
    subprocess.run(["uv", "venv", "--python", "3.11", str(venv), "-q"], check=True)
    subprocess.run(
        ["uv", "pip", "install", "--python", str(venv), "-q", f"kaggle-environments=={version}"],
        check=True,
    )
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(py)


def run_verify(artefact: Path, seed: int, steps: int, no_fresh_venv: bool) -> tuple[bool, dict | None]:
    version = engine_version()
    work = Path(tempfile.mkdtemp(prefix="kaggriculture-verify-"))
    try:
        sub = work / "submission"
        sub.mkdir()
        unpack(artefact, sub)
        py = sys.executable if no_fresh_venv else fresh_venv_python(work, version)
        print(f"verify   : {'current interpreter' if no_fresh_venv else 'fresh venv'} -> {py}")
        runner = work / "runner.py"
        runner.write_text(RUNNER, encoding="utf-8")
        res = subprocess.run(
            [py, "-I", str(runner), str(sub), ENV_NAME, str(steps), str(seed)],
            text=True, capture_output=True, cwd=str(work),
        )
        lines = [ln for ln in res.stdout.splitlines() if ln.startswith("VERIFY ")]
        if res.returncode != 0 or not lines:
            print(res.stdout[-3000:])
            print(res.stderr[-3000:], file=sys.stderr)
            print("VERIFY FAILED")
            return False, None
        info = json.loads(lines[-1][len("VERIFY "):])
        print(f"result   : {info}")
        ok = info["statuses"][0] == "DONE" and info["steps"] == steps
        print("VERIFY " + ("OK" if ok else "FAILED"))
        return ok, info
    finally:
        shutil.rmtree(work, ignore_errors=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha(short: bool = True) -> str:
    try:
        args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        return subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def find_ledger(explicit: str | None) -> Path:
    """Locate workshop/notes/submissions.md. The competition repo is normally cloned inside
    a Guildhall subsidiary tree (guilds/kaggler/artisans/<op>/bench/repo); walk up from the
    repo root looking for a sibling workshop/ -- works regardless of which operative's bench
    the repo is cloned into, or how deep."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("KAGGLER_LEDGER")
    if env:
        return Path(env)
    here = REPO_ROOT
    for _ in range(8):
        candidate = here / "workshop" / "notes" / "submissions.md"
        if candidate.exists():
            return candidate
        here = here.parent
    raise SystemExit(
        "Cannot locate workshop/notes/submissions.md by walking up from the repo. "
        "Pass --ledger explicitly or set $KAGGLER_LEDGER."
    )


def latest_submission_id(env: dict) -> str:
    """Best-effort: read back the most recent submission id via `kaggle competitions
    submissions`. Returns "" if this fails for any reason -- never blocks the ledger row."""
    try:
        cmd = kaggle_cmd() + ["competitions", "submissions", COMPETITION, "-v"]
        res = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=30)
        if res.returncode != 0:
            return ""
        rows = list(csv.DictReader(res.stdout.splitlines()))
        if not rows:
            return ""
        # Kaggle lists submissions newest-first; the id/ref column name has varied across
        # kaggle CLI versions, so take whichever of the likely id columns is present.
        row0 = rows[0]
        for key in ("ref", "submissionId", "id"):
            if key in row0 and row0[key]:
                return row0[key]
        return ""
    except Exception:
        return ""


def append_ledger_row(ledger: Path, row: dict) -> None:
    new_file = not ledger.exists()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    line = "| " + " | ".join(row[c] for c in LEDGER_COLUMNS) + " |\n"
    if new_file:
        header = "# Submission ledger — Kaggriculture\n\n"
        header += "| " + " | ".join(LEDGER_COLUMNS) + " |\n"
        header += "|" + "|".join(["---"] * len(LEDGER_COLUMNS)) + "|\n"
        ledger.write_text(header + line, encoding="utf-8")
    else:
        with open(ledger, "a", encoding="utf-8") as f:
            f.write(line)


def kaggle_cmd() -> list[str]:
    exe = shutil.which("kaggle")
    if exe:
        return [exe]
    return [sys.executable, "-m", "kaggle.cli"]


def do_submit(artefact: Path, args) -> int:
    """Guarded submit path. Never called unless --submit was passed."""
    if os.environ.get("KAGGLER_SUBMITTER") != "ceo":
        raise SystemExit(
            "--submit refused: $KAGGLER_SUBMITTER is not 'ceo'. Only the CEO submits to "
            "Kaggle (workshop/notes/team-charter.md) -- this is a hard guard, not a suggestion."
        )
    if not args.message:
        raise SystemExit("--submit requires --message")
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("KAGGLE_API_TOKEN is not set in the environment (export it in-shell; never write it to a file)")

    ok, info = run_verify(artefact, args.seed, args.steps, args.no_fresh_venv)
    if not ok:
        raise SystemExit("--submit refused: --verify failed on the packaged artefact. Fix it before submitting.")

    env = dict(os.environ)
    if "KAGGLE_CONFIG_DIR" not in env:
        tmp = tempfile.mkdtemp(prefix="nokaggle-")
        env["KAGGLE_CONFIG_DIR"] = tmp

    if args.dry_run:
        print(f"[dry-run] would run: kaggle competitions submit {COMPETITION} -f {artefact} -m {args.message!r}")
        print("[dry-run] would append a ledger row to " + str(find_ledger(args.ledger)))
        return 0

    cmd = kaggle_cmd() + ["competitions", "submit", COMPETITION, "-f", str(artefact), "-m", args.message]
    print("$ kaggle " + " ".join(cmd[1:]))
    res = subprocess.run(cmd, env=env, text=True, capture_output=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        raise SystemExit(f"kaggle competitions submit failed ({res.returncode})")

    sub_id = latest_submission_id(env)
    ledger = find_ledger(args.ledger)
    append_ledger_row(ledger, {
        "date (UTC)": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "team": args.team or "unspecified",
        "agent@sha": f"agent/main.py@{git_sha()}",
        "artefact sha256": sha256_of(artefact),
        "Pit gate": args.pit_gate or "n/a",
        "Auditor": args.auditor or "n/a",
        "Kaggle sub id": sub_id or "unknown (check Kaggle UI)",
        "slot (active?)": "?",
        "validation result": "pending (Kaggle runs a self-play validation episode on upload)",
        "ladder record (W-L-T, rating) as of": "n/a (Ops fills from ingestion)",
        "notes": args.message,
    })
    print(f"ledger   : appended row to {ledger}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verify", action="store_true", help="play one episode with the packaged artefact")
    p.add_argument("--no-fresh-venv", action="store_true", help="verify with the current interpreter instead of a fresh uv venv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=720)
    p.add_argument("--submit", action="store_true", help="submit to Kaggle -- guarded, CEO only (see module docstring)")
    p.add_argument("--message", help="submission message (required with --submit)")
    p.add_argument("--team", help="team label for the ledger row (e.g. teamA/teamB)")
    p.add_argument("--pit-gate", help="Pit gate summary for the ledger row (e.g. a run id or verdict)")
    p.add_argument("--auditor", help="Auditor verdict reference for the ledger row")
    p.add_argument("--ledger", help="override path to workshop/notes/submissions.md")
    p.add_argument("--dry-run", action="store_true", help="with --submit: do everything except call `kaggle competitions submit` and append the ledger row")
    args = p.parse_args(argv)

    files = collect_files()
    artefact = build(files)
    print(f"artefact : {artefact}  ({artefact.stat().st_size} B, {len(files)} file(s): {[f.name for f in files]})")

    if args.submit:
        return do_submit(artefact, args)

    if not args.verify:
        return 0

    ok, _info = run_verify(artefact, args.seed, args.steps, args.no_fresh_venv)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
