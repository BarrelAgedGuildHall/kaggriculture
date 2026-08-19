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

This script never submits. Submitting is a deliberate, separate command:
    kaggle competitions submit kaggriculture -f dist/main.py -m "<message>"
"""
from __future__ import annotations

import argparse
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verify", action="store_true", help="play one episode with the packaged artefact")
    p.add_argument("--no-fresh-venv", action="store_true", help="verify with the current interpreter instead of a fresh uv venv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=720)
    args = p.parse_args(argv)

    files = collect_files()
    artefact = build(files)
    print(f"artefact : {artefact}  ({artefact.stat().st_size} B, {len(files)} file(s): {[f.name for f in files]})")

    if not args.verify:
        return 0

    version = engine_version()
    work = Path(tempfile.mkdtemp(prefix="kaggriculture-verify-"))
    try:
        sub = work / "submission"
        sub.mkdir()
        unpack(artefact, sub)
        py = sys.executable if args.no_fresh_venv else fresh_venv_python(work, version)
        print(f"verify   : {'current interpreter' if args.no_fresh_venv else 'fresh venv'} -> {py}")
        runner = work / "runner.py"
        runner.write_text(RUNNER, encoding="utf-8")
        res = subprocess.run(
            [py, "-I", str(runner), str(sub), ENV_NAME, str(args.steps), str(args.seed)],
            text=True, capture_output=True, cwd=str(work),
        )
        lines = [ln for ln in res.stdout.splitlines() if ln.startswith("VERIFY ")]
        if res.returncode != 0 or not lines:
            print(res.stdout[-3000:])
            print(res.stderr[-3000:], file=sys.stderr)
            print("VERIFY FAILED")
            return 1
        info = json.loads(lines[-1][len("VERIFY "):])
        print(f"result   : {info}")
        ok = info["statuses"][0] == "DONE" and info["steps"] == args.steps
        print("VERIFY " + ("OK" if ok else "FAILED"))
        return 0 if ok else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
