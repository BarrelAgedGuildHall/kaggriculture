"""Run a match on a remote node over plain ssh and import the results back.

Sync is by *content of the working tree* (tracked + untracked-but-not-ignored files, i.e. what
`git ls-files` sees) plus the arena registry and agent snapshots, shipped as one tar.gz over
scp -- the node does not need credentials for the private GitHub repo (workflows-box has none).
On the node: `~/kaggri-arena/repo` is (re)populated, `uv sync --frozen` builds/reuses the venv
(uv fetches Python 3.11 itself), the match runs with `--db data/arena/runs/<run_id>.sqlite`, and
that file is copied back and merged into the local store by game_id.

Node prerequisites: ssh reachable in BatchMode (key auth), `curl` or `uv` on PATH (the bootstrap
installs uv into ~/.local/bin if missing). Crash replays stay on the node (data/arena/replays/).
"""
from __future__ import annotations

import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

from .paths import AGENTS_DIR, ARENA_DIR, REGISTRY_PATH, REPO_ROOT, RUNS_DIR, ensure_dirs

REMOTE_DIR = "~/kaggri-arena/repo"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30"]


def _sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, **kw)


def _tracked_files() -> list[str]:
    out = _sh(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=REPO_ROOT,
              capture_output=True, check=True).stdout
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def build_bundle(dest: Path) -> Path:
    """tar.gz of the working tree (git-visible files) + data/arena/registry.json + data/arena/agents/."""
    files = _tracked_files()
    with tarfile.open(dest, "w:gz") as tar:
        for rel in files:
            p = REPO_ROOT / rel
            if p.is_file():
                tar.add(p, arcname=rel)
        if REGISTRY_PATH.exists():
            tar.add(REGISTRY_PATH, arcname="data/arena/registry.json")
        if AGENTS_DIR.exists():
            tar.add(AGENTS_DIR, arcname="data/arena/agents")
    return dest


BOOTSTRAP = r"""
set -e
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "[node] installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
fi
mkdir -p {dir}
cd {dir}
tar xzf {bundle}
rm -f {bundle}
echo "[node] $(hostname) cpus=$(nproc) uv=$(uv --version)"
uv sync --frozen --no-dev -q
echo "[node] venv ready: $(uv run python -c 'import kaggle_environments as k; print(k.__version__)')"
"""


def run_remote(args) -> int:
    node = args.node
    ensure_dirs()
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        bundle = build_bundle(Path(td) / f"arena-{run_id}.tar.gz")
        size_mb = bundle.stat().st_size / 1e6
        print(f"[local] bundle {bundle.name} {size_mb:.1f} MB -> {node}", flush=True)
        remote_bundle = f"/tmp/{bundle.name}"
        r = _sh(["scp", *SSH_OPTS, str(bundle), f"{node}:{remote_bundle}"])
        if r.returncode != 0:
            print("scp failed", file=sys.stderr)
            return r.returncode
    script = BOOTSTRAP.format(dir=REMOTE_DIR, bundle=remote_bundle)
    # bytes + explicit LF: in text mode Windows would turn the newlines into CRLF and bash chokes
    r = subprocess.run(["ssh", *SSH_OPTS, node, "bash", "-s"], input=script.replace("\r\n", "\n").encode())
    if r.returncode != 0:
        print("remote bootstrap failed", file=sys.stderr)
        return r.returncode

    # Rebuild the match command for the node (same arguments, remote db path, remote workers).
    argv = ["match", *args.agents, "--mode", args.mode, "--run-id", run_id,
            "--db", f"data/arena/runs/{run_id}.sqlite", "--keep-replays", args.keep_replays,
            "--steps", str(args.steps), "--game-timeout", str(args.game_timeout)]
    if args.n:
        argv += ["--n", str(args.n)]
    if args.seeds:
        argv += ["--seeds", args.seeds]
    if args.remote_workers:
        argv += ["--workers", str(args.remote_workers)]
    if args.quiet:
        argv += ["--quiet"]
    # The node runs `arena match` itself, so the guard fires there -- and the node-side run also
    # TAKES the lease, against the Pit host's own lease file, which is the one that matters.
    # `cmd_match` skips the local guard entirely for --node runs, so every lease flag has to be
    # forwarded or it silently does nothing (Auditor L-4: --lease-hub was not).
    for flag in ("lease", "lease_owner", "lease_hub", "lease_ttl"):
        val = getattr(args, flag, None)
        if val:
            argv += ["--" + flag.replace("_", "-"), str(val)]
    for flag in ("ignore_lease", "no_lease"):
        if getattr(args, flag, False):
            argv += ["--" + flag.replace("_", "-")]
    from . import engine as eng
    remote_cmd = (f'export PATH="$HOME/.local/bin:$PATH" KAGGRI_ARENA_SHA={shlex.quote(eng.arena_git_sha())}; '
                  f"cd {REMOTE_DIR} && uv run python -m arena {' '.join(shlex.quote(a) for a in argv)}")
    print(f"[node] {remote_cmd}", flush=True)
    r = _sh(["ssh", *SSH_OPTS, node, remote_cmd])
    if r.returncode != 0:
        print(f"remote match exited {r.returncode}", file=sys.stderr)
    # Pull the delta back and merge.
    local_delta = RUNS_DIR / f"{run_id}.sqlite"
    r2 = _sh(["scp", *SSH_OPTS, f"{node}:{REMOTE_DIR}/data/arena/runs/{run_id}.sqlite", str(local_delta)])
    if r2.returncode != 0:
        print("could not fetch the remote results sqlite", file=sys.stderr)
        return r2.returncode or 1
    from . import store
    con = store.connect()
    n = store.import_db(con, local_delta)
    con.close()
    print(f"[local] imported {n} games from {node} ({local_delta}) into {ARENA_DIR / 'arena.sqlite'}; "
          f"total wall {time.perf_counter() - t0:.0f}s", flush=True)
    return r.returncode
