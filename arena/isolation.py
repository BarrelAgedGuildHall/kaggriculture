"""Hard per-agent module isolation: every file agent runs in its own interpreter.

Why (2026-08-19 bug): kaggle-environments loads a file agent by exec'ing it with its
directory on sys.path, INSIDE the worker process. A multi-module submission resolves its
siblings with bare imports (`import policy`) -- correct on Kaggle, where one agent owns the
sandbox -- but in a shared worker the first such import wins `sys.modules` for the whole
process, so two directory-agents with colliding module names were silently played as the
SAME policy (byte-identical banks at --workers 1; scheduling-dependent corruption above).
Running each file agent behind a subprocess reproduces Kaggle's actual
one-agent-per-sandbox model and makes the collision impossible.

`SubprocessAgent` is a callable class instance, so the framework's Agent wrapper passes it
(observation, configuration) and measures its wall time exactly as before (now including a
small JSON-IPC overhead, which is the honest number: Kaggle's own runner is out-of-process
too). A child error is re-raised here; the framework prints the traceback into the step's
stderr log, so the arena's existing crash detection and crash-is-a-loss scoring apply
unchanged. `play_game` closes the children after every game -- no state leaks across games.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

CHILD = Path(__file__).resolve().parent / "agent_child.py"
START_TIMEOUT_S = 120.0   # heavy imports (numpy etc.) happen once, on the first call
TURN_MARGIN_S = 30.0      # slack on top of actTimeout + remainingOverageTime


class AgentProcessError(RuntimeError):
    pass


class SubprocessAgent:
    """Callable proxy the framework can play: the agent itself lives in a child process."""

    def __init__(self, entry: Path | str, name: str = "") -> None:
        self.entry = Path(entry)
        self.name = name or self.entry.name
        self.proc: subprocess.Popen | None = None
        self._q: queue.Queue | None = None
        self._err_path: str | None = None

    # -- lifecycle -------------------------------------------------------------------
    def _start(self) -> None:
        fd, self._err_path = tempfile.mkstemp(prefix="agent-stderr-", suffix=".log")
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(CHILD), str(self.entry)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=fd,
            cwd=str(self.entry.parent), text=True, encoding="utf-8")
        os.close(fd)
        self._q = queue.Queue()
        threading.Thread(target=self._reader, args=(self.proc, self._q), daemon=True).start()
        first = self._read(START_TIMEOUT_S)
        if not first.get("ready"):
            err = first.get("error", "") or self._stderr_tail()
            self._kill()
            raise AgentProcessError(f"agent {self.name} failed to load:\n{err}")

    @staticmethod
    def _reader(proc: subprocess.Popen, q: queue.Queue) -> None:
        try:
            for line in proc.stdout:
                q.put(line)
        finally:
            q.put(None)

    def close(self) -> None:
        proc, self.proc = self.proc, None
        if proc is not None:
            try:
                proc.stdin.write(json.dumps({"exit": True}) + "\n")
                proc.stdin.flush()
                proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=3)
            except Exception:
                proc.kill()
                try:
                    proc.wait(timeout=3)
                except Exception:
                    pass
        if self._err_path:
            try:
                os.unlink(self._err_path)
            except OSError:
                pass
            self._err_path = None

    def __del__(self):  # safety net; play_game closes explicitly
        try:
            self.close()
        except Exception:
            pass

    # -- protocol --------------------------------------------------------------------
    def _kill(self) -> None:
        proc, self.proc = self.proc, None
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=3)   # reap: a killed-but-unwaited child is a zombie on POSIX
            except Exception:
                pass

    def _stderr_tail(self, n: int = 2000) -> str:
        try:
            with open(self._err_path, encoding="utf-8", errors="replace") as f:
                return f.read()[-n:]
        except Exception:
            return ""

    def _read(self, timeout: float) -> dict:
        try:
            line = self._q.get(timeout=timeout)
        except queue.Empty:
            self._kill()
            raise AgentProcessError(f"agent {self.name}: no reply within {timeout:.0f}s (child killed)") from None
        if line is None:
            tail = self._stderr_tail()
            self._kill()
            raise AgentProcessError(f"agent {self.name}: child process died\n{tail}")
        try:
            return json.loads(line)
        except ValueError as exc:
            self._kill()
            raise AgentProcessError(f"agent {self.name}: bad protocol line {line[:200]!r}") from exc

    def __call__(self, observation, configuration):
        if self.proc is None:
            self._start()
        try:
            self.proc.stdin.write(json.dumps({"obs": observation, "config": configuration}) + "\n")
            self.proc.stdin.flush()
        except OSError as exc:
            tail = self._stderr_tail()
            self._kill()
            raise AgentProcessError(f"agent {self.name}: write to child failed ({exc!r})\n{tail}") from exc
        budget = TURN_MARGIN_S
        try:
            budget += float(configuration.get("actTimeout", 1) or 1)
            budget += float(observation.get("remainingOverageTime", 60) or 60)
        except Exception:
            budget += 61.0
        reply = self._read(budget)
        if "error" in reply:
            raise AgentProcessError(f"agent {self.name} raised:\n{reply['error']}")
        return reply.get("action")
