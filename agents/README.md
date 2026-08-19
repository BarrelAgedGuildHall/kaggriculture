# agents/ -- frozen generations (the one home for "past agents")

Every agent that is ever measured in the arena lives here, committed, never edited after the
fact. Both teams and the arena registry use this layout -- the Modeler's earlier `gen/` folder
and any per-team scratch copies are superseded by it.

```
agents/A/<name>.py          Team A generations (single-file agents)
agents/A/<name>/main.py     ... or a directory when an agent needs helper files
agents/B/<name>.py          Team B generations
agents/resident/<name>.py   neutral residents that are not engine built-ins (e.g. a PASS file)
agent/main.py               the CURRENT submission candidate (copied from a frozen generation)
```

Rules
- **Freeze = copy + commit.** `agent/main.py` is working state; when a generation is worth
  measuring, copy it to `agents/<team>/<name>.py` (name like `v1`, `v1b`, `v2-fert`) and commit.
  Never edit a frozen file; make a new name.
- **Register it**: `uv run python -m arena register agents/A/v1.py --name v1 --team A --notes "exp-004 market module"`.
  The arena id is `v1@<git short sha>` when the file is committed and clean (else a content
  hash), and the registry keeps its own snapshot under `data/arena/agents/`, so the id is stable
  across clones and nodes.
- The champion pointers (`arena champion set v1 --team A`) and the submission ledger refer to
  these ids; `make_submission.py` should be pointed at the frozen file that was gated, not at
  `agent/main.py`.
- Ghost residents are not files here; they are built from `data/replays/official/` with
  `arena register-ghost`.
