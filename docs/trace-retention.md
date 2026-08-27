# Package-bound trace retention

`scripts/trace_pipeline.py` is neutral measurement infrastructure. It does not interpret,
classify, replace, or score actions. For each acting turn and both seats, it retains the
exact runtime observation immediately before the agent call and the exact JSON action the
agent emitted. A call fault is retained with a deterministic message hash and a null action;
the episode then stops before the engine transition. A separate state stream retains every
framework state and the materialized per-seat runtime observations associated with actionable
states.

## One-command pipeline

```bash
uv run python scripts/trace_pipeline.py run \
  --config trace_retention/configs/development.json \
  --out data/traces/cycle05-exact-competition-v2 --workers 8
```

The command snapshots every file/directory package, binds every built-in to the full pinned
engine source hash, expands exact seed/seat jobs, captures traces, validates them, and writes
a deterministic timing-free extraction. Output is under git-ignored `data/traces/`.

The config names the exact engine value `episode_steps: 720`. Under the pinned competition
semantics this produces exactly 720 stored framework states indexed 0 through 719 and exactly
719 paired action calls per seat at steps 0 through 718. The plan derives those call and final
index values from `episode_steps`; it never extends the episode to manufacture another call.
Validation rejects a normal trace with 718 or 720 calls per seat, any missing paired-seat call,
any state/observation/action mismatch, or a terminal state other than index 719 with `DONE/DONE`.
The engine, agents, and competition are not modified.

## Bundle contract

- `input-config.json`: normalized requested agents, pairings, seeds, and episode steps.
- `schema/turn-trace-v2.schema.json`: versioned action-call schema, copied and SHA-256 bound.
- `schema/framework-state-v1.schema.json`: versioned stored-state schema, likewise bound.
- `packages/`: exact retained file/directory packages; every path, size, and SHA-256 is bound.
- `plan.json`: full engine identity, package identities, configuration, seeds, seats, and
  deterministic 64-hex game identities.
- `traces/<game-id>.jsonl.gz`: timing-bearing raw turn records. Gzip metadata is normalized.
- `states/<game-id>.jsonl.gz`: all stored framework states plus the exact materialized runtime
  observations for actionable states; the terminal state's runtime observation is null.
- `extracted/turns.jsonl.gz`: deterministic canonical state/action/fault records with timing
  removed; observations and actions remain complete.
- `manifest.json`: per-episode completion, faults, timings, exact file hashes, semantic hashes,
  and extraction identity.
- `validation-manifest.json`: the deterministic list of validation checks and bound hashes.

Raw trace hashes bind observed timing. Semantic trace and manifest hashes deliberately omit
timing so an exact rerun can prove state/action determinism without pretending wall-clock
latency is deterministic.

## Validate, extract, reproduce

```bash
uv run python scripts/trace_pipeline.py validate --bundle data/traces/cycle05-exact-competition-v2
uv run python scripts/trace_pipeline.py extract --bundle data/traces/cycle05-exact-competition-v2
uv run python scripts/trace_pipeline.py reproduce \
  --bundle data/traces/cycle05-exact-competition-v2 \
  --out data/traces/cycle05-exact-competition-v2-reproduction --workers 8
```

Reproduction loads only the retained packages and the hash-bound plan. It reruns every job,
validates the new bundle, and requires every normal action-trace semantic hash, every normal
stored-state semantic hash, the stable fault fingerprint, and the semantic manifest hash to
match.
