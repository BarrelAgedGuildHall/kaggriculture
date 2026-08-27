# Package-bound trace retention

`scripts/trace_pipeline.py` is neutral measurement infrastructure. It does not interpret,
classify, replace, or score actions. For each acting turn and both seats, it retains the
exact runtime observation immediately before the agent call and the exact JSON action the
agent emitted. A call fault is retained with a deterministic message hash and a null action;
the episode then stops before the engine transition.

## One-command pipeline

```bash
uv run python scripts/trace_pipeline.py run \
  --config trace_retention/configs/development.json \
  --out data/traces/cycle05-development --workers 8
```

The command snapshots every file/directory package, binds every built-in to the full pinned
engine source hash, expands exact seed/seat jobs, captures traces, validates them, and writes
a deterministic timing-free extraction. Output is under git-ignored `data/traces/`.

The pinned framework records one terminal state in `episodeSteps`. Therefore 720 complete
pre-action/action turns require `episodeSteps=721`. The config names `turns: 720`; the plan
binds both values and validation rejects 719-turn normal traces. The engine, agents, and
competition are not modified.

## Bundle contract

- `input-config.json`: normalized requested agents, pairings, seeds, and turns.
- `schema/turn-trace-v1.schema.json`: versioned record schema, copied and SHA-256 bound.
- `packages/`: exact retained file/directory packages; every path, size, and SHA-256 is bound.
- `plan.json`: full engine identity, package identities, configuration, seeds, seats, and
  deterministic 64-hex game identities.
- `traces/<game-id>.jsonl.gz`: timing-bearing raw turn records. Gzip metadata is normalized.
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
uv run python scripts/trace_pipeline.py validate --bundle data/traces/cycle05-development
uv run python scripts/trace_pipeline.py extract --bundle data/traces/cycle05-development
uv run python scripts/trace_pipeline.py reproduce \
  --bundle data/traces/cycle05-development \
  --out data/traces/cycle05-reproduction --workers 8
```

Reproduction loads only the retained packages and the hash-bound plan. It reruns every job,
validates the new bundle, and requires every episode semantic hash plus the semantic manifest
hash to match.
