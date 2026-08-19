# Engine version record

The Kaggriculture engine is the `kaggriculture` environment inside the PyPI package
`kaggle-environments`. We pin it in `pyproject.toml` / `uv.lock`; this file records what
that pin resolves to, so the rulebook copies next to it can be trusted.

| Item | Value |
|---|---|
| `kaggle-environments` (pinned) | **1.32.7** (PyPI release 2026-08-15) |
| Engine file | `<site-packages>/kaggle_environments/envs/kaggriculture/kaggriculture.py` (40,356 B) |
| `kaggriculture.py` sha256 | `bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e` |
| `kaggriculture.json` sha256 | `a82c89c1a2315b93f39775d8e025471a01b738647c9772658368ee6b1b6f4867` |
| `README.md` sha256 | `3081e52baf8eb2da5d861acc63a3636ce29425f6bdb79a67036ba234ac4ade00` |
| `AGENTS.md` sha256 | `e1a80501a7b02a212eaac9370ada4129a64e0ee6cb3cbc790f3d77d22863fe22` |
| Env spec version (`kaggriculture.json` → `version`) | 0.1.0 |
| Recorded | 2026-08-19 by Engineer |

## Evidence that Kaggle runs this version

- Every replay JSON carries `module_version` (the `kaggle-environments` version that produced
  it) and `info.seed`. Episodes from the 2026-08-16 daily dataset (ids 93459920, 93459927,
  93459932) carry `module_version: 1.32.7` — the release of 2026-08-15 — i.e. Kaggle's
  ladder picked up the new release within ~1 day.
- `scripts/replay_check.py` re-steps those three replays through the local 1.32.7 engine with
  the recorded seed and actions: **0 mismatched steps out of 720, identical final rewards**.
- Older replays (`module_version` 1.32.2 from 2026-07-30, 1.32.6 from 2026-08-08) do **not**
  reproduce under 1.32.7: first divergence at step 1, `market/prices/CARROT` 36 vs 35. The
  1.32.6 → 1.32.7 diff changed the market price curves (CARROT/TOMATO/EGG moved to a new
  `hinge` shape function). So the economy can change between releases, mid-competition.
- PyPI 1.32.7 and GitHub `Kaggle/kaggle-environments@master` were byte-identical for all four
  engine files on 2026-08-19 (same sha256 as above).

## Keeping this current

When a new `kaggle-environments` release appears (check `pip index versions kaggle-environments`
or PyPI): bump the pin, `uv lock && uv sync`, re-copy the rulebook files into `docs/engine/`,
update the table above, fetch a few fresh replays (`scripts/fetch_replays.py --daily <date>`)
and re-run `scripts/replay_check.py`. Record the engine version on every experiment row.
