# `nav` — the shared economic substrate (S-1)

`nav/` turns an observation into a marked balance sheet, a demand book, an opponent book and
the λ pair. **Both teams import the same package.** Spec:
`workshop/deliverables/2026-08-21-balance-sheet-program/s1-navpy-interface.md`.
Validity study it is lifted from: exp-N1 (`artisans/modeler/bench/nav/exp-N1-results.md`).

## The one rule

> **`nav` never decides anything.** No policy, no thresholds, no "should I plant", no
> team-specific defaults, no constant that is not either a rule of the game or measured at
> runtime. It answers *what is this worth* and *what is exactly inferable*.

That is what lets two competing teams share one file without either being able to steer the
other, and it is the review criterion: a PR that adds a decision, a tuned constant, or a
branch on which team is calling is rejected.

## Quick start

```python
import nav

n = nav.Nav()                       # once per episode
n.observe(obs)                      # every turn, cheap (p50 ~0.03 ms)
sheet = n.sheet()                   # on demand      (p50 ~0.3 ms)
sheet.nav, sheet.phi                # cash + marks, and the non-cash potential
n.phi_rel()                         # (nav-cash)_us - (nav-cash)_them
sheet.as_dict()                     # JSON-safe, versioned: the S-3 telemetry row
                                    # (the Pit records these per episode -- docs/pit.md,
                                    #  "NAV telemetry"; `arena nav --calibrate`)
```

The mark variant and λ are **always explicit**:

```python
sheet = n.sheet(variant="drain", lam=nav.LambdaPair(opp=15.0, wage=15.0))
```

| | meaning | when |
|---|---|---|
| `variant="spot"` | every unit at the current quoted price | fastest, most optimistic |
| `variant="walk"` | the closed-form curve integral from the pool's current position | exp-N1's best calibration arm |
| `variant="drain"` | as `walk`, and a future delivery is sold into a pool the town has already eaten | the default, most conservative |
| `lam.opp = 15` | exp-N1 Test 2 arm — predicts the final bank | critic features (exp-014) |
| `lam.opp = 50` | exp-N1 Test 1b arm — what a day of liquidation converts | liquidation accounting |

These are *documented recommendations*, not defaults `nav` applies behind your back —
`nav.LAMBDA_CRITIC` and `nav.LAMBDA_LIQUIDATION` name them so an experiment row can say which
it used.

## What is measured and what is inlined

| | where | why |
|---|---|---|
| price curves | **fitted at runtime** (`nav.curves`) from observed `(inventory, price)` pairs; `base` and `I0` are observed exactly at step 0 | the agent cannot read the curve: `market["params"]` only exists in an overridden episode. This is what makes the sheet survive an engine rebalance. |
| crop/animal/shop tables | inlined in `nav.rules` | rules of the game, not prices, and not inferable from one observation. A test asserts they equal the engine's, so a release that changes one fails loudly. |
| town drain | exact from `obs.town` | lifted from `probe_core._drain_per_day`, already validated |
| future shop unlocks | posterior over the engine's own uniform-with-replacement draw | reported as an option value, never as a decision |
| opponent's crops/animals/cash | marked from `obs.farms[them]`, which is **public for both seats** | this is what makes `phi_rel` computable |
| opponent's shed/seeds | **not marked** — reported as the unmarkable residual | reported, never guessed |

### The fallback ladder (a mark never silently becomes a guess)

* `confidence == 0` on the side a mark needs (no points observed there yet) ⇒ `price_at`
  returns `base` and the mark is flagged `provisional`.
* A partially fitted curve is used as fitted; the fit's RMSE and out-of-sample residual are in
  `CurveSet.diagnostics()`.
* Worth knowing: in a *quiet* episode nobody pushes a pool into glut, so the **above-I0 side
  is often unobserved** and a future sale marked there marks at `base` — optimistically.
  `scripts/nav_tripwire.py` prints exactly which pools that applies to.

## The engine-bump tripwire

Plan §1 row 16, three layers, one command:

```bash
uv run python scripts/nav_tripwire.py          # exit 1 = do not trust a balance sheet number
```

1. `nav.rules` versus the installed engine's tables (exact equality).
2. Fit accuracy over the observed inventory range, on the shipped params **and** on the g4
   perturbation (`agents/team_b/tools/g4_perturb.json`: amplitudes ±30%, MELON's glut shape
   swapped) with nav given no params.
3. `CurveSet.tripwire()`: the median **out-of-sample** prediction error per product, measured
   against each new point *before* it is folded into the fit. It must stay quiet on curves nav
   can re-derive, and the built-in negative control — a market outside the five shape families
   — must make it fire. A tripwire that has never been seen firing is decoration.

Run it after every `kaggle-environments` bump, before re-running any experiment.

## Vendoring — and the rule that goes with it

A Kaggle bundle ships one agent directory and the loader puts **only that directory** on
`sys.path`. So `nav` is copied into the bundle:

```bash
uv run python scripts/make_submission.py --agent-dir agents/team_b/routed --vendor nav --verify
```

`--vendor` leaves out `nav/analysis.py` (the numpy half), refuses any vendored file that
imports numpy/pandas/scipy/torch/kaggle_environments, caps the tree at 200 KB, and `--verify`
imports the package standalone in the fresh venv before playing the episode.

> ### Never bump `nav` mid-experiment or during a promotion window
>
> An arena registration hashes **every file** of the agent, so a vendored `nav` is part of the
> agent's identity. Changing one line here:
> * re-ids every agent that vendors it (`name@sha` changes),
> * invalidates its Pit gates — they must be re-run,
> * makes its ladder rows non-comparable with rows measured before the change.
>
> Treat a nav bump exactly like an engine bump: sequence it **between** experiments, never
> inside one, and never while a promotion is being decided. Record `nav.SCHEMA_VERSION` and
> `nav.source_hash()` in the agent's registry notes at registration time.

`SCHEMA_VERSION` bumps when a mark's *meaning* changes; `Sheet.as_dict()` carries it, so any
telemetry row or experiment result can be attributed to the version that produced it.

## Terminal identity

`H(i) = 719 - i`. Snapshot **719** is terminal — a SELL still executes on step 718 (exp-N1 §1,
measured; the spec's `718` was wrong). Every mark asks what its asset can deliver within `H`,
so at `H = 0` every non-cash line is zero *by the same formula that values it at H = 700*.
There is no `if last_day:` branch anywhere in `nav/sheet.py`, and
`tests/test_nav.py::test_no_terminal_special_casing` fails the suite if one appears — without
it, `Φ(terminal) == 0` would be unfalsifiable rather than measured.

## Tests

```bash
uv run pytest -q tests/test_nav.py
```

Covers: rulebook-vs-engine, shape algebra and its closed-form integrals, the closed form
against the engine's per-unit walk, curve recovery (shipped and perturbed params), the
tripwire and its negative control, `Φ(719) = 0` on a real replayed episode across every
variant and λ arm, the anti-degeneracy guard (`Φ > 0` mid-episode), the static no-terminal-
special-casing check, purity, online/offline replay equality, opponent-sheet equality with
the opponent's own public lines, exact flow inference, the demand book against
`probe_core._drain_per_day` and against the engine's actual drain, λ handling, isolation
(runtime half is stdlib-only), vendorability, and the runtime budget.
