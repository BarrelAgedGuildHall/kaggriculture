"""`nav` -- the shared economic substrate: what a farm is worth, and what is exactly inferable.

Both teams import this one package; **neither team's use of it is the library's business**:

> `nav` never decides anything. It contains no policy, no thresholds, no "should I plant", no
> team-specific defaults. It answers *what is this worth* and *what is exactly inferable*.
> Every judgement about what to do with the number stays in the team's own agent code.

That rule is what lets two competing teams share one file without either being able to steer
the other, and it is the review criterion: a PR to `nav` that adds a decision, a tuned
constant that is not measured at runtime, or a branch on which team is calling, is rejected.

Layout
------
    nav.rules      the engine's RULEBOOK tables (never prices) + the horizon constants
    nav.curves     PriceCurve/CurveSet: price curves FITTED AT RUNTIME + the closed-form integral
    nav.book       DemandBook: exact town drain + the posterior over future shop unlocks
    nav.sheet      Line/Sheet/MarkContext + one pure `mark_*` per plan §1 row
    nav.opponent   OpponentBook: their public marks and their inferable net flow
    nav.lam        LambdaPair/LiveLambda: the price of an action, always an explicit argument
    nav.api        Nav: observe() every turn, sheet()/opponent_sheet()/phi_rel() on demand
    nav.analysis   OFFLINE ONLY (numpy). Not imported here, not vendored into a submission.

Runtime half is stdlib-only
---------------------------
Everything above except `analysis` imports nothing but the standard library -- no numpy, no
pandas, and above all no `kaggle_environments`, which does not exist inside a submission.
`scripts/make_submission.py --vendor nav` copies this package next to `main.py` so the import
statement is identical in the repo and in the bundle; `tests/test_nav.py` asserts both halves
of that (the isolation test and the vendorability test).

!! NEVER BUMP `nav` MID-EXPERIMENT OR DURING A PROMOTION WINDOW !!
------------------------------------------------------------------
A registered arena snapshot hashes **every file** of an agent, so a vendored `nav` is part of
the agent's identity: changing one line here re-ids every agent that vendors it, its gates
have to be re-run, and its ladder rows stop being comparable with the rows measured before
the change. Treat a nav bump exactly like an engine bump -- sequence it BETWEEN experiments,
never inside one, and never while a promotion is being decided (spec §9.3, docs/nav.md).

`SCHEMA_VERSION` bumps whenever a mark's *meaning* changes; `Sheet.as_dict()` carries it, so
any S-3 telemetry row or experiment result can be attributed to the version that produced it.
Registrations should record `nav.SCHEMA_VERSION` and `nav.source_hash()` in the agent's notes.
It lives in `nav.sheet` beside the dataclass that stamps it, so the two cannot drift apart.

What backed a read (schema 2, ruling-05 R-52)
---------------------------------------------
Every read now says what it rested on -- `observed`, `fitted`, `extrapolated`, `base_fallback`
(the side had no fit, so the answer was the flat `base`) or `floor`. `Sheet.provenance` is the
worst class the sheet touched and `Sheet.void` is the gate clause:

> **Any offline projection, back-test or calibration pair whose read touched `base_fallback`
> is VOID** -- excluded from the denominator, reported as a separate count, never scored as a
> miss. A model cannot be refuted on data where it was never consulted. And no irreversible
> action -- a purchase, a plant, a hire -- may be taken on such a read.

`nav` states the classes and reports them; enforcing either clause is the caller's job.
"""
from __future__ import annotations

from .api import Nav, NavConfig, obs_get, step_of                       # noqa: E402
from .book import DemandBook                                            # noqa: E402
from .curves import (CurveSet, EXTRAPOLATION_FACTOR, PROVENANCE_ORDER, PriceCurve,
                     VOID_PROVENANCE, read_is_provisional, read_is_void, shape,
                     shape_integral, worst_provenance)                  # noqa: E402
from .lam import (LAMBDA_CRITIC, LAMBDA_LIQUIDATION, LambdaPair, LiveLambda,
                  fixed_lambda)                                         # noqa: E402
from .opponent import OpponentBook                                      # noqa: E402
from .sheet import (SCHEMA_VERSION, Line, MarkContext, Sheet, VARIANTS, build_sheet,
                    covenant_charge, intangible_diagnostic, mark_animal, mark_carried,
                    mark_cash, mark_crop, mark_feed_liability, mark_fertilizer_entitlement,
                    mark_land, mark_seeds, mark_shed, mark_structures)  # noqa: E402

__all__ = [
    "SCHEMA_VERSION", "source_hash",
    "Nav", "NavConfig", "obs_get", "step_of",
    "CurveSet", "PriceCurve", "shape", "shape_integral",
    "EXTRAPOLATION_FACTOR", "PROVENANCE_ORDER", "VOID_PROVENANCE",
    "read_is_provisional", "read_is_void", "worst_provenance",
    "DemandBook", "OpponentBook",
    "LambdaPair", "LiveLambda", "fixed_lambda", "LAMBDA_CRITIC", "LAMBDA_LIQUIDATION",
    "Line", "Sheet", "MarkContext", "VARIANTS", "build_sheet",
    "mark_cash", "mark_shed", "mark_carried", "mark_seeds", "mark_crop", "mark_animal",
    "mark_fertilizer_entitlement", "mark_feed_liability", "mark_structures", "mark_land",
    "covenant_charge", "intangible_diagnostic",
]

RUNTIME_MODULES = ("__init__.py", "api.py", "book.py", "curves.py", "lam.py", "opponent.py",
                   "rules.py", "sheet.py")


def source_hash() -> str:
    """sha256 over the runtime half, in a fixed order. Record it with every registration: it
    is the half of the agent's identity that lives outside the agent's own directory."""
    import hashlib
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for name in RUNTIME_MODULES:
        path = os.path.join(here, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                h.update(name.encode())
                h.update(f.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]
