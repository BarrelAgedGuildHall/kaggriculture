"""OFFLINE ONLY: numpy helpers for calibrating and auditing the marks.

**This module is never imported by `nav/__init__.py` and never vendored into a submission.**
`scripts/make_submission.py --vendor nav` leaves it out and `tests/test_nav.py` asserts both:
the runtime half of `nav` must stay stdlib-only, because Team B's agents are pure stdlib and
Kaggle's sandbox has no repo to import from.

What lives here: everything that wants numpy -- the calibration tables exp-N1 published
(Test 1 terminal residuals, Test 1b realisation ratios, Test 2 R^2 against the final bank),
the curve-accuracy table the engine-bump tripwire reports, and the trace-marking loop that
S-3 telemetry and exp-014's feature extraction both build on.
"""
from __future__ import annotations

import numpy as np

from .api import Nav, NavConfig
from .rules import TERMINAL_INDEX, TURNS_PER_DAY


# ------------------------------------------------------------------------------- marking
def mark_trace(observations, sample_steps=None, variant: str = "drain", lam=None,
               config: NavConfig | None = None) -> list[dict]:
    """Replay a recorded observation trace and return one `Sheet.as_dict()` per sample point.

    The curves only ever see points up to each sample step, exactly as the online object
    would -- which is what makes an offline table comparable with what the agent saw.
    """
    at = None if sample_steps is None else set(int(s) for s in sample_steps)
    out = []
    for step, sheet in Nav.replay_sheets(observations, at=at, config=config,
                                         variant=variant, lam=lam):
        out.append(sheet.as_dict())
    return out


def day_hour0_steps(days=None) -> list[int]:
    """The exp-N1 sample grid: hour 0 of every day, plus 718 and the terminal 719."""
    days = range(30) if days is None else days
    return [d * TURNS_PER_DAY for d in days] + [TERMINAL_INDEX - 1, TERMINAL_INDEX]


# ------------------------------------------------------------------------------- statistics
def r2(x, y) -> float:
    """R^2 of the OLS fit y ~ x (the exp-N1 Test 2 statistic)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 3 or np.allclose(x, x[0]):
        return float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def pearson(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def terminal_residuals(sheets: list[dict]) -> dict:
    """Test 1: |Phi| at the terminal snapshot, absolutely and relative to the final bank."""
    rows = [s for s in sheets if s["step"] == TERMINAL_INDEX]
    if not rows:
        return {"n": 0}
    phi = np.array([r["phi"] for r in rows], dtype=float)
    cash = np.array([max(1.0, r["cash"]) for r in rows], dtype=float)
    rel = np.abs(phi) / cash
    per_line: dict[str, float] = {}
    for r in rows:
        for name, value in r["lines"].items():
            per_line[name] = per_line.get(name, 0.0) + abs(value)
    return {"n": len(rows), "max_abs": float(np.max(np.abs(phi))),
            "mean_abs": float(np.mean(np.abs(phi))), "max_rel": float(np.max(rel)),
            "mean_rel": float(np.mean(rel)), "violations": int(np.sum(rel >= 0.02)),
            "per_line_abs": per_line}


def realisation_ratio(sheets_at: list[dict], realised: list[float], floor: float = 50.0) -> dict:
    """Test 1b: `realised / Phi(sample)` -- how much of the marked potential the agent
    actually converted. Traces whose Phi is below `floor` are excluded: a near-zero
    denominator is not a ratio."""
    phi = np.array([s["phi"] for s in sheets_at], dtype=float)
    got = np.array(realised, dtype=float)
    keep = phi > floor
    if keep.sum() < 3:
        return {"n": int(keep.sum())}
    ratio = got[keep] / phi[keep]
    return {"n": int(keep.sum()), "mean": float(ratio.mean()), "median": float(np.median(ratio)),
            "pearson": pearson(phi[keep], got[keep]), "mean_phi": float(phi[keep].mean()),
            "mean_realised": float(got[keep].mean())}


def calibration_table(sheets_by_day: dict[int, list[dict]], final_bank: list[float]) -> dict:
    """Test 2: R^2 of `final_bank ~ NAV(day d)` with the mandatory cash-alone control."""
    out = {}
    for day, sheets in sorted(sheets_by_day.items()):
        nav_v = [s["nav"] for s in sheets]
        cash_v = [s["cash"] for s in sheets]
        out[day] = {"r2_nav": r2(nav_v, final_bank), "r2_cash": r2(cash_v, final_bank),
                    "n": len(sheets)}
    return out


# ------------------------------------------------------------------------------- curves
def curve_error_table(curves, true_price_fn, only_unobserved: bool = True) -> dict:
    """`|quote - true|` over each product's observed inventory range.

    `true_price_fn(product, inventory)` is the oracle -- the engine's `market_price` with the
    episode's real params. This is the table the engine-bump tripwire reports (spec T4), and
    it is the only place `nav` is ever compared against params it was not given.
    """
    table = {}
    for product, curve in curves.curves.items():
        seen = curve._observed
        if len(seen) < 2:
            continue
        lo, hi = min(seen), max(seen)
        errs = [abs(curve.quote(inv) - true_price_fn(product, inv))
                for inv in range(lo, hi + 1) if not (only_unobserved and inv in seen)]
        if not errs:
            continue
        a = np.array(errs, dtype=float)
        table[product] = {"n": int(a.size), "median": float(np.median(a)),
                          "p95": float(np.percentile(a, 95)), "max": float(a.max()),
                          "range": (lo, hi), "n_points": curve.n_points,
                          "confidence": curve.confidence}
    return table


def revenue_error_table(curves, oracle_fn, products, inventories, sizes) -> dict:
    """`marginal_revenue` vs a per-unit walk on the true curve, relative. The T4 gate."""
    rows = []
    for p in products:
        for inv in inventories:
            for n in sizes:
                true = float(oracle_fn(p, inv, n))
                got = curves[p].marginal_revenue(inv, n)
                rows.append(abs(got - true) / max(1.0, true))
    a = np.array(rows, dtype=float)
    return {"n": int(a.size), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max())}
