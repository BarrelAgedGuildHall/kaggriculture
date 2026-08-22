"""Critics for the PPO loop: the scalar control arm and `exp-014`'s factored critic.

`exp-014` (plan.md section 2; id collision closed by the 2026-08-22 halt ruling R-4.0 --
plan.md governs, spec section 8's "P2 full fine-tune" is re-issued as exp-018) changes ONE
thing against `exp-013b`: the critic. The trainable policy parameters, the reward, the PFSP
pool, the temperatures and the gradient-step count are identical, so the comparison is
one-variable and `exp-013b` doubles as the scalar-critic control (ruling R-4.4).

Two mechanisms, one object:

**1. The value head is decomposed over terminal-bank lines** (`lines.py`). The advantage still
uses a single scalar -- `V(s) = sum_k v_k(s)` -- so nothing downstream of the critic changes
and the optimum is untouched. What changes is the *supervision*: instead of one MSE against
one scalar per episode, the trunk sees `N_LINES` MSEs against an exactly-decomposed target
(the lines sum to the bank to the cent, `lines.Accountant.residual`). A scalar critic has to
learn "this state is worth 0.31" from one number; the factored critic is told which lines that
0.31 is made of. `explained_var` per line is the diagnostic that says whether it helped.

**2. Auxiliary supervised heads share the trunk and never touch the value.** Opponent net
sales and pool headroom (S-2 labels) and the NAV(lambda=15) marks are things the critic's
features *should* be predictive of; regressing them adds gradient signal to the trunk without
adding a term to the return. They are excluded from `value()` by construction, so an aux head
cannot leak into the advantage even if it is badly scaled.

Why per-line MSE is a strictly richer supervision of the SAME scalar, not a different
objective: the line targets sum to the scalar target by the engine identity, so the sum of the
per-line regressions is an unbiased estimator of the scalar regression's target. There is no
reweighting of what the critic is asked to predict -- only of how the error is attributed.

Targets are standardised per line before the loss, because the lines differ by more than an
order of magnitude in scale (`cost_labour` is a few thousand dollars against `sell_premium`'s
tens of thousands) and an unstandardised sum-of-MSE would silently become "fit the biggest
line". The predictions are de-standardised before summing, so `value()` is in return units.
"""
from __future__ import annotations

import numpy as np

from .bcpath import N
from .lines import N_LINES

EPS = 1e-6


def _norm(glogits):
    """L2 over a {head: dL/dlogits} dict -- the R-5.4 per-term gradient-norm convention."""
    tot = 0.0
    for g in glogits.values():
        f = np.asarray(g).reshape(-1)
        tot += float(np.dot(f, f))
    return float(np.sqrt(tot))


class ScalarCritic:
    """Train-only value net on the market-turn features. Never exported, so unconstrained.

    Byte-for-byte the behaviour of `ppo.Critic`, which it replaces; it keeps that name's
    semantics so `exp-013b` (the control arm) and `exp-014` differ only by the class chosen.
    """

    kind = "scalar"

    def __init__(self, in_dim, hidden=256, lr=1e-3, seed=0):
        self.net = N.MLP(in_dim, hidden, {"v": 1}, seed=seed)
        self.opt = N.Adam(self.net.params(), lr=lr)
        self.mu = np.zeros(in_dim, np.float32)
        self.sd = np.ones(in_dim, np.float32)

    # `target_lines` / `aux` are accepted and ignored so the two critics are interchangeable at
    # every call site -- exp-013b's control arm must run the SAME loop as exp-014, or the critic
    # is not the one variable it is advertised as.
    aux_dims = {}

    def fit_norm(self, x, target_lines=None, aux=None):
        self.mu = x.mean(0).astype(np.float32)
        self.sd = np.maximum(x.std(0), 1e-3).astype(np.float32)

    def value(self, x):
        return self.net.forward((x - self.mu) / self.sd)["v"][:, 0]

    def update(self, x, target, target_lines=None, aux=None):
        xn = (x - self.mu) / self.sd
        out, cache = self.net.forward(xn, cache=True)
        pred = out["v"][:, 0]
        diff = pred - target
        loss = float((diff ** 2).mean())
        g = (2.0 * diff / len(target)).astype(np.float32)[:, None]
        grads = self.net.backward(cache, {"v": g})
        self.opt.step(self.net.params(), grads)
        return {"critic_loss": loss, "aux_loss": 0.0,
                "gnorm_critic": _norm({"v": g}), "gnorm_critic_aux": 0.0}


class FactoredCritic:
    """Value decomposed over `lines.LINES`, plus auxiliary supervised heads on the trunk.

    `aux_dims` maps an auxiliary head name to its output width, e.g.
    `{"opp_net": 9, "headroom": 9, "nav": 4}`. A head with no target in a given `update` call
    simply contributes no gradient that step -- so a partially-labelled batch is legal and the
    S-2 corpus can be joined on whichever rows it covers.
    """

    kind = "factored"

    def __init__(self, in_dim, hidden=256, lr=1e-3, seed=0, n_lines=N_LINES,
                 aux_dims=None, alpha_aux=0.1):
        self.n_lines = int(n_lines)
        self.aux_dims = dict(aux_dims or {})
        heads = {"v": self.n_lines}
        heads.update(self.aux_dims)
        self.net = N.MLP(in_dim, hidden, heads, seed=seed)
        self.opt = N.Adam(self.net.params(), lr=lr)
        self.alpha_aux = float(alpha_aux)
        self.mu = np.zeros(in_dim, np.float32)
        self.sd = np.ones(in_dim, np.float32)
        # per-line target standardisation, fitted once with the feature norm
        self.line_mu = np.zeros(self.n_lines, np.float32)
        self.line_sd = np.ones(self.n_lines, np.float32)
        self.aux_mu = {k: np.zeros(d, np.float32) for k, d in self.aux_dims.items()}
        self.aux_sd = {k: np.ones(d, np.float32) for k, d in self.aux_dims.items()}

    # -- normalisation --------------------------------------------------------
    def fit_norm(self, x, target_lines=None, aux=None):
        self.mu = x.mean(0).astype(np.float32)
        self.sd = np.maximum(x.std(0), 1e-3).astype(np.float32)
        if target_lines is not None:
            t = np.asarray(target_lines, dtype=np.float64)
            self.line_mu = t.mean(0).astype(np.float32)
            self.line_sd = np.maximum(t.std(0), 1e-3).astype(np.float32)
        for k, v in (aux or {}).items():
            if k not in self.aux_dims or v is None:
                continue
            v = np.asarray(v, dtype=np.float64)
            self.aux_mu[k] = v.mean(0).astype(np.float32)
            self.aux_sd[k] = np.maximum(v.std(0), 1e-3).astype(np.float32)

    # -- forward --------------------------------------------------------------
    def lines(self, x):
        """Per-line values in RETURN units, shape (batch, n_lines)."""
        z = self.net.forward((x - self.mu) / self.sd)["v"]
        return z * self.line_sd + self.line_mu

    def value(self, x):
        """The scalar the advantage uses: the sum of the lines. Aux heads never enter."""
        return self.lines(x).sum(1)

    # -- update ---------------------------------------------------------------
    def update(self, x, target=None, target_lines=None, aux=None):
        """One Adam step. `target_lines` is (batch, n_lines) in return units.

        `target` (the scalar) is accepted and used only when no line targets are supplied,
        so a FactoredCritic degrades to a one-line critic rather than crashing a caller that
        has not been taught about lines yet.
        """
        if target_lines is None:
            if target is None:
                raise ValueError("FactoredCritic.update needs target_lines or target")
            target_lines = np.repeat(np.asarray(target, float)[:, None] / self.n_lines,
                                     self.n_lines, axis=1)
        t = (np.asarray(target_lines, dtype=np.float64) - self.line_mu) / self.line_sd
        n = t.shape[0]
        xn = (x - self.mu) / self.sd
        out, cache = self.net.forward(xn, cache=True)
        pred = out["v"]
        diff = pred - t
        # mean over rows, SUM over lines: each line contributes one full MSE, so adding a line
        # does not shrink the gradient the others already had.
        v_loss = float((diff ** 2).mean(0).sum())
        grads_out = {"v": (2.0 * diff / n).astype(np.float32)}

        aux_loss = 0.0
        for k in self.aux_dims:
            v = (aux or {}).get(k)
            if v is None:
                continue
            a = (np.asarray(v, dtype=np.float64) - self.aux_mu[k]) / self.aux_sd[k]
            ad = out[k] - a
            aux_loss += float((ad ** 2).mean(0).sum())
            grads_out[k] = (self.alpha_aux * 2.0 * ad / n).astype(np.float32)

        grads = self.net.backward(cache, grads_out)
        self.opt.step(self.net.params(), grads)
        return {"critic_loss": v_loss, "aux_loss": aux_loss,
                "gnorm_critic": _norm({"v": grads_out["v"]}),
                "gnorm_critic_aux": _norm({k: g for k, g in grads_out.items() if k != "v"}),
                "line_loss": (diff ** 2).mean(0).round(6).tolist()}

    # -- diagnostics ----------------------------------------------------------
    def explained_var(self, x, target_lines):
        """Per-line and pooled 1 - Var(residual)/Var(target), in return units."""
        p = self.lines(x)
        t = np.asarray(target_lines, dtype=np.float64)
        per = [float(1.0 - np.var(t[:, k] - p[:, k]) / max(1e-12, np.var(t[:, k])))
               for k in range(self.n_lines)]
        tot_t, tot_p = t.sum(1), p.sum(1)
        return {"per_line": [round(v, 4) for v in per],
                "pooled": round(float(1.0 - np.var(tot_t - tot_p)
                                      / max(1e-12, np.var(tot_t))), 4)}


def make_critic(kind, in_dim, **kw):
    if kind == "scalar":
        return ScalarCritic(in_dim, **{k: v for k, v in kw.items()
                                       if k in ("hidden", "lr", "seed")})
    if kind == "factored":
        return FactoredCritic(in_dim, **kw)
    raise ValueError(f"unknown critic kind {kind!r}")
