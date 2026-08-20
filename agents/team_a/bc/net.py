"""A minimal multi-head MLP in numpy, used for BOTH training and inference.

Inference inside the Kaggle sandbox is stdlib + numpy, so the policy is trained in numpy
too. That removes the export step entirely -- there is no framework boundary at which a
"correct model behind an incorrect interface" can appear.

Shape: x -> Dense/ReLU -> Dense/ReLU -> one linear head per action head.
"""
from __future__ import annotations

import numpy as np

NEG = -1e9


class MLP:
    def __init__(self, in_dim: int, hidden: int, head_dims: dict, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.head_dims = dict(head_dims)
        self.hidden = hidden
        self.in_dim = in_dim

        def he(a, b):
            return (rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)

        self.W1, self.b1 = he(in_dim, hidden), np.zeros(hidden, np.float32)
        self.W2, self.b2 = he(hidden, hidden), np.zeros(hidden, np.float32)
        self.heads = {k: [he(hidden, d) * 0.1, np.zeros(d, np.float32)]
                      for k, d in head_dims.items()}

    # -- forward ------------------------------------------------------------
    def forward(self, x, cache=False):
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(z2, 0.0)
        out = {k: h2 @ W + b for k, (W, b) in self.heads.items()}
        if cache:
            return out, (x, z1, h1, z2, h2)
        return out

    # -- params -------------------------------------------------------------
    def params(self):
        ps = [self.W1, self.b1, self.W2, self.b2]
        for k in sorted(self.heads):
            ps += self.heads[k]
        return ps

    def backward(self, cache, grads_out):
        """grads_out: {head: dL/dlogits (batch, dim)} -> list of grads matching params()."""
        x, z1, h1, z2, h2 = cache
        gh2 = np.zeros_like(h2)
        head_grads = {}
        for k in sorted(self.heads):
            g = grads_out.get(k)
            W = self.heads[k][0]
            if g is None:
                head_grads[k] = [np.zeros_like(W), np.zeros_like(self.heads[k][1])]
                continue
            head_grads[k] = [h2.T @ g, g.sum(0)]
            gh2 += g @ W.T
        gz2 = gh2 * (z2 > 0)
        gW2, gb2 = h1.T @ gz2, gz2.sum(0)
        gh1 = gz2 @ self.W2.T
        gz1 = gh1 * (z1 > 0)
        gW1, gb1 = x.T @ gz1, gz1.sum(0)
        out = [gW1, gb1, gW2, gb2]
        for k in sorted(self.heads):
            out += head_grads[k]
        return out

    # -- serialisation ------------------------------------------------------
    def state(self):
        d = {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}
        for k, (W, b) in self.heads.items():
            d[f"head_{k}_W"] = W
            d[f"head_{k}_b"] = b
        return d

    @classmethod
    def from_state(cls, d, head_dims):
        m = cls.__new__(cls)
        m.W1, m.b1 = np.asarray(d["W1"]), np.asarray(d["b1"])
        m.W2, m.b2 = np.asarray(d["W2"]), np.asarray(d["b2"])
        m.in_dim, m.hidden = m.W1.shape
        m.head_dims = dict(head_dims)
        m.heads = {k: [np.asarray(d[f"head_{k}_W"]), np.asarray(d[f"head_{k}_b"])]
                   for k in head_dims}
        return m


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads, scale=1.0):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for i, (p, g) in enumerate(zip(params, grads)):
            g = g * scale
            if self.wd and p.ndim == 2:
                g = g + self.wd * p
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            p -= self.lr * (self.m[i] / bc1) / (np.sqrt(self.v[i] / bc2) + self.eps)


def masked_softmax(logits, mask=None):
    z = logits if mask is None else np.where(mask > 0, logits, NEG)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def ce_grad(logits, labels, mask=None):
    """Cross-entropy over rows with label >= 0. Returns (loss_sum, n_valid, dL/dlogits)."""
    p = masked_softmax(logits, mask)
    valid = labels >= 0
    n = int(valid.sum())
    if n == 0:
        return 0.0, 0, np.zeros_like(logits)
    idx = np.nonzero(valid)[0]
    lab = labels[idx]
    loss = -np.log(np.maximum(p[idx, lab], 1e-12)).sum()
    g = np.zeros_like(p)
    g[idx] = p[idx]
    g[idx, lab] -= 1.0
    return float(loss), n, g


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def bce_grad(logits, targets, pos_weight=None):
    """Multi-label BCE. targets are 0/1, same shape as logits.
    Returns (mean loss, dL/dlogits) with the gradient already averaged over the batch."""
    p = sigmoid(logits)
    w = 1.0 if pos_weight is None else (1.0 + targets * (pos_weight - 1.0))
    eps = 1e-9
    loss = -(w * (targets * np.log(p + eps) + (1 - targets) * np.log(1 - p + eps))).mean()
    g = (w * (p - targets)) / logits.shape[0]
    return float(loss), g
