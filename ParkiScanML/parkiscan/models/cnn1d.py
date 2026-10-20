"""A small 1-D convolutional neural network implemented in NumPy (no deep-learning framework required).

Architecture (for multichannel time windows, e.g. 3-axis accelerometer, 4 s @ 64 Hz):
    Conv1d(C→16, k=9) → ReLU → MaxPool(4) → Conv1d(16→32, k=7) → ReLU → GlobalAvgPool → Dense(32→2) → softmax
Trained with Adam, class-weighted cross-entropy, mini-batches, L2 weight decay, and early stopping on a validation
split. Forward and backward passes use im2col, so training on ~15k windows takes a couple of minutes on a laptop CPU.
Weights save to .npz (portable; inference needs only NumPy).
"""
from __future__ import annotations

import numpy as np


def _im2col(x, k):
    # x: (N, C, L) -> (N, L-k+1, C*k)
    N, C, L = x.shape
    Lo = L - k + 1
    s0, s1, s2 = x.strides
    cols = np.lib.stride_tricks.as_strided(x, shape=(N, Lo, C, k), strides=(s0, s2, s1, s2))
    return cols.reshape(N, Lo, C * k)


class TinyCNN1D:
    def __init__(self, in_ch=3, n_classes=2, c1=16, k1=9, c2=32, k2=7, pool=4, seed=0):
        rng = np.random.default_rng(seed)
        self.cfg = dict(in_ch=in_ch, n_classes=n_classes, c1=c1, k1=k1, c2=c2, k2=k2, pool=pool)
        he = lambda fan_in, shape: rng.normal(0, np.sqrt(2.0 / fan_in), shape).astype(np.float32)
        self.p = {
            "W1": he(in_ch * k1, (in_ch * k1, c1)), "b1": np.zeros(c1, np.float32),
            "W2": he(c1 * k2, (c1 * k2, c2)), "b2": np.zeros(c2, np.float32),
            "W3": he(c2, (c2, n_classes)) * 0.5, "b3": np.zeros(n_classes, np.float32),
        }
        self.mu = None
        self.sd = None

    # ---------------------------------------------------------------- forward / backward
    def _forward(self, x, keep=False):
        c = self.cfg
        X1 = _im2col(x, c["k1"])                       # (N, L1, C*k1)
        Z1 = X1 @ self.p["W1"] + self.p["b1"]          # (N, L1, c1)
        A1 = np.maximum(Z1, 0)
        N, L1, C1 = A1.shape
        Lp = L1 // c["pool"]
        A1r = A1[:, :Lp * c["pool"]].reshape(N, Lp, c["pool"], C1)
        P1 = A1r.max(axis=2)                           # (N, Lp, c1)
        P1t = np.ascontiguousarray(P1.transpose(0, 2, 1))   # (N, c1, Lp)
        X2 = _im2col(P1t, c["k2"])                     # (N, L2, c1*k2)
        Z2 = X2 @ self.p["W2"] + self.p["b2"]
        A2 = np.maximum(Z2, 0)                         # (N, L2, c2)
        G = A2.mean(axis=1)                            # (N, c2)
        logits = G @ self.p["W3"] + self.p["b3"]
        if not keep:
            return logits
        return logits, dict(X1=X1, Z1=Z1, A1r=A1r, P1=P1, X2=X2, Z2=Z2, A2=A2, G=G, L1=L1, Lp=Lp)

    def _backward(self, cache, dlogits):
        c = self.cfg
        g = {}
        N = dlogits.shape[0]
        g["W3"] = cache["G"].T @ dlogits
        g["b3"] = dlogits.sum(0)
        dG = dlogits @ self.p["W3"].T                  # (N, c2)
        L2 = cache["A2"].shape[1]
        dA2 = np.repeat(dG[:, None, :] / L2, L2, axis=1)
        dZ2 = dA2 * (cache["Z2"] > 0)
        g["W2"] = np.einsum("nlk,nlc->kc", cache["X2"], dZ2, optimize=True)
        g["b2"] = dZ2.sum((0, 1))
        dX2 = dZ2 @ self.p["W2"].T                     # (N, L2, c1*k2)
        C1, k2, Lp = c["c1"], c["k2"], cache["Lp"]
        dX2 = dX2.reshape(N, L2, C1, k2)
        dP1t = np.zeros((N, C1, Lp), np.float32)
        for j in range(k2):                            # col2im
            dP1t[:, :, j:j + L2] += dX2[:, :, :, j].transpose(0, 2, 1)
        dP1 = dP1t.transpose(0, 2, 1)                  # (N, Lp, c1)
        A1r = cache["A1r"]
        mask = (A1r == cache["P1"][:, :, None, :])
        mask = mask / np.maximum(mask.sum(axis=2, keepdims=True), 1)
        dA1r = mask * dP1[:, :, None, :]
        L1 = cache["L1"]
        dA1 = np.zeros((N, L1, C1), np.float32)
        dA1[:, :Lp * c["pool"]] = dA1r.reshape(N, Lp * c["pool"], C1)
        dZ1 = dA1 * (cache["Z1"] > 0)
        g["W1"] = np.einsum("nlk,nlc->kc", cache["X1"], dZ1, optimize=True)
        g["b1"] = dZ1.sum((0, 1))
        return g

    # ---------------------------------------------------------------- training
    def _norm(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)

    def fit(self, X, y, epochs=12, batch=128, lr=2e-3, wd=1e-4, val_frac=0.15, patience=3, seed=0, verbose=False):
        """X: (N, C, L) float, y: (N,) int. Returns self (best validation-loss weights kept)."""
        rng = np.random.default_rng(seed)
        X = X - X.mean(axis=2, keepdims=True)           # remove per-window gravity / offset
        self.mu = X.mean(axis=(0, 2), keepdims=True)[0]
        self.sd = X.std(axis=(0, 2), keepdims=True)[0] + 1e-6
        Xn = self._norm(X)
        idx = rng.permutation(len(y))
        nv = int(len(y) * val_frac)
        vi, ti = idx[:nv], idx[nv:]
        counts = np.bincount(y[ti], minlength=self.cfg["n_classes"]).astype(np.float32)
        cw = (counts.sum() / (len(counts) * np.maximum(counts, 1))).astype(np.float32)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v2 = {k: np.zeros_like(v) for k, v in self.p.items()}
        step, best, best_p, bad = 0, np.inf, None, 0
        for ep in range(epochs):
            rng.shuffle(ti)
            for b0 in range(0, len(ti), batch):
                bi = ti[b0:b0 + batch]
                logits, cache = self._forward(Xn[bi], keep=True)
                pr = _softmax(logits)
                oh = np.eye(self.cfg["n_classes"], dtype=np.float32)[y[bi]]
                w = cw[y[bi]][:, None]
                dlog = (pr - oh) * w / w.sum()
                grads = self._backward(cache, dlog)
                step += 1
                for k in self.p:
                    gk = grads[k] + wd * self.p[k]
                    m[k] = 0.9 * m[k] + 0.1 * gk
                    v2[k] = 0.999 * v2[k] + 0.001 * gk * gk
                    mh = m[k] / (1 - 0.9 ** step); vh = v2[k] / (1 - 0.999 ** step)
                    self.p[k] -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
            if nv:
                pv = _softmax(self._forward(Xn[vi]))
                loss = float(-np.mean(np.log(pv[np.arange(nv), y[vi]] + 1e-9) * cw[y[vi]]))
                if verbose:
                    print(f"  epoch {ep + 1}: val loss {loss:.4f}")
                if loss < best - 1e-4:
                    best, best_p, bad = loss, {k: v.copy() for k, v in self.p.items()}, 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_p is not None:
            self.p = best_p
        return self

    def predict_proba(self, X, batch=512):
        X = X - X.mean(axis=2, keepdims=True)
        Xn = self._norm(X)
        out = [_softmax(self._forward(Xn[i:i + batch])) for i in range(0, len(Xn), batch)]
        return np.vstack(out)

    def save(self, path):
        np.savez_compressed(path, mu=self.mu, sd=self.sd, **self.p, **{f"cfg_{k}": v for k, v in self.cfg.items()})

    @classmethod
    def load(cls, path):
        d = np.load(path)
        cfg = {k[4:]: int(d[k]) for k in d.files if k.startswith("cfg_")}
        m = cls(**cfg)
        m.p = {k: d[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3")}
        m.mu, m.sd = d["mu"], d["sd"]
        return m


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
