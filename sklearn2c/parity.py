"""Check that compiled C predictions equal scikit-learn's, sample by sample."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from .codegen import float32_at_or_below, inspect_model, transpile
from .native import compile_shared

# Published regression tolerance:  |c - sk| <= REG_RTOL * max(1, |sk|, S)
# where S = |b| + sum_i |w_i x_i| for linear models (S = |sk| otherwise).
# Covers float32 accumulation for up to ~80 features; classification is exact.
REG_RTOL = 1e-5


@dataclass
class ParityReport:
    model: str
    is_classifier: bool
    n_samples: int
    n_mismatches: int
    max_abs_error: float = 0.0
    tolerance: str = "exact label match"
    mismatches: list = field(default_factory=list)  # (row, sklearn, c) for the first few

    @property
    def ok(self) -> bool:
        return self.n_mismatches == 0

    def summary(self) -> str:
        status = "OK" if self.ok else "FAIL"
        extra = "" if self.is_classifier else f", max |err| {self.max_abs_error:.3g}"
        return (f"parity {status}: {self.n_samples - self.n_mismatches}/{self.n_samples} samples match "
                f"({self.n_mismatches} mismatches, {self.tolerance}{extra})")


def _split_points(model):
    trees = [model] if hasattr(model, "tree_") else list(getattr(model, "estimators_", []))
    pts = []
    for est in trees:
        t = est.tree_
        inner = t.children_left != -1
        pts.extend(zip(t.feature[inner].tolist(), t.threshold[inner].tolist()))
    return pts


def probe_inputs(model, n_random: int = 1000, seed: int = 0, X_ref=None, max_boundary: int = 2000) -> np.ndarray:
    """float32 inputs that exercise ``model``: random rows plus, for tree models,
    rows sitting exactly on each split (the float32 at or below the threshold,
    which must go left) and one ulp above it (which must go right)."""
    spec = inspect_model(model)
    rng = np.random.default_rng(seed)
    if X_ref is not None:
        X_ref = np.asarray(X_ref, dtype=np.float64)
        lo, hi = X_ref.min(axis=0), X_ref.max(axis=0)
    else:
        lo, hi = np.full(spec.n_features, -3.0), np.full(spec.n_features, 3.0)
        pts = _split_points(model)
        for f, t in pts:
            lo[f], hi[f] = min(lo[f], t - 1), max(hi[f], t + 1)
    span = np.where(hi > lo, hi - lo, 1.0)
    X = (lo - 0.1 * span + rng.random((n_random, spec.n_features)) * 1.2 * span).astype(np.float32)
    rows = [X]
    pts = _split_points(model)
    if pts:
        if len(pts) > max_boundary:
            pts = [pts[i] for i in rng.choice(len(pts), max_boundary, replace=False)]
        base = (lo - 0.1 * span + rng.random((len(pts), spec.n_features)) * 1.2 * span).astype(np.float32)
        on, above = base.copy(), base.copy()
        for i, (f, t) in enumerate(pts):
            le = float32_at_or_below(t)
            on[i, f] = le
            above[i, f] = np.nextafter(le, np.float32(np.inf), dtype=np.float32)
        rows += [on, above]
    return np.concatenate(rows).astype(np.float32)


_UNIT_ROUNDOFF = {"float": 2.0 ** -24, "double": 2.0 ** -53}


def exact_decision_function(model, X) -> np.ndarray:
    """Binary linear model: ``intercept_ + coef_ @ x`` in exact rational arithmetic on the
    float32 inputs the C code receives, rounded once to float64 (the sign is exact).

    This is the reference for inputs near the decision boundary: there, sklearn's own
    ``predict`` depends on its dtype and BLAS summation order (scikit-learn 1.9 keeps
    ``coef_`` in float32 for float32 training data), so it is not a stable oracle."""
    coef = np.atleast_2d(model.coef_)
    if coef.shape[0] != 1:
        raise ValueError("exact_decision_function supports binary / single-output linear models")
    w = [Fraction(float(v)) for v in coef[0]]
    b = Fraction(float(np.atleast_1d(model.intercept_)[0]))
    X = np.asarray(X, dtype=np.float32)
    return np.array([float(b + sum(wi * Fraction(float(xi)) for wi, xi in zip(w, row))) for row in X])


def linear_rounding_bound(n_features: int, real: str = "float") -> float:
    """Relative error bound gamma_{n+2} = (n+2)u / (1-(n+2)u) of the C dot product
    (n products, n additions + intercept, coefficients rounded to SKLEARN2C_REAL).
    If ``|z_exact| > bound * (|b| + sum|w_i x_i|)``, the C label equals the exact sign."""
    k = (n_features + 2) * _UNIT_ROUNDOFF[real]
    return k / (1 - k)


def linear_boundary_probes(model, rel_distances=(0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 1e-4), n_per: int = 300,
                           seed: int = 0) -> np.ndarray:
    """Random points moved to ``z = +-d * scale`` of a binary linear model's decision boundary
    (float64 construction, then cast to the float32 the C code takes)."""
    w = np.atleast_2d(model.coef_)[0].astype(np.float64)
    b = float(np.atleast_1d(model.intercept_)[0])
    rng = np.random.default_rng(seed)
    out = []
    for d in rel_distances:
        X = rng.normal(size=(n_per, w.size))
        X -= np.outer((X @ w + b) / (w @ w), w)
        scale = np.abs(X * w).sum(1) + abs(b)
        X += np.outer(rng.choice([-1.0, 1.0], n_per) * d * scale / (w @ w), w)
        out.append(X)
    return np.concatenate(out).astype(np.float32)


def _linear_scale(model, X):
    coef = np.atleast_2d(model.coef_).astype(np.float64)
    b = np.abs(np.atleast_1d(model.intercept_)).astype(np.float64)
    return (np.abs(X.astype(np.float64)) @ np.abs(coef).T + b).max(axis=1)


def verify_predictions(model, c_source: str | None = None, X=None, *, prefix: str = "model",
                       cc: str | None = None, cflags: tuple[str, ...] = ("-O2",), seed: int = 0,
                       compiled=None) -> ParityReport:
    """Compile the generated C, run it on X and compare with ``model.predict``.

    ``X`` defaults to :func:`probe_inputs`. Inputs are rounded to float32 (the
    C ABI) before both sides see them, so the comparison is like for like.
    """
    spec = inspect_model(model)
    if c_source is None:
        c_source = transpile(model, prefix=prefix)
    X = probe_inputs(model, seed=seed) if X is None else np.asarray(X, dtype=np.float32)
    lib = compiled or compile_shared(c_source, prefix=prefix, is_classifier=spec.is_classifier,
                                     n_features=spec.n_features, cc=cc, cflags=cflags)
    got = lib.predict(X)
    expected = model.predict(X)

    if spec.is_classifier:
        bad = np.flatnonzero(np.asarray(expected) != got)
        return ParityReport(spec.name, True, len(X), int(bad.size),
                            mismatches=[(int(i), expected[i].item(), int(got[i])) for i in bad[:10]])

    expected = np.asarray(expected, dtype=np.float64).ravel()
    err = np.abs(got - expected)
    scale = np.maximum(1.0, np.abs(expected))
    if spec.family == "linear":
        scale = np.maximum(scale, _linear_scale(model, X))
    bad = np.flatnonzero(err > REG_RTOL * scale)
    return ParityReport(spec.name, False, len(X), int(bad.size), float(err.max(initial=0.0)),
                        tolerance=f"|err| <= {REG_RTOL:g} * max(1, |y|, sum|w*x|)",
                        mismatches=[(int(i), float(expected[i]), float(got[i])) for i in bad[:10]])
