"""C code generation for fitted scikit-learn estimators.

Every generated file exposes the same contract (``<prefix>`` defaults to ``model``)::

    #define MODEL_N_FEATURES n
    /* classifiers */
    int32_t model_predict_index(const float *x);  /* index into classes_  */
    int32_t model_predict(const float *x);        /* classes_[index]      */
    /* regressors */
    SKLEARN2C_REAL model_predict(const float *x); /* float unless -DSKLEARN2C_REAL=double */

Inputs are ``float`` because scikit-learn's tree models cast inputs to
float32 before predicting.  Tree thresholds are emitted as the largest
float32 that is ``<=`` the float64 threshold, which makes ``x <= t`` in C
bit-identical to scikit-learn's ``(double)x <= threshold`` for every float input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .errors import UnsupportedLabelError, UnsupportedModelError

INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1

# estimator class name -> (family, is_classifier)
SUPPORTED_MODELS = {
    "DecisionTreeClassifier": ("tree", True),
    "DecisionTreeRegressor": ("tree", False),
    "ExtraTreeClassifier": ("tree", True),
    "ExtraTreeRegressor": ("tree", False),
    "RandomForestClassifier": ("forest", True),
    "RandomForestRegressor": ("forest", False),
    "ExtraTreesClassifier": ("forest", True),
    "ExtraTreesRegressor": ("forest", False),
    "LinearRegression": ("linear", False),
    "Ridge": ("linear", False),
    "Lasso": ("linear", False),
    "ElasticNet": ("linear", False),
    "LogisticRegression": ("linear", True),
    "RidgeClassifier": ("linear", True),
    "LinearSVC": ("linear", True),
}

_HINTS = {
    "Pipeline": "export the final estimator and apply the preprocessing in your firmware "
    "(or fold a StandardScaler into the linear coefficients) before calling predict().",
    "SVC": "kernel SVMs are not supported; LinearSVC is.",
    "GradientBoostingClassifier": "gradient boosting is on the roadmap; RandomForestClassifier is supported.",
    "GradientBoostingRegressor": "gradient boosting is on the roadmap; RandomForestRegressor is supported.",
}


@dataclass
class ModelSpec:
    """What sklearn2c learned about a model before generating code."""

    name: str
    family: str
    is_classifier: bool
    n_features: int
    classes: list[int] = field(default_factory=list)

    @property
    def n_classes(self) -> int:
        return len(self.classes)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _int_labels(model) -> list[int]:
    classes = np.asarray(model.classes_)
    name = type(model).__name__
    preview = ", ".join(repr(c.item() if hasattr(c, "item") else c) for c in classes[:4])
    if len(classes) > 4:
        preview += ", ..."
    hint = (
        "sklearn2c returns labels as C int32_t. Encode the target with "
        "sklearn.preprocessing.LabelEncoder (y = le.fit_transform(y)) before fitting, "
        "and keep le.classes_ on the host to decode the integer the C code returns."
    )
    if classes.dtype.kind == "b":
        return [int(c) for c in classes]
    if classes.dtype.kind in "iu":
        values = [int(c) for c in classes]
    elif classes.dtype.kind == "f" and np.all(np.isfinite(classes)) and np.all(classes == np.round(classes)):
        values = [int(c) for c in classes]
    else:
        raise UnsupportedLabelError(
            f"{name}.classes_ = [{preview}] (dtype {classes.dtype}) is not integer-valued. {hint}"
        )
    bad = [v for v in values if not INT32_MIN <= v <= INT32_MAX]
    if bad:
        raise UnsupportedLabelError(f"{name}.classes_ contains {bad[0]}, outside the int32_t range. {hint}")
    return values


def inspect_model(model) -> ModelSpec:
    """Validate ``model`` and describe it, or raise a clear error."""
    name = type(model).__name__
    if name not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        hint = _HINTS.get(name, "")
        raise UnsupportedModelError(
            f"{name} is not supported by sklearn2c. {hint + ' ' if hint else ''}"
            f"Supported estimators: {supported}."
        )
    check_is_fitted(model)
    family, is_clf = SUPPORTED_MODELS[name]
    n_features = int(model.n_features_in_)

    if family in ("tree", "forest") and getattr(model, "n_outputs_", 1) != 1:
        raise UnsupportedModelError(f"{name} with n_outputs_={model.n_outputs_} (multi-output) is not supported.")
    if family == "linear" and not is_clf and np.ndim(model.coef_) != 1:
        if np.shape(model.coef_)[0] != 1:
            raise UnsupportedModelError(f"{name} with a 2-D target (multi-output regression) is not supported.")
    if family == "tree" and model.tree_.max_depth > 256:
        raise UnsupportedModelError(f"{name} has depth {model.tree_.max_depth}; sklearn2c limits trees to depth 256.")

    classes = _int_labels(model) if is_clf else []
    if is_clf and len(classes) < 2:
        raise UnsupportedModelError(f"{name} was fitted on a single class; nothing to predict.")
    return ModelSpec(name=name, family=family, is_classifier=is_clf, n_features=n_features, classes=classes)


# ---------------------------------------------------------------------------
# literals
# ---------------------------------------------------------------------------


def float32_at_or_below(t: float) -> np.float32:
    """Largest float32 ``f`` with ``f <= t``: for float32 x, ``x <= f`` iff ``x <= t``."""
    f = np.float32(t)
    if float(f) > t:
        f = np.nextafter(f, np.float32(-np.inf), dtype=np.float32)
    return f


def c_float(v) -> str:
    """Shortest decimal literal that a C compiler parses back to the same float32."""
    f = np.float32(v)
    if not np.isfinite(f):
        raise UnsupportedModelError(f"non-finite parameter {v!r} cannot be emitted as C")
    s = np.format_float_scientific(f, unique=True, trim="-") if (f != 0 and not 1e-4 <= abs(f) < 1e7) \
        else np.format_float_positional(f, unique=True, trim="0")
    return s + "f"


def c_double(v) -> str:
    """Shortest decimal literal that round-trips the float64 value exactly."""
    v = float(v)
    if not np.isfinite(v):
        raise UnsupportedModelError(f"non-finite parameter {v!r} cannot be emitted as C")
    s = repr(v)
    return s if any(ch in s for ch in ".e") else s + ".0"


def _int_type(max_value: int) -> str:
    return "int16_t" if max_value <= 32767 else "int32_t"


def _array(ctype: str, name: str, values: list[str], per_line: int = 8) -> str:
    rows = [", ".join(values[i : i + per_line]) for i in range(0, len(values), per_line)]
    body = ",\n    ".join(rows)
    return f"static const {ctype} {name}[{len(values)}] = {{\n    {body}\n}};\n"


# ---------------------------------------------------------------------------
# emitters
# ---------------------------------------------------------------------------

_PRELUDE = """\
#include <stdint.h>

#ifndef SKLEARN2C_REAL
#define SKLEARN2C_REAL float /* accumulator/output type for regressors and linear models */
#endif
"""


def _tree_classifier_value(tree, node: int) -> int:
    return int(np.argmax(tree.value[node, 0]))


def _emit_tree(spec: ModelSpec, model, p: str) -> str:
    tree = model.tree_

    def leaf(node: int) -> str:
        if spec.is_classifier:
            return str(_tree_classifier_value(tree, node))
        return f"(SKLEARN2C_REAL){c_double(tree.value[node, 0, 0])}"

    lines: list[str] = []

    def walk(node: int, depth: int) -> None:
        pad = "    " * depth
        if tree.children_left[node] == -1:
            lines.append(f"{pad}return {leaf(node)};")
            return
        thr = c_float(float32_at_or_below(tree.threshold[node]))
        lines.append(f"{pad}if (x[{tree.feature[node]}] <= {thr}) {{")
        walk(int(tree.children_left[node]), depth + 1)
        lines.append(f"{pad}}} else {{")
        walk(int(tree.children_right[node]), depth + 1)
        lines.append(f"{pad}}}")

    walk(0, 1)
    body = "\n".join(lines)
    if spec.is_classifier:
        return f"int32_t {p}_predict_index(const float *x) {{\n{body}\n}}\n"
    return f"SKLEARN2C_REAL {p}_predict(const float *x) {{\n{body}\n}}\n"


def _emit_forest(spec: ModelSpec, model, p: str) -> str:
    """Table-driven forest. Nodes are in sklearn's pre-order, so the left child
    of an internal node is always ``node + 1``; ``NEXT`` holds the right child
    for internal nodes and the leaf-table row for leaves (``FEATURE < 0``).
    Identical leaf rows (e.g. pure leaves) are stored once."""
    P = p.upper()
    feats, thrs, nexts, roots = [], [], [], []
    leaf_rows: dict[tuple, int] = {}
    k = spec.n_classes
    for est in model.estimators_:
        t = est.tree_
        base = len(feats)
        roots.append(base)
        if spec.is_classifier:
            proba = t.value[:, 0, :k].copy()
            norm = proba.sum(axis=1)[:, np.newaxis]
            norm[norm == 0.0] = 1.0
            proba /= norm
        for node in range(t.node_count):
            if t.children_left[node] == -1:
                row = tuple(proba[node].tolist()) if spec.is_classifier else (float(t.value[node, 0, 0]),)
                feats.append(-1)
                thrs.append(0.0)
                nexts.append(leaf_rows.setdefault(row, len(leaf_rows)))
            else:
                if t.children_left[node] != node + 1:  # pragma: no cover - sklearn invariant
                    raise UnsupportedModelError("tree nodes are not in pre-order; cannot emit a compact forest")
                feats.append(int(t.feature[node]))
                thrs.append(float32_at_or_below(t.threshold[node]))
                nexts.append(base + int(t.children_right[node]))

    idx_t = _int_type(max(len(feats), len(leaf_rows)))
    feat_t = "int8_t" if spec.n_features <= 127 else _int_type(spec.n_features)
    leaf_vals = [v for row in leaf_rows for v in row]
    out = [
        f"#define {P}_N_TREES {len(roots)}  /* {len(feats)} nodes, {len(leaf_rows)} distinct leaves */\n",
        _array(feat_t, f"{P}_FEATURE", [str(v) for v in feats], 16),
        _array("float", f"{P}_THRESHOLD", [c_float(v) for v in thrs]),
        _array(idx_t, f"{P}_NEXT", [str(v) for v in nexts], 16),
        _array(idx_t, f"{P}_ROOT", [str(v) for v in roots], 16),
    ]
    walk = f"""        int32_t node = {P}_ROOT[t];
        while ({P}_FEATURE[node] >= 0)
            node = (x[{P}_FEATURE[node]] <= {P}_THRESHOLD[node]) ? node + 1 : {P}_NEXT[node];
"""
    if spec.is_classifier:
        out.append(_array("double", f"{P}_LEAF_PROBA", [c_double(v) for v in leaf_vals], 4))
        out.append(f"""
/* Mirrors RandomForestClassifier.predict: sum per-tree class probabilities in
 * tree order (double), divide by the number of trees, take the first argmax. */
int32_t {p}_predict_index(const float *x) {{
    double acc[{P}_N_CLASSES] = {{0}};
    for (int32_t t = 0; t < {P}_N_TREES; t++) {{
{walk}        const double *proba = &{P}_LEAF_PROBA[{P}_NEXT[node] * {P}_N_CLASSES];
        for (int32_t c = 0; c < {P}_N_CLASSES; c++) acc[c] += proba[c];
    }}
    int32_t best = 0;
    double best_v = acc[0] / {P}_N_TREES;
    for (int32_t c = 1; c < {P}_N_CLASSES; c++) {{
        double v = acc[c] / {P}_N_TREES;
        if (v > best_v) {{ best_v = v; best = c; }}
    }}
    return best;
}}
""")
    else:
        out.append(_array("SKLEARN2C_REAL", f"{P}_LEAF_VALUE", [c_double(v) for v in leaf_vals], 4))
        out.append(f"""
/* Mirrors RandomForestRegressor.predict: mean of the per-tree leaf values. */
SKLEARN2C_REAL {p}_predict(const float *x) {{
    SKLEARN2C_REAL acc = 0;
    for (int32_t t = 0; t < {P}_N_TREES; t++) {{
{walk}        acc += {P}_LEAF_VALUE[{P}_NEXT[node]];
    }}
    return acc / {P}_N_TREES;
}}
""")
    return "".join(out)


def _emit_linear(spec: ModelSpec, model, p: str) -> str:
    P = p.upper()
    coef = np.atleast_2d(np.asarray(model.coef_, dtype=np.float64))
    intercept = np.atleast_1d(np.asarray(model.intercept_, dtype=np.float64))
    if intercept.size == 1 and coef.shape[0] > 1:
        intercept = np.repeat(intercept, coef.shape[0])
    n_rows = coef.shape[0]
    out = [
        f"#define {P}_N_SCORES {n_rows}\n",
        _array("SKLEARN2C_REAL", f"{P}_INTERCEPT", [c_double(v) for v in intercept], 4),
        _array("SKLEARN2C_REAL", f"{P}_COEF", [c_double(v) for v in coef.ravel()], 4),
        f"""
/* scores[r] = intercept[r] + sum_i coef[r][i] * x[i]  (sklearn's decision_function) */
void {p}_decision_function(const float *x, SKLEARN2C_REAL *scores) {{
    for (int32_t r = 0; r < {P}_N_SCORES; r++) {{
        SKLEARN2C_REAL z = {P}_INTERCEPT[r];
        for (int32_t i = 0; i < {P}_N_FEATURES; i++) z += {P}_COEF[r * {P}_N_FEATURES + i] * (SKLEARN2C_REAL)x[i];
        scores[r] = z;
    }}
}}
""",
    ]
    if not spec.is_classifier:
        out.append(f"""
SKLEARN2C_REAL {p}_predict(const float *x) {{
    SKLEARN2C_REAL z;
    {p}_decision_function(x, &z);
    return z;
}}
""")
    elif n_rows == 1:
        out.append(f"""
/* binary: class 1 when the decision function is strictly positive */
int32_t {p}_predict_index(const float *x) {{
    SKLEARN2C_REAL z;
    {p}_decision_function(x, &z);
    return z > 0 ? 1 : 0;
}}
""")
    else:
        out.append(f"""
/* multiclass: first argmax of the decision function */
int32_t {p}_predict_index(const float *x) {{
    SKLEARN2C_REAL s[{P}_N_SCORES];
    {p}_decision_function(x, s);
    int32_t best = 0;
    for (int32_t r = 1; r < {P}_N_SCORES; r++) if (s[r] > s[best]) best = r;
    return best;
}}
""")
    return "".join(out)


def _emit_main(spec: ModelSpec, p: str) -> str:
    P = p.upper()
    fmt, cast = ('"%ld\\n"', "(long)") if spec.is_classifier else ('"%.9g\\n"', "(double)")
    return f"""
#ifndef SKLEARN2C_NO_MAIN
#include <stdio.h>
#include <stdlib.h>

/* usage: ./model f0 f1 ... f{spec.n_features - 1}   -> prints the prediction */
int main(int argc, char **argv) {{
    float x[{P}_N_FEATURES];
    if (argc != {P}_N_FEATURES + 1) {{
        fprintf(stderr, "usage: %s <%d features>\\n", argv[0], {P}_N_FEATURES);
        return 2;
    }}
    for (int i = 0; i < {P}_N_FEATURES; i++) x[i] = strtof(argv[i + 1], NULL);
    printf({fmt}, {cast}{p}_predict(x));
    return 0;
}}
#endif
"""


def transpile(model, prefix: str = "model", with_main: bool = False) -> str:
    """Return C99 source implementing ``model.predict`` for one sample.

    Raises :class:`UnsupportedModelError` or :class:`UnsupportedLabelError`
    instead of emitting code that would disagree with scikit-learn.
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", prefix):
        raise ValueError(f"prefix {prefix!r} is not a valid C identifier")
    spec = inspect_model(model)
    p, P = prefix, prefix.upper()
    from . import __version__

    kind = f"{spec.n_classes}-class classifier" if spec.is_classifier else "regressor"
    head = [
        f"/* Generated by sklearn2c {__version__} from {spec.name} ({kind}, {spec.n_features} features).\n"
        f" * {p}_predict(x) returns what {spec.name}.predict() returns for x. Do not edit. */\n",
        _PRELUDE,
        f"\n#define {P}_N_FEATURES {spec.n_features}\n",
    ]
    if spec.is_classifier:
        head.append(f"#define {P}_N_CLASSES {spec.n_classes}\n")
        head.append(_array("int32_t", f"{P}_CLASSES", [str(c) for c in spec.classes], 16))
    head.append("\n")

    emit = {"tree": _emit_tree, "forest": _emit_forest, "linear": _emit_linear}[spec.family]
    body = emit(spec, model, p)
    if spec.is_classifier:
        body += f"""
/* label = classes_[index], exactly what {spec.name}.predict() returns */
int32_t {p}_predict(const float *x) {{
    return {P}_CLASSES[{p}_predict_index(x)];
}}
"""
    src = "".join(head) + body
    if with_main:
        src += _emit_main(spec, p)
    return src
