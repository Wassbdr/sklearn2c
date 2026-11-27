#!/usr/bin/env python3
"""Parity table: compiled C vs scikit-learn for every supported estimator.

Fixed synthetic data (sklearn make_classification / make_regression, seed 0),
compiled with the host gcc. Writes benchmarks/results/parity.{md,json}.

    python benchmarks/parity_table.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import sklearn
from sklearn import ensemble, linear_model, svm, tree
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

import sklearn2c
from sklearn2c import inspect_model, probe_inputs, transpile, verify_predictions
from sklearn2c.native import compile_shared, latency_ns, object_size

OUT = Path(__file__).parent / "results"
ARM_CC = "arm-none-eabi-gcc"
ARM_FLAGS = ("-Os", "-mcpu=cortex-m4", "-mthumb", "-mfloat-abi=hard", "-mfpu=fpv4-sp-d16")
SEED = 0


def datasets():
    Xc, yc = make_classification(2000, 8, n_informative=5, n_redundant=1, n_classes=3, random_state=SEED)
    yc = yc * 10 + 10  # labels 10/20/30
    Xr, yr = make_regression(2000, 8, n_informative=6, noise=10.0, random_state=SEED)
    return (train_test_split(Xc.astype(np.float32), yc, test_size=0.25, random_state=SEED),
            train_test_split(Xr.astype(np.float32), yr, test_size=0.25, random_state=SEED))


def models(n_trees: int):
    return [
        ("DecisionTreeClassifier", "max_depth=8", tree.DecisionTreeClassifier(max_depth=8, random_state=SEED)),
        ("ExtraTreeClassifier", "max_depth=8", tree.ExtraTreeClassifier(max_depth=8, random_state=SEED)),
        ("RandomForestClassifier", f"{n_trees} trees, max_depth=8",
         ensemble.RandomForestClassifier(n_trees, max_depth=8, random_state=SEED)),
        ("ExtraTreesClassifier", f"{n_trees} trees, max_depth=8",
         ensemble.ExtraTreesClassifier(n_trees, max_depth=8, random_state=SEED)),
        ("LogisticRegression", "3 classes", linear_model.LogisticRegression(max_iter=2000)),
        ("RidgeClassifier", "3 classes", linear_model.RidgeClassifier()),
        ("LinearSVC", "3 classes", svm.LinearSVC(random_state=SEED)),
        ("DecisionTreeRegressor", "max_depth=8", tree.DecisionTreeRegressor(max_depth=8, random_state=SEED)),
        ("ExtraTreeRegressor", "max_depth=8", tree.ExtraTreeRegressor(max_depth=8, random_state=SEED)),
        ("RandomForestRegressor", f"{n_trees} trees, max_depth=8",
         ensemble.RandomForestRegressor(n_trees, max_depth=8, random_state=SEED)),
        ("ExtraTreesRegressor", f"{n_trees} trees, max_depth=8",
         ensemble.ExtraTreesRegressor(n_trees, max_depth=8, random_state=SEED)),
        ("LinearRegression", "", linear_model.LinearRegression()),
        ("Ridge", "alpha=1", linear_model.Ridge(alpha=1.0)),
        ("Lasso", "alpha=0.1", linear_model.Lasso(alpha=0.1)),
        ("ElasticNet", "alpha=0.1", linear_model.ElasticNet(alpha=0.1)),
    ]


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def gcc_version(cc: str) -> str:
    return subprocess.run([cc, "-dumpfullversion"], capture_output=True, text=True).stdout.strip()


def linear_boundary_stress(clf_data, n=2000):
    """Points projected onto a binary LogisticRegression decision boundary: the
    worst case for float32 accumulation (labels decided by the last bits)."""
    (Xtr, _, ytr, _) = clf_data
    yb = (ytr == 10).astype(int)
    model = linear_model.LogisticRegression(max_iter=2000).fit(Xtr, yb)
    rng = np.random.default_rng(SEED)
    w, b = model.coef_[0], model.intercept_[0]
    X = rng.normal(size=(n, Xtr.shape[1]))
    X = X - np.outer((X @ w + b) / (w @ w), w)  # z == 0 in exact arithmetic
    X = X.astype(np.float32)
    rows = []
    for real in ("float", "double"):
        r = verify_predictions(model, X=X, cflags=("-O2", f"-DSKLEARN2C_REAL={real}"))
        rows.append({"accumulator": real, "n_samples": r.n_samples, "n_mismatches": r.n_mismatches})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="10 trees per forest instead of 50")
    args = ap.parse_args()
    n_trees = 10 if args.quick else 50
    clf_data, reg_data = datasets()
    have_arm = shutil.which(ARM_CC) is not None
    rows = []
    for name, params, model in models(n_trees):
        Xtr, Xte, ytr, yte = clf_data if sklearn.base.is_classifier(model) else reg_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Xtr, ytr)
        spec = inspect_model(model)
        src = transpile(model)
        lib = compile_shared(src, prefix="model", is_classifier=spec.is_classifier, n_features=spec.n_features)
        X = np.concatenate([Xte, probe_inputs(model, n_random=500, seed=SEED, X_ref=Xtr)])
        rep = verify_predictions(model, src, X, compiled=lib)
        x86 = object_size(src, cflags=("-Os",))
        row = {
            "model": name, "params": params, "task": "classification" if spec.is_classifier else "regression",
            "n_samples": rep.n_samples, "n_mismatches": rep.n_mismatches,
            "max_abs_error": None if spec.is_classifier else rep.max_abs_error,
            "x86_64_object_bytes": x86["text"] + x86["data"],
            "latency_ns": latency_ns(src, Xte),
            "c_lines": len(src.splitlines()),
        }
        if have_arm:
            arm = object_size(src, cc=ARM_CC, cflags=ARM_FLAGS)
            row["cortex_m4_flash_bytes"] = arm["text"] + arm["data"]
            row["cortex_m4_static_ram_bytes"] = arm["data"] + arm["bss"]
            row["cortex_m4_stack_bytes"] = arm["stack"]
        rows.append(row)
        print(f"{name:<24} {rep.summary()}")

    env = {
        "data": "make_classification(2000, 8 features, 3 classes -> labels 10/20/30) and "
                "make_regression(2000, 8 features, noise=10), random_state=0, 75/25 split",
        "compared": "held-out test rows (500) + 500 random rows + split-boundary probes "
                    "(float32 on each threshold and one ulp above)",
        "hardware": f"x86 VM, not a microcontroller: {cpu_name()}",
        "compiler": f"gcc {gcc_version('gcc')} (-O2 for latency, -Os for size)",
        "arm_compiler": f"{ARM_CC} {gcc_version(ARM_CC)} {' '.join(ARM_FLAGS)}" if have_arm else None,
        "versions": f"python {platform.python_version()}, scikit-learn {sklearn.__version__}, "
                    f"numpy {np.__version__}, sklearn2c {sklearn2c.__version__}",
        "regression_tolerance": "|C - sklearn| <= 1e-5 * max(1, |y|, sum|w*x|)",
        "latency": "best of 7 passes over the 500 test rows, per sample, single thread",
    }
    stress = linear_boundary_stress(clf_data)
    OUT.mkdir(exist_ok=True)
    (OUT / "parity.json").write_text(json.dumps({"setup": env, "rows": rows, "linear_boundary_stress": stress},
                                                indent=2) + "\n")
    (OUT / "parity.md").write_text(render_md(env, rows, stress, have_arm))
    print((OUT / "parity.md").read_text())


def render_md(env, rows, stress, have_arm) -> str:
    head = "| Model | Samples compared | Mismatches | Max abs err | x86-64 .o (B) |"
    sep = "|---|---:|---:|---:|---:|"
    if have_arm:
        head += " Cortex-M4 flash (B) | Cortex-M4 RAM: static + stack (B) |"
        sep += "---:|---:|"
    head += " Latency / sample (x86 VM) |"
    sep += "---:|"
    lines = [head, sep]
    for r in rows:
        err = "exact" if r["max_abs_error"] is None else f"{r['max_abs_error']:.2g}"
        label = r["model"] + (f" ({r['params']})" if r["params"] else "")
        line = f"| {label} | {r['n_samples']} | {r['n_mismatches']} | {err} | {r['x86_64_object_bytes']:,} |"
        if have_arm:
            line += f" {r['cortex_m4_flash_bytes']:,} | {r['cortex_m4_static_ram_bytes']} + {r['cortex_m4_stack_bytes']} |"
        line += f" {r['latency_ns']:.1f} ns |"
        lines.append(line)
    setup = "\n".join(f"- **{k}**: {v}" for k, v in env.items() if v)
    st = "\n".join(f"| {s['accumulator']} | {s['n_samples']} | {s['n_mismatches']} |" for s in stress)
    return (
        "\n".join(lines)
        + "\n\n" + setup
        + "\n\n**Linear classifier on its decision boundary** (binary LogisticRegression, inputs projected "
        "onto z = 0, the worst case for float rounding):\n\n| Accumulator | Samples | Label mismatches |\n"
        "|---|---:|---:|\n" + st + "\n"
    )


if __name__ == "__main__":
    main()
