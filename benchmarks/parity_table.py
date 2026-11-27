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
from sklearn2c.parity import exact_decision_function, linear_boundary_probes, linear_rounding_bound

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


BINS = (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, np.inf)


def linear_boundary_stress(clf_data):
    """Binary LogisticRegression probed at controlled distances from its decision boundary.

    Reference = exact sign of z (rational arithmetic on the float32 inputs), not sklearn.predict:
    near z = 0 sklearn's own label depends on its dtype and BLAS order. Flips are reported per bin of
    relative distance r = |z| / (|b| + sum|w*x|); the C label is guaranteed for r > gamma_{n+2}."""
    (Xtr, _, ytr, _) = clf_data
    yb = (ytr == 10).astype(int)
    model = linear_model.LogisticRegression(max_iter=2000).fit(Xtr, yb)
    X = linear_boundary_probes(model, seed=SEED)
    z = exact_decision_function(model, X)
    r = np.abs(z) / (np.abs(X.astype(np.float64) * np.asarray(model.coef_[0], np.float64)).sum(1)
                     + abs(float(model.intercept_[0])))
    truth = model.classes_[(z > 0).astype(int)]
    labels = {"sklearn.predict": model.predict(X)}
    bound = {}
    for real in ("float", "double"):
        lib = compile_shared(transpile(model), prefix="model", is_classifier=True, n_features=X.shape[1],
                             cflags=("-O2", f"-DSKLEARN2C_REAL={real}"))
        labels[f"C {real}"] = lib.predict(X)
        bound[real] = linear_rounding_bound(X.shape[1], real)
    bins = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (r >= lo) & (r < hi)
        bins.append({"r_min": lo, "r_max": None if np.isinf(hi) else hi, "n": int(m.sum()),
                     **{k: int((v[m] != truth[m]).sum()) for k, v in labels.items()}})
    return {
        "model": "LogisticRegression (binary)", "n_samples": int(len(X)),
        "sklearn_coef_dtype": str(model.coef_.dtype),
        "sklearn_decision_function_dtype": str(model.decision_function(X[:1]).dtype),
        "bound": bound,
        "flips_total": {k: int((v != truth).sum()) for k, v in labels.items()},
        "flips_beyond_bound": {f"C {k}": int((labels[f"C {k}"] != truth)[r > b].sum()) for k, b in bound.items()},
        "bins": bins,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="10 trees per forest instead of 50")
    ap.add_argument("-v", "--verbose", action="store_true", help="print each parity report")
    ap.add_argument("--stress-only", action="store_true", help="only run the linear decision-boundary stress test")
    args = ap.parse_args()
    if args.stress_only:
        stress = linear_boundary_stress(datasets()[0])
        print(json.dumps(stress, indent=1))
        return
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
        if args.verbose or not rep.ok:
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
    stem = "parity_quick" if args.quick else "parity"  # --quick never overwrites the published table
    (OUT / f"{stem}.json").write_text(json.dumps({"setup": env, "rows": rows, "linear_boundary_stress": stress},
                                                 indent=2) + "\n")
    (OUT / f"{stem}.md").write_text(render_md(env, rows, stress, have_arm))
    print_terminal(rows, stress, have_arm, env, stem)


def print_terminal(rows, stress, have_arm, env, stem):
    """Compact table that fits an 80-column terminal (the full one is in parity.md)."""
    print(f"\n{'model':<24}{'compared':>9}{'mismatch':>9}{'M4 flash':>10}{'latency':>11}")
    for r in rows:
        flash = f"{r['cortex_m4_flash_bytes']:,}B" if have_arm else "-"
        print(f"{r['model']:<24}{r['n_samples']:>9}{r['n_mismatches']:>9}{flash:>10}{r['latency_ns']:>9.1f}ns")
    print(f"latency: {env['hardware']}")
    print(f"boundary stress (binary LogisticRegression, {stress['n_samples']} inputs near z=0, vs exact sign): "
          + ", ".join(f"{k} {v} flips" for k, v in stress["flips_total"].items())
          + "; beyond the rounding bound: "
          + ", ".join(f"{k} {v}" for k, v in stress["flips_beyond_bound"].items()))
    print(f"wrote {OUT.relative_to(Path.cwd()) if OUT.is_relative_to(Path.cwd()) else OUT}/{stem}.md")


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
    keys = list(stress["flips_total"])
    st = [f"| r = \\|z\\| / (\\|b\\| + sum\\|w*x\\|) | samples | " + " | ".join(f"{k} flips" for k in keys) + " |",
          "|---|---:|" + "---:|" * len(keys)]
    for b in stress["bins"]:
        rng = f"[{b['r_min']:.0e}, {b['r_max']:.0e})" if b["r_max"] else f">= {b['r_min']:.0e}"
        st.append(f"| {rng} | {b['n']} | " + " | ".join(str(b[k]) for k in keys) + " |")
    bd = stress["bound"]
    return (
        "\n".join(lines)
        + "\n\n" + setup
        + "\n\n**Linear classifier near its decision boundary** (binary LogisticRegression; inputs placed at "
        "relative distance r from z = 0, then cast to float32). A flip is a label that differs from the exact "
        "sign of z, computed in rational arithmetic on the float32 inputs. sklearn's own `predict` is shown for "
        f"comparison: this scikit-learn keeps `coef_` in {stress['sklearn_coef_dtype']} and evaluates the "
        f"decision function in {stress['sklearn_decision_function_dtype']}.\n\n" + "\n".join(st)
        + f"\n\nRounding bound gamma_(n+2): float {bd['float']:.1e}, double {bd['double']:.1e}. Flips beyond it: "
        + ", ".join(f"{k} {v}" for k, v in stress["flips_beyond_bound"].items()) + " (guaranteed 0).\n"
    )


if __name__ == "__main__":
    main()
