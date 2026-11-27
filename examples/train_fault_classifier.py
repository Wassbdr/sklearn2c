#!/usr/bin/env python3
"""Train a vibration-fault classifier on synthetic sensor data (fixed seed).

Fault codes are the labels a firmware would log: 10 = healthy, 20 = imbalance,
30 = bearing wear. They are deliberately not 0/1/2, so a transpiler that
returns the class index instead of the label is caught immediately.

    python examples/train_fault_classifier.py                # -> fault_tree.joblib
    python examples/train_fault_classifier.py --model forest # -> fault_forest.joblib
    python examples/train_fault_classifier.py --string-labels  # labels "healthy", ... (unsupported)
"""
import argparse

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["rms_g", "peak_g", "kurtosis", "temp_c"]
CODES = {10: "healthy", 20: "imbalance", 30: "bearing_wear"}


def make_data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice([10, 20, 30], size=n, p=[0.6, 0.25, 0.15])
    rms = rng.normal(0.30, 0.05, n) + 0.35 * (y == 20) + 0.10 * (y == 30)
    peak = rms * rng.normal(3.0, 0.3, n) + 1.5 * (y == 30)
    kurt = rng.normal(3.0, 0.4, n) + 2.5 * (y == 30)
    temp = rng.normal(45.0, 4.0, n) + 6.0 * (y != 10)
    return np.c_[rms, peak, kurt, temp].astype(np.float32), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tree", "forest"], default="tree")
    ap.add_argument("--string-labels", action="store_true")
    ap.add_argument("-o", "--output")
    args = ap.parse_args()

    X, y = make_data()
    if args.string_labels:
        y = np.array([CODES[v] for v in y])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    if args.model == "tree":
        model = DecisionTreeClassifier(max_depth=5, random_state=0)
    else:
        model = RandomForestClassifier(n_estimators=20, max_depth=6, random_state=0)
    model.fit(Xtr, ytr)

    out = args.output or f"fault_{args.model}{'_str' if args.string_labels else ''}.joblib"
    joblib.dump(model, out)
    np.savetxt("fault_test.csv", Xte, delimiter=",", header=",".join(FEATURES), comments="", fmt="%.9g")
    print(f"{type(model).__name__} on {len(X)} synthetic sensor rows (seed 0), features {FEATURES}")
    print(f"labels {model.classes_.tolist()}  test accuracy {model.score(Xte, yte):.3f}  -> {out}, fault_test.csv")


if __name__ == "__main__":
    main()
