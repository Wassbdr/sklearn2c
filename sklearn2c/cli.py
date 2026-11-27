"""``sklearn2c model.joblib -o model.c [--verify]``"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

from . import __version__
from .codegen import SUPPORTED_MODELS, inspect_model, transpile
from .errors import Sklearn2CError
from .parity import verify_predictions


def _load_matrix(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    with open(path) as fh:
        first = fh.readline()
    skip = 0 if all(_is_number(c) for c in first.strip().split(",")) else 1
    return np.loadtxt(path, delimiter=",", skiprows=skip, ndmin=2)


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="sklearn2c",
        description="Transpile a fitted scikit-learn model (.joblib) to dependency-free C99.",
        epilog="Supported: " + ", ".join(sorted(SUPPORTED_MODELS)),
    )
    ap.add_argument("model", help="path to a fitted estimator saved with joblib.dump (trusted files only)")
    ap.add_argument("-o", "--output", help="output .c file (default: <model>.c)")
    ap.add_argument("--prefix", default="model", help="C symbol prefix (default: model -> model_predict)")
    ap.add_argument("--main", action="store_true", help="append a main() that predicts from argv")
    ap.add_argument("--verify", action="store_true",
                    help="compile with $CC/gcc and compare C vs sklearn predictions sample by sample")
    ap.add_argument("--verify-data", metavar="X.csv|X.npy",
                    help="feature matrix to verify on (default: random + split-boundary probes)")
    ap.add_argument("--cc", default=None, help="C compiler for --verify (default: $CC or gcc)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the probe inputs (default: 0)")
    ap.add_argument("--version", action="version", version=f"sklearn2c {__version__}")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_path = Path(args.model)
    out = Path(args.output) if args.output else model_path.with_suffix(".c")
    try:
        model = joblib.load(model_path)
        spec = inspect_model(model)
        src = transpile(model, prefix=args.prefix, with_main=args.main)
    except FileNotFoundError:
        print(f"sklearn2c: error: {model_path} not found", file=sys.stderr)
        return 2
    except Sklearn2CError as exc:
        print(f"sklearn2c: error: {exc}", file=sys.stderr)
        return 2

    out.write_text(src)
    what = f"{spec.n_classes} classes {spec.classes}" if spec.is_classifier else "regressor"
    print(f"{spec.name}: {spec.n_features} features, {what}")
    print(f"wrote {out} ({len(src.splitlines())} lines) -> {args.prefix}_predict(const float *x)")

    if args.verify or args.verify_data:
        X = _load_matrix(args.verify_data) if args.verify_data else None
        try:
            report = verify_predictions(model, src, X, prefix=args.prefix, cc=args.cc, seed=args.seed)
        except Sklearn2CError as exc:
            print(f"sklearn2c: error: {exc}", file=sys.stderr)
            return 2
        print(report.summary())
        for row, sk, c in report.mismatches:
            print(f"  row {row}: sklearn={sk} C={c}")
        return 0 if report.ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
