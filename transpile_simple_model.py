#!/usr/bin/env python3
"""Deprecated entry point kept for old instructions; use the ``sklearn2c`` CLI.

    python transpile_simple_model.py model.joblib   ==   sklearn2c model.joblib -o generated_model.c --main --verify
"""
import sys

from sklearn2c.cli import main

if __name__ == "__main__":
    args = sys.argv[1:] or ["regression.joblib"]
    if "-o" not in args and "--output" not in args:
        args += ["-o", "generated_model.c"]
    sys.exit(main(args + ["--main", "--verify"]))
