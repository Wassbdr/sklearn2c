"""Parity of compiled C labels vs scikit-learn for decision trees.

Regression test for the label-contract bug: leaves returned the class
*index* (0/1) instead of the label (10/20) stored in ``classes_``.
"""
import ctypes
import shutil
import subprocess

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

import transpile_simple_model as t

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not found")

X4 = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [0.0, 0.0]], dtype=np.float32)


def compile_and_predict(model, tmp_path):
    coef, icpt, n_features, kind, extra = t.extract_parameters(model)
    src = tmp_path / "model.c"
    src.write_text(t.generate_c_code(coef, icpt, n_features, kind, extra))
    lib = tmp_path / "model.so"
    subprocess.run(["gcc", "-shared", "-fPIC", "-O2", str(src), "-o", str(lib)], check=True)
    so = ctypes.CDLL(str(lib))
    so.predict.restype = ctypes.c_int
    so.predict.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    out = []
    for row in X4:
        buf = (ctypes.c_float * n_features)(*row)
        out.append(so.predict(buf, n_features))
    return np.array(out)


@pytest.mark.parametrize("labels", [(10, 20), (0, 1)], ids=["labels_10_20", "control_0_1"])
def test_tree_labels_match_sklearn(labels, tmp_path):
    y = np.array([labels[0], labels[0], labels[1], labels[1]])
    model = DecisionTreeClassifier(random_state=0).fit(X4, y)
    expected = model.predict(X4)
    got = compile_and_predict(model, tmp_path)
    mismatches = [(i, int(e), int(g)) for i, (e, g) in enumerate(zip(expected, got)) if e != g]
    assert not mismatches, f"(sample, sklearn, C) mismatches: {mismatches}"
