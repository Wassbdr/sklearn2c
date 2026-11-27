"""Parity of compiled C labels vs scikit-learn for decision trees.

Regression test for the label-contract bug: leaves returned the class
*index* (0/1) instead of the label (10/20) stored in ``classes_``.
This test calls the generated ``model_predict`` symbol directly through
ctypes, independently of sklearn2c's own parity harness.
"""
import ctypes
import subprocess

import numpy as np
import pytest
from conftest import requires_gcc
from sklearn.tree import DecisionTreeClassifier

from sklearn2c import transpile

pytestmark = requires_gcc

X4 = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [0.0, 0.0]], dtype=np.float32)


def compile_and_predict(model, tmp_path, X):
    src = tmp_path / "model.c"
    src.write_text(transpile(model))
    lib = tmp_path / "model.so"
    subprocess.run(["gcc", "-shared", "-fPIC", "-O2", str(src), "-o", str(lib)], check=True)
    so = ctypes.CDLL(str(lib))
    so.model_predict.restype = ctypes.c_int32
    so.model_predict.argtypes = [ctypes.POINTER(ctypes.c_float)]
    n_features = X.shape[1]
    return np.array([so.model_predict((ctypes.c_float * n_features)(*row)) for row in X])


@pytest.mark.parametrize("labels", [(10, 20), (0, 1)], ids=["labels_10_20", "control_0_1"])
def test_tree_labels_match_sklearn(labels, tmp_path):
    y = np.array([labels[0], labels[0], labels[1], labels[1]])
    model = DecisionTreeClassifier(random_state=0).fit(X4, y)
    expected = model.predict(X4)
    got = compile_and_predict(model, tmp_path, X4)
    mismatches = [(i, int(e), int(g)) for i, (e, g) in enumerate(zip(expected, got)) if e != g]
    assert not mismatches, f"(sample, sklearn, C) mismatches: {mismatches}"


def test_negative_and_large_labels(tmp_path):
    y = np.array([-7, -7, 2_000_000_000, 2_000_000_000])
    model = DecisionTreeClassifier(random_state=0).fit(X4, y)
    np.testing.assert_array_equal(compile_and_predict(model, tmp_path, X4), model.predict(X4))


def test_float_integer_labels(tmp_path):
    y = np.array([1.0, 1.0, 3.0, 3.0])
    model = DecisionTreeClassifier(random_state=0).fit(X4, y)
    np.testing.assert_array_equal(compile_and_predict(model, tmp_path, X4), model.predict(X4).astype(int))
