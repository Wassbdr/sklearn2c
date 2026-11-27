"""Every supported estimator: compiled C vs sklearn on fixed synthetic data."""
import warnings

import numpy as np
import pytest
from conftest import requires_gcc
from sklearn import ensemble, linear_model, svm, tree
from sklearn.datasets import make_classification, make_regression

from sklearn2c import SUPPORTED_MODELS, probe_inputs, transpile, verify_predictions
from sklearn2c.native import compile_shared
from sklearn2c.parity import (REG_RTOL, exact_decision_function, linear_boundary_probes,
                              linear_rounding_bound)

pytestmark = requires_gcc

Xc, yc = make_classification(400, 6, n_informative=4, n_classes=3, random_state=0)
yc = yc * 10 + 10  # labels 10/20/30: never equal to the class index
Xb, yb = make_classification(400, 6, random_state=1)
yb = yb + 5
Xr, yr = make_regression(400, 6, noise=5.0, random_state=0)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    MODELS = {
        "DecisionTreeClassifier": tree.DecisionTreeClassifier(random_state=0).fit(Xc, yc),
        "DecisionTreeRegressor": tree.DecisionTreeRegressor(max_depth=8, random_state=0).fit(Xr, yr),
        "ExtraTreeClassifier": tree.ExtraTreeClassifier(random_state=0).fit(Xc, yc),
        "ExtraTreeRegressor": tree.ExtraTreeRegressor(max_depth=8, random_state=0).fit(Xr, yr),
        "RandomForestClassifier": ensemble.RandomForestClassifier(25, max_depth=8, random_state=0).fit(Xc, yc),
        "RandomForestRegressor": ensemble.RandomForestRegressor(25, max_depth=8, random_state=0).fit(Xr, yr),
        "ExtraTreesClassifier": ensemble.ExtraTreesClassifier(25, max_depth=8, random_state=0).fit(Xc, yc),
        "ExtraTreesRegressor": ensemble.ExtraTreesRegressor(25, max_depth=8, random_state=0).fit(Xr, yr),
        "LinearRegression": linear_model.LinearRegression().fit(Xr, yr),
        "Ridge": linear_model.Ridge(alpha=1.0).fit(Xr, yr),
        "Lasso": linear_model.Lasso(alpha=0.1).fit(Xr, yr),
        "ElasticNet": linear_model.ElasticNet(alpha=0.1).fit(Xr, yr),
        "LogisticRegression": linear_model.LogisticRegression(max_iter=2000).fit(Xb, yb),
        "LogisticRegression_multiclass": linear_model.LogisticRegression(max_iter=2000).fit(Xc, yc),
        "RidgeClassifier": linear_model.RidgeClassifier().fit(Xc, yc),
        "LinearSVC": svm.LinearSVC(random_state=0).fit(Xb, yb),
    }


def test_every_supported_model_is_covered():
    assert {k.split("_")[0] for k in MODELS} == set(SUPPORTED_MODELS)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_parity_probe_inputs(name):
    report = verify_predictions(MODELS[name], seed=0)
    assert report.n_samples >= 400
    assert report.ok, report.summary() + f" first mismatches: {report.mismatches}"


@pytest.mark.parametrize("name", sorted(MODELS))
def test_parity_training_data(name):
    model = MODELS[name]
    X = {6: Xc}[model.n_features_in_]
    report = verify_predictions(model, X=X)
    assert report.ok, report.summary()


@pytest.mark.parametrize("name", sorted(MODELS))
def test_parity_double_accumulator(name):
    """-DSKLEARN2C_REAL=double is a supported build option."""
    report = verify_predictions(MODELS[name], cflags=("-O2", "-DSKLEARN2C_REAL=double"))
    assert report.ok, report.summary()
    if not report.is_classifier:
        assert report.max_abs_error < 1e-6 * max(1.0, float(np.abs(yr).max()))


def test_regression_tolerance_is_published():
    r = verify_predictions(MODELS["LinearRegression"])
    assert f"{REG_RTOL:g}" in r.tolerance and r.max_abs_error > 0  # float32 C really differs a little


@pytest.mark.parametrize("name", ["DecisionTreeClassifier", "RandomForestClassifier", "DecisionTreeRegressor"])
def test_probe_inputs_hit_every_split_exactly(name):
    """Inputs exactly on a split value (float32 at/below the threshold) must go left,
    one ulp above must go right - checked against sklearn on every split."""
    model = MODELS[name]
    X = probe_inputs(model, n_random=0)
    est = model.estimators_[0] if hasattr(model, "estimators_") else model
    n_splits = sum(int((e.tree_.children_left != -1).sum())
                   for e in getattr(model, "estimators_", [model]))
    assert len(X) == 2 * min(n_splits, 2000)
    assert est.tree_.node_count > 1
    assert verify_predictions(model, X=X).ok


def test_threshold_exactly_representable():
    """Training values 0 and 1 give threshold 0.5; x = 0.5 must go left (x <= t)."""
    X = np.array([[0.0], [1.0]], dtype=np.float32)
    model = tree.DecisionTreeClassifier().fit(X, [10, 20])
    assert model.tree_.threshold[0] == 0.5
    probes = np.array([[0.5], [np.nextafter(np.float32(0.5), np.float32(1))],
                       [np.nextafter(np.float32(0.5), np.float32(0))]], dtype=np.float32)
    assert list(model.predict(probes)) == [10, 20, 10]
    lib = compile_shared(transpile(model), prefix="model", is_classifier=True, n_features=1)
    assert list(lib.predict(probes)) == [10, 20, 10]


def test_threshold_not_representable_in_float32():
    """Training values 3 ulps apart: the float64 midpoint sits between two float32
    values. The float32 just below must go left, the one just above right."""
    up = lambda v, n: v if n == 0 else up(np.nextafter(v, np.float32(2)), n - 1)  # noqa: E731
    a = np.float32(1.0)
    X = np.array([[a], [up(a, 3)]], dtype=np.float32)
    model = tree.DecisionTreeClassifier().fit(X, [1, 2])
    t = model.tree_.threshold[0]
    assert model.tree_.node_count == 3 and np.float32(t) != t
    probes = np.array([[a], [up(a, 1)], [up(a, 2)], [up(a, 3)]], dtype=np.float32)
    assert list(model.predict(probes)) == [1, 1, 2, 2]
    lib = compile_shared(transpile(model), prefix="model", is_classifier=True, n_features=1)
    assert list(lib.predict(probes)) == [1, 1, 2, 2]


@pytest.mark.parametrize("real", ["float", "double"])
def test_linear_label_matches_exact_sign_beyond_rounding_bound(real):
    """Near z = 0 the reference is the exact sign of z (sklearn's own predict there depends on its dtype
    and BLAS order). Beyond the published rounding bound the C label must equal it, on every compiler."""
    X, y = make_classification(600, 8, n_informative=5, random_state=1)
    X = X.astype(np.float32)
    model = linear_model.LogisticRegression(max_iter=2000).fit(X, y)
    P = linear_boundary_probes(model, n_per=150, seed=1)
    z = exact_decision_function(model, P)
    r = np.abs(z) / (np.abs(P.astype(np.float64) * np.asarray(model.coef_[0], np.float64)).sum(1)
                     + abs(float(model.intercept_[0])))
    lib = compile_shared(transpile(model), prefix="model", is_classifier=True, n_features=8,
                         cflags=("-O2", f"-DSKLEARN2C_REAL={real}"))
    far = r > linear_rounding_bound(8, real)
    assert far.sum() > 300
    np.testing.assert_array_equal(lib.predict(P)[far], model.classes_[(z[far] > 0).astype(int)])
