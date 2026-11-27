"""Every supported estimator: compiled C vs sklearn on fixed synthetic data."""
import warnings

import numpy as np
import pytest
from conftest import requires_gcc
from sklearn import ensemble, linear_model, svm, tree
from sklearn.datasets import make_classification, make_regression

from sklearn2c import SUPPORTED_MODELS, probe_inputs, transpile, verify_predictions
from sklearn2c.native import compile_shared
from sklearn2c.parity import REG_RTOL

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
    """A float64 threshold between two float32 values: C must route like sklearn."""
    a = np.float32(1.0)
    b = np.nextafter(a, np.float32(2))
    X = np.array([[a], [b]], dtype=np.float32)
    model = tree.DecisionTreeClassifier().fit(X, [1, 2])
    t = model.tree_.threshold[0]
    assert np.float32(t) != t  # the midpoint has no float32 representation
    lib = compile_shared(transpile(model), prefix="model", is_classifier=True, n_features=1)
    np.testing.assert_array_equal(lib.predict(X), model.predict(X))
