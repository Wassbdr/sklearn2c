"""Unsupported models and labels fail loudly instead of producing wrong C."""
import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn2c import UnsupportedLabelError, UnsupportedModelError, transpile

X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])


@pytest.mark.parametrize("cls", [DecisionTreeClassifier, LogisticRegression])
def test_string_labels_rejected(cls):
    model = cls().fit(X, ["cat", "cat", "dog", "dog"])
    with pytest.raises(UnsupportedLabelError, match=r"\['cat', 'dog'\].*LabelEncoder"):
        transpile(model)


def test_fractional_float_labels_rejected():
    model = DecisionTreeClassifier().fit(X, [0, 0, 1, 1])
    model.classes_ = np.array([0.5, 1.5])  # sklearn itself rejects continuous y at fit time
    with pytest.raises(UnsupportedLabelError, match="not integer-valued"):
        transpile(model)


def test_labels_outside_int32_rejected():
    model = DecisionTreeClassifier().fit(X, [0, 0, 2**40, 2**40])
    with pytest.raises(UnsupportedLabelError, match="int32"):
        transpile(model)


@pytest.mark.parametrize("model", [
    KNeighborsClassifier(n_neighbors=1).fit(X, [0, 0, 1, 1]),
    SVC().fit(X, [0, 0, 1, 1]),
    GradientBoostingClassifier(n_estimators=2).fit(X, [0, 0, 1, 1]),
    KMeans(n_clusters=2, n_init=1, random_state=0).fit(X),
    make_pipeline(StandardScaler(), LogisticRegression()).fit(X, [0, 0, 1, 1]),
], ids=lambda m: type(m).__name__)
def test_unsupported_model_types_rejected(model):
    with pytest.raises(UnsupportedModelError, match="not supported.*Supported estimators"):
        transpile(model)


def test_lookalike_names_are_not_accepted():
    """The old substring check accepted anything with 'Tree' or 'Linear' in its name."""
    class MyLinearThing:
        pass

    with pytest.raises(UnsupportedModelError):
        transpile(MyLinearThing())


def test_unfitted_model_rejected():
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        transpile(DecisionTreeClassifier())


def test_multi_output_rejected():
    with pytest.raises(UnsupportedModelError, match="multi-output"):
        transpile(DecisionTreeRegressor().fit(X, np.c_[X[:, 0], X[:, 1]]))
    with pytest.raises(UnsupportedModelError, match="multi-output"):
        transpile(LinearRegression().fit(X, np.c_[X[:, 0], X[:, 1]]))


def test_invalid_prefix():
    with pytest.raises(ValueError):
        transpile(DecisionTreeClassifier().fit(X, [0, 0, 1, 1]), prefix="1bad-name")
