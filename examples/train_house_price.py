#!/usr/bin/env python3
"""LinearRegression on examples/data/houses.csv (40 rows shipped with the repo) -> house_price.joblib"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

data = np.genfromtxt(Path(__file__).parent / "data" / "houses.csv", delimiter=",", names=True,
                     dtype=None, encoding="utf-8")
X = np.c_[data["size"], data["nb_rooms"], data["garden"]]
model = LinearRegression().fit(X, data["price"])
joblib.dump(model, "house_price.joblib")
print(f"LinearRegression R^2 {model.score(X, data['price']):.3f} -> house_price.joblib")
