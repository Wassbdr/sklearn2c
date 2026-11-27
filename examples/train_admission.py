#!/usr/bin/env python3
"""LogisticRegression on examples/data/students.csv (synthetic, seed 42) -> admission.joblib

students.csv: exam1, exam2 ~ U(30, 100), hours_studied ~ U(0, 10);
admitted = mean(exam1, exam2) > 65 and hours_studied > 3.
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

d = np.loadtxt(Path(__file__).parent / "data" / "students.csv", delimiter=",", skiprows=1)
X, y = d[:, :3], d[:, 3].astype(int)
model = LogisticRegression(max_iter=1000).fit(X, y)
joblib.dump(model, "admission.joblib")
print(f"LogisticRegression accuracy {model.score(X, y):.3f} -> admission.joblib")
