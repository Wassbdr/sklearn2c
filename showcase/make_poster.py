#!/usr/bin/env python3
"""showcase/poster.png (1280x720): Python model | generated C | parity table.

Everything is computed here: the model is trained (seed 0), transpiled, and the
table is read from benchmarks/results/parity.json written by parity_table.py."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))
from train_fault_classifier import make_data  # noqa: E402

from sklearn.model_selection import train_test_split  # noqa: E402

from sklearn2c import transpile, verify_predictions  # noqa: E402

BG, FG, DIM, ACC, OK = "#1e1e2e", "#cdd6f4", "#7f849c", "#89b4fa", "#a6e3a1"

X, y = make_data()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
model = DecisionTreeClassifier(max_depth=5, random_state=0).fit(Xtr, ytr)  # as in the example
report = verify_predictions(model, X=Xte)
c_lines = transpile(model).splitlines()
start = next(i for i, l in enumerate(c_lines) if l.startswith("int32_t model_predict_index"))
c_snippet = c_lines[start:start + 15] + ["    ...", "}", "", "/* label = classes_[index] */"] + c_lines[-3:]

py = """import joblib
from sklearn.tree import DecisionTreeClassifier

# synthetic vibration data (seed 0)
# features: rms_g, peak_g, kurtosis, temp_c
# labels: 10 healthy, 20 imbalance,
#         30 bearing wear
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)
joblib.dump(model, "fault_tree.joblib")
""".splitlines() + ["", "$ sklearn2c fault_tree.joblib \\", "    -o fault_tree.c \\", "    --verify-data fault_test.csv",
                    report.summary().split(" (")[0], "(" + report.summary().split(" (", 1)[1]]

res = json.loads((ROOT / "benchmarks" / "results" / "parity.json").read_text())
pick = ["DecisionTreeClassifier", "RandomForestClassifier", "LogisticRegression", "LinearSVC",
        "DecisionTreeRegressor", "RandomForestRegressor", "LinearRegression", "Ridge"]
rows = [r for r in res["rows"] if r["model"] in pick]

fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
fig.text(0.03, 0.93, "sklearn2c", color=ACC, fontsize=30, weight="bold", family="DejaVu Sans")
fig.text(0.235, 0.935, "scikit-learn models -> dependency-free C for microcontrollers,\n"
         "checked sample by sample against the compiled code", color=FG, fontsize=13, va="center")


def panel(x, w, title, lines, size=9.6, colors=None):
    ax = fig.add_axes([x, 0.10, w, 0.76])
    ax.set_facecolor("#181825")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#313244")
    ax.set_title(title, color=DIM, fontsize=11, loc="left")
    for i, line in enumerate(lines):
        col = (colors or {}).get(i, FG)
        ax.text(0.03, 0.97 - i * 0.045, line, color=col, fontsize=size, family="DejaVu Sans Mono",
                va="top", transform=ax.transAxes)


py_colors = {i: DIM for i, l in enumerate(py) if l.startswith("#")}
py_colors.update({len(py) - 2: OK, len(py) - 1: OK})
panel(0.03, 0.285, "Python  (examples/train_fault_classifier.py)", py, 8.4, py_colors)
panel(0.33, 0.30, "generated C  (fault_tree.c)", c_snippet, 8.8,
      {i: ACC for i, l in enumerate(c_snippet) if "return MODEL_CLASSES" in l or "MODEL_CLASSES" in l})

ax = fig.add_axes([0.645, 0.10, 0.335, 0.76])
ax.set_facecolor("#181825"); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_color("#313244")
ax.set_title("parity: compiled C vs sklearn.predict()", color=DIM, fontsize=11, loc="left")
hdr = ["model", "compared", "mism.", "M4 flash", "x86 ns"]
xs = [0.03, 0.50, 0.63, 0.82, 0.97]
for x, h in zip(xs, hdr):
    ax.text(x, 0.95, h, color=DIM, fontsize=9.5, ha="left" if x < 0.1 else "right", transform=ax.transAxes,
            family="DejaVu Sans Mono")
for i, r in enumerate(rows):
    yy = 0.88 - i * 0.075
    name = r["model"].replace("Classifier", "Clf").replace("Regressor", "Reg").replace("Regression", "Reg")
    vals = [name, f"{r['n_samples']:,}", str(r["n_mismatches"]),
            f"{r.get('cortex_m4_flash_bytes', 0) / 1024:.1f} KB" if r.get("cortex_m4_flash_bytes", 0) >= 1024
            else f"{r.get('cortex_m4_flash_bytes', 0)} B", f"{r['latency_ns']:.0f}"]
    for x, v in zip(xs, vals):
        ax.text(x, yy, v, color=OK if v == "0" else FG, fontsize=9.5, ha="left" if x < 0.1 else "right",
                transform=ax.transAxes, family="DejaVu Sans Mono")
s = res["setup"]
ax.text(0.03, 0.22, "labels: exact match;  regression: |err| <= 1e-5*scale\n"
        "data: make_classification / make_regression, seed 0\n"
        "flash: arm-none-eabi-gcc -Os, Cortex-M4F\n"
        "latency: x86 VM, not a microcontroller", color=DIM, fontsize=8.6, va="top",
        transform=ax.transAxes, family="DejaVu Sans Mono", linespacing=1.5)
bs = {b["accumulator"]: b for b in res["linear_boundary_stress"]}
fig.text(0.03, 0.045, f"{sum(r['n_samples'] for r in res['rows']):,} samples compared across "
         f"{len(res['rows'])} estimator types, {sum(r['n_mismatches'] for r in res['rows'])} mismatches.   "
         "pip install .  ->  sklearn2c model.joblib -o model.c --verify", color=FG, fontsize=11)
out = ROOT / "showcase" / "poster.png"
fig.savefig(out, dpi=100, facecolor=BG)
print(f"wrote {out.relative_to(ROOT)}")
