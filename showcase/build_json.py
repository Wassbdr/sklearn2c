#!/usr/bin/env python3
"""Write showcase/showcase.json from the measured results (no hand-typed numbers)."""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
res = json.loads((ROOT / "benchmarks" / "results" / "parity.json").read_text())
rows = {r["model"]: r for r in res["rows"]}
setup = res["setup"]
hw = setup["hardware"]
data = "synthetic make_classification/make_regression, seed 0"
arm = (ROOT / "showcase" / "results" / "arm_cortex_m4.txt").read_text()
fw = re.search(r"fault_forest.*?firmware : flash (\d+) B.*?RAM (\d+) B", arm, re.S)
tests = (ROOT / "showcase" / "results" / "pytest.txt").read_text().strip().splitlines()[-1]
n_total = sum(r["n_samples"] for r in res["rows"])
n_bad = sum(r["n_mismatches"] for r in res["rows"])
stress = {s["accumulator"]: s for s in res["linear_boundary_stress"]}
PT = "python benchmarks/parity_table.py"

results = [
    {"label": "Compiled-C vs sklearn.predict mismatches, 15 estimator types",
     "value": f"{n_bad} / {n_total:,} samples", "command": PT,
     "setup": f"{data}; held-out + random + split-boundary probes; gcc -O2 -shared via ctypes; {hw}"},
    {"label": f"DecisionTreeClassifier ({rows['DecisionTreeClassifier']['params']}) latency per sample",
     "value": f"{rows['DecisionTreeClassifier']['latency_ns']:.1f} ns", "command": PT,
     "setup": f"{data}; gcc -O2; {hw}"},
    {"label": f"RandomForestClassifier ({rows['RandomForestClassifier']['params']}) latency per sample",
     "value": f"{rows['RandomForestClassifier']['latency_ns'] / 1000:.2f} us", "command": PT,
     "setup": f"{data}; gcc -O2; {hw}"},
    {"label": f"Cortex-M4F flash, DecisionTreeClassifier ({rows['DecisionTreeClassifier']['params']})",
     "value": f"{rows['DecisionTreeClassifier']['cortex_m4_flash_bytes']:,} B", "command": PT,
     "setup": f"{data}; {setup.get('arm_compiler')}; object size, compiled not run"},
    {"label": f"Cortex-M4F flash, RandomForestClassifier ({rows['RandomForestClassifier']['params']})",
     "value": f"{rows['RandomForestClassifier']['cortex_m4_flash_bytes'] / 1024:.1f} KiB", "command": PT,
     "setup": f"{data}; {setup.get('arm_compiler')}; object size, compiled not run"},
    {"label": "Bare-metal firmware with the 20-tree fault forest (flash / RAM)",
     "value": f"{int(fw.group(1)):,} B / {fw.group(2)} B" if fw else "n/a",
     "command": "examples/arm_cortex_m/build.sh fault_forest.c",
     "setup": "examples/train_fault_classifier.py --model forest (seed 0); arm-none-eabi-gcc -Os, linked, not run"},
    {"label": "Linear classifier inputs exactly on the decision boundary, label flips",
     "value": f"float {stress['float']['n_mismatches']}/{stress['float']['n_samples']}, "
              f"double {stress['double']['n_mismatches']}/{stress['double']['n_samples']}",
     "command": PT, "setup": f"binary LogisticRegression, inputs projected onto z=0, seed 0; {hw}"},
    {"label": "Test suite (pytest, compiles every model with gcc)", "value": tests,
     "command": "python -m pytest", "setup": f"{setup['versions']}; {hw}"},
]

doc = {
    "slug": "sklearn2c",
    "title": "sklearn2c",
    "tagline": "Transpile fitted scikit-learn models to dependency-free C for microcontrollers, "
               "verified sample by sample against the compiled code.",
    "problem": "scikit-learn does not run on microcontrollers, and runtimes such as TFLite Micro or ONNX add flash, "
               "RAM and a toolchain. Hand-porting a model to C is error-prone: the original transpiler returned "
               "class indices instead of labels and its 'verification' never executed the C.",
    "approach": "sklearn2c reads the fitted parameters (tree nodes, forest tables, coefficients) and emits C99 with "
                "them as const data and a single predict(const float *x). Leaves map through classes_, tree "
                "thresholds are emitted as the largest float32 <= the float64 threshold so routing is bit-identical "
                "to sklearn, and verify_predictions compiles the C with gcc, calls it through ctypes and compares "
                "every sample, including inputs exactly on each split.",
    "stack": ["Python", "scikit-learn", "NumPy", "C99", "gcc", "arm-none-eabi-gcc", "ctypes", "pytest",
              "GitHub Actions"],
    "demo": {
        "gif": "showcase/demo.gif", "mp4": "showcase/demo.mp4", "poster": "showcase/poster.png", "kind": "live",
        "caption": "Live terminal run: train a fault classifier with labels 10/20/30, export it to C, compile it "
                   "with gcc, check parity on held-out data, print the parity table for every supported model, "
                   "then show the clear error raised for string labels.",
    },
    "reproduce": ["pip install .", "python -m pytest", "bash showcase/make_showcase.sh"],
    "results": results,
    "limits": [
        "Integer class labels only (int32); string labels must be encoded with LabelEncoder first.",
        "Linear models accumulate in float by default; inputs within float rounding of the decision boundary can "
        "flip label (compile with -DSKLEARN2C_REAL=double for exact parity).",
        "Forest classifiers accumulate probabilities in double to match sklearn exactly; this is software-emulated "
        "on Cortex-M4F.",
        "No Pipeline / preprocessing export, no NaN (missing value) routing, no gradient boosting, SVM kernels or MLPs.",
        "Latency measured on an x86 VM, not on a microcontroller; ARM builds are compiled and linked, not run.",
    ],
    "role": "TODO",
    "repo": "https://github.com/Wassbdr/sklearn2c",
}
(ROOT / "showcase" / "showcase.json").write_text(json.dumps(doc, indent=2) + "\n")
print("wrote showcase/showcase.json")
