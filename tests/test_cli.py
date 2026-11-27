import subprocess
import sys

import joblib
import numpy as np
import pytest
from conftest import requires_gcc
from sklearn.tree import DecisionTreeClassifier

from sklearn2c.cli import main

X = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [0.0, 0.0]])


@pytest.fixture
def tree_path(tmp_path):
    p = tmp_path / "tree.joblib"
    joblib.dump(DecisionTreeClassifier(random_state=0).fit(X, [10, 10, 20, 20]), p)
    return p


@requires_gcc
def test_cli_writes_and_verifies(tree_path, tmp_path, capsys):
    out = tmp_path / "tree.c"
    assert main([str(tree_path), "-o", str(out), "--verify"]) == 0
    assert "model_predict" in out.read_text()
    assert "parity OK" in capsys.readouterr().out


@requires_gcc
def test_cli_verify_data_csv(tree_path, tmp_path, capsys):
    csv = tmp_path / "x.csv"
    csv.write_text("a,b\n" + "\n".join(",".join(map(str, r)) for r in X))
    assert main([str(tree_path), "-o", str(tmp_path / "t.c"), "--verify-data", str(csv)]) == 0
    assert "4/4 samples match" in capsys.readouterr().out


@requires_gcc
def test_cli_main_binary(tree_path, tmp_path):
    out = tmp_path / "tree.c"
    assert main([str(tree_path), "-o", str(out), "--main"]) == 0
    exe = tmp_path / "tree"
    subprocess.run(["gcc", "-std=c99", "-Wall", "-Werror", str(out), "-o", str(exe)], check=True)
    preds = [subprocess.run([str(exe), *map(str, row)], capture_output=True, text=True).stdout.strip()
             for row in X]
    assert preds == ["10", "10", "20", "20"]


def test_cli_unsupported_labels_exit_2(tmp_path, capsys):
    p = tmp_path / "s.joblib"
    joblib.dump(DecisionTreeClassifier().fit(X, ["a", "a", "b", "b"]), p)
    assert main([str(p), "-o", str(tmp_path / "s.c")]) == 2
    assert "LabelEncoder" in capsys.readouterr().err
    assert not (tmp_path / "s.c").exists()


def test_cli_missing_file(tmp_path, capsys):
    assert main([str(tmp_path / "nope.joblib")]) == 2


def test_module_entry_point():
    r = subprocess.run([sys.executable, "-m", "sklearn2c", "--version"], capture_output=True, text=True)
    assert r.returncode == 0 and "sklearn2c" in r.stdout
