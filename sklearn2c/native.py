"""Compile generated C with a real compiler and call it from Python (ctypes)."""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import CompilationError


def find_compiler(cc: str | None = None) -> str:
    cc = cc or os.environ.get("CC", "gcc")
    if shutil.which(cc) is None:
        raise CompilationError(f"C compiler {cc!r} not found on PATH (set --cc or $CC)")
    return cc


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise CompilationError(f"{' '.join(cmd)} failed:\n{proc.stderr.strip()}")
    return proc


@dataclass
class CompiledModel:
    """A generated model compiled to a shared library and loaded with ctypes."""

    lib: ctypes.CDLL
    path: Path
    is_classifier: bool
    n_features: int

    def predict(self, X) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"expected X of shape (n, {self.n_features}), got {X.shape}")
        n = X.shape[0]
        out = np.empty(n, dtype=np.int32 if self.is_classifier else np.float64)
        self.lib.sklearn2c_batch(
            X.ctypes.data_as(ctypes.c_void_p), ctypes.c_int32(n), out.ctypes.data_as(ctypes.c_void_p)
        )
        return out


def _harness(c_file: Path, prefix: str, is_classifier: bool) -> str:
    P = prefix.upper()
    out_t = "int32_t" if is_classifier else "double"
    return f"""#define SKLEARN2C_NO_MAIN
#include "{c_file.name}"
void sklearn2c_batch(const float *X, int32_t n, {out_t} *out) {{
    for (int32_t i = 0; i < n; i++) out[i] = ({out_t}){prefix}_predict(X + (long)i * {P}_N_FEATURES);
}}
"""


def compile_shared(c_source: str, *, prefix: str, is_classifier: bool, n_features: int,
                   cc: str | None = None, cflags: tuple[str, ...] = ("-O2",),
                   workdir: str | os.PathLike | None = None) -> CompiledModel:
    """Compile ``c_source`` with ``cc -shared -fPIC`` and load it."""
    cc = find_compiler(cc)
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="sklearn2c_"))
    work.mkdir(parents=True, exist_ok=True)
    c_file = work / f"{prefix}.c"
    c_file.write_text(c_source)
    harness = work / f"{prefix}_harness.c"
    harness.write_text(_harness(c_file, prefix, is_classifier))
    so = work / f"{prefix}.so"
    _run([cc, "-std=c99", "-Wall", "-Werror", *cflags, "-shared", "-fPIC", str(harness), "-o", str(so)])
    return CompiledModel(ctypes.CDLL(str(so.resolve())), so, is_classifier, n_features)


def object_size(c_source: str, *, cc: str | None = None, cflags: tuple[str, ...] = ("-Os",),
                size_tool: str | None = None) -> dict:
    """Compile to an object file and return ``{text, data, bss}`` in bytes."""
    cc = find_compiler(cc)
    size_tool = size_tool or (cc[: -len("gcc")] + "size" if cc.endswith("gcc") else "size")
    with tempfile.TemporaryDirectory(prefix="sklearn2c_") as d:
        src, obj = Path(d) / "m.c", Path(d) / "m.o"
        src.write_text(c_source)
        _run([cc, "-std=c99", "-Wall", "-Werror", *cflags, "-c", str(src), "-o", str(obj)])
        out = _run([size_tool, "--format=berkeley", str(obj)]).stdout.splitlines()[1].split()
    text, data, bss = (int(v) for v in out[:3])
    return {"text": text, "data": data, "bss": bss}


_BENCH = r"""#define SKLEARN2C_NO_MAIN
#include "{c_name}"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}}

int main(int argc, char **argv) {{
    int n = atoi(argv[2]);
    float *X = malloc(sizeof(float) * n * {P}_N_FEATURES);
    FILE *f = fopen(argv[1], "rb");
    if (!f || fread(X, sizeof(float), (size_t)n * {P}_N_FEATURES, f) != (size_t)n * {P}_N_FEATURES) return 1;
    fclose(f);
    volatile double sink = 0;
    long reps = 1;
    double best = 1e30;
    for (;;) {{ /* grow until one pass takes >= 20 ms */
        double t0 = now();
        for (long r = 0; r < reps; r++)
            for (int i = 0; i < n; i++) sink += (double){prefix}_predict(X + (long)i * {P}_N_FEATURES);
        double dt = now() - t0;
        if (dt >= 0.02) {{ best = dt; break; }}
        reps *= 2;
    }}
    for (int k = 0; k < 7; k++) {{ /* best of 7 */
        double t0 = now();
        for (long r = 0; r < reps; r++)
            for (int i = 0; i < n; i++) sink += (double){prefix}_predict(X + (long)i * {P}_N_FEATURES);
        double dt = now() - t0;
        if (dt < best) best = dt;
    }}
    printf("%.3f\n", best / ((double)reps * n) * 1e9);
    return 0;
}}
"""


def latency_ns(c_source: str, X, *, prefix: str = "model", cc: str | None = None,
               cflags: tuple[str, ...] = ("-O2",)) -> float:
    """Per-sample latency of ``<prefix>_predict`` in ns (best of 7 timed passes over X)."""
    cc = find_compiler(cc)
    X = np.ascontiguousarray(X, dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix="sklearn2c_") as d:
        d = Path(d)
        (d / "m.c").write_text(c_source)
        (d / "bench.c").write_text(_BENCH.format(c_name="m.c", prefix=prefix, P=prefix.upper()))
        X.tofile(d / "x.bin")
        _run([cc, "-std=c99", *cflags, str(d / "bench.c"), "-o", str(d / "bench")])
        out = _run([str(d / "bench"), str(d / "x.bin"), str(X.shape[0])]).stdout
    return float(re.findall(r"[\d.]+", out)[0])
