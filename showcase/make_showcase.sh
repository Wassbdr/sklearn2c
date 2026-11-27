#!/usr/bin/env bash
# Rebuild every showcase artifact from a clean clone (fixed seeds).
#
# Requirements:
#   python3 >= 3.9 with venv, gcc + binutils (size)
#   arm-none-eabi-gcc + libnewlib-arm-none-eabi   (Cortex-M4 sizes; apt install gcc-arm-none-eabi libnewlib-arm-none-eabi)
#   vhs >= 0.10 (charmbracelet) + ttyd + ffmpeg  (terminal demo -> gif/mp4)
#
# Outputs: benchmarks/results/parity.{md,json}, showcase/results/*.txt,
#          showcase/demo.gif, showcase/demo.mp4, showcase/poster.png, showcase/showcase.json
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD

for tool in python3 gcc size arm-none-eabi-gcc vhs ttyd ffmpeg ffprobe; do
  command -v "$tool" >/dev/null || { echo "missing required tool: $tool" >&2; exit 1; }
done

if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q ".[test,showcase]"

mkdir -p showcase/results showcase/work
rm -rf showcase/work/*

echo "== tests"
python -m pytest -q -p no:cacheprovider | tee showcase/results/pytest.txt

echo "== parity table"
python benchmarks/parity_table.py

echo "== Cortex-M4 firmware builds"
(
  cd showcase/work
  python "$ROOT/examples/train_fault_classifier.py" >/dev/null
  python "$ROOT/examples/train_fault_classifier.py" --model forest >/dev/null
  for m in fault_tree fault_forest; do
    sklearn2c "$m.joblib" -o "$m.c" >/dev/null
    echo "## $m"
    "$ROOT/examples/arm_cortex_m/build.sh" "$m.c" "arm/$m"
  done
) | tee showcase/results/arm_cortex_m4.txt

echo "== demo (vhs, live run)"
vhs showcase/demo.tape
# palette-optimised GIF (<= 1280 px wide, <= 40 s) and an H.264 MP4 of the same recording
ffmpeg -loglevel error -y -i showcase/work/demo.webm \
  -vf "fps=10,scale=1280:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" \
  showcase/demo.gif
ffmpeg -loglevel error -y -i showcase/work/demo.webm -c:v libx264 -pix_fmt yuv420p -crf 23 -movflags +faststart \
  -vf "scale=1280:-2" showcase/demo.mp4

echo "== poster + showcase.json"
python showcase/make_poster.py
python showcase/build_json.py

python - <<'PY'
import json, os, subprocess
d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          "showcase/demo.mp4"], capture_output=True, text=True).stdout)
gif = os.path.getsize("showcase/demo.gif") / 1e6
print(f"demo: {d:.1f} s, gif {gif:.2f} MB")
assert d <= 40.5 and gif < 10, "demo exceeds 40 s or 10 MB"
json.load(open("showcase/showcase.json"))
PY
echo "done"
