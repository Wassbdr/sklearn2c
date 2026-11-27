#!/usr/bin/env bash
# Cross-compile a generated model for an ARM Cortex-M4F and report flash / RAM.
#   requires: arm-none-eabi-gcc, arm-none-eabi-size (apt install gcc-arm-none-eabi)
#   usage:    examples/arm_cortex_m/build.sh model.c [out_dir]
# Compiles and links only; nothing is flashed or run.
set -euo pipefail
MODEL_C=${1:?usage: build.sh model.c [out_dir]}
OUT=${2:-out/arm}
HERE=$(cd "$(dirname "$0")" && pwd)
CFLAGS=(-std=c99 -Wall -Werror -Os -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16
        -ffunction-sections -fdata-sections -fstack-usage)
mkdir -p "$OUT"
arm-none-eabi-gcc "${CFLAGS[@]}" -DSKLEARN2C_NO_MAIN -c "$MODEL_C" -o "$OUT/model.o"
arm-none-eabi-gcc "${CFLAGS[@]}" -c "$HERE/main.c" -o "$OUT/main.o"
arm-none-eabi-gcc "${CFLAGS[@]}" -nostartfiles --specs=nano.specs --specs=nosys.specs -T "$HERE/cortex_m4.ld" -Wl,--gc-sections \
    "$OUT/main.o" "$OUT/model.o" -o "$OUT/firmware.elf"

read -r mt md mb _ < <(arm-none-eabi-size "$OUT/model.o" | tail -1)
read -r ft fd fb _ < <(arm-none-eabi-size "$OUT/firmware.elf" | tail -1)
stack=$(awk -F'\t' '{s += $2} END {print s + 0}' "$OUT"/model.su)
echo "target   : Cortex-M4F (thumb, hard float), arm-none-eabi-gcc $(arm-none-eabi-gcc -dumpversion) -Os"
echo "model.o  : flash $((mt + md)) B, static RAM $((md + mb)) B, stack <= $stack B"
echo "firmware : flash $((ft + fd)) B of 1048576, RAM $((fd + fb)) B of 131072 (+ stack)  -> $OUT/firmware.elf"
