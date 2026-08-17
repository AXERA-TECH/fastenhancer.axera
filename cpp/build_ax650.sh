#!/usr/bin/env bash
# Cross-compile FastEnhancer C++ demo for AX650.
#
# Prerequisites (run `bash cpp/download_toolchains.sh` once, or export the paths):
#   TOOLCHAIN_ROOT  aarch64 cross toolchain (Arm GNU 9.2)
#   BSP_MSP_DIR     AX650 BSP msp/out (ax650n_bsp_sdk)
#
# Usage:
#   bash build_ax650.sh          # 16kHz
#   bash build_ax650.sh 48k      # 48kHz
set -euo pipefail

CPP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:-16k}"
BUILD_DIR="${CPP_DIR}/build/${MODEL}"
TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-${CPP_DIR}/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu}"
BSP_MSP_DIR="${BSP_MSP_DIR:-${CPP_DIR}/toolchains/ax650n_bsp_sdk/msp/out}"

if [[ ! -x "${TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-g++" ]]; then
  echo "ERROR: AArch64 toolchain not found: ${TOOLCHAIN_ROOT}" >&2
  echo "Run: bash cpp/download_toolchains.sh" >&2
  exit 2
fi
if [[ ! -f "${BSP_MSP_DIR}/include/ax_engine_api.h" ]]; then
  echo "ERROR: AX650 BSP msp/out not found: ${BSP_MSP_DIR}" >&2
  echo "Run: bash cpp/download_toolchains.sh" >&2
  exit 2
fi

rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR"
cmake -S "$CPP_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER="${TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-gcc" \
    -DCMAKE_CXX_COMPILER="${TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-g++" \
    -DAX650=ON \
    -DBSP_MSP_DIR="$BSP_MSP_DIR" \
    -DFE_MODEL="$MODEL"
cmake --build "$BUILD_DIR" -j"$(nproc)"

mkdir -p "$CPP_DIR/bin"
cp "$BUILD_DIR/fastenhancer_demo" "$CPP_DIR/bin/fastenhancer_ax650_${MODEL}"
"${TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-strip" --strip-unneeded \
  "$CPP_DIR/bin/fastenhancer_ax650_${MODEL}"

echo ""
echo "Built: $CPP_DIR/bin/fastenhancer_ax650_${MODEL} (${MODEL})"
echo "Deploy to AX650 board:"
echo "  scp $CPP_DIR/bin/fastenhancer_ax650_${MODEL} model.axmodel noisy.wav root@<board>:/tmp/"
echo "  ssh root@<board> 'cd /tmp && LD_LIBRARY_PATH=/soc/lib ./fastenhancer_ax650_${MODEL} model.axmodel noisy.wav enhanced.wav'"
