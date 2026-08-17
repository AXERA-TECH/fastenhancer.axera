# FastEnhancer C++ SDK

AX650 板端 C++ 推理（`fastenhancer.hpp/cpp` + `main.cpp`），与 `python/fastenhancer_sdk`
逐行对齐：STFT → compress → NPU core（2×GRU）→ decompress → ISTFT。
编译宏 `FE_MODEL_48K` 切换 16k/48k。

## 交叉编译工具链

一键下载（Arm GNU 9.2 交叉编译器 + AX650 BSP SDK，放到 `toolchains/`，不入库）：

```bash
bash download_toolchains.sh
```

或手动准备（已装系统 `aarch64-linux-gnu-g++` 或已有工具链时）：

```bash
wget https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
tar -xf gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
export TOOLCHAIN_ROOT=$(pwd)/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu

git clone https://github.com/AXERA-TECH/ax650n_bsp_sdk.git --depth=1
export BSP_MSP_DIR=$(pwd)/ax650n_bsp_sdk/msp/out
```

## 编译

```bash
bash build_ax650.sh 16k    # -> bin/fastenhancer_ax650_16k
bash build_ax650.sh 48k    # -> bin/fastenhancer_ax650_48k
```

或手动 CMake：

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=$TOOLCHAIN_ROOT/bin/aarch64-none-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=$TOOLCHAIN_ROOT/bin/aarch64-none-linux-gnu-g++ \
  -DAX650=ON -DBSP_MSP_DIR=$BSP_MSP_DIR -DFE_MODEL=16k
make -j$(nproc)
```

## 板端运行

```bash
# 16kHz
LD_LIBRARY_PATH=/soc/lib ./bin/fastenhancer_ax650_16k model_convert/16k/model.axmodel noisy.wav enhanced.wav
# 48kHz
LD_LIBRARY_PATH=/soc/lib ./bin/fastenhancer_ax650_48k model_convert/48k/model.axmodel noisy.wav enhanced.wav
```

## 性能（AX650 板端实测）

| 模型 | Python RTF | C++ RTF |
|------|:--:|:--:|
| 16k | 0.20 | 0.02 |
| 48k | 0.31 | 0.06 |

> RTF = 推理时间 / 音频时长（不含模型加载）。
