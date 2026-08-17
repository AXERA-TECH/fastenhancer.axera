# FastEnhancer.AXERA Export

[FastEnhancer](https://github.com/aask1357/fastenhancer) (ICASSP 2026) 的 AX650/AX630C 导出和量化。

## 模型

| 模型 | 采样率 | 参数 | n_fft | hop |
|------|--------|------|-------|-----|
| fastenhancer_t_16k | 16 kHz | 21,774 | 512 | 256 |
| fastenhancer_t_48k | 48 kHz | 28,078 | 1024 | 512 |

## 目录结构

```
fastenhancer-ax650-export/
├── python/              # 导出脚本 + 推理 SDK
│   ├── fastenhancer_sdk/    # 推理 SDK (numpy + pyaxengine)
│   │   ├── inference.py         # FastEnhancer 类 (STFT + compress + NPU core + decompress + ISTFT)
│   │   ├── 16k/window_istft.npy # 16kHz STFT/ISTFT 窗
│   │   └── 48k/window_istft.npy # 48kHz STFT/ISTFT 窗
│   ├── build_corenet.py     # CoreNet ONNX 导出
│   ├── generate_calib.py    # 校准数据生成
│   ├── demo.py              # 板端降噪 demo
│   └── requirements.txt     # 推理依赖 (numpy + pyaxengine)
├── cpp/                 # C++ 推理 SDK
│   ├── config.h              # 模型维度 (#ifdef FE_MODEL_48K)
│   ├── windows.h             # STFT/ISTFT 窗 (#ifdef FE_MODEL_48K)
│   ├── fastenhancer.hpp/cpp  # FastEnhancer 类实现
│   ├── main.cpp + CMakeLists.txt
├── model_convert/       # Pulsar2 编译配置 (16k + 48k 共用 compile.sh)
│   ├── compile.sh            # bash compile.sh 16k   或   48k
│   ├── 16k/                  # model_meta.json + pulsar2_config.json
│   └── 48k/
└── README.md
```

## 环境

```bash
conda create -n fastenhancer python=3.12
conda activate fastenhancer
pip install -r requirements.txt
```

## 步骤 1：获取原始模型

从 [FastEnhancer Releases](https://github.com/aask1357/fastenhancer/releases) 下载预训练权重：

| 模型 | Release | 文件 |
|------|---------|------|
| 16kHz | [ckpt-vd-v1.0.0](https://github.com/aask1357/fastenhancer/releases/tag/ckpt-vd-v1.0.0) | `fastenhancer_t.zip` |
| 48kHz | [ckpt-v1.0.0-48khz](https://github.com/aask1357/fastenhancer/releases/tag/ckpt-v1.0.0-48khz) | `fastenhancer_t.zip` |

同时克隆源码：

```bash
git clone --depth 1 https://github.com/aask1357/fastenhancer.git origin/
```

解压权重到对应位置：

```bash
# 16kHz
unzip fastenhancer_t.zip -d origin/logs/fastenhancer_t/

# 48kHz（注意：48kHz 版 zip 内无子目录）
mkdir -p origin/logs/fastenhancer_t48
unzip fastenhancer_t.zip -d origin/logs/fastenhancer_t48/
```

最终目录结构：

```text
origin/
├── models/fastenhancer/default/model.py   # ONNXModel 定义
├── functional/audio_modules.py            # ONNXSTFT (流式缓存 FFT)
├── wrappers/, utils/, configs/
├── scripts/                               # 官方导出/测试脚本
└── logs/
    ├── fastenhancer_t/                    # 16kHz checkpoint
    │   ├── config.yaml
    │   └── 00500.pth
    └── fastenhancer_t48/                  # 48kHz checkpoint
        ├── config.yaml
        └── 00500.pth
```

## 步骤 2：导出 CoreNet ONNX

CoreNet 架构：压缩后的 STFT → 复数 mask，无 Pow/ReduceL2，NPU 编译友好。
压缩/解压与 STFT/ISTFT 保留在主机 CPU。

```bash
cd origin

# 16kHz
python ../python/build_corenet.py --model 16k

# 48kHz
python ../python/build_corenet.py --model 48k
```

产物：

```text
model_convert/16k/model.onnx    # comp_spec[1,256,1,2] + 2×GRU → mask[1,256,1,2]
model_convert/48k/model.onnx    # comp_spec[1,512,1,2] + 2×GRU → mask[1,512,1,2]
```

## 步骤 3：生成校准数据

需要真实语音音频。校准脚本生成的 `.tar.gz` 会自动放到 `model_convert/16k/calib/` 或 `model_convert/48k/calib/`。

```bash
cd origin

python ../python/generate_calib.py --model 16k \
  --audio <path/to/audio1.wav> <path/to/audio2.wav>

python ../python/generate_calib.py --model 48k \
  --audio <path/to/audio1.wav> <path/to/audio2.wav>
```

产物：`model_convert/{16k,48k}/calib/{comp_spec,cache_in_0,cache_in_1}.tar.gz`

## 步骤 4：Pulsar2 量化

```bash
# 16kHz
cd model_convert
bash compile.sh 16k

# 48kHz
bash compile.sh 48k
```

编译配置（`pulsar2_config.json` 关键参数）：

- 量化：全 U16（DEFAULT start/end tensor），权重 S8
- 校准：MinMax，16 样本
- 目标：AX650 NPU3
- highest_mix_precision: false

需要 Pulsar2 7.0 Docker 镜像：

```bash
docker images | grep pulsar2
```

产物：`model.axmodel`（16k ~148KB / 48k ~153KB）

## 验证

```bash
# Pulsar2 仿真
docker run --rm --network host -v $(pwd):/workspace pulsar2:7.0 \
  pulsar2 run --model /workspace/model.axmodel \
  --input_dir /workspace/input --output_dir /workspace/output
```

## 板端推理

编译完成后将 `model.axmodel` 放入 `python/` 同目录，即可在 AX650 板端运行：

### Python

```bash
conda create -n fastenhancer python=3.10
conda activate fastenhancer

# 安装 pyaxengine (https://github.com/AXERA-TECH/pyaxengine/releases/latest)
pip install axengine-x.x.x-py3-none-any.whl
pip install -r python/requirements.txt

# 运行
cd python
# 16kHz
python3 demo.py --model ../model_convert/16k/model.axmodel --input noisy.wav --output enhanced.wav
# 48kHz
python3 demo.py --model ../model_convert/48k/model.axmodel --input noisy.wav --output enhanced.wav
```

```python
from fastenhancer_sdk import FastEnhancer

fe = FastEnhancer("model_convert/16k/model.axmodel")   # 或 model_convert/48k/
enhanced = fe.enhance(noisy_wav)  # np.float32 [T] 16k/48kHz mono [-1,1]
```

SDK 自动从 `model_meta.json` 读取采样率和 STFT 参数，一套代码服务两个模型。
依赖仅 numpy + pyaxengine，无 torch/onnxruntime。

### C++

#### 交叉编译工具链

一键下载（Arm GNU 9.2 交叉编译器 + AX650 BSP SDK，放到 `cpp/toolchains/`，不入库）：

```bash
bash cpp/download_toolchains.sh
```

或手动准备（已装系统 `aarch64-linux-gnu-g++` 或已有工具链时）：

```bash
# 交叉编译器
wget https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
tar -xf gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
export TOOLCHAIN_ROOT=$(pwd)/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu

# AX650 BSP SDK（含 ax_engine/ax_sys/ax_interpreter 头文件和库）
git clone https://github.com/AXERA-TECH/ax650n_bsp_sdk.git --depth=1
export BSP_MSP_DIR=$(pwd)/ax650n_bsp_sdk/msp/out
```

#### 编译

用 `build_ax650.sh`（编译宏 `FE_MODEL_48K` 切换 16k/48k，无需改源码）：

```bash
cd cpp

# 16kHz（默认）
bash build_ax650.sh 16k    # -> bin/fastenhancer_ax650_16k

# 48kHz
bash build_ax650.sh 48k    # -> bin/fastenhancer_ax650_48k
```

或手动 CMake：

```bash
cd cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=$TOOLCHAIN_ROOT/bin/aarch64-none-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=$TOOLCHAIN_ROOT/bin/aarch64-none-linux-gnu-g++ \
  -DAX650=ON -DBSP_MSP_DIR=$BSP_MSP_DIR -DFE_MODEL=16k
make -j$(nproc)
```

#### 板端运行

```bash
# 16kHz
LD_LIBRARY_PATH=/soc/lib ./bin/fastenhancer_ax650_16k model_convert/16k/model.axmodel noisy.wav enhanced.wav
# 48kHz
LD_LIBRARY_PATH=/soc/lib ./bin/fastenhancer_ax650_48k model_convert/48k/model.axmodel noisy.wav enhanced.wav
```

#### 性能

| 模型 | Python RTF | C++ RTF |
|------|:--:|:--:|
| 16k | 0.20 | 0.02 |
| 48k | 0.31 | 0.06 |

> RTF = 推理时间 / 音频时长（不含模型加载），AX650 板端实测。

预编译模型和 SDK 也提供 HuggingFace 即取即用版（含 C++ 可执行文件）：
[FastEnhancer.AXERA](https://huggingface.co/AXERA-TECH/fastenhancer.axera)。

## 参考

- [FastEnhancer](https://github.com/aask1357/fastenhancer) — 原始模型
- [FastEnhancer.AXERA（HuggingFace 预编译模型+SDK）](https://huggingface.co/AXERA-TECH/fastenhancer.axera)
- [Magnetar](https://github.com/AXERA-TECH/Magnetar) — AXERA 模型部署工具链

## 引用

```bibtex
@inproceedings{fastenhancer2026,
  title     = {FastEnhancer: Speed-Optimized Streaming Neural Speech Enhancement},
  booktitle = {ICASSP},
  year      = {2026}
}
```
