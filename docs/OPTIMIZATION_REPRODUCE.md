# LingBot-Depth v0.5 优化复现手册

本手册复现 RTX 5080 上已经完成的 level 3、`torch.compile`、TensorRT FP16 和组合优化，以及 8 个官方场景的视觉对比。所有项目路径均相对当前仓库根目录，命令从克隆得到的 `lingbot-depth-realsense/` 执行。本文性能数字是 RTX 5080 的历史测量；RTX 5060 Ti 的复现见 [独立记录](OPTIMIZATION_5060_TI.txt)。

已安装环境、下载权重和构建引擎的本机，可以直接从“复现编译优化”或“复现 TensorRT 测试”开始。新环境先完成环境与权重准备。原始测量汇总已保存为 [2026 年 10 月 13 日结果快照](optimization-results-20261013.json)，重新测试产生的报告保存在 `result/optimization/`。

## 已完成的结果与计时口径

| 配置 | 平均耗时 | 测量范围 |
|---|---:|---|
| 默认 level 9，BF16 | 129.73 ms | 场景 0，预热后 5 次 |
| 原始 level 3，BF16 | 70.70 ms | 8 场景，深度、掩码及点云 |
| level 3，只编译解码器，reduce-overhead | 67.84 ms | 8 场景，深度、掩码及点云 |
| level 3，编译前向路径，default | 55.96 ms | 8 场景，深度、掩码及点云 |
| level 3，编译前向路径，reduce-overhead | 55.63 ms | 8 场景，深度、掩码及点云 |
| level 3，编译前向路径，max-autotune | 51.75 ms | 8 场景，深度、掩码及点云 |
| TensorRT FP16，2000 tokens | 31.82 ms | 8 场景，仅深度引擎 |
| TensorRT FP16 加 GPU 点云投影 | 31.89 ms | 8 场景，无输出掩码 |
| TensorRT CUDA Graph 加编译后点云投影 | 31.82 ms | 8 场景，无输出掩码 |

输入和输出分辨率均为 **640×480**，batch size 为 **1**。level 3 对应目标 `num_tokens=2000`；实际 patch 网格会按宽高比取整。参考仓库 README 的 `1200 tokens` 对应 level 0，不能当作 level 3 比较。

模型常驻 GPU，输入已完成预处理和 GPU 传输。每次计时前后调用 `torch.cuda.synchronize()`，记录逐帧主机墙钟时间。**启动、模型加载、编译、引擎构建、预处理、输入传输、引擎输入缓冲区复制和文件读写均不计入 latency。** 编译测试每场景预热后测量 20 次；最终 TRT 测试每场景每种模式测量 50 次，再对 8 个场景平均。

`max-autotune` 首次编译第一帧约 108 秒，未计入推理耗时。遇到新的动态 token 长度仍可能触发编译，因此常驻服务应完成代表性输入的预热后再开始记录延迟。`reduce-overhead` 日志中部分计算因涉及 CPU 数据而跳过 CUDA Graph。

### 输出数值差异

`torch.compile` 保留原模型的深度 token 掩码、输出有效性掩码和 BF16 autocast，但算子融合会产生浮点数值差异。

成功的 TRT 路线来自 [Ar-Ray-code/lingbot-depth-trt](https://github.com/Ar-Ray-code/lingbot-depth-trt/blob/60e69ab6ea5c65a2f19a22388a20d5969a8f556b/tools/export_trt.py)，它采用整模型 FP16 和固定形状 ONNX 导出，并有以下行为变化：

- 关闭可变深度 token 掩码，保留固定长度的 RGB 和深度 token。
- 关闭 RGB 插值的 antialias。
- 引擎只输出 `depth_reg`，没有输出有效性掩码；点云投影在引擎外执行。

平均相对差异在两种输出都具有有限正深度的像素上计算：`mean(abs(candidate - baseline) / baseline) × 100%`。本次 TRT 与原始 level 3 BF16 的逐场景平均值为 **0.287%–0.630%**；单独比较 TRT 与参考导出路径的 PyTorch FP16 输出，逐场景平均值为 **0.021%–0.125%**。

场景 3 与原始 level 3 的绝对差异 P99 为 **1.284 m**，最大为 **5.894 m**，所以平均差异较小不代表逐像素等价。这些是预测输出之间的差异；样例没有深度真值，不能据此判定测量精度优劣。同一 TRT 引擎的普通执行与 CUDA Graph 执行，深度输出已验证逐像素完全一致。

## 环境与版本

| 项目 | 已验证版本 |
|---|---|
| GPU | NVIDIA GeForce RTX 5080，16 GB |
| NVIDIA 驱动 | 595.84 |
| Python | 3.12.3 |
| PyTorch / CUDA runtime | 2.8.0+cu128 / CUDA 12.8 |
| torchvision | 0.23.0+cu128 |
| xFormers | 0.0.32.post2 |
| TensorRT | 10.12.0.36 |
| Torch-TensorRT | 2.8.0，保留原失败探测使用的依赖 |
| ONNX / ONNX Runtime | 1.23.2 / 1.31.0 |
| 官方源码提交 | `f3a237e434ae987bc38281476d6cfb5df3e4d739` |
| TRT 参考源码提交 | `60e69ab6ea5c65a2f19a22388a20d5969a8f556b` |

所有 Python 依赖固定在 [requirements.lock.txt](../tools/optimization/requirements.lock.txt)。`nvidia-smi` 显示的 CUDA 13.2 是驱动能力；本次 PyTorch 实际使用 CUDA 12.8 runtime。

复现脚本已经随本仓库保存在 [tools/optimization](../tools/optimization/)，8 个官方输入场景保存在 `examples/0`–`examples/7`。生成物放在 `result/`，权重放在 `checkpoints/`；不需要从其他机器复制文件。

从头准备当前仓库：

```bash
git clone git@github.com:terryma2024/lingbot-depth-realsense.git
cd lingbot-depth-realsense
```

### 准备独立 Python 环境

使用 Python 3.12、`git` 和 `uv`，按 [README](../README.md) 的环境步骤安装。为便于复现，这里列出核心命令：

```bash
uv venv --python 3.12 .venv-optimization
env -u PYTHONPATH uv pip install --python .venv-optimization/bin/python \
  --index https://download.pytorch.org/whl/cu128 \
  torch==2.8.0+cu128 torchvision==0.23.0+cu128 xformers==0.0.32.post2
env -u PYTHONPATH uv pip install --python .venv-optimization/bin/python -e '.[build,test]'
env -u PYTHONPATH uv pip check --python .venv-optimization/bin/python
```

依赖锁文件是原实验的版本快照，其中包含失败 Torch-TensorRT 探测所使用的依赖；当前成功路线不需要安装 Torch-TensorRT 或修改项目版本声明。优先使用 README 的安装方式。

命令中的 `env -u PYTHONPATH` 避免当前终端注入的 ROS 路径干扰 Python，不修改其他终端。构建依赖包含 CPU `onnxruntime`，以满足参考脚本导入；实际 TRT 测试跳过 ORT。

### 下载和校验 v0.5 权重

```bash
mkdir -p result/logs result/optimization

.venv-optimization/bin/hf download robbyant/lingbot-depth-pretrain-vitl-14-v0.5 \
  --revision 79204ed6b837f4fdd192cf563e59481fecfa0295 \
  --local-dir checkpoints/lingbot-depth-v0.5

.venv-optimization/bin/hf cache verify robbyant/lingbot-depth-pretrain-vitl-14-v0.5 \
  --revision 79204ed6b837f4fdd192cf563e59481fecfa0295 \
  --local-dir checkpoints/lingbot-depth-v0.5 --fail-on-missing-files

sha256sum checkpoints/lingbot-depth-v0.5/model.pt
```

`model.pt` 大小为 **1,284,837,952 字节**，SHA256 应为：

```text
b60cf27ddbd0e51e9b59b03475c0d39d02d2e48ecf8dbb5866f04d46802b3c23
```

## 运行默认样例和 level 3

RTX 5080 上 xFormers 自动选择的 FA3 曾报 `CUDA invalid argument`。本仓库在 `mdm/model/dinov2_rgbd/layers/attention.py` 中对 Blackwell 禁用 FA3；编译探测脚本也显式使用这一设置，保留其他可用 attention 后端。

```bash
mkdir -p result/logs result/optimization
env -u PYTHONPATH .venv-optimization/bin/python example.py \
  --model checkpoints/lingbot-depth-v0.5/model.pt \
  --device cuda --example 0 --output result/example_0

for scene in 0 1 2 3 4 5 6 7; do
  env -u PYTHONPATH .venv-optimization/bin/python example.py \
    --model checkpoints/lingbot-depth-v0.5/model.pt \
    --device cuda --example "$scene" --output "result/example_$scene" \
    > "result/logs/example_$scene.log" 2>&1 || break
done
```

`example.py` 默认是 level 9。level 3 在 Python 中通过 `model.infer(..., resolution_level=3)` 指定，不是该命令行的分辨率参数。

输入 RGB tensor 为 `[1,3,H,W]`、float32、范围 `[0,1]`；输入深度为 `[1,H,W]` 或 `[H,W]`，单位米。官方样例的 `raw_depth.png` 按毫米保存，需要除以 1000。相机实时数据必须使用 SDK 读取的比例和归一化内参，不能固定毫米比例或设备 ID。

## 复现编译优化

[probe_compile.py](../tools/optimization/probe_compile.py) 在每个配置的同一进程内先测原始 level 3，再编译模型并检查 8 个场景的输出。依次运行：

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
  .venv-optimization/bin/python -u tools/optimization/probe_compile.py \
  --scope decoder --mode reduce-overhead

env -u PYTHONPATH HF_HUB_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
  .venv-optimization/bin/python -u tools/optimization/probe_compile.py \
  --scope forward --mode default

env -u PYTHONPATH HF_HUB_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
  .venv-optimization/bin/python -u tools/optimization/probe_compile.py \
  --scope forward --mode reduce-overhead

env -u PYTHONPATH HF_HUB_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
  .venv-optimization/bin/python -u tools/optimization/probe_compile.py \
  --scope forward --mode max-autotune
```

生成 `result/optimization/compile-{scope}-{mode}.json`，包含逐场景基线、平均和 P95 耗时、输出数值差异与掩码差异。在常驻程序中，核心设置为：

```python
import torch
import xformers.ops.fmha as fmha
from mdm.model.v2 import MDMModel

fmha._set_use_fa3(False)
model = MDMModel.from_pretrained('checkpoints/lingbot-depth-v0.5/model.pt').cuda().eval()
model.forward = torch.compile(model.forward, mode='max-autotune', dynamic=True)

# image、depth、intrinsics 按上面的输入格式准备并放到 GPU。
output = model.infer(image, depth_in=depth, intrinsics=intrinsics, resolution_level=3)
```

## 复现 TensorRT 测试

### 准备固定版本参考仓库

```bash
git clone https://github.com/Ar-Ray-code/lingbot-depth-trt.git \
  result/optimization/reference-lingbot-depth-trt
git -C result/optimization/reference-lingbot-depth-trt checkout \
  60e69ab6ea5c65a2f19a22388a20d5969a8f556b
```

已有参考仓库时，只需检查 `git rev-parse HEAD` 与上面的提交一致。使用当前仓库的 `.venv-optimization` 执行参考脚本；参考仓库自身的依赖文件固定了较旧的 PyTorch，不用于本次 RTX 5080 环境。

### 构建 2000-token FP16 引擎

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 .venv-optimization/bin/python -u \
  result/optimization/reference-lingbot-depth-trt/tools/export_trt.py \
  --model checkpoints/lingbot-depth-v0.5/model.pt \
  --capture examples/0 \
  --precision fp16 --num-tokens 2000 --width 640 --height 480 \
  --work-dir result/optimization/reference-trt-nt2000-fp16 \
  --skip-ort --workspace-gb 4 --tactic-gb 2 --trt-warmup 10 --trt-runs 100 \
  > result/logs/reference-trt-export.log 2>&1
```

本次引擎构建约 **32.2 秒**，导出和构建时间不计入 latency。主要文件为：

| 文件 | 用途 |
|---|---|
| `reference-trt-nt2000-fp16/lingbot_depth_nt2000.onnx` | 固定形状 ONNX，约 607 MiB |
| `reference-trt-nt2000-fp16/lingbot_depth_nt2000.engine` | 2000-token FP16 引擎，约 617 MiB |
| `reference-trt-nt2000-fp16/conversion_report.json` | 导出、构建和单场景检查报告 |

以上路径均相对 `result/optimization/`。引擎输入 `image` 为 FP16 `[1, 3, 480, 640]`，`depth` 为 FP16 `[1, 480, 640]`，输出 `depth_refined` 为 FP16 `[1, 480, 640]`。RGB 仍归一化到 `[0, 1]`，深度仍以米为单位。其他 GPU、TensorRT 版本或输入尺寸应重新构建引擎。

转换报告中的 `torch.infer_s` 是未预热的单次调用，不能用它比较常驻 latency；采用下面的 8 场景测试结果。

### 测量引擎和组合方案

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
  .venv-optimization/bin/python -u tools/optimization/probe_reference_trt.py \
  > result/logs/reference-trt-benchmark.log 2>&1
```

[probe_reference_trt.py](../tools/optimization/probe_reference_trt.py) 先保存原始 level 3 基线，再加载参考 PyTorch FP16 路径和实际 TRT 引擎。TRT 与后处理使用同一个专用非默认 CUDA stream，测量以下三项：

1. TRT 深度引擎本身。
2. TRT 加 GPU float32 转换和点云投影。
3. 捕获整个 TRT 执行为 CUDA Graph，重放后执行 `torch.compile` 编译的点云投影。

第三项中的 `torch.compile` 编译的是 **TRT 引擎外的后处理**，不是再次编译 TensorRT 引擎。测试同时检查有限正深度、点云形状、点云 Z 与深度一致，以及 CUDA Graph 与普通 TRT 执行的深度完全一致。

生成 `result/optimization/reference-trt-benchmark.json` 和 `reference-trt-scene-0.pt` 到 `reference-trt-scene-7.pt`。报告分别记录 `vs_reference_fp16` 与 `vs_standard_level3`，避免混淆转换本身的差异和整个推理路径变化的差异。

### 原 BF16 Torch-TensorRT 路线的失败记录

原先尝试的官方 Torch-TensorRT 2.8 路线未形成可用的完整 BF16 引擎，因此未记为加速成功。整模型导出在 `mask_utils.py` 的数据依赖 token 长度处失败；解码器构建还遇到了 TensorRT 10.12 的 BF16 Resize 和转置卷积格式限制。

失败原因已在 [优化过程](MODEL_OPTIMIZATION_PROCESS.md) 汇总；原失败探测日志和 `result/optimization/tensorrt-bf16-attempt.json` 不随仓库提交。复现成功路线不需要重跑该失败尝试。

## 复现 8 行 10 列视觉大图

先完成默认样例、四种编译测试和 TRT 测试，再按下面的顺序执行。图片需要当前系统支持中文的 `Noto Sans CJK` 字体。

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 \
  .venv-optimization/bin/python tools/optimization/prepare_visual_inputs.py

env -u PYTHONPATH HF_HUB_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
  .venv-optimization/bin/python tools/optimization/dump_visual_outputs.py --scope decoder --mode reduce-overhead
env -u PYTHONPATH HF_HUB_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
  .venv-optimization/bin/python tools/optimization/dump_visual_outputs.py --scope forward --mode default
env -u PYTHONPATH HF_HUB_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
  .venv-optimization/bin/python tools/optimization/dump_visual_outputs.py --scope forward --mode reduce-overhead
env -u PYTHONPATH HF_HUB_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
  .venv-optimization/bin/python tools/optimization/dump_visual_outputs.py --scope forward --mode max-autotune

env -u PYTHONPATH MPLBACKEND=Agg \
  .venv-optimization/bin/python tools/optimization/plot_all_cases.py
```

每个 case 一行，列依次为 RGB、原始 Depth、默认 level 9、level 3 基线、四种编译方案、TRT、TRT 加 CUDA Graph 和编译后处理。最后两列使用同一份已验证逐像素一致的 TRT 深度，不是两组有独立差异的预测。

同一行所有深度图使用统一色阶，范围来自原始 Depth、level 9 和 level 3 有效深度的 P1–P99；只在显示时截断，底层 `.npy` 保留完整数值。黑色表示无效深度。

运行后生成以下文件（图片不随仓库提交）：

- 高清大图：`result/optimization/visuals/all_cases_all_optimizations.png`
- 预览图：`result/optimization/visuals/all_cases_all_optimizations_preview.jpg`
- 图像来源和色阶：`result/optimization/visuals/all_cases_manifest.json`

## 检查复现结果

首先检查环境，再查看两个成功路线的报告：

```bash
env -u PYTHONPATH uv pip check --python .venv-optimization/bin/python

env -u PYTHONPATH .venv-optimization/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('result/optimization')
c = json.loads((p / 'compile-forward-max-autotune.json').read_text())
t = json.loads((p / 'reference-trt-benchmark.json').read_text())
assert len(c['compiled']) == 8
assert t['status'] == 'PASS'
assert len(t['results']) == len(t['cuda_graph_plus_compiled_projection']) == 8
assert t['num_tokens'] == 2000
assert Path(t['engine']).is_file()
print('torch.compile 平均 ms:', c['compiled_mean_ms'])
print('TRT 加点云平均 ms:', t['engine_and_projection_mean_ms'])
print('组合方案平均 ms:', t['combined_mean_ms'])
print('8/8 场景及引擎文件检查通过')
PY
```

性能会随 GPU 温度、时钟、其他负载和 tactic 选择浮动；先确保软件版本、输入尺寸、token 数和计时口径相同，再比较延迟。数值差异应同时查看均值、P99、最大值和输出掩码变化，不能只看平均百分比。

## 参考资料

- [官方 LingBot-Depth 源码](https://github.com/Robbyant/lingbot-depth/tree/f3a237e434ae987bc38281476d6cfb5df3e4d739)
- [v0.5 模型权重](https://huggingface.co/robbyant/lingbot-depth-pretrain-vitl-14-v0.5/tree/79204ed6b837f4fdd192cf563e59481fecfa0295)
- [PyTorch 2.8 编译接口](https://docs.pytorch.org/docs/2.8/generated/torch.compile.html)
- [Torch-TensorRT 2.8 版本说明](https://github.com/pytorch/TensorRT/releases/tag/v2.8.0)
- [参考仓库固定版本导出脚本](https://github.com/Ar-Ray-code/lingbot-depth-trt/blob/60e69ab6ea5c65a2f19a22388a20d5969a8f556b/tools/export_trt.py)
- [TensorRT 10.12 Resize 支持的数据类型](https://docs.nvidia.com/deeplearning/tensorrt/10.12.0/_static/operators/Resize.html)
