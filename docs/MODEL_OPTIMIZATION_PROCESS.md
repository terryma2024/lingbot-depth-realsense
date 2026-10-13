# LingBot-Depth-v0.5 模型优化过程

本文记录当前 RealSense GUI 所采用的优化路线、失败尝试、选择依据及测量结果。所有项目路径和命令均相对仓库根目录；先按 [README](../README.md) 安装环境、下载权重并构建引擎。

RTX 5080 的历史实验见 [完整复现手册](OPTIMIZATION_REPRODUCE.md) 和 [原始结果快照](optimization-results-20261013.json)。RTX 5060 Ti 的复现实验见 [本机复现记录](OPTIMIZATION_5060_TI.txt) 和 [原始结果快照](optimization-results-5060-ti-20261013.json)。这些快照是 2026-10-13 的测量记录，不是其他机器的性能保证。

## 1. 先建立可运行基线

固定官方源码 commit `f3a237e434ae987bc38281476d6cfb5df3e4d739`，v0.5 权重 revision `79204ed6b837f4fdd192cf563e59481fecfa0295`。使用官方 8 个场景检查深度、有效像素和点云 Z，避免只比较耗时而忽略输出。

Blackwell GPU 不能沿用官方旧版 torch 2.6 和 xFormers 0.0.29.post2 的环境。本次最初使用 torch 2.7.1+cu128 / xFormers 0.0.31，优化复现使用 torch 2.8.0+cu128 / xFormers 0.0.32.post2。

xFormers 自动选择 FA3 时曾触发 `CUDA invalid argument`。处理方式是在 `mdm/model/dinov2_rgbd/layers/attention.py` 中对 Blackwell 禁用 FA3，保留其他可用 attention 后端。直接禁用整个 xFormers 会让原模型所需的 nested-tensor attention 失败，不能作为替代方案。

最初 torch 2.7.1 环境下，默认 level 9、场景 0、BF16 预热后均值为 310.29ms。这是早期单场景参考值；下表的 level 3 和 TRT 使用另一组固定版本与 8 场景测量，不应把二者视为严格同条件比较。

## 2. 降低 token 数并探测 torch.compile

将 `model.infer(..., resolution_level=3)` 作为同分辨率优化基线，目标约 2000 tokens。输入和输出仍为 640×480、batch=1；内部 token 数减少，不代表与 level 9 输出等价。

依次探测解码器编译及整个前向编译，分别使用 `default`、`reduce-overhead` 和 `max-autotune`。对应实现为 `tools/optimization/probe_compile.py`。

RTX 5080 上，level 3 BF16 从 70.70ms 降至前向 `max-autotune` 的 51.75ms；仅编译解码器为 67.84ms。首次 `max-autotune` 编译约 108 秒，未计入常驻推理耗时。原模型深度 token 长度受输入影响，新的长度仍可能触发编译，服务启动必须预热。

```bash
mkdir -p result/logs result/optimization
env -u PYTHONPATH HF_HUB_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
  .venv-optimization/bin/python -u tools/optimization/probe_compile.py \
  --scope forward --mode max-autotune
```

`torch.compile` 保留原 BF16 路径及输出掩码，但融合计算可能产生浮点差异。RTX 5060 Ti 本次直接复现最优 TRT 路线，没有重新测量所有编译配置；5080 编译结果不能套用为 5060 Ti 的结果。

## 3. BF16 Torch-TensorRT 尝试为何未采用

Torch-TensorRT 2.8 的整模型 BF16 路线遇到数据依赖的 token 长度导出问题；解码器还遇到 TensorRT 10.12 的 BF16 Resize 和转置卷积格式限制。本次没有形成可用的完整 BF16 引擎，不能把失败探测计为优化成功。

转向固定形状、固定 token 长度的 FP16 ONNX 导出和原生 TensorRT。参考导出器固定为 `Ar-Ray-code/lingbot-depth-trt` commit `60e69ab6ea5c65a2f19a22388a20d5969a8f556b`，下载及构建命令见 README。

## 4. 固定形状 FP16 TRT 与 GPU 后处理

采用 TensorRT 10.12.0.36、FP16、batch=1、640×480、目标 2000 tokens。按宽高比取整后，patch 网格为 39×52，即 2028 个位置。

为导出固定计算图，参考路径做了三项行为改变：关闭可变深度 token masking、关闭 RGB resize antialias、省略原始输出置信度掩码头。引擎只返回深度，FP32 转换和 XYZ 投影在 GPU 上另行完成。

引擎和后处理使用同一专用非默认 CUDA stream；输入输出显存地址保持稳定。先预热 TRT 并捕获 CUDA Graph，再用 `torch.compile(mode='reduce-overhead', fullgraph=True, dynamic=False)` 编译点云投影。这里编译的是引擎外的投影函数，不是再次编译 TensorRT。

点云完整形状为 `[1,480,640,3]`，单位为米。检查普通 TRT 与图重放输出逐像素一致、XYZ 的 Z 与深度一致。GUI 的显示抽样不改变完整 GPU 投影。

## 5. 性能比较与最终选择

| 配置 | RTX 5080 | RTX 5060 Ti |
|---|---:|---:|
| Level 3 BF16 基线 | 70.70ms | 147.27ms |
| 前向 torch.compile / max-autotune | 51.75ms | 未重新测量 |
| TRT FP16 深度引擎 | 31.82ms | 67.77ms |
| TRT＋GPU 点云投影 | 31.89ms | 67.89ms |
| TRT CUDA Graph＋编译投影 | 31.82ms | 67.91ms |

RTX 5060 Ti 上组合版相对同环境 level 3 基线约 2.17 倍加速。与普通 TRT＋投影只差约 0.017ms，本次没有可测的额外加速。当前 GUI 采用已验证的组合接口，但没有据此宣称 CUDA Graph 或投影编译显著更快。

计时对象是 GPU 常驻模型和已经预处理、传输的输入；逐帧 CUDA 同步后测主机墙钟。每个场景基线测 20 次，TRT 每种模式测 50 次，再对 8 场景平均。启动、预处理、CPU/GPU 传输、输入缓冲区复制、编译、引擎构建和文件 I/O 均不计入上述 latency。

RTX 5060 Ti 上 ONNX 导出 3.95 秒、引擎构建 50.17 秒，引擎 646,536,156 字节（616.59MiB）。其他 GPU、TensorRT 版本、输入形状或权重需要重新构建。

## 6. 数值差异与适用边界

RTX 5060 Ti 的 8 个场景通过形状、有限正深度、普通 TRT/Graph 等价和 Z 一致性验证。TRT 对参考 FP16 PyTorch 的平均相对差异为 0.019%–0.131%；对原始 level 3 BF16 为 0.288%–0.649%。

场景 3 对原始 level 3 的 P99 绝对差异为 1.28125m，最大为 6.14375m。平均差异小不能说明逐像素等价，特别是此路线已经移除输出掩码。官方样例没有深度真值；这些检查衡量预测一致性，不衡量测量精度。

RealSense 输入的缺失深度保留为零，网络可能进行补全。显示范围筛选只处理非有限、非正或范围外数据，不替代置信度掩码。

## 7. 显存与 GUI 端到端结果

单引擎、单执行上下文、单帧的普通 TRT＋GPU 投影独立测量峰值为 1122MiB（约 1.10GiB）。其中引擎文件约 616.59MiB，上下文报告的设备工作内存约 208.82MiB。torch allocator 不能反映原生 TensorRT 的全部分配，须看本进程 `nvidia-smi` 总量。

实际 GUI 加入图重放、编译、复制和显示后，D435i / RTX 5060 Ti 连续 60 秒的结果为 9.98Hz、完整转换均值 73.16ms、应用延迟均值 111.97ms、进程峰值 1231MiB。显存以 1 秒采样测量；与纯 GPU 基准的范围不同，不能直接用 67.91ms 倒数宣称 GUI 帧率。

相机型号和序列号不固定；运行时用 SDK 枚举，查询命令见 README。物理拔插未实测；设备错误、停止和恢复范围见 [开发记录](DEVELOPMENT_RECORD.md)。

## 8. 在当前仓库复现与查看结果

准备环境、权重及参考导出仓库后，运行：

```bash
mkdir -p result/logs result/optimization
env -u PYTHONPATH HF_HUB_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
  .venv-optimization/bin/python -u tools/optimization/probe_reference_trt.py \
  > result/logs/reference-trt-benchmark.log 2>&1

env -u PYTHONPATH HF_HUB_OFFLINE=1 \
  .venv-optimization/bin/python tools/optimization/prepare_visual_inputs.py

env -u PYTHONPATH MPLBACKEND=Agg \
  .venv-optimization/bin/python tools/optimization/plot_best_cases.py
```

生成的基准报告为 `result/optimization/reference-trt-benchmark.json`，5 列对比图为 `result/optimization/visuals/best_optimization_comparison.png`。完整 10 列大图需先完成所有编译模式及默认样例，具体步骤见完整复现手册。中文绘图需要可用的 Noto Sans CJK 字体。

优化会占用 GPU 并加载基线及参考模型；跑基准前先停止实时 GUI，以免影响计时和显存。生成物不提交 Git，历史结果快照保存在 `docs/` 供核对。
