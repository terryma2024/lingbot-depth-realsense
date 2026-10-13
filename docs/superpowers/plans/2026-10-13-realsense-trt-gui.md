# RealSense TRT GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从当前 RealSense D435i 获取 RGB 和深度，用现有 TRT CUDA Graph＋编译 GPU 点云投影转换，在 PyQt 四视图中以 10Hz 显示最新结果。

**Architecture:** 采集线程、GPU 推理线程与 Qt 主线程通过容量一的最新帧槽协作。相机输出已经对齐、米制化的帧，推理输出同帧的增强深度和抽样彩色点云；Qt 定时器只呈现帧号变化的新结果。相机与 GPU 通过数据接口解耦，纯函数、线程生命周期、真实 GPU 和真实相机分别验证。

**Tech Stack:** Python 3.12、pyrealsense2、PyQt6、pyqtgraph、PyOpenGL、NumPy、OpenCV、torch 2.8.0+cu128、TensorRT 10.12.0.36、pytest。

**Spec:** `docs/superpowers/specs/2026-10-13-realsense-trt-gui-design.md`（用户已确认）。

## Global Constraints

- 已验证硬件为 D435I；设备型号、序列号和比例在运行时通过 SDK 查询，不固定设备 ID。
- RGB8＋Z16 640×480、30fps 采集；单相机、单引擎、batch=1、FP16、目标 2000 tokens。
- 使用 `.venv-optimization` 与现有 `result/optimization/reference-trt-nt2000-fp16/lingbot_depth_nt2000.engine`。
- 转换调度目标 10Hz；Qt 精确定时器 100ms；稳定期至少 60 秒，实际新结果显示率目标至少 9.5Hz。
- GPU 推理和编译投影在同一专用非默认 CUDA stream 上，稳定缓冲区，输出消费完成后才能下一次 replay。
- 深度单位米；RGB 顺序；归一化相机内参；GPU 完整投影 `[1,480,640,3]`；点云显示默认二维步长 2，最多 76800 点。
- 四视图同帧；深度共用色标；无效深度黑色；显示范围默认 0.1–5m。
- 主线程独占 Qt/OpenGL；帧与结果槽容量一、覆盖旧帧；取帧有界超时，停止和关闭释放 SDK 与线程。
- 不计重复旧帧为刷新；不将纯 GPU 基准或历史 1122MiB 显存作为 GUI 实測值。
- 保留既有用户修改；不提交、不创建新分支、不改动已有基准脚本；以任务检查点和工作区差异记录进度。

## Review Focus

- SDK 帧缓冲被复用后，已发布 RGB/深度必须仍然有效且同帧：Task 1 测试拥有数据副本。
- 相机比例非 0.001、非零畸变或内参变化不能产生错误米制几何：Task 1 几何测试及 Task 2 内参更新测试。
- 编译/CUDA 图重放更新输入时，不能重复首帧或覆盖上一帧输出：Task 2 连续变化输入 GPU 测试。
- 用户快速开始/停止、取帧超时或异常后，不能留下占用相机的线程：Task 3 状态与退出测试。
- 定时器运行而 GPU 没有新结果时，不能报告虚假的 10Hz：Task 4 新结果计数测试及 Task 6 实机计时。

## 文件结构

- `tools/realsense_gui/__init__.py`：包入口。
- `tools/realsense_gui/data.py`：帧契约、最新数据槽、显示筛选纯函数。
- `tools/realsense_gui/camera.py`：SDK 设备与对齐、比例、内参、去畸变。
- `tools/realsense_gui/runtime.py`：TRT、CUDA Graph、编译投影、完整 GPU 结果与显示数据拷回。
- `tools/realsense_gui/pipeline.py`：线程、初始化、10Hz 转换、停止和状态。
- `tools/realsense_gui/window.py`：四视图与交互、Qt 调度。
- `tools/realsense_gui/app.py`：CLI、生命周期、验证入口。
- `tests/realsense_gui/`：纯函数、生命周期、GUI 和可选 GPU 测试。
- `run_gui.sh`、`tools/realsense_gui/requirements.txt`、`docs/REALSENSE_GUI.txt`：环境、启动与交付说明。
- `result/realsense_gui/`：生成的日志、截图和实机验收 JSON；沿用现有 result 忽略规则。

### Task 1: 相机帧、最新帧槽与正确米制几何

**Files:** Create `tools/realsense_gui/{__init__,data,camera}.py`; tests `tests/realsense_gui/test_data.py`、`test_camera.py`。

**Interfaces:**
- `CameraFrame(sequence: int, captured_at: float, rgb: np.ndarray, depth_m: np.ndarray, intrinsics: np.ndarray)`；RGB 为 `(480,640,3)` uint8，深度为 `(480,640)` FP32，内参为归一化 `(3,3)` FP32；时间为主机 monotonic。
- `LatestSlot.publish(value) -> None`、`LatestSlot.take() -> object | None`、`LatestSlot.clear() -> None`：线程安全、读取后清空。
- `normalize_intrinsics(fx: float, fy: float, cx: float, cy: float, width: int, height: int) -> np.ndarray`。
- `RealSenseCamera(serial: str | None)`；`start() -> dict` 返回设备信息；`read(timeout_ms: int = 1000) -> CameraFrame | None`；`stop() -> None` 幂等。
- `display_cloud(points: np.ndarray, rgb: np.ndarray, near_m: float, far_m: float) -> tuple[np.ndarray, np.ndarray]`：共同有效掩码，输出 OpenGL 位置及 RGBA 颜色，保持颜色和点对应。

- [ ] 写失败测试：发布帧 1 后帧 2，只读出帧 2，第二次读取为 None；0.002 深度比例将 1000 转为 2m；归一化后 fx×640、fy×480 恢复原值；非有限和非正点不会显示。
- [ ] 写相机测试：SDK 数据源复用后上一帧副本不变；非零支持的畸变同时重映射 RGB 和深度，深度用最近邻；SDK 不支持的畸变明确报错而不是静默忽略。
- [ ] 运行 `env -u PYTHONPATH .venv-optimization/bin/python -m pytest tests/realsense_gui/test_data.py tests/realsense_gui/test_camera.py -q`，记录功能缺失导致的 RED。
- [ ] 实现数据接口与相机封装；枚举零设备报错，多设备未指定 serial 报明确选择提示；读取深度比例及对齐目标内参，复制 SDK 数据。
- [ ] 同命令确认 GREEN；以 `rs-enumerate-devices -s` 核对设备。此任务安装必要 pyrealsense2 与 pytest，不改现有 torch/TRT 版本。
- [ ] 检查差异并记录任务完成、RED/GREEN 证据；不执行 git commit。

### Task 2: TRT CUDA Graph 与编译 GPU 投影运行时

**Files:** Create `tools/realsense_gui/runtime.py`; tests `tests/realsense_gui/test_runtime.py`。

**Interfaces:**
- Consumes Task 1 `CameraFrame`。
- `ConvertedFrame(frame: CameraFrame, depth_m: np.ndarray, points: np.ndarray, colors: np.ndarray, conversion_ms: float, gpu_ms: float, completed_at: float)`；depth 为完整 `(480,640)` FP32，points 与 colors 是相同步长抽样的相机坐标 XYZ、RGB，拥有 CPU 内存。
- `TrtRuntime(engine_path: Path)`；`warmup(frame: CameraFrame) -> None`；`convert(frame: CameraFrame, stride: int = 2) -> ConvertedFrame`；`close() -> None`。
- `prepare_inputs(frame: CameraFrame) -> tuple[np.ndarray, np.ndarray]`：RGB FP16 NCHW `[0,1]`、深度 FP16 引擎要求形状。

- [ ] 写失败测试：RGB 红色样本没有交换通道，0/255 转为 0/1；输入输出名称、dtype、固定形状错误得到清晰错误；有效 frame 深度不变为 mm。
- [ ] 添加显式开启的真实 GPU 集成测试：两帧不同 RGB/深度输入产生变化结果；单帧完整 GPU points 的 Z 与 depth 相同；第二帧改变 K 时 XY 按对应 K 变化；保存的第一帧 CPU 数组在后续 replay 后不变。
- [ ] 跑 CPU 用例，确认 RED，再实现输入验证、单上下文、稳定 GPU 缓冲和 stream。
- [ ] 复用 `mdm.utils.geo.depth_to_pointcloud`；捕获 TRT，编译 FP32 投影；首次 warmup 完成前不发布运行状态。使用 CUDA event 测 GPU 时间，主机计时覆盖预处理、输入与结果拷贝。
- [ ] GPU 完整投影后抽样再拷回显示点；深度完整拷回；同步消费与拷贝后构造独立结果，避免图输出别名。
- [ ] 运行 `env -u PYTHONPATH LINGBOT_GUI_GPU_TESTS=1 .venv-optimization/bin/python -m pytest tests/realsense_gui/test_runtime.py -q`，确认 GREEN；默认普通测试跳过明确标注的 GPU 用例。
- [ ] 检查差异并记任务证据；不执行 git commit。

### Task 3: 后台管线与可停止的 10Hz 调度

**Files:** Create `tools/realsense_gui/pipeline.py`; tests `tests/realsense_gui/test_pipeline.py`。

**Interfaces:**
- Consumes Tasks 1–2 的相机、运行时、帧和结果接口。
- `Pipeline(camera_factory, runtime_factory, hz: float = 10.0)`；`start() -> None`、`stop() -> None`、`join(timeout: float) -> bool`、`take_result() -> ConvertedFrame | None`、`snapshot() -> dict`。
- 状态 `stopped / starting / running / stopping / error`；snapshot 包含状态、设备元数据与错误说明。转换 stride 可线程安全设置。

- [ ] 失败测试：连续快速 start 仅运行一个采集者和一个转换者；停止后再次开始有效；read 超时不会发布空帧；设备异常停止两线程且只清理自身资源；停止唤醒调度等待，不需等满一个周期。
- [ ] 写有界最新帧测试：模拟采集比转换快，结果帧号前进并跳过旧帧；每结果 RGB/raw/增强/点云来自同一 CameraFrame；conversion 过慢时不追赶积压时间片。
- [ ] 跑 `env -u PYTHONPATH .venv-optimization/bin/python -m pytest tests/realsense_gui/test_pipeline.py -q`，确认 RED。
- [ ] 实现 Event 停止、Lock 状态与最新槽、monotonic 调度；推理线程内部创建/预热/使用/关闭 GPU runtime，采集线程内部管理 SDK 生命周期；异常保留可读原因并确保 finally 释放。
- [ ] 同命令确认 GREEN；记录检查点和所有针对计划歧义的决定。

### Task 4: 四视图、交互与真实刷新统计

**Files:** Create `tools/realsense_gui/window.py`; tests `tests/realsense_gui/test_window.py`。

**Interfaces:**
- Consumes Task 3 `Pipeline` 与 Task 2 `ConvertedFrame`。
- `MainWindow(pipeline: Pipeline)`；2×2 RGB/raw/refined/GL 点云布局，开始/停止/重置视角/距离/抽样控制。
- `RefreshMetrics.record(sequence: int, captured_at: float, presented_at: float) -> bool`；重复 sequence 返回 False，不增加刷新次数；输出实际 Hz、间隔 P95 和应用延迟。
- `depth_rgb(depth_m: np.ndarray, near_m: float, far_m: float) -> np.ndarray`：原始和增强共用相同色标、无效像素黑色。

- [ ] 写失败测试：重复帧不增加显示计数；两深度视图同一深度值同颜色；NaN/Inf/零为黑；显示范围改变不修改源深度。
- [ ] 写 Qt 事件循环测试：发布单结果更新四视图同一 sequence；按钮状态匹配 starting/running/stopping/error；启动错误可见；关闭通知 pipeline.stop 并等待有界退出。
- [ ] 安装 PyQt6、pyqtgraph、PyOpenGL；无显示测试使用 `QT_QPA_PLATFORM=offscreen`，真实 OpenGL 验证使用 `DISPLAY=:1`。Qt 错误路径测试不依赖实际相机或加载引擎。
- [ ] 跑 `env -u PYTHONPATH QT_QPA_PLATFORM=offscreen .venv-optimization/bin/python -m pytest tests/realsense_gui/test_window.py -q`，确认 RED，再实现 UI 和 Qt PreciseTimer 100ms。
- [ ] 初始视角展示直立点云，鼠标旋转/缩放，重置回初始角度；映射相机坐标只影响绘图数据。
- [ ] 同命令确认 GREEN，随后在真实显示上验证 OpenGL；记录检查点。

### Task 5: 可运行入口、验证模式与运行说明

**Files:** Create `tools/realsense_gui/app.py`, `run_gui.sh`, `tools/realsense_gui/requirements.txt`, `docs/REALSENSE_GUI.txt`; tests `tests/realsense_gui/test_app.py`。

**Interfaces:**
- `python -m tools.realsense_gui.app` 参数：`--serial`、`--engine`、`--verify-seconds`、`--report`、`--screenshot`。
- `./run_gui.sh` 默认当前引擎和 10Hz，使用 `.venv-optimization/bin/python`。
- 验证模式在运行/预热结束后开始稳定期计时，收集唯一新帧呈现事件并输出 JSON；达到运行时长后安全退出。报告字段含真实设备、依赖版本、分辨率、实际 Hz、间隔 P95、应用延迟、转换/GPU 时间、显存、线程退出与最终判定。

- [ ] 写失败测试：引擎路径不存在时错误可读；非法验证时长拒绝；验证周期未开始时不计启动/编译时间；无新帧时不能 PASS；退出失败不能 PASS。
- [ ] 跑 `env -u PYTHONPATH .venv-optimization/bin/python -m pytest tests/realsense_gui/test_app.py -q`，确认 RED。
- [ ] 实现 CLI、Qt 实例和生命周期；验证模式只生成测试产物，不增加录制或导出功能。
- [ ] 启动脚本去除 ROS PYTHONPATH 干扰；在导入 cv2 后避免其 Qt 插件目录覆盖 PyQt，并使用实际 PyQt 插件路径；诊断 DISPLAY 和插件错误。
- [ ] 固定实测 GUI 依赖版本，执行 `uv pip check --python .venv-optimization/bin/python`；文档列安装、启动、设备选择、点云操作、语义差异、错误恢复与验收命令。
- [ ] 同测试命令确认 GREEN；检查 `./run_gui.sh --help` 输出与脚本可执行权限；记录检查点。

### Task 6: 实机 60 秒验收、故障恢复与整体审查

**Files:** Generated `result/realsense_gui/verification.json`, `gui.png`, 日志；更新 `docs/REALSENSE_GUI.txt` 实测数据和计划完成状态。

**Interfaces:** Consumes Task 5 CLI 与 JSON 报告。

- [ ] 全套运行 `env -u PYTHONPATH QT_QPA_PLATFORM=offscreen .venv-optimization/bin/python -m pytest tests/realsense_gui -q`；GPU 用例另外显式开启，并记录每项通过/跳过原因。
- [ ] 真实显示运行 `./run_gui.sh --verify-seconds 60 --report result/realsense_gui/verification.json --screenshot result/realsense_gui/gui.png`，预热后测稳定期；多设备时通过 SDK 查询目标序列号再传给 `--serial`。每个运行阶段检查实际异常，不能只凭退出码判定正确。
- [ ] 截图检查四视图、色标和 3D 内容；检查记录覆盖多个变化帧，Z 与深度一致，目标实际 Hz≥9.5，报告 interval P95、端到端时延、GPU 时间与 own-PID 总显存。
- [ ] 测试停止/重新开始、关闭后设备重开；拔插恢复仅在人为可执行时实测，否则清楚标明由模拟错误覆盖，没有实际拔插。
- [ ] 性能不满足时分解采集/对齐/转换/传输/绘制耗时，使用 systematic-debugging；行为修复先增加失败测试，再修复、跑相关用例和全套用例。
- [ ] 使用独立 reviewer 做整体审查，聚焦图重放、线程结束、同帧展示和刷新率计数；重要问题以 RED→GREEN 验证修复，不扩展首版功能。
- [ ] 执行 `git diff --check`、`uv pip check`、必要的 Python 编译检查。使用 verification-before-completion 核验报告、截图、测试输出后交付一键启动入口和测量结果；不自动提交或创建分支。

## 计划自检与执行交接

已覆盖设计中的相机、数值几何、GPU 图重放、四视图、生命周期、安装、故障和实机验收。Tasks 1–5 的接口依次衔接，Task 6 验证整个结果；测试不将定时器调用次数混作画面更新率。完成记录和决定写入此计划对应的 Superpowers ledger。

推荐 Native：本会话依次实现，按 TDD 逐任务验证，最后独立整体审查。理由是相机帧、GPU 输出和 GUI 生命周期共享接口，顺序开发便于实机问题定位；不需要为每个小模块付出独立实现者的上下文成本。Subagent-driven 也可选择，它会为每个任务加入独立实现和审查。

状态：用户已确认，采用 Native。Tasks 1–6 已完成；实机验收及审查记录见 `docs/DEVELOPMENT_RECORD.md`。用户后续授权将代码和文档提交到指定独立仓库，不提交原工作区。
