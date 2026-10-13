# RealSense + LingBot-Depth TRT 实时 GUI 设计

状态：用户已确认设计及 Native 实施计划；开发和最终 60 秒实机验收完成，详见 `docs/DEVELOPMENT_RECORD.md`。

## 目标与已确认需求

使用 PyQt 和 pyrealsense2，从当前连接的 RealSense 获取 RGB 与深度，经过现有 LingBot-Depth FP16 TensorRT 引擎及 GPU 点云投影，实时展示结果。用户已选择四个视图：RGB、原始深度、TRT 增强深度、可旋转的彩色 3D 点云。目标是预热后以 10Hz 展示新的转换结果。

已验证硬件为 RealSense D435I，固件 `5.17.3.10`，USB 类型 `3.2`。应用通过 `rs.context().query_devices()` 枚举设备，使用 `rs.camera_info.name` 和 `rs.camera_info.serial_number` 获取实际型号与序列号。单设备自动选择，多设备时指定 SDK 查询到的序列号；不固定到测量时使用的设备 ID。

设备支持 RGB8 和 Z16 两路 640×480、30fps。当前 TensorRT 引擎固定 batch=1、640×480、FP16、目标 2000 tokens。相机用 30fps 采集，转换和显示以 10Hz 调度；10Hz 不作为硬件流配置。

## 方案选择

1. 推荐：独立采集线程、独立推理线程、Qt 主线程绘制。采集和结果之间各使用容量为一的最新数据槽，覆盖未消费旧帧。采集持续进行，推理不会阻塞界面，显示不会积累历史帧。
2. 单工作线程顺序采集和推理：代码更少，但推理期间取帧间隔增大，SDK 内部排队可能增大画面延迟。
3. 多进程及共享内存：故障隔离更强，但引入进程生命周期、共享缓冲区与 GPU 所有权管理；当前单相机和单引擎应用没有这一需求。

选择方案 1。GUI 主线程使用 PyQt6，3D 视图使用 pyqtgraph 的 OpenGL 点云控件及 PyOpenGL。

## 数据流与模块边界

```text
RealSense 640×480 RGB8 + Z16 @30fps
  → 采集线程：深度对齐 RGB、读取米制比例与内参、复制数据
  → 最新相机帧槽（容量 1）
  → 推理线程 @10Hz：预处理、TRT CUDA Graph、编译 GPU 投影
  → 最新结果槽（容量 1，同一帧的 RGB / 原始深度 / 增强深度 / XYZ）
  → Qt 主线程 100ms 定时器：四视图及运行指标
```

计划在 `tools/realsense_gui/` 放置应用模块，提供根目录 `run_gui.sh` 启动入口：

- `camera.py`：设备枚举、指定序列号、流启动、对齐、相机元数据与安全关闭。
- `runtime.py`：加载现有引擎、验证输入输出、稳定 CUDA 缓冲区、图捕获、编译投影、预热与同步拷回。
- `pipeline.py`：帧数据结构、容量一数据槽、采集与推理生命周期及计时。
- `window.py`：四视图、控制按钮、显示计时及状态展示；仅在主线程访问 Qt/OpenGL。
- `app.py`：参数解析、Qt 应用启动、异常与退出管理。

相机和运行时通过帧数据结构解耦，便于使用已有示例数据验证 GPU 管线和 GUI，实机测试另行覆盖 SDK 行为。应用不导入会加载模型或执行基准的优化脚本；只复用现有投影函数和已生成引擎的接口。

## 相机与数值约定

- 用 `rs.align(rs.stream.color)` 对齐深度与 RGB；使用对齐后的目标相机内参。
- 原始 Z16 乘设备读取的 `depth_scale` 转为米，保留零深度作为输入缺失值；不假定固定 0.001 比例。
- RGB 输入为 RGB 顺序、`[0,1]`、NCHW，输入 RGB 和深度均转为连续 FP16，深度形状依引擎检查结果固定。
- 读取实际内参，按宽高归一化 fx、fy、cx、cy，传入现有 GPU 投影。检查彩色相机畸变；若系数非零，RGB 和对齐深度一致去畸变后再使用对应的针孔内参，不能直接忽略畸变。
- TRT 深度输出转 FP32，完整 GPU 点云输出为 `[1,480,640,3]`，单位米。投影验证 XYZ 的 Z 与增强深度一致。
- 数据层保持相机坐标：X 向右、Y 向下、Z 向前。3D 视图只在显示阶段执行明确的坐标转换，让画面直立；原始数组保持相机坐标。
- TRT 沿用已验证的语义：无原始输出置信度掩码，关闭 depth-token masking 和 RGB antialias。增强结果包含网络补全深度，不能作为逐像素测量精度保证。
- 非有限或非正增强深度不绘制。显示距离范围默认 0.1–5m，可在界面调整；范围只影响显示，不改变网络输入或保存于结果对象的米制深度。

## GPU 执行与缓冲区

复用 `result/optimization/reference-trt-nt2000-fp16/lingbot_depth_nt2000.engine`。使用 `.venv-optimization` 的 torch 2.8.0+cu128、TensorRT 10.12.0.36。单引擎、单上下文，由推理线程独占一个非默认 CUDA stream。

预分配输入输出及归一化内参 GPU 缓冲区，复制相机帧之后 replay TRT CUDA Graph。点云投影使用 `torch.compile(mode='reduce-overhead', fullgraph=True, dynamic=False)`，保持形状和缓冲区稳定；每帧标记编译图步骤并保证输出在下一次 replay 前完成消费或复制。

GPU 创建、图捕获和编译预热放在后台初始化，界面显示启动状态。测量时区分初始化耗时、GPU 转换耗时和相机帧到结果呈现的应用延迟。实时路径计时包含预处理及拷贝，不能直接将先前的 67.9ms 纯 GPU 基准作为实际刷新率。

完整 307200 个像素在 GPU 投影；3D 显示默认在二维像素网格上步长 2 抽样，最多 76800 个彩色点，颜色来自同帧 RGB。显示抽样减少 GPU→CPU 传输和 OpenGL 绘制负担，不改变完整投影结果。深度图保持 640×480。

## 界面与操作

四视图 2×2 布局：左上 RGB、右上原始深度、左下 TRT 增强深度、右下彩色 3D 点云。原始和增强深度使用相同固定米制色标，缺失或无效深度显示黑色。窗口缩放保持图像宽高比。

控制项为开始、停止、重置 3D 视角、显示距离上限和点云抽样步长。鼠标支持点云旋转和缩放。窗口可先打开后后台启动当前设备；多台设备时需要明确选择序列号。

状态区展示实际型号与序列号、初始化/运行/停止/错误状态、转换耗时、新结果显示 Hz、应用延迟及显示点数。初始化期间不将定时器触发次数计为结果刷新率。

Qt 使用精确定时器每 100ms 尝试取最新结果，仅在结果帧号变化时更新四视图。四个视图始终来自同一推理帧，避免 RGB 与深度错位。性能下降时如实展示实际刷新率，不重复旧帧伪装 10Hz。

## 生命周期与错误处理

- 无相机、设备被占用、流配置失败、引擎不存在或不兼容：界面给出具体错误，允许用户修正后重试。
- 取帧使用有限超时，连续超时或设备断开转入错误状态，停止推理并释放设备；用户重新连接后点击开始恢复。
- 开始/停止按钮按状态启用，防止重复启动工作线程和并发操作上下文。
- 关闭窗口先请求工作线程停止，在有界取帧和当前 GPU 工作完成后清理 SDK pipeline，再释放 GPU 与 OpenGL 资源；不强行 terminate 线程。
- 结果槽内是独立拥有的 CPU 数组，不能引用即将被 SDK 或 CUDA 图覆盖的底层帧缓冲区。

## 安装与交付

在 `.venv-optimization` 加入 pyrealsense2、PyQt6、pyqtgraph、PyOpenGL；记录实测版本于独立 GUI 依赖文件。启动脚本清除 ROS 注入的 PYTHONPATH，并处理 OpenCV 自带 Qt 插件路径对 PyQt 的潜在冲突。保持现有命令行示例和优化环境可用。

交付启动脚本、应用模块、安装运行说明、实机测试报告及 GUI 截图。提供有限时长的验证入口，便于记录刷新率、帧延迟、GPU 耗时、关闭结果及进程显存。

## 验证与验收

按 Superpowers TDD 流程先验证容易造成错帧或错误几何的行为：深度比例和内参归一化、最新帧覆盖语义、帧号一致性、无效深度筛选、工作线程退出。GPU 验证检查引擎输入输出形状、连续变化的帧输入、XYZ 与深度一致性。

GUI 使用 Qt 事件循环验证四视图更新、开始/停止和错误反馈，并在真实 DISPLAY=:1 下检查 OpenGL 视图及截图。实机验证必须用当前 D435i 连续运行至少 60 秒，初始化和预热不计入稳定期：

1. 测量新结果的实际显示率，目标稳定期平均至少 9.5Hz，定时调度目标 10Hz；同时报告显示间隔 P95，不能只报告定时器频率。
2. 记录应用延迟、端到端转换时间和 GPU 时间；若不能达到目标，定位采集、转换、拷贝和绘制瓶颈后再调整。
3. 用至少两个不同取景确认增强结果和点云随相机输入变化，检查四视图同帧、颜色与投影匹配及米制尺度。
4. 测量实际 GUI 进程显存；先前 1122MiB 是纯 TRT＋投影独立测量，GUI 加上编译、显示和缓冲区后须重新实测。
5. 检查停止后设备可重新打开、再次开始正常、关闭后无残留相机线程。设备错误通过可控模拟覆盖，并在不影响验证数据的情况下执行拔插恢复检查。

## 范围边界

本次交付聚焦单相机实时转换与交互显示。IMU、SLAM、录制、离线点云导出、多相机融合、引擎重新训练和重新评估预测精度不纳入首版。

## 参考

- [pyrealsense2 PyPI](https://pypi.org/project/pyrealsense2/)：Python SDK 发行包，当前页面提供 Python 3.12 Linux wheel。
- [RealSense 官方对齐示例](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/align-depth2color.py)：深度对齐到彩色及设备 depth_scale 读取。
- [pyqtgraph GLScatterPlotItem](https://pyqtgraph.readthedocs.io/en/latest/api_reference/3dgraphics/glscatterplotitem.html)：OpenGL 点云显示。
- 本地 `tools/optimization/probe_reference_trt.py`：现有 TRT CUDA Graph 与编译 GPU 投影组合。
- 本地 `result/optimization/reference-trt-benchmark.json` 和 `reference-trt-memory.json`：已测纯 GPU 性能及显存，实际 GUI 需重新验收。
