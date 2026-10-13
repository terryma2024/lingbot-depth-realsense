from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import tensorrt as trt
import torch

from mdm.utils.geo import depth_to_pointcloud
from .data import CameraFrame


@dataclass(frozen=True)
class ConvertedFrame:
    frame: CameraFrame
    depth_m: np.ndarray
    points: np.ndarray
    colors: np.ndarray
    conversion_ms: float
    gpu_ms: float
    completed_at: float


def prepare_inputs(frame):
    if frame.rgb.shape != (480, 640, 3) or frame.depth_m.shape != (480, 640):
        raise ValueError('Camera input must be RGB/depth 640×480')
    image = np.ascontiguousarray((frame.rgb.astype(np.float32) / 255).transpose(2, 0, 1)[None], dtype=np.float16)
    depth = np.ascontiguousarray(frame.depth_m[None], dtype=np.float16)
    return image, depth


def validate_layout(layout):
    expected = {'image': ((1, 3, 480, 640), 'float16'),
                'depth': ((1, 480, 640), 'float16'),
                'depth_refined': ((1, 480, 640), 'float16')}
    if layout != expected:
        raise ValueError(f'Incompatible TensorRT engine layout: {layout}; expected {expected}')


def project(depth, intrinsics):
    depth = depth.float()
    return depth, depth_to_pointcloud(depth, intrinsics)


class TrtRuntime:
    def __init__(self, engine_path: Path):
        engine_path = Path(engine_path)
        if not engine_path.is_file():
            raise FileNotFoundError(f'TensorRT 引擎不存在：{engine_path}')
        if not torch.cuda.is_available():
            raise RuntimeError('未检测到可用 CUDA GPU')
        torch.set_num_threads(1)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError('TensorRT 引擎加载失败，请检查 GPU 与 TensorRT 版本')
        layout = {self.engine.get_tensor_name(index):
                  (tuple(self.engine.get_tensor_shape(self.engine.get_tensor_name(index))),
                   str(np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.engine.get_tensor_name(index))))))
                  for index in range(self.engine.num_io_tensors)}
        validate_layout(layout)
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('TensorRT 执行上下文创建失败')
        self.stream = torch.cuda.Stream()
        self.graph = None
        self.compiled_project = None
        self.closed = False
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.stream):
            self.image_buffer = torch.empty((1, 3, 480, 640), dtype=torch.float16, device='cuda')
            self.depth_buffer = torch.empty((1, 480, 640), dtype=torch.float16, device='cuda')
            self.output = torch.empty((1, 480, 640), dtype=torch.float16, device='cuda')
            self.intrinsics = torch.empty((1, 3, 3), dtype=torch.float32, device='cuda')
        self.host_image = torch.empty(self.image_buffer.shape, dtype=torch.float16, pin_memory=True)
        self.host_depth = torch.empty(self.depth_buffer.shape, dtype=torch.float16, pin_memory=True)
        for name, tensor in [('image', self.image_buffer), ('depth', self.depth_buffer), ('depth_refined', self.output)]:
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f'TensorRT buffer binding failed: {name}')

    def _copy_inputs(self, frame):
        image, depth = prepare_inputs(frame)
        np.copyto(self.host_image.numpy(), image)
        np.copyto(self.host_depth.numpy(), depth)
        self.image_buffer.copy_(self.host_image, non_blocking=True)
        self.depth_buffer.copy_(self.host_depth, non_blocking=True)
        self.intrinsics.copy_(torch.from_numpy(frame.intrinsics[None]))

    def _execute(self):
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError('TensorRT GPU execution failed')

    @torch.inference_mode()
    def warmup(self, frame):
        with torch.cuda.stream(self.stream):
            self._copy_inputs(frame)
            for _ in range(3):
                self._execute()
            self.stream.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.stream):
                self._execute()
            self.compiled_project = torch.compile(project, mode='reduce-overhead', fullgraph=True, dynamic=False)
            for _ in range(3):
                torch.compiler.cudagraph_mark_step_begin()
                self.graph.replay()
                self.compiled_project(self.output, self.intrinsics)
                self.stream.synchronize()

    @torch.inference_mode()
    def convert(self, frame, stride=2):
        if self.closed or self.graph is None:
            raise RuntimeError('TensorRT runtime is not warmed up or has been closed')
        if stride not in (1, 2, 4, 8):
            raise ValueError('Point cloud stride must be 1, 2, 4 or 8')
        started = time.monotonic()
        with torch.cuda.stream(self.stream):
            self._copy_inputs(frame)
            self.start_event.record(self.stream)
            torch.compiler.cudagraph_mark_step_begin()
            self.graph.replay()
            depth, points = self.compiled_project(self.output, self.intrinsics)
            self.end_event.record(self.stream)
            depth_cpu = depth[0].cpu().numpy().copy()
            points_cpu = points[0, ::stride, ::stride].contiguous().cpu().numpy().copy()
            self.stream.synchronize()
        completed = time.monotonic()
        return ConvertedFrame(frame, depth_cpu, points_cpu, frame.rgb[::stride, ::stride].copy(),
                              (completed - started) * 1000, self.start_event.elapsed_time(self.end_event), completed)

    def close(self):
        if self.closed:
            return
        self.stream.synchronize()
        self.graph = self.compiled_project = None
        self.context = self.engine = self.runtime = None
        self.image_buffer = self.depth_buffer = self.output = self.intrinsics = None
        self.host_image = self.host_depth = None
        self.closed = True
        torch.cuda.empty_cache()
