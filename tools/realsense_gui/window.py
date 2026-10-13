from collections import deque
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PyQt6.QtCore import QLibraryInfo, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QVector3D
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel,
                            QMainWindow, QPushButton, QVBoxLayout, QWidget)
import pyqtgraph.opengl as gl

from .data import display_cloud


def configure_qt():
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)) / 'platforms')
    if 'cv2' in os.environ.get('QT_QPA_FONTDIR', ''):
        os.environ.pop('QT_QPA_FONTDIR', None)


def depth_rgb(depth_m, near_m, far_m):
    valid = np.isfinite(depth_m) & (depth_m > 0)
    safe = np.where(valid, depth_m, near_m)
    scaled = np.clip((safe - near_m) * (255 / (far_m - near_m)), 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    return rgb


class RefreshMetrics:
    def __init__(self):
        self.count = 0
        self.sequence = None
        self.timestamps = deque(maxlen=6000)
        self.latencies = deque(maxlen=6000)

    def record(self, sequence, captured_at, presented_at):
        if sequence == self.sequence:
            return False
        self.sequence = sequence
        self.count += 1
        self.timestamps.append(presented_at)
        self.latencies.append((presented_at - captured_at) * 1000)
        return True

    def summary(self, now=None):
        now = now if now is not None else (self.timestamps[-1] if self.timestamps else 0)
        recent = [timestamp for timestamp in self.timestamps if now - 2 <= timestamp <= now]
        intervals = np.diff(recent) * 1000
        duration = now - recent[0] if len(recent) > 1 else 0
        return {'frames': self.count, 'hz': (len(recent) - 1) / duration if duration else 0,
                'interval_p95_ms': float(np.percentile(intervals, 95)) if len(intervals) else 0,
                'latency_p95_ms': float(np.percentile(self.latencies, 95)) if self.latencies else 0}


class ImageView(QLabel):
    def __init__(self):
        super().__init__('等待数据')
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(160, 120)
        self.setStyleSheet('background: #0b1016; color: #8191a3;')
        self._image = None

    def set_image(self, rgb):
        rgb = np.ascontiguousarray(rgb)
        self._image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self._redraw()

    def _redraw(self):
        if self._image is not None:
            self.setPixmap(QPixmap.fromImage(self._image).scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                               Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._redraw()


class MainWindow(QMainWindow):
    presented = pyqtSignal(object, float)
    shutdown_finished = pyqtSignal(bool)

    def __init__(self, pipeline):
        super().__init__()
        configure_qt()
        self.pipeline = pipeline
        self.metrics = RefreshMetrics()
        self.last_result = None
        self._point_count = 0
        self.shutdown_ok = None
        self._closing = False
        self._previous_state = 'stopped'
        self.setWindowTitle('LingBot-Depth · RealSense 实时深度与点云')
        self.resize(1400, 950)
        self.setStyleSheet('QMainWindow, QWidget { background: #151d28; color: #dde7f2; } '
                          'QPushButton { background: #26364a; padding: 7px 16px; border-radius: 4px; } '
                          'QPushButton:disabled { color: #607083; } '
                          'QComboBox, QDoubleSpinBox { background: #26364a; padding: 5px; }')
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        controls = QHBoxLayout()
        self.start_button = QPushButton('开始')
        self.stop_button = QPushButton('停止')
        reset_button = QPushButton('重置 3D 视角')
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self.pipeline.stop)
        reset_button.clicked.connect(self.reset_view)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(reset_button)
        controls.addStretch()
        controls.addWidget(QLabel('显示距离上限（m）'))
        self.far_control = QDoubleSpinBox()
        self.far_control.setRange(0.2, 20)
        self.far_control.setSingleStep(0.5)
        self.far_control.setValue(5)
        self.far_control.valueChanged.connect(self._redraw_last)
        controls.addWidget(self.far_control)
        controls.addWidget(QLabel('点云步长'))
        self.stride_control = QComboBox()
        self.stride_control.addItems(['1', '2', '4', '8'])
        self.stride_control.setCurrentText('2')
        self.stride_control.currentTextChanged.connect(lambda value: self.pipeline.set_stride(int(value)))
        controls.addWidget(self.stride_control)
        layout.addLayout(controls)
        views = QGridLayout()
        self.rgb_view, self.raw_view, self.refined_view = ImageView(), ImageView(), ImageView()
        self.cloud_view = gl.GLViewWidget()
        self.cloud_view.setBackgroundColor('#0b1016')
        self.cloud_view.setMinimumSize(160, 120)
        self.cloud_item = gl.GLScatterPlotItem(pos=np.empty((0, 3), dtype=np.float32), size=2, pxMode=True)
        self.cloud_item.setGLOptions('opaque')
        self.cloud_view.addItem(self.cloud_item)
        grid = gl.GLGridItem()
        grid.setSize(10, 10)
        grid.setSpacing(1, 1)
        grid.translate(0, 0, -1)
        self.cloud_view.addItem(grid)
        for index, (title, view) in enumerate([('RGB', self.rgb_view), ('原始深度', self.raw_view),
                                             ('TRT 增强深度', self.refined_view), ('彩色 3D 点云 · 拖动旋转 / 滚轮缩放', self.cloud_view)]):
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(0, 0, 0, 0)
            panel_layout.addWidget(QLabel(title))
            panel_layout.addWidget(view, 1)
            views.addWidget(panel, index // 2, index % 2)
        views.setRowStretch(0, 1)
        views.setRowStretch(1, 1)
        views.setColumnStretch(0, 1)
        views.setColumnStretch(1, 1)
        layout.addLayout(views, 1)
        self.scale_label = QLabel('深度色标：0.1–5.0 m；黑色为无效深度；增强深度含网络补全')
        layout.addWidget(self.scale_label)
        self.status_label = QLabel('准备就绪')
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status_label)
        self.stats_label = QLabel('目标 10 Hz · 等待相机和 GPU 初始化')
        layout.addWidget(self.stats_label)
        self.reset_view()
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self.timer.start(100)
        self._tick()

    def reset_view(self):
        self.cloud_view.opts['center'] = QVector3D(0, 2, 0)
        self.cloud_view.setCameraPosition(distance=4.5, elevation=15, azimuth=-90)

    def _start(self):
        self.metrics = RefreshMetrics()
        self.last_result = None
        self.pipeline.start()

    def _redraw_last(self):
        if self.last_result is not None:
            self._point_count = self._render(self.last_result)

    def _render(self, result):
        far_m = self.far_control.value()
        self.rgb_view.set_image(result.frame.rgb)
        self.raw_view.set_image(depth_rgb(result.frame.depth_m, 0.1, far_m))
        self.refined_view.set_image(depth_rgb(result.depth_m, 0.1, far_m))
        positions, colors = display_cloud(result.points, result.colors, 0.1, far_m)
        view_positions = positions[:, [0, 2, 1]].copy()
        view_positions[:, 2] *= -1
        self.cloud_item.setData(pos=view_positions, color=colors, size=2, pxMode=True)
        self.scale_label.setText(f'深度色标：0.1–{far_m:.1f} m；黑色为无效深度；增强深度含网络补全')
        return len(positions)

    def _tick(self):
        snapshot = self.pipeline.snapshot()
        state = snapshot['state']
        if state == 'starting' and self._previous_state != 'starting':
            self.metrics = RefreshMetrics()
        self._previous_state = state
        self.start_button.setEnabled(state in ('stopped', 'error') and snapshot['workers_alive'] == 0)
        self.stop_button.setEnabled(state in ('starting', 'running'))
        device = snapshot['device']
        prefix = f"{device.get('name', 'RealSense')} · {device.get('serial', '未连接')}"
        states = {'starting': '正在启动相机与 GPU，请等待预热完成', 'running': '运行中',
                  'stopped': '已停止', 'stopping': '正在停止', 'error': '错误'}
        self.status_label.setText(f"{prefix} · {states[state]} {snapshot['error']}")
        result = self.pipeline.take_result()
        if result is None or result.frame.sequence == self.metrics.sequence:
            self._update_stats(time.monotonic())
            return
        self.last_result = result
        self._point_count = self._render(result)
        presented_at = time.monotonic()
        self.metrics.record(result.frame.sequence, result.frame.captured_at, presented_at)
        self._update_stats(presented_at)
        self.presented.emit(result, presented_at)

    def _update_stats(self, now):
        if self.last_result is None:
            return
        result = self.last_result
        summary = self.metrics.summary(now=now)
        self.stats_label.setText(f"新结果 {summary['hz']:.2f} Hz / 目标 10 Hz · 转换 {result.conversion_ms:.1f} ms · "
                                 f"GPU {result.gpu_ms:.1f} ms · 应用延迟 {(now - result.frame.captured_at) * 1000:.1f} ms · "
                                 f"显示 {self._point_count:,} 点")

    def closeEvent(self, event):
        if self._closing:
            self.shutdown_ok = self.pipeline.join(0)
            if not self.shutdown_ok:
                event.ignore()
                return
            self.timer.stop()
            self.shutdown_finished.emit(True)
            event.accept()
            return
        self.pipeline.stop()
        self.shutdown_ok = self.pipeline.join(0)
        if self.shutdown_ok:
            self.timer.stop()
            self.shutdown_finished.emit(True)
            event.accept()
        else:
            self._closing = True
            self.start_button.setEnabled(False)
            event.ignore()
            self.close_timer = QTimer(self)
            self.close_timer.timeout.connect(self.close)
            self.close_timer.start(100)
