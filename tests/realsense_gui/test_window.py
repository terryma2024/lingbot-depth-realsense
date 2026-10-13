import importlib

import numpy as np

from tools.realsense_gui.data import LatestSlot
from tools.realsense_gui.runtime import ConvertedFrame
from test_runtime import make_frame


def window_module():
    try:
        return importlib.import_module('tools.realsense_gui.window')
    except ModuleNotFoundError:
        assert False, 'Four-view GUI has not been implemented'


def test_duplicate_frame_does_not_inflate_refresh_rate():
    metrics = window_module().RefreshMetrics()
    assert metrics.record(1, 0.9, 1)
    assert not metrics.record(1, 0.9, 1.05)
    assert metrics.record(2, 1, 1.1)
    assert metrics.count == 2
    assert abs(metrics.summary()['hz'] - 10) < 1e-6
    assert abs(metrics.summary()['latency_p95_ms'] - 100) < 1e-6


def test_refresh_rate_decays_to_zero_during_stall():
    metrics = window_module().RefreshMetrics()
    metrics.record(1, 0.9, 1)
    metrics.record(2, 1, 1.1)
    assert metrics.summary(now=4)['hz'] == 0


def test_depth_color_scale_masks_invalid_without_mutating_source():
    depth = np.array([[2, 0, np.nan, np.inf]], dtype=np.float32)
    saved = depth.copy()
    first = window_module().depth_rgb(depth, 0.1, 5)
    second = window_module().depth_rgb(np.array([[2]], dtype=np.float32), 0.1, 5)
    np.testing.assert_array_equal(first[0, 0], second[0, 0])
    np.testing.assert_array_equal(first[0, 1:], 0)
    window_module().depth_rgb(depth, 0.1, 3)
    np.testing.assert_array_equal(depth, saved)


class DisplayPipeline:
    def __init__(self):
        self.results = LatestSlot()
        self.state = 'running'
        self.stopped = False

    def snapshot(self):
        return {'state': self.state, 'error': 'camera unavailable' if self.state == 'error' else '',
                'device': {'name': 'test device', 'serial': '123'}, 'workers_alive': 0}

    def take_result(self):
        return self.results.take()

    def start(self):
        self.state = 'starting'

    def stop(self):
        self.state = 'stopped'
        self.stopped = True

    def join(self, timeout):
        return True

    def set_stride(self, stride):
        pass


def test_event_loop_displays_one_same_frame_result_and_closes(qtbot):
    module = window_module()
    pipeline = DisplayPipeline()
    window = module.MainWindow(pipeline)
    qtbot.addWidget(window)
    frame = make_frame(9)
    points = np.zeros((240, 320, 3), dtype=np.float32)
    points[..., 2] = 2
    pipeline.results.publish(ConvertedFrame(frame, frame.depth_m.copy(), points, frame.rgb[::2, ::2].copy(), 70, 68, frame.captured_at + 0.07))
    qtbot.waitUntil(lambda: window.metrics.count == 1)
    assert window.last_result.frame.sequence == 9
    assert window.rgb_view.pixmap().toImage().pixelColor(0, 0).red() == 255
    assert window.raw_view.pixmap() is not None
    assert window.refined_view.pixmap() is not None
    assert window.cloud_item.pos.shape == (76800, 3)
    qtbot.wait(130)
    assert window.metrics.count == 1
    window.close()
    assert pipeline.stopped


def test_buttons_and_error_message_follow_pipeline_state(qtbot):
    module = window_module()
    pipeline = DisplayPipeline()
    window = module.MainWindow(pipeline)
    qtbot.addWidget(window)
    qtbot.waitUntil(lambda: window.stop_button.isEnabled())
    assert not window.start_button.isEnabled()
    pipeline.state = 'error'
    qtbot.waitUntil(lambda: 'camera unavailable' in window.status_label.text())
    assert window.start_button.isEnabled()
    assert not window.stop_button.isEnabled()
    window.close()
