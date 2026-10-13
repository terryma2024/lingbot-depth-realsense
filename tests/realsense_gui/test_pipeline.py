import importlib
import time

import numpy as np

from tools.realsense_gui.data import CameraFrame
from tools.realsense_gui.runtime import ConvertedFrame


def pipeline_module():
    try:
        return importlib.import_module('tools.realsense_gui.pipeline')
    except ModuleNotFoundError:
        assert False, 'Background pipeline has not been implemented'


class FakeCamera:
    def __init__(self, fail=False, timeout=False):
        self.sequence = 0
        self.fail = fail
        self.timeout = timeout
        self.open = False
        self.starts = 0
        self.info = {'name': 'test camera', 'serial': 'test'}

    def start(self):
        self.open = True
        self.starts += 1
        return self.info

    def read(self, timeout_ms=1000):
        time.sleep(0.005)
        if self.fail:
            raise RuntimeError('camera disconnected')
        if self.timeout:
            return None
        self.sequence += 1
        return CameraFrame(self.sequence, time.monotonic(),
                           np.full((2, 2, 3), self.sequence % 256, dtype=np.uint8),
                           np.full((2, 2), self.sequence, dtype=np.float32), np.eye(3, dtype=np.float32))

    def stop(self):
        self.open = False


class FakeRuntime:
    def __init__(self, delay=0.01):
        self.delay = delay

    def warmup(self, frame):
        pass

    def convert(self, frame, stride=2):
        time.sleep(self.delay)
        return ConvertedFrame(frame, frame.depth_m.copy(), np.zeros((1, 1, 3)), frame.rgb[:1, :1].copy(),
                              self.delay * 1000, 1, time.monotonic())

    def close(self):
        pass


def await_result(pipeline, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = pipeline.take_result()
        if result is not None:
            return result
        time.sleep(0.005)
    raise AssertionError(f'No conversion result: {pipeline.snapshot()}')


def test_latest_frames_skip_backlog_and_keep_same_frame():
    camera = FakeCamera()
    pipeline = pipeline_module().Pipeline(lambda: camera, FakeRuntime)
    try:
        pipeline.start()
        first = await_result(pipeline)
        second = await_result(pipeline)
        assert second.frame.sequence > first.frame.sequence + 1
        np.testing.assert_array_equal(second.depth_m, second.frame.sequence)
        np.testing.assert_array_equal(second.frame.rgb, second.frame.sequence % 256)
        assert second.completed_at - second.frame.captured_at < 0.08
    finally:
        pipeline.stop()
        assert pipeline.join(2)
    assert not camera.open


def test_duplicate_start_and_restart_do_not_duplicate_workers():
    camera = FakeCamera()
    pipeline = pipeline_module().Pipeline(lambda: camera, FakeRuntime, hz=1)
    pipeline.start()
    pipeline.start()
    await_result(pipeline)
    assert camera.starts == 1
    started = time.monotonic()
    pipeline.stop()
    assert pipeline.join(0.3)
    assert time.monotonic() - started < 0.3
    assert pipeline.snapshot()['state'] == 'stopped'
    pipeline.start()
    await_result(pipeline)
    pipeline.stop()
    assert pipeline.join(1)
    assert camera.starts == 2


def test_disconnect_stops_workers_and_releases_camera():
    camera = FakeCamera(fail=True)
    pipeline = pipeline_module().Pipeline(lambda: camera, FakeRuntime)
    pipeline.start()
    assert pipeline.join(2)
    assert pipeline.snapshot()['state'] == 'error'
    assert 'disconnected' in pipeline.snapshot()['error']
    assert pipeline.take_result() is None
    assert not camera.open


def test_repeated_timeouts_report_error_without_empty_results():
    camera = FakeCamera(timeout=True)
    pipeline = pipeline_module().Pipeline(lambda: camera, FakeRuntime)
    pipeline.start()
    assert pipeline.join(2)
    assert pipeline.snapshot()['state'] == 'error'
    assert pipeline.take_result() is None
    assert not camera.open


def test_slow_conversion_does_not_burst_to_catch_up():
    camera = FakeCamera()
    pipeline = pipeline_module().Pipeline(lambda: camera, lambda: FakeRuntime(0.12))
    try:
        pipeline.start()
        first = await_result(pipeline)
        second = await_result(pipeline)
        assert second.completed_at - first.completed_at >= 0.115
        assert second.frame.sequence > first.frame.sequence + 5
    finally:
        pipeline.stop()
        assert pipeline.join(2)
