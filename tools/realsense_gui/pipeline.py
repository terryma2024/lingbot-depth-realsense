from threading import Event, Lock, Thread
import time
import traceback

from .data import LatestSlot


class Pipeline:
    def __init__(self, camera_factory, runtime_factory, hz=10.0):
        if not 0 < hz <= 30:
            raise ValueError('Conversion rate must be in (0, 30] Hz')
        self.camera_factory = camera_factory
        self.runtime_factory = runtime_factory
        self.period = 1 / hz
        self.frames = LatestSlot()
        self.results = LatestSlot()
        self._lock = Lock()
        self._stop = Event()
        self._threads = []
        self._state = 'stopped'
        self._error = ''
        self._traceback = ''
        self._device = {}
        self._stride = 2
        self._live_workers = 0

    def start(self):
        with self._lock:
            if any(thread.is_alive() for thread in self._threads):
                return
            self._stop.clear()
            self.frames.clear()
            self.results.clear()
            self._state, self._error, self._traceback = 'starting', '', ''
            self._device = {}
            self._live_workers = 2
            self._threads = [Thread(target=self._capture, name='realsense-capture'),
                             Thread(target=self._infer, name='lingbot-inference')]
            for thread in self._threads:
                thread.start()

    def stop(self):
        with self._lock:
            if self._state in ('starting', 'running'):
                self._state = 'stopping'
            self._stop.set()

    def join(self, timeout):
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0, deadline - time.monotonic()))
        return not any(thread.is_alive() for thread in self._threads)

    def take_result(self):
        return self.results.take()

    def set_stride(self, stride):
        if stride not in (1, 2, 4, 8):
            raise ValueError('Stride must be 1, 2, 4 or 8')
        with self._lock:
            self._stride = stride

    def snapshot(self):
        with self._lock:
            return {'state': self._state, 'error': self._error, 'traceback': self._traceback,
                    'device': self._device.copy(), 'workers_alive': self._live_workers}

    def _fail(self, error):
        with self._lock:
            if not self._error:
                self._error = str(error)
                self._traceback = traceback.format_exc()
            self._state = 'error'
            self._stop.set()

    def _worker_done(self):
        with self._lock:
            self._live_workers -= 1
            if self._live_workers == 0:
                if self._state != 'error':
                    self._state = 'stopped'
                self.frames.clear()

    def _capture(self):
        camera = None
        try:
            camera = self.camera_factory()
            info = camera.start()
            with self._lock:
                self._device = info.copy()
            timeouts = 0
            while not self._stop.is_set():
                frame = camera.read(timeout_ms=1000)
                if frame is None:
                    timeouts += 1
                    if timeouts >= 3:
                        raise RuntimeError('相机连续取帧超时，请检查 USB 或重新连接设备')
                    continue
                timeouts = 0
                with self._lock:
                    self._device = camera.info.copy()
                self.frames.publish(frame)
        except Exception as error:
            self._fail(error)
        finally:
            try:
                if camera is not None:
                    camera.stop()
            except Exception as error:
                self._fail(error)
            self._worker_done()

    def _infer(self):
        runtime = None
        try:
            frame = None
            while frame is None and not self._stop.is_set():
                frame = self.frames.take()
                if frame is None:
                    self._stop.wait(0.005)
            if self._stop.is_set():
                return
            runtime = self.runtime_factory()
            runtime.warmup(frame)
            with self._lock:
                if self._stop.is_set():
                    return
                self._state = 'running'
            next_due = time.monotonic()
            while not self._stop.is_set():
                if self._stop.wait(max(0, next_due - time.monotonic())):
                    break
                frame = self.frames.take()
                if frame is None:
                    self._stop.wait(0.005)
                    continue
                with self._lock:
                    stride = self._stride
                result = runtime.convert(frame, stride=stride)
                if not self._stop.is_set():
                    self.results.publish(result)
                next_due = max(next_due + self.period, time.monotonic())
        except Exception as error:
            self._fail(error)
        finally:
            try:
                if runtime is not None:
                    runtime.close()
            except Exception as error:
                self._fail(error)
            self._worker_done()
