import json
import os
import subprocess
from threading import Event, Lock, Thread
import time

import numpy as np


def process_memory(csv_text, pid):
    values = []
    for line in csv_text.splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) == 2 and fields[0] == str(pid):
            try:
                values.append(float(fields[1]))
            except ValueError:
                pass
    return sum(values) if values else None


class MemorySampler:
    def __init__(self):
        self.samples = []
        self.error = ''
        self._lock = Lock()
        self._stop = Event()
        self._thread = Thread(target=self._run, name='gpu-memory-sampler', daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(3)

    def snapshot(self):
        with self._lock:
            return self.samples.copy()

    def _run(self):
        while not self._stop.is_set():
            try:
                result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                         '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2, check=True)
                memory = process_memory(result.stdout, os.getpid())
                if memory is not None:
                    with self._lock:
                        self.samples.append((time.monotonic(), memory))
            except (OSError, subprocess.SubprocessError) as error:
                self.error = str(error)
            self._stop.wait(1)


class Verification:
    def __init__(self, seconds):
        self.seconds = seconds
        self.started_at = None
        self.last_sequence = None
        self.presented = []
        self.latencies = []
        self.conversion = []
        self.gpu = []
        self.input_signatures = []
        self.depth_signatures = []
        self.geometry_ok = True
        self.geometry_checks = []
        self.geometry_failures = []

    def record(self, result, presented_at):
        if result.frame.sequence == self.last_sequence:
            return
        self.last_sequence = result.frame.sequence
        if self.started_at is None:
            self.started_at = presented_at
        self.presented.append(presented_at)
        self.latencies.append((presented_at - result.frame.captured_at) * 1000)
        self.conversion.append(result.conversion_ms)
        self.gpu.append(result.gpu_ms)
        self.input_signatures.append(hash(result.frame.rgb[::16, ::16].tobytes()))
        self.depth_signatures.append(hash(result.depth_m[::16, ::16].tobytes()))
        stride = 480 // result.points.shape[0]
        nonfinite = int((~np.isfinite(result.depth_m)).sum())
        nonpositive = int((result.depth_m <= 0).sum())
        mismatches = int((result.points[..., 2] != result.depth_m[::stride, ::stride]).sum())
        check = nonfinite == nonpositive == mismatches == 0
        self.geometry_checks.append(check)
        self.geometry_ok &= check
        if not check:
            failure = {'sequence': result.frame.sequence, 'presented_at': presented_at,
                       'nonfinite_depth_pixels': nonfinite, 'nonpositive_depth_pixels': nonpositive,
                       'z_mismatch_pixels': mismatches}
            self.geometry_failures.append(failure)
            if len(self.geometry_failures) <= 5:
                print(json.dumps({'geometry_failure': failure}), flush=True)

    def report(self, end_at, shutdown_ok, device, memory_samples, error=''):
        duration = end_at - self.started_at if self.started_at is not None else 0
        eligible = [index for index, timestamp in enumerate(self.presented) if timestamp <= end_at]
        presented = [self.presented[index] for index in eligible]
        hz = (len(presented) - 1) / duration if duration > 0 else 0
        memory = [value for timestamp, value in memory_samples
                  if self.started_at is not None and self.started_at <= timestamp <= end_at]
        intervals = np.diff(presented) * 1000
        rgb_count = len({self.input_signatures[index] for index in eligible})
        depth_count = len({self.depth_signatures[index] for index in eligible})
        variation_ok = rgb_count > 1 and depth_count > 1
        geometry_ok = bool(eligible) and all(self.geometry_checks[index] for index in eligible)
        def summary(values):
            return {'mean': float(np.mean(values)), 'p95': float(np.percentile(values, 95))} if len(values) else None
        passed = (duration >= self.seconds and hz >= 9.5 and geometry_ok and variation_ok
                  and shutdown_ok and not error and bool(memory))
        return {'status': 'PASS' if passed else 'FAIL', 'device': device, 'stable_seconds': duration,
                'target_hz': 10, 'actual_display_hz': hz, 'unique_presented_frames': len(presented),
                'display_interval_ms': summary(intervals), 'application_latency_ms': summary([self.latencies[index] for index in eligible]),
                'conversion_ms': summary([self.conversion[index] for index in eligible]), 'gpu_ms': summary([self.gpu[index] for index in eligible]),
                'geometry_ok': geometry_ok, 'geometry_failures': [failure for failure in self.geometry_failures if failure['presented_at'] <= end_at],
                'variation_ok': variation_ok, 'distinct_rgb_frames': rgb_count,
                'distinct_depth_results': depth_count, 'memory_peak_mib': max(memory) if memory else None,
                'memory_sampling': 'Own PID nvidia-smi process total, 1s sampling; includes native TRT and graphics',
                'shutdown_ok': shutdown_ok, 'error': error,
                'latency_definition': 'Host SDK frame receipt to GUI update submission; excludes exposure/USB transport and monitor scanout'}
