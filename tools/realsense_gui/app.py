import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENGINE = ROOT / 'result/optimization/reference-trt-nt2000-fp16/lingbot_depth_nt2000.engine'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='RealSense + LingBot-Depth TRT 四视图实时 GUI（10Hz）')
    parser.add_argument('--serial', help='RealSense 序列号，单设备时自动选择')
    parser.add_argument('--engine', type=Path, default=DEFAULT_ENGINE, help='固定 640×480 FP16 TensorRT 引擎')
    parser.add_argument('--verify-seconds', type=float, help='预热后稳定期验证时长，结束后退出')
    parser.add_argument('--report', type=Path, default=ROOT / 'result/realsense_gui/verification.json')
    parser.add_argument('--screenshot', type=Path, help='验证结束时保存 GUI 截图')
    args = parser.parse_args(argv)
    if not args.engine.is_file():
        parser.error(f'TensorRT 引擎不存在：{args.engine}')
    if args.verify_seconds is not None and (args.verify_seconds <= 0 or not math.isfinite(args.verify_seconds)):
        parser.error('--verify-seconds 必须为正数')
    return args


def final_report(verification, end_at, shutdown_ok, pipeline, memory_samples, error):
    snapshot = pipeline.snapshot()
    return verification.report(end_at, shutdown_ok, snapshot['device'], memory_samples, error or snapshot['error'])


def main(argv=None):
    args = parse_args(argv)
    from .camera import RealSenseCamera
    from .pipeline import Pipeline
    from .runtime import TrtRuntime
    from .verification import MemorySampler, Verification
    from .window import MainWindow, configure_qt
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    configure_qt()
    application = QApplication(sys.argv[:1])
    pipeline = Pipeline(lambda: RealSenseCamera(args.serial), lambda: TrtRuntime(args.engine))
    window = MainWindow(pipeline)
    screen = application.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        window.resize(min(1400, available.width() - 80), min(950, available.height() - 80))
    verification = Verification(args.verify_seconds) if args.verify_seconds else None
    sampler = MemorySampler() if verification is not None else None
    started = time.monotonic()
    end_at = None
    verification_error = ''
    last_state = None

    if verification is not None:
        sampler.start()
        window.presented.connect(verification.record)

    def monitor():
        nonlocal end_at, verification_error, last_state
        snapshot = pipeline.snapshot()
        if snapshot['state'] != last_state:
            print(json.dumps({'state': snapshot['state'], 'device': snapshot['device'], 'error': snapshot['error']}, ensure_ascii=False), flush=True)
            last_state = snapshot['state']
        if verification is None or end_at is not None:
            return
        now = time.monotonic()
        finished = verification.started_at is not None and now - verification.started_at >= verification.seconds
        failed = snapshot['state'] == 'error' or (verification.started_at is None and now - started > 120)
        if failed:
            verification_error = snapshot['error'] or '初始化超过 120 秒，未收到转换结果'
        if finished or failed:
            end_at = now
            if args.screenshot is not None and window.last_result is not None:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                if not window.grab().save(str(args.screenshot)):
                    verification_error = 'GUI 截图保存失败'
            window.close()

    monitor_timer = QTimer(application)
    monitor_timer.timeout.connect(monitor)
    monitor_timer.start(100)
    window.show()
    QTimer.singleShot(0, pipeline.start)
    exit_code = application.exec()
    pipeline.stop()
    shutdown_ok = pipeline.join(5)
    if sampler is not None:
        sampler.stop()
        report = final_report(verification, end_at if end_at is not None else time.monotonic(), shutdown_ok,
                              pipeline, sampler.snapshot(), verification_error)
        report.update({'engine': str(args.engine), 'versions': {
            package: importlib.metadata.version(package)
            for package in ['torch', 'tensorrt', 'pyrealsense2', 'PyQt6', 'pyqtgraph', 'PyOpenGL']},
            'memory_sampling_error': sampler.error, 'screenshot': str(args.screenshot) if args.screenshot else None,
            'semantic_changes': ['depth-token masking disabled', 'RGB antialias disabled', 'output confidence mask omitted']})
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        if report['status'] != 'PASS':
            exit_code = 1
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
