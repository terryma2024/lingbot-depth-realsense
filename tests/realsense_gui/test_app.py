import importlib
from dataclasses import replace

import numpy as np
import pytest

from tools.realsense_gui.runtime import ConvertedFrame
from test_runtime import make_frame


def new_module(name):
    try:
        return importlib.import_module(f'tools.realsense_gui.{name}')
    except ModuleNotFoundError:
        assert False, f'GUI {name} has not been implemented'


def sample(sequence, presented_at):
    frame = replace(make_frame(sequence), captured_at=presented_at - 0.08)
    frame.rgb[..., 1] = sequence % 256
    enhanced = np.full((480, 640), 2 + sequence / 1000, dtype=np.float32)
    points = np.zeros((240, 320, 3), dtype=np.float32)
    points[..., 2] = enhanced[::2, ::2]
    return ConvertedFrame(frame, enhanced, points, frame.rgb[::2, ::2], 70, 68, presented_at - 0.01)


def test_cli_rejects_missing_engine_and_nonpositive_duration(tmp_path):
    app = new_module('app')
    with pytest.raises(SystemExit):
        app.parse_args(['--engine', str(tmp_path / 'missing.engine')])
    with pytest.raises(SystemExit):
        app.parse_args(['--verify-seconds', '0'])
    with pytest.raises(SystemExit):
        app.parse_args(['--verify-seconds', 'inf'])


def test_verification_excludes_initialization_and_counts_real_duration():
    validation = new_module('verification').Verification(1)
    assert validation.started_at is None
    validation.record(sample(1, 100), 100)
    for sequence in range(2, 12):
        validation.record(sample(sequence, 100 + (sequence - 1) / 10), 100 + (sequence - 1) / 10)
    report = validation.report(101, True, {}, [(100.5, 1200)])
    assert report['status'] == 'PASS'
    assert report['actual_display_hz'] == 10
    assert report['stable_seconds'] == 1
    assert report['memory_peak_mib'] == 1200


def test_verification_rejects_no_frames_stale_frames_and_failed_shutdown():
    module = new_module('verification')
    empty = module.Verification(1)
    assert empty.report(101, True, {}, [])['status'] == 'FAIL'
    stalled = module.Verification(1)
    stalled.record(sample(1, 100), 100)
    stalled.record(sample(1, 100.1), 100.1)
    assert stalled.report(101, True, {}, [(100.5, 1200)])['status'] == 'FAIL'
    complete = module.Verification(1)
    complete.record(sample(1, 100), 100)
    for sequence in range(2, 12):
        complete.record(sample(sequence, 100 + (sequence - 1) / 10), 100 + (sequence - 1) / 10)
    assert complete.report(101, False, {}, [(100.5, 1200)])['status'] == 'FAIL'


def test_memory_parser_only_attributes_own_process():
    module = new_module('verification')
    assert module.process_memory('111, 400\n222, 1234\n333, N/A\n', 222) == 1234
    assert module.process_memory('111, 400\n', 222) is None


def test_shutdown_tail_frames_are_excluded_from_stable_period():
    validation = new_module('verification').Verification(1)
    for sequence in range(1, 13):
        timestamp = 100 + (sequence - 1) / 10
        validation.record(sample(sequence, timestamp), timestamp)
    report = validation.report(101, True, {}, [(100.5, 1200)])
    assert report['actual_display_hz'] == 10
    assert report['unique_presented_frames'] == 11


def test_verification_requires_changing_rgb_and_depth_outputs():
    validation = new_module('verification').Verification(1)
    frozen = sample(1, 100)
    for sequence in range(1, 12):
        timestamp = 100 + (sequence - 1) / 10
        moving_frame = replace(frozen.frame, sequence=sequence, captured_at=timestamp - 0.08)
        validation.record(replace(frozen, frame=moving_frame), timestamp)
    report = validation.report(101, True, {}, [(100.5, 1200)])
    assert report['status'] == 'FAIL'
    assert not report['variation_ok']


def test_final_cleanup_error_prevents_passing_report():
    app = new_module('app')
    validation = new_module('verification').Verification(1)
    for sequence in range(1, 12):
        timestamp = 100 + (sequence - 1) / 10
        validation.record(sample(sequence, timestamp), timestamp)
    class FailedCleanupPipeline:
        def snapshot(self):
            return {'device': {}, 'state': 'error', 'error': 'camera stop failed'}
    report = app.final_report(validation, 101, True, FailedCleanupPipeline(), [(100.5, 1200)], '')
    assert report['status'] == 'FAIL'
    assert report['error'] == 'camera stop failed'


def test_geometry_report_explains_failed_pixels():
    validation = new_module('verification').Verification(1)
    result = sample(1, 100)
    result.points[..., 2] = 1
    validation.record(result, 100)
    report = validation.report(101, True, {}, [(100.5, 1200)])
    assert not report['geometry_ok']
    assert report.get('geometry_failures', [{}])[0].get('z_mismatch_pixels') == 76800


def test_geometry_after_measurement_cutoff_does_not_change_verdict():
    validation = new_module('verification').Verification(1)
    for sequence in range(1, 12):
        timestamp = 100 + (sequence - 1) / 10
        validation.record(sample(sequence, timestamp), timestamp)
    tail = sample(12, 101.1)
    tail.points[..., 2] = 1
    validation.record(tail, 101.1)
    assert validation.report(101, True, {}, [(100.5, 1200)])['status'] == 'PASS'
