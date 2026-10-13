import importlib
import os
from pathlib import Path
import time

import numpy as np
import pytest

from tools.realsense_gui.data import CameraFrame, normalize_intrinsics


def runtime_module():
    try:
        return importlib.import_module('tools.realsense_gui.runtime')
    except ModuleNotFoundError:
        assert False, 'TRT runtime has not been implemented'


def make_frame(sequence=1, depth=2, fx=600):
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    return CameraFrame(sequence, time.monotonic(), rgb,
                       np.full((480, 640), depth, dtype=np.float32),
                       normalize_intrinsics(fx, 600, 320, 240, 640, 480))


def test_preprocessing_preserves_rgb_order_and_meters():
    image, depth = runtime_module().prepare_inputs(make_frame())
    assert image.shape == (1, 3, 480, 640)
    assert depth.shape == (1, 480, 640)
    assert image.dtype == depth.dtype == np.float16
    np.testing.assert_array_equal(image[0, 0], 1)
    np.testing.assert_array_equal(image[0, 1:], 0)
    np.testing.assert_array_equal(depth, 2)
    assert image.flags.c_contiguous and depth.flags.c_contiguous


def test_runtime_rejects_incompatible_engine_layout():
    module = runtime_module()
    with pytest.raises(ValueError, match='layout'):
        module.validate_layout({'image': ((1, 3, 240, 320), 'float32')})


@pytest.mark.skipif(os.environ.get('LINGBOT_GUI_GPU_TESTS') != '1', reason='explicit GPU integration opt-in')
def test_gpu_replay_consumes_new_frames_and_preserves_owned_outputs():
    module = runtime_module()
    engine = Path('result/optimization/reference-trt-nt2000-fp16/lingbot_depth_nt2000.engine')
    runtime = module.TrtRuntime(engine)
    try:
        runtime.warmup(make_frame())
        first = runtime.convert(make_frame(1, 2))
        saved_depth = first.depth_m.copy()
        saved_points = first.points.copy()
        second = runtime.convert(make_frame(2, 3))
        assert second.frame.sequence == 2
        assert np.isfinite(second.depth_m).all() and (second.depth_m > 0).all()
        assert not np.array_equal(first.depth_m, second.depth_m)
        np.testing.assert_array_equal(first.depth_m, saved_depth)
        np.testing.assert_array_equal(first.points, saved_points)
        np.testing.assert_array_equal(second.points[..., 2], second.depth_m[::2, ::2])
        changed_intrinsics = runtime.convert(make_frame(3, 3, fx=300))
        np.testing.assert_array_equal(changed_intrinsics.depth_m, second.depth_m)
        np.testing.assert_allclose(changed_intrinsics.points[..., 0], second.points[..., 0] * 2, atol=1e-5)
        assert changed_intrinsics.colors.shape == (240, 320, 3)
        blue_frame = make_frame(4, 3, fx=300)
        blue_frame.rgb[:] = [0, 0, 255]
        changed_rgb = runtime.convert(blue_frame)
        assert not np.array_equal(changed_rgb.depth_m, changed_intrinsics.depth_m)
    finally:
        runtime.close()
