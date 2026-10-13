import importlib

import numpy as np
import pytest


def camera_module():
    try:
        return importlib.import_module('tools.realsense_gui.camera')
    except ModuleNotFoundError:
        assert False, 'RealSense frame conversion has not been implemented'


def test_camera_frame_owns_rgb_and_scaled_depth():
    module = camera_module()
    rgb = np.full((480, 640, 3), 127, dtype=np.uint8)
    raw = np.full((480, 640), 1000, dtype=np.uint16)
    converter = module.FrameConverter(640, 480, 600, 610, 320, 240, 'none', [0] * 5)
    frame = converter.convert(rgb, raw, 0.002, 7, 123.0)
    rgb[:] = 0
    raw[:] = 0
    assert frame.sequence == 7
    assert frame.captured_at == 123.0
    np.testing.assert_array_equal(frame.rgb, 127)
    np.testing.assert_array_equal(frame.depth_m, 2)
    assert frame.depth_m.dtype == np.float32


def test_nonzero_brown_distortion_remaps_color_and_depth_together():
    module = camera_module()
    converter = module.FrameConverter(640, 480, 400, 400, 320, 240, 'brown_conrady', [0.2, 0, 0, 0, 0])
    raw = np.tile(np.arange(640, dtype=np.uint16), (480, 1))
    rgb = np.repeat((raw % 256).astype(np.uint8)[..., None], 3, axis=2)
    frame = converter.convert(rgb, raw, 1, 1, 0)
    valid = frame.depth_m > 0
    np.testing.assert_array_equal(frame.rgb[..., 0][valid], (frame.depth_m[valid] % 256).astype(np.uint8))
    assert not np.array_equal(frame.depth_m, raw)


def test_unsupported_nonzero_distortion_is_rejected():
    module = camera_module()
    with pytest.raises(ValueError, match='distortion'):
        module.FrameConverter(640, 480, 400, 400, 320, 240, 'ftheta', [0.2, 0, 0, 0, 0])
