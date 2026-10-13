import importlib

import numpy as np


def data_module():
    try:
        return importlib.import_module('tools.realsense_gui.data')
    except ModuleNotFoundError:
        assert False, 'GUI data contracts have not been implemented'


def test_latest_slot_replaces_unconsumed_frame():
    slot = data_module().LatestSlot()
    slot.publish('older')
    slot.publish('newer')
    assert slot.take() == 'newer'
    assert slot.take() is None
    slot.publish('discard')
    slot.clear()
    assert slot.take() is None


def test_intrinsics_recover_pixel_focal_length_and_center():
    normalized = data_module().normalize_intrinsics(600, 610, 320, 240, 640, 480)
    np.testing.assert_allclose(normalized, [[0.9375, 0, 0.5], [0, 610 / 480, 0.5], [0, 0, 1]])


def test_cloud_filters_geometry_and_matching_colors():
    points = np.array([[1, 2, 2], [0, 0, np.nan], [0, 0, 0], [0, 0, 8]], dtype=np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]], dtype=np.uint8)
    positions, rgba = data_module().display_cloud(points, colors, 0.1, 5)
    np.testing.assert_array_equal(positions, [[1, 2, 2]])
    np.testing.assert_array_equal(rgba, [[1, 0, 0, 1]])
    np.testing.assert_array_equal(points[0], [1, 2, 2])
