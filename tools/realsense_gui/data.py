from dataclasses import dataclass
from threading import Lock
from typing import Generic, TypeVar

import numpy as np


ValueType = TypeVar('ValueType')


@dataclass(frozen=True)
class CameraFrame:
    sequence: int
    captured_at: float
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray


class LatestSlot(Generic[ValueType]):
    def __init__(self):
        self._lock = Lock()
        self._value = None

    def publish(self, value: ValueType) -> None:
        with self._lock:
            self._value = value

    def take(self) -> ValueType | None:
        with self._lock:
            value, self._value = self._value, None
            return value

    def clear(self) -> None:
        self.take()


def normalize_intrinsics(fx, fy, cx, cy, width, height) -> np.ndarray:
    if min(fx, fy, width, height) <= 0:
        raise ValueError('Focal lengths and image dimensions must be positive')
    return np.array([[fx / width, 0, cx / width], [0, fy / height, cy / height], [0, 0, 1]], dtype=np.float32)


def display_cloud(points, rgb, near_m, far_m):
    positions = np.asarray(points).reshape(-1, 3)
    colors = np.asarray(rgb).reshape(-1, 3)
    valid = np.isfinite(positions).all(axis=1) & (positions[:, 2] >= near_m) & (positions[:, 2] <= far_m)
    rgba = np.ones((int(valid.sum()), 4), dtype=np.float32)
    rgba[:, :3] = colors[valid].astype(np.float32) / 255
    return np.ascontiguousarray(positions[valid], dtype=np.float32), rgba
