import time

import cv2
import numpy as np
import pyrealsense2 as rs

from .data import CameraFrame, normalize_intrinsics


class FrameConverter:
    def __init__(self, width, height, fx, fy, cx, cy, distortion, coefficients):
        self.intrinsics = normalize_intrinsics(fx, fy, cx, cy, width, height)
        self.maps = None
        if np.any(np.asarray(coefficients) != 0):
            if distortion != 'brown_conrady':
                raise ValueError(f'Unsupported nonzero camera distortion: {distortion}')
            matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            self.maps = cv2.initUndistortRectifyMap(matrix, np.asarray(coefficients), None, matrix,
                                                  (width, height), cv2.CV_32FC1)

    def convert(self, rgb, depth_raw, scale, sequence, captured_at):
        if self.maps is not None:
            rgb = cv2.remap(rgb, *self.maps, cv2.INTER_NEAREST)
            depth_raw = cv2.remap(depth_raw, *self.maps, cv2.INTER_NEAREST)
        return CameraFrame(sequence, captured_at, np.array(rgb, dtype=np.uint8, copy=True),
                           np.array(depth_raw, dtype=np.float32) * np.float32(scale),
                           self.intrinsics.copy())


class RealSenseCamera:
    def __init__(self, serial=None):
        self.serial = serial
        self.pipeline = None
        self.converter = None
        self.info = {}

    def start(self):
        devices = list(rs.context().query_devices())
        serials = [device.get_info(rs.camera_info.serial_number) for device in devices]
        if not serials:
            raise RuntimeError('未检测到 RealSense 相机，请检查 USB 连接')
        if self.serial is None and len(serials) > 1:
            raise RuntimeError(f'检测到多台相机，请使用 --serial 选择：{serials}')
        selected = self.serial or serials[0]
        if selected not in serials:
            raise RuntimeError(f'RealSense {selected} 未连接；可用设备：{serials}')
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(selected)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        profile = pipeline.start(config)
        self.pipeline = pipeline
        device = profile.get_device()
        self.scale = device.first_depth_sensor().get_depth_scale()
        self.aligner = rs.align(rs.stream.color)
        self.info = {
            'name': device.get_info(rs.camera_info.name), 'serial': selected,
            'firmware': device.get_info(rs.camera_info.firmware_version),
            'depth_scale': self.scale, 'width': 640, 'height': 480, 'capture_fps': 30,
        }
        return self.info.copy()

    def read(self, timeout_ms=1000):
        success, frames = self.pipeline.try_wait_for_frames(timeout_ms)
        if not success:
            return None
        captured_at = time.monotonic()
        aligned = self.aligner.process(frames)
        depth, color = aligned.get_depth_frame(), aligned.get_color_frame()
        if not depth or not color:
            return None
        if self.converter is None:
            calibration = depth.profile.as_video_stream_profile().get_intrinsics()
            distortion = str(calibration.model).split('.')[-1]
            self.converter = FrameConverter(calibration.width, calibration.height, calibration.fx,
                                            calibration.fy, calibration.ppx, calibration.ppy,
                                            distortion, calibration.coeffs)
            self.info.update({'distortion': distortion, 'coefficients': list(calibration.coeffs),
                              'intrinsics_normalized': self.converter.intrinsics.tolist()})
        return self.converter.convert(np.asanyarray(color.get_data()), np.asanyarray(depth.get_data()),
                                      self.scale, color.get_frame_number(), captured_at)

    def stop(self):
        if self.pipeline is not None:
            pipeline, self.pipeline = self.pipeline, None
            try:
                pipeline.stop()
            finally:
                self.converter = None
