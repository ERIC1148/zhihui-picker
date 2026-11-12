"""视觉：相机接口、作物检测、像素 → 相机系 → 底座系定位。

坐标约定
  光学系 (optical)：z 向前、x 向右、y 向下（深度相机原生约定）
  camera_link    ：x 向前、y 向左、z 向上（ROS 机体约定，手眼标定给的是它在 base_link 下的位姿）
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class Frame:
    rgb: np.ndarray        # HxWx3 uint8
    depth_m: np.ndarray    # HxW float32，单位米，0 表示无效
    intrinsics: Intrinsics
    stamp: float


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]            # x0, y0, x1, y1（像素）
    position_base: tuple[float, float, float]  # 果实中心，base_link，米
    width_m: float                             # 水平方向果径（夹爪开合方向）
    height_m: float
    depth_m: float


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


class HandEye:
    """手眼标定：camera_link 在 base_link 下的位姿（眼在手外）。"""

    def __init__(self, xyz, rpy):
        self.R = rpy_matrix(*rpy)
        self.t = np.asarray(xyz, dtype=float)

    def optical_to_base(self, p_opt) -> np.ndarray:
        x, y, z = p_opt
        return self.R @ np.array([z, -x, -y]) + self.t

    def base_to_optical(self, p_base) -> np.ndarray:
        c = self.R.T @ (np.asarray(p_base, dtype=float) - self.t)
        return np.array([-c[1], -c[2], c[0]])


def deproject(intr: Intrinsics, u: float, v: float, z: float) -> tuple[float, float, float]:
    """小孔模型反投影：像素 (u,v) + 深度 z → 光学系三维点。"""
    return (u - intr.cx) * z / intr.fx, (v - intr.cy) * z / intr.fy, z


# ---------------------------------------------------------------- 检测器
def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 → H∈[0,360), S,V∈[0,1]。"""
    a = rgb.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(-1), a.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-6
    rm, gm, bm = nz & (mx == r), nz & (mx == g) & (mx != r), nz & (mx == b) & (mx != r) & (mx != g)
    h[rm] = (60 * ((g - b)[rm] / d[rm])) % 360
    h[gm] = 60 * ((b - r)[gm] / d[gm]) + 120
    h[bm] = 60 * ((r - g)[bm] / d[bm]) + 240
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0)
    return np.stack([h, s, mx], -1)


def _localize(label, conf, mask, bbox, frame: Frame, hand_eye: HandEye,
              depth_range) -> Detection | None:
    """由掩膜/框 + 深度图得到果实中心的底座系坐标与果径。"""
    x0, y0, x1, y1 = bbox
    d = frame.depth_m[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1] if mask is not None else np.ones_like(d, bool)
    vals = d[m & (d > depth_range[0]) & (d < depth_range[1])]
    if vals.size < 20:
        return None                                   # 深度空洞太多，不可信
    z_surf = float(np.percentile(vals, 25))           # 取近表面，抗背景混入
    intr = frame.intrinsics
    width = (x1 - x0) * z_surf / intr.fx
    height = (y1 - y0) * z_surf / intr.fy
    z_c = z_surf + min(width, height) / 2.0           # 表面 → 果实中心
    if mask is not None:
        ys, xs = np.nonzero(m)
        u, v = x0 + xs.mean(), y0 + ys.mean()
    else:
        u, v = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    p = hand_eye.optical_to_base(deproject(intr, u, v, z_c))
    return Detection(label, float(conf), (int(x0), int(y0), int(x1), int(y1)),
                     (float(p[0]), float(p[1]), float(p[2])), float(width), float(height), z_surf)


class ColorDetector:
    """HSV 颜色分割检测器：零依赖（numpy + scipy），用于演示与无 GPU 环境的兜底。"""

    def __init__(self, crops: dict, hand_eye: HandEye, min_area_px: int = 300,
                 depth_range=(0.15, 1.5)):
        self.crops, self.hand_eye = crops, hand_eye
        self.min_area, self.depth_range = min_area_px, depth_range

    def detect(self, frame: Frame) -> list[Detection]:
        from scipy import ndimage
        hsv = rgb_to_hsv(frame.rgb)
        out = []
        for label, c in self.crops.items():
            hs = c.get('hsv')
            if not hs:
                continue
            h_lo, h_hi = hs['h']
            in_h = ((hsv[..., 0] >= h_lo) & (hsv[..., 0] <= h_hi) if h_lo <= h_hi
                    else (hsv[..., 0] >= h_lo) | (hsv[..., 0] <= h_hi))   # 红色跨 0°
            mask = in_h & (hsv[..., 1] >= hs['s_min']) & (hsv[..., 2] >= hs['v_min'])
            mask = ndimage.binary_opening(mask, iterations=2)
            lab, n = ndimage.label(mask)
            for i, sl in enumerate(ndimage.find_objects(lab), start=1):
                comp = lab[sl] == i
                area = int(comp.sum())
                if area < self.min_area:
                    continue
                bbox = (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop)
                full = np.zeros_like(mask)
                full[sl] = comp
                fill = area / ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                det = _localize(label, min(0.99, 0.5 + 0.5 * fill), full, bbox, frame,
                                self.hand_eye, self.depth_range)
                if det:
                    out.append(det)
        return sorted(out, key=lambda d: -d.confidence)


class YoloDetector:
    """Ultralytics YOLO 检测器（生产环境推荐）。需要: pip install ultralytics，并提供训练好的权重。"""

    def __init__(self, weights: str, crops: dict, hand_eye: HandEye, conf: float = 0.5,
                 class_map: dict | None = None, depth_range=(0.15, 1.5)):
        from ultralytics import YOLO   # 延迟导入：没装也不影响其它后端
        self.model = YOLO(weights)
        self.crops, self.hand_eye, self.conf = crops, hand_eye, conf
        self.class_map, self.depth_range = class_map or {}, depth_range

    def detect(self, frame: Frame) -> list[Detection]:
        res = self.model.predict(frame.rgb[..., ::-1], conf=self.conf, verbose=False)[0]
        out = []
        h, w = frame.depth_m.shape
        for box in res.boxes:
            name = res.names[int(box.cls)]
            label = self.class_map.get(name, name)
            if label not in self.crops:
                continue
            x0, y0, x1, y1 = (int(round(float(v))) for v in box.xyxy[0])
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            # 无分割掩膜时只用框中心 50% 区域取深度，减少背景混入
            mx, my = (x1 - x0) // 4, (y1 - y0) // 4
            mask = np.zeros((h, w), bool)
            mask[y0 + my:y1 - my, x0 + mx:x1 - mx] = True
            det = _localize(label, float(box.conf), mask, (x0, y0, x1, y1), frame,
                            self.hand_eye, self.depth_range)
            if det:
                out.append(det)
        return sorted(out, key=lambda d: -d.confidence)


# ---------------------------------------------------------------- 相机
class RealSenseCamera:
    """Intel RealSense D4xx（彩色与深度对齐）。需要: pip install pyrealsense2。"""

    def __init__(self, width=640, height=480, fps=30, serial: str | None = None):
        import pyrealsense2 as rs
        self._rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        prof = self.pipe.start(cfg)
        self.scale = prof.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intr = Intrinsics(i.fx, i.fy, i.ppx, i.ppy, i.width, i.height)

    def read(self) -> Frame:
        import time
        fs = self.align.process(self.pipe.wait_for_frames(timeout_ms=2000))
        c, d = fs.get_color_frame(), fs.get_depth_frame()
        if not c or not d:
            raise RuntimeError('RealSense 未返回完整的彩色/深度帧')
        return Frame(np.asanyarray(c.get_data()).copy(),
                     np.asanyarray(d.get_data()).astype(np.float32) * self.scale,
                     self.intr, time.time())

    def close(self):
        self.pipe.stop()


class SimCamera:
    """仿真相机：把 SimWorld 里的作物渲染成 RGB-D 图，让检测与定位走真实代码路径。"""

    def __init__(self, world, hand_eye: HandEye, width=640, height=480, fov_fx=600.0,
                 noise_m: float = 0.002, seed: int = 0):
        self.world, self.hand_eye = world, hand_eye
        self.intr = Intrinsics(fov_fx, fov_fx, width / 2, height / 2, width, height)
        self.noise = noise_m
        self.rng = np.random.default_rng(seed)

    def read(self) -> Frame:
        import time
        it = self.intr
        rgb = np.empty((it.height, it.width, 3), np.uint8)
        rgb[:] = (70, 96, 62)                                  # 暗绿叶背景
        rgb += self.rng.integers(0, 12, rgb.shape, dtype=np.uint8)
        depth = np.full((it.height, it.width), 1.2, np.float32)
        vv, uu = np.mgrid[0:it.height, 0:it.width]
        for o in sorted(self.world.visible_objects(), key=lambda o: -self.hand_eye.base_to_optical(o.position)[2]):
            x, y, z = self.hand_eye.base_to_optical(o.position)
            if z <= 0.1:
                continue
            u0, v0 = it.fx * x / z + it.cx, it.fy * y / z + it.cy
            a, b = it.fx * (o.width_m / 2) / z, it.fy * (o.height_m / 2) / z
            e = ((uu - u0) / a) ** 2 + ((vv - v0) / b) ** 2
            m = e <= 1.0
            shade = (1.0 - 0.35 * np.clip(e[m], 0, 1))[:, None]
            rgb[m] = np.clip(np.asarray(o.color) * shade, 0, 255).astype(np.uint8)
            depth[m] = z - (o.width_m / 2) * np.sqrt(np.clip(1 - e[m], 0, 1))
        depth += self.rng.normal(0, self.noise, depth.shape).astype(np.float32)
        return Frame(rgb, depth, it, time.time())

    def close(self):
        pass
