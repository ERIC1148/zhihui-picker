"""仿真后端：仿真世界、仿真机械臂、仿真夹爪。与实机后端实现同一套接口。"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from .kinematics import ArmGeometry, forward


class HardwareFault(RuntimeError):
    pass


@dataclass
class SimObject:
    label: str
    position: tuple[float, float, float]
    width_m: float
    height_m: float
    mass_g: float
    true_min_force_n: float      # 该果实真实的最小不滑力（已量化）
    color: tuple[int, int, int]
    state: str = 'hanging'       # hanging | held | in_basket


class SimWorld:
    def __init__(self, objects: list[SimObject]):
        self.objects = objects
        self.lock = threading.Lock()

    def visible_objects(self):
        with self.lock:
            return [o for o in self.objects if o.state == 'hanging']

    def nearest(self, pos, max_dist=0.04):
        best, bd = None, max_dist
        for o in self.visible_objects():
            d = math.dist(o.position, pos)
            if d < bd:
                best, bd = o, d
        return best


class SimArm:
    def __init__(self, geo: ArmGeometry, home_q: list[float]):
        self.geo, self.q = geo, list(home_q)
        self.enabled = False

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def estop(self):
        self.enabled = False

    def read(self) -> list[float]:
        return list(self.q)

    def command(self, q: list[float], payload_kg: float = 0.0):
        if not self.enabled:
            raise HardwareFault('机械臂未使能')
        self.q = list(q)

    def close(self):
        pass


class SimGripper:
    """仿真夹爪：施加力 >= 果实真实最小力才抓得住，否则提起时滑脱。"""

    def __init__(self, world: SimWorld, arm: SimArm, open_gap_m: float = 0.096):
        self.world, self.arm, self.open_gap = world, arm, open_gap_m
        self.gap, self.force, self.held = open_gap_m, 0.0, None

    def open(self):
        if self.held:
            with self.world.lock:
                self.held.state = 'in_basket'
            self.held = None
        self.gap, self.force = self.open_gap, 0.0

    def grasp(self, object_width_m: float, force_n: float):
        self.gap, self.force = max(0.0, object_width_m - 0.004), force_n
        obj = self.world.nearest(forward(self.arm.geo, self.arm.q), 0.03)
        if obj and force_n >= obj.true_min_force_n - 1e-9:
            with self.world.lock:
                obj.state = 'held'
            self.held = obj

    def finger_gap(self) -> float:
        return self.gap

    def object_lost(self) -> bool:
        return self.force > 0 and self.held is None

    def close(self):
        pass
