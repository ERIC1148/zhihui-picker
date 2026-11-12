"""智汇蔬香 4 自由度采摘臂：正/逆运动学、最小加加速度轨迹、重力矩前馈。

关节定义与 urdf/picker_arm.urdf 一致：
  q1 腰部偏航；q2 肩（0 = 大臂竖直向上）；q3 肘；q4 腕（保持夹爪水平：q2+q3+q4 = 0）
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


class UnreachableError(ValueError):
    pass


@dataclass
class ArmGeometry:
    z_base: float = 0.135
    l1: float = 0.326
    l2: float = 0.326
    l_ee: float = 0.100
    lower: list[float] = field(default_factory=lambda: [-3.14, -1.57, -2.4, -3.14])
    upper: list[float] = field(default_factory=lambda: [3.14, 1.57, 2.4, 3.14])
    # 重力前馈用的集总质量 (kg) 与质心位置 (m)，需按实机称重后修改
    m1: float = 0.60
    m2: float = 0.55
    lc1: float = 0.163
    lc2: float = 0.200


def forward(geo: ArmGeometry, q: list[float]) -> tuple[float, float, float]:
    """关节角 → 夹持中心在 base_link 下的坐标（假设夹爪水平）。"""
    q1, q2, q3 = q[0], q[1], q[2]
    r = geo.l1 * math.sin(q2) + geo.l2 * math.sin(q2 + q3) + geo.l_ee
    z = geo.z_base + geo.l1 * math.cos(q2) + geo.l2 * math.cos(q2 + q3)
    return r * math.cos(q1), r * math.sin(q1), z


def inverse(geo: ArmGeometry, x: float, y: float, z: float, tol: float = 2e-3) -> list[float]:
    """夹持中心目标点 → [q1..q4]。超出工作空间或关节限位时抛 UnreachableError（不做静默截断）。"""
    rho = math.hypot(x, y)
    if rho < geo.l_ee + 0.03:
        raise UnreachableError(f'目标离腰部轴线过近 (ρ={rho:.3f} m)')
    q1 = math.atan2(y, x)
    r, dz = rho - geo.l_ee, z - geo.z_base
    d = math.hypot(r, dz)
    if d > geo.l1 + geo.l2 - tol or d < abs(geo.l1 - geo.l2) + tol:
        raise UnreachableError(f'目标超出臂展: 腕部距肩 {d:.3f} m，最大 {geo.l1 + geo.l2:.3f} m')
    c3 = (d * d - geo.l1 ** 2 - geo.l2 ** 2) / (2 * geo.l1 * geo.l2)
    q3 = math.acos(max(-1.0, min(1.0, c3)))
    q2 = math.atan2(r, dz) - math.atan2(geo.l2 * math.sin(q3), geo.l1 + geo.l2 * math.cos(q3))
    q = [q1, q2, q3, -(q2 + q3)]
    for i, (v, lo, hi) in enumerate(zip(q, geo.lower, geo.upper)):
        if not lo <= v <= hi:
            raise UnreachableError(f'关节 {i + 1} 超限: {v:.3f} rad 不在 [{lo}, {hi}]')
    return q


def min_jerk(s: float) -> float:
    """五次多项式时间缩放：起止速度、加速度均为 0。"""
    s = max(0.0, min(1.0, s))
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def gravity_torque(geo: ArmGeometry, q: list[float], payload_kg: float = 0.0) -> list[float]:
    """抵消重力所需的关节保持力矩 (N·m，URDF 正方向)；腰部与腕部近似为 0。

    重力使 q2、q3 增大（向前倒），所以保持力矩为负。"""
    g = 9.81
    s2, s23 = math.sin(q[1]), math.sin(q[1] + q[2])
    pay_arm2 = geo.l2 * s23 + geo.l_ee          # 负载相对肘的水平力臂
    t3 = geo.m2 * g * geo.lc2 * s23 + payload_kg * g * pay_arm2
    t2 = (geo.m1 * g * geo.lc1 * s2 + geo.m2 * g * (geo.l1 * s2 + geo.lc2 * s23)
          + payload_kg * g * (geo.l1 * s2 + pay_arm2))
    return [0.0, -t2, -t3, 0.0]
