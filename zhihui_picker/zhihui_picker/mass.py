"""质量估计接口。框架内置仿真用的模拟估计器；真实的视觉大模型估计器以插件形式接入。"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MassEstimate:
    mass_g: float
    source: str        # 'model' | 'geometry' | 'mock' …
    detail: str = ''


def geometric_mass_g(width_m: float, height_m: float, density: float = 950.0) -> float:
    """把果实近似为旋转椭球，按密度估质量。"""
    return density * (4.0 / 3.0) * math.pi * (width_m / 2) ** 2 * (height_m / 2) * 1000.0


def geometric_estimate(detection, crop: dict, detail: str = '') -> MassEstimate:
    lo, hi = crop['mass_range_g']
    m = geometric_mass_g(detection.width_m, detection.height_m, crop.get('density_kg_m3', 950.0))
    return MassEstimate(min(hi, max(lo, m)), 'geometry', detail)


class MassEstimator:
    """质量估计接口：返回 MassEstimate；估不出来返回 None（框架会退回按果径的几何估计）。"""

    def estimate(self, frame, detection, crop: dict) -> MassEstimate | None:
        raise NotImplementedError


class MockMassEstimator(MassEstimator):
    """仿真用：真值质量 × 对数正态噪声，模拟"看图估重"的误差。"""

    def __init__(self, world, sigma: float = 0.12, seed: int = 0):
        self.world, self.sigma = world, sigma
        self.rng = np.random.default_rng(seed)

    def estimate(self, frame, detection, crop):
        obj = self.world.nearest(detection.position_base, 0.06) if self.world else None
        if obj is None:
            return None
        return MassEstimate(float(obj.mass_g * math.exp(self.rng.normal(0, self.sigma))), 'mock')
