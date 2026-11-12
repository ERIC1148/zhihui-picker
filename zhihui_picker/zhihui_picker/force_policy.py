"""抓取力策略接口，以及一个教科书式的参考实现。

流程框架只依赖这个接口：给定作物和估计质量，决定用多大的夹持力；抓取结束后把结果反馈给策略。
力的口径：两指法向力之和，单位 N。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

G = 9.81


@dataclass(frozen=True)
class ForceDecision:
    force_n: float                    # 下发给夹爪的力
    experience_used: bool = False     # 这次决策是否用到了以往的抓取经验
    note: str = ''                    # 给操作员看的简短说明
    details: dict = field(default_factory=dict)   # 策略内部细节，只进日志


class ForcePolicy:
    """抓取力策略接口。"""

    def decide(self, label: str, mass_g: float, crop: dict, last_failed_n: float | None = None) -> ForceDecision:
        """last_failed_n: 本次采摘中刚刚失败（没夹牢）的那次用的力；首次尝试为 None。"""
        raise NotImplementedError

    def feedback(self, label: str, mass_g: float, decision: ForceDecision, slipped: bool) -> None:
        """抓取验证后的反馈。有学习能力的策略在这里更新自己。"""


class ReferencePolicy(ForcePolicy):
    """参考实现：库仑摩擦模型 F = m·g / μ，没夹牢就加一档。

    它不区分作物形状，也不积累经验——每个果实都从头试。仅用于让框架开箱即用和做对照。
    """

    def __init__(self, mu: float = 1.0, step_n: float = 0.25, f_min_n: float = 0.25, f_max_n: float = 7.0):
        if not (mu > 0 and step_n > 0 and 0 < f_min_n < f_max_n):
            raise ValueError('ReferencePolicy 参数非法')
        self.mu, self.step, self.f_min, self.f_max = mu, step_n, f_min_n, f_max_n

    @classmethod
    def from_config(cls, cfg, ctx):
        p = cfg['force_policy']
        return cls(p['friction_mu'], p['step_n'], p['f_min_n'], p['f_max_n'])

    def quantize(self, f_n: float) -> float:
        return min(self.f_max, max(self.f_min, math.ceil(f_n / self.step - 1e-9) * self.step))

    def decide(self, label, mass_g, crop, last_failed_n=None):
        if not (mass_g > 0 and math.isfinite(mass_g)):
            raise ValueError(f'质量必须为正的有限数，收到 {mass_g!r}')
        force = self.quantize(mass_g / 1000.0 * G / self.mu)
        note = '摩擦模型'
        if last_failed_n is not None:
            force, note = max(force, self.quantize(last_failed_n + self.step)), '上次偏轻，加一档'
        limit = crop.get('max_force_n')
        if limit is not None and force > limit:        # 防压伤优先
            force = max(self.f_min, math.floor(limit / self.step + 1e-9) * self.step)
            note += '（已到防压伤上限）'
        return ForceDecision(force, False, note)
