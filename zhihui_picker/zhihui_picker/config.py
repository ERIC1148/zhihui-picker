"""配置加载与校验：尽早失败，错误信息指出是哪个字段。"""
from __future__ import annotations

import copy
import os

import yaml


def default_config_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, '..', 'config', 'picker.yaml')       # 直接从源码树运行时
    if os.path.exists(src):
        return os.path.normpath(src)
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('zhihui_picker'), 'config', 'picker.yaml')


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=None, overrides: dict | None = None) -> dict:
    """读取包内默认配置，再按顺序叠加用户配置（每份只写需要改的字段），最后叠加代码内覆盖。

    path 可以是单个路径、列表，或用逗号分隔的多个路径，后面的覆盖前面的。
    """
    with open(default_config_path(), encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    paths = path.split(',') if isinstance(path, str) else list(path or [])
    for p in (os.path.expanduser(x.strip()) for x in paths if x and x.strip()):
        with open(p, encoding='utf-8') as f:
            _merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        _merge(cfg, copy.deepcopy(overrides))
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    def need(cond, msg):
        if not cond:
            raise ValueError(f'配置错误: {msg}')

    need(cfg.get('mode') in ('sim', 'real'), "mode 必须是 'sim' 或 'real'")
    need(cfg['crops'], 'crops 不能为空')
    for name, c in cfg['crops'].items():
        lo, hi = c['mass_range_g']
        need(0 < lo < hi, f'crops.{name}.mass_range_g 必须满足 0 < min < max')
        need(c['max_force_n'] > 0, f'crops.{name}.max_force_n 必须为正')
    a = cfg['arm']
    need(len(a['joint_lower']) == 4 and len(a['joint_upper']) == 4, 'arm.joint_lower/upper 必须各 4 个')
    need(0 < a['max_joint_speed'] <= 3.0, 'arm.max_joint_speed 必须在 (0, 3] rad/s')
    need(0 < a['max_cart_speed'] <= 0.5, 'arm.max_cart_speed 必须在 (0, 0.5] m/s')
    need(10 <= a['control_hz'] <= 500, 'arm.control_hz 必须在 [10, 500]')
    for ax in 'xyz':
        need(a['workspace'][ax][0] < a['workspace'][ax][1], f'arm.workspace.{ax} 下限必须小于上限')
    need(cfg['gripper']['open_gap_m'] > 0, 'gripper.open_gap_m 必须为正')
    p = cfg['pick']
    need(1 <= p['max_attempts'] <= 5, 'pick.max_attempts 必须在 [1, 5]')
    need(p['lift_m'] > p['verify_radius_m'] + 0.02, 'pick.lift_m 必须明显大于 verify_radius_m，否则无法区分抓起与没夹牢')
