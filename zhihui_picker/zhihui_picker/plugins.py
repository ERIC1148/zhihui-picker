"""插件加载：配置里写 "包名.模块:类名" 即可接入自己的实现（抓取力策略、质量估计、机械臂、夹爪）。

插件类需要提供类方法 from_config(cfg, ctx)，其中
  cfg  完整配置字典
  ctx  运行时上下文：{'geo': 机械臂几何, 'hand_eye': 手眼标定, 'world': 仿真世界或 None, 'arm': 已创建的机械臂}
"""
from __future__ import annotations

from importlib import import_module


def is_plugin(spec) -> bool:
    return isinstance(spec, str) and ':' in spec


def load_plugin(spec: str, cfg: dict, ctx: dict):
    module, _, name = spec.partition(':')
    try:
        cls = getattr(import_module(module), name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f'无法加载插件 {spec!r}: {e}') from e
    return cls.from_config(cfg, ctx)
