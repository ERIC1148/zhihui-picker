"""抓取流程编排：感知 → 估质量 → 定力 → 运动 → 抓取 → 验证 → 放置（没夹牢时重试，并把结果反馈给策略）。

本模块不依赖 ROS，可直接在命令行与单元测试里运行；ROS 节点只是它的薄封装。
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field

from .force_policy import ForcePolicy, ReferencePolicy
from .kinematics import ArmGeometry, UnreachableError, forward, inverse, min_jerk
from .mass import MockMassEstimator, geometric_estimate
from .plugins import is_plugin, load_plugin
from .sim import HardwareFault, SimArm, SimGripper, SimObject, SimWorld
from .vision import ColorDetector, HandEye, SimCamera

log = logging.getLogger('zhihui_picker.pipeline')


class Aborted(Exception):
    pass


@dataclass
class PickResult:
    status: str                 # success | no_target | slipped | force_limited | aborted | fault
    label: str = ''
    mass_g: float = 0.0
    mass_source: str = ''
    attempts: list = field(default_factory=list)   # 每次尝试: {force_n, experience_used, outcome}
    message: str = ''
    duration_s: float = 0.0

    @property
    def final_force_n(self) -> float:
        return self.attempts[-1]['force_n'] if self.attempts else 0.0


class PickPipeline:
    def __init__(self, cfg, camera, detector, estimator, arm, gripper, policy: ForcePolicy,
                 geo: ArmGeometry, on_event=None, realtime: bool = True):
        self.cfg, self.camera, self.detector, self.estimator = cfg, camera, detector, estimator
        self.arm, self.gripper, self.policy, self.geo = arm, gripper, policy, geo
        self.last_view = None       # (frame, detections)：最近一次用于决策的图像，供可视化
        self.on_event = on_event or (lambda stage, data: None)
        self.realtime = realtime
        self.abort = threading.Event()
        self.dt = 1.0 / cfg['arm']['control_hz']
        self.payload_kg = 0.0
        self.world = None           # 仅仿真
        log_dir = cfg['pick'].get('log_dir')
        self._log_path = None
        if log_dir and realtime:
            os.makedirs(os.path.expanduser(log_dir), exist_ok=True)
            self._log_path = os.path.join(os.path.expanduser(log_dir), time.strftime('picks_%Y%m%d.jsonl'))

    # ------------------------------------------------------------ 工具
    def _emit(self, stage: str, **data):
        log.info('[%s] %s', stage, json.dumps(data, ensure_ascii=False, default=str))
        self.on_event(stage, data)

    def _sleep(self, s: float):
        if self.abort.is_set():
            raise Aborted()
        if self.realtime:
            time.sleep(s)

    def _steps(self, duration: float) -> int:
        return max(2, int(math.ceil(duration / self.dt)))

    def _move_joints(self, q_goal: list[float]):
        """关节空间最小加加速度插值（回待机位、去果篮等大范围运动）。"""
        q0 = self.arm.read()
        span = max(abs(a - b) for a, b in zip(q_goal, q0))
        if span < 1e-4:
            return
        # 五次多项式峰值速度是平均速度的 1.875 倍
        n = self._steps(max(0.5, 1.875 * span / self.cfg['arm']['max_joint_speed']))
        for i in range(1, n + 1):
            s = min_jerk(i / n)
            self.arm.command([a + s * (b - a) for a, b in zip(q0, q_goal)], self.payload_kg)
            self._sleep(self.dt)

    def _move_line(self, p_goal):
        """笛卡尔直线运动（接近、上提、退出），每步解 IK 并限制关节步长。"""
        q_prev = self.arm.read()
        p0 = forward(self.geo, q_prev)
        dist = math.dist(p0, p_goal)
        if dist < 1e-4:
            return
        n = self._steps(max(0.5, 1.875 * dist / self.cfg['arm']['max_cart_speed']))
        max_step = self.cfg['arm']['max_joint_speed'] * self.dt * 1.875 * 1.5
        for i in range(1, n + 1):
            s = min_jerk(i / n)
            q = inverse(self.geo, *[a + s * (b - a) for a, b in zip(p0, p_goal)])
            if max(abs(a - b) for a, b in zip(q, q_prev)) > max_step:
                # 直线上出现关节速度超限（靠近奇异位形）：改用限速的关节插值走完剩余路程
                self._move_joints(inverse(self.geo, *p_goal))
                return
            self.arm.command(q, self.payload_kg)
            q_prev = q
            self._sleep(self.dt)

    def _waypoints(self, target):
        """由果实中心算预抓取点与上提点，并确认全部可达。"""
        x, y, z = target
        yaw = math.atan2(y, x)
        a = self.cfg['pick']['approach_m']
        pre = (x - a * math.cos(yaw), y - a * math.sin(yaw), z)
        lift = (x, y, z + self.cfg['pick']['lift_m'])
        pre_lift = (pre[0], pre[1], lift[2])
        for p in (pre, target, lift, pre_lift):
            inverse(self.geo, *p)          # 不可达会抛 UnreachableError
        return pre, lift, pre_lift

    def _select_target(self, frame):
        ws = self.cfg['arm']['workspace']
        open_gap = self.cfg['gripper']['open_gap_m']
        skipped = []
        dets = self.detector.detect(frame)
        self.last_view = (frame, dets)
        # 先摘离机械臂最近的：伸展最短、最稳，也不会被前面的果实挡住
        for d in sorted(dets, key=lambda d: math.hypot(d.position_base[0], d.position_base[1])):
            x, y, z = d.position_base
            if d.confidence < self.cfg['detector']['min_confidence']:
                skipped.append((d.label, '置信度不足'))
            elif not (ws['x'][0] <= x <= ws['x'][1] and ws['y'][0] <= y <= ws['y'][1]
                      and ws['z'][0] <= z <= ws['z'][1]):
                skipped.append((d.label, f'在安全工作区外 ({x:.2f},{y:.2f},{z:.2f})'))
            elif d.width_m > open_gap - 0.01:
                skipped.append((d.label, f'果径 {d.width_m * 100:.1f} cm 超过夹爪开口'))
            else:
                try:
                    return d, self._waypoints(d.position_base), skipped
                except UnreachableError as e:
                    skipped.append((d.label, f'不可达: {e}'))
        return None, None, skipped

    def _slipped(self, label: str, position) -> bool:
        """上提后复拍：原位置附近仍有同类果实 → 没抓起来。"""
        frame = self.camera.read()
        r = self.cfg['pick']['verify_radius_m']
        return any(d.label == label and math.dist(d.position_base, position) < r
                   for d in self.detector.detect(frame))

    # ------------------------------------------------------------ 主流程
    def go_home(self):
        if not getattr(self.arm, 'enabled', False):
            self.arm.enable()
        self.gripper.open()
        self.payload_kg = 0.0
        self._move_joints(inverse(self.geo, *self.cfg['arm']['home_xyz']))

    def run_once(self) -> PickResult:
        t0 = time.time()
        self.abort.clear()
        res = PickResult('fault')
        try:
            res = self._run()
        except Aborted:
            self.arm.estop()
            res = PickResult('aborted', message='收到急停/中止指令')
        except (HardwareFault, UnreachableError) as e:
            log.error('流程因故障中止: %s', e)
            try:
                self.arm.estop()
            except Exception as e2:
                log.error('急停时出错: %s', e2)
            res = PickResult('fault', message=str(e))
        res.duration_s = time.time() - t0
        self._emit('done', status=res.status, label=res.label, mass_g=round(res.mass_g, 1),
                   force_n=res.final_force_n, attempts=len(res.attempts), message=res.message,
                   duration_s=round(res.duration_s, 2))
        if self._log_path:
            with open(self._log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'ts': t0, **res.__dict__}, ensure_ascii=False) + '\n')
        return res

    def _run(self) -> PickResult:
        pk = self.cfg['pick']
        self._emit('home')
        self.go_home()
        self._sleep(pk['settle_s'])

        # 1) 视觉识别 + 三维定位
        frame = self.camera.read()
        det, wps, skipped = self._select_target(frame)
        if det is None:
            self._emit('no_target', skipped=skipped)
            return PickResult('no_target', message='视野内没有可抓取的作物')
        pre, lift, pre_lift = wps
        crop = self.cfg['crops'][det.label]
        self._emit('detected', label=det.label, name_zh=crop.get('name_zh', det.label),
                   position=[round(v, 3) for v in det.position_base],
                   width_cm=round(det.width_m * 100, 1), confidence=round(det.confidence, 2))

        # 2) 估计质量
        try:
            me = self.estimator.estimate(frame, det, crop)
        except Exception as e:      # 质量估计失败不应让整机停摆
            log.warning('质量估计器异常: %s', e)
            me = None
        if me is None:
            me = geometric_estimate(det, crop, '估计器不可用，按果径几何估计')
        self._emit('mass', mass_g=round(me.mass_g, 1), source=me.source, detail=me.detail)
        result = PickResult('slipped', det.label, me.mass_g, me.source)

        last_failed = None
        for attempt in range(1, pk['max_attempts'] + 1):
            # 3) 抓取力策略决定夹持力
            dec = self.policy.decide(det.label, me.mass_g, crop, last_failed)
            if last_failed is not None and dec.force_n <= last_failed:
                result.status = 'force_limited'
                result.message = (f'{last_failed:.2f} N 没夹牢，但已到该作物的防压伤上限 '
                                  f"{crop['max_force_n']:.2f} N，放弃以免夹伤")
                break
            self._emit('force', attempt=attempt, force_n=dec.force_n, experience_used=dec.experience_used,
                       note=dec.note, **{k: v for k, v in dec.details.items() if not k.startswith('_')})

            # 4) 接近 → 抓取 → 上提
            self._move_joints(inverse(self.geo, *pre))
            self._move_line(det.position_base)
            self._emit('grasp', force_n=dec.force_n, width_cm=round(det.width_m * 100, 1))
            self.gripper.grasp(det.width_m, dec.force_n)
            self._sleep(pk['settle_s'])
            self.payload_kg = me.mass_g / 1000.0
            self._move_line(lift)
            self._sleep(pk['settle_s'])

            # 5) 验证。两路证据任一成立即判没夹牢：夹爪指间已空，或复拍在原位置仍看到果实
            slipped = self.gripper.object_lost() or self._slipped(det.label, det.position_base)
            self.policy.feedback(det.label, me.mass_g, dec, slipped)
            result.attempts.append({'force_n': dec.force_n, 'experience_used': dec.experience_used,
                                    'outcome': 'slip' if slipped else 'success'})
            if slipped:
                self._emit('slip', attempt=attempt, force_n=dec.force_n)
                last_failed = dec.force_n
                self.payload_kg = 0.0
                self.gripper.open()
                self._move_line(pre_lift)
                continue

            # 6) 放入果篮
            self._emit('lifted', attempt=attempt, force_n=dec.force_n)
            self._move_line(pre_lift)
            self._move_joints(inverse(self.geo, *self.cfg['arm']['basket_xyz']))
            self.gripper.open()
            self.payload_kg = 0.0
            self._sleep(pk['settle_s'])
            self._emit('placed')
            result.status = 'success'
            break
        else:
            result.message = f"连续 {pk['max_attempts']} 次没夹牢"
        self._move_joints(inverse(self.geo, *self.cfg['arm']['home_xyz']))
        return result

    def shutdown(self):
        for dev in (self.gripper, self.arm, self.camera):
            try:
                dev.close()
            except Exception as e:
                log.warning('关闭设备时出错: %s', e)


# ---------------------------------------------------------------- 装配
def build_pipeline(cfg: dict, on_event=None, realtime: bool = True) -> PickPipeline:
    """按配置装配整套系统。

    mode=sim 时所有设备用仿真；mode=real 时每个设备按各自的 backend 选择，
    任何一个都可以单独设为 sim，便于分步联调（例如真相机 + 仿真机械臂）。
    抓取力策略、质量估计、机械臂、夹爪都可以用 "包名.模块:类名" 接入自己的实现。
    """
    a, c, g, m, dc = cfg['arm'], cfg['camera'], cfg['gripper'], cfg['mass_estimator'], cfg['detector']
    all_sim = cfg['mode'] == 'sim'
    backend = lambda section: 'sim' if all_sim else section['backend']
    geo = ArmGeometry(lower=list(a['joint_lower']), upper=list(a['joint_upper']), **a['geometry'])
    hand_eye = HandEye(cfg['hand_eye']['xyz'], cfg['hand_eye']['rpy'])

    # 仿真世界：只有仿真相机才往里放果实（真相机 + 仿真执行器时世界是空的）
    world = None
    if 'sim' in (backend(c), backend(a), backend(g)):
        objs = cfg['sim']['objects'] if backend(c) == 'sim' else []
        world = SimWorld([SimObject(o['label'], tuple(o['position']), o['width_m'], o['height_m'],
                                    o['mass_g'], o['true_min_force_n'], tuple(o['color'])) for o in objs])
    ctx = {'geo': geo, 'hand_eye': hand_eye, 'world': world, 'arm': None}

    if backend(c) == 'sim':
        camera = SimCamera(world, hand_eye, c['width'], c['height'], seed=cfg['sim']['seed'])
    elif c['backend'] == 'realsense':
        from .vision import RealSenseCamera
        camera = RealSenseCamera(c['width'], c['height'], c['fps'], c.get('serial') or None)
    else:
        raise ValueError(f"未知的 camera.backend: {c['backend']}")

    if backend(a) == 'sim':
        arm = SimArm(geo, inverse(geo, *a['home_xyz']))
    elif is_plugin(a['backend']):
        arm = load_plugin(a['backend'], cfg, ctx)
    else:
        raise ValueError(f"未知的 arm.backend: {a['backend']}")
    ctx['arm'] = arm

    if backend(g) == 'sim':
        if not isinstance(arm, SimArm):
            raise ValueError('仿真夹爪只能配仿真机械臂（gripper.backend=sim 需要 arm.backend=sim）')
        gripper = SimGripper(world, arm, g['open_gap_m'])
    elif is_plugin(g['backend']):
        gripper = load_plugin(g['backend'], cfg, ctx)
    else:
        raise ValueError(f"未知的 gripper.backend: {g['backend']}")

    if all_sim or m['backend'] == 'mock':
        estimator = MockMassEstimator(world, m.get('mock_noise_sigma', 0.12), cfg['sim']['seed'])
    elif is_plugin(m['backend']):
        estimator = load_plugin(m['backend'], cfg, ctx)
    else:
        raise ValueError(f"未知的 mass_estimator.backend: {m['backend']}")

    spec = cfg['force_policy']['plugin']
    if spec == 'reference':
        policy = ReferencePolicy.from_config(cfg, ctx)
    elif is_plugin(spec):
        policy = load_plugin(spec, cfg, ctx)
    else:
        raise ValueError(f'未知的 force_policy.plugin: {spec}')

    if dc['backend'] == 'yolo' and backend(c) != 'sim':
        from .vision import YoloDetector
        detector = YoloDetector(dc['yolo_weights'], cfg['crops'], hand_eye, dc['min_confidence'],
                                dc.get('yolo_class_map'), tuple(dc['depth_range_m']))
    else:
        detector = ColorDetector(cfg['crops'], hand_eye, dc['min_area_px'], tuple(dc['depth_range_m']))
    pipe = PickPipeline(cfg, camera, detector, estimator, arm, gripper, policy, geo, on_event, realtime)
    pipe.world = world
    return pipe
