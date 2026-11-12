import math

import numpy as np
import pytest

from zhihui_picker.config import load_config
from zhihui_picker.force_policy import ForceDecision, ForcePolicy, ReferencePolicy
from zhihui_picker.kinematics import ArmGeometry, UnreachableError, forward, gravity_torque, inverse
from zhihui_picker.mass import MassEstimate, geometric_estimate
from zhihui_picker.pipeline import build_pipeline
from zhihui_picker.sim import HardwareFault
from zhihui_picker.vision import Detection, HandEye

CROP = {'mass_range_g': [80, 600], 'density_kg_m3': 700, 'max_force_n': 5.0}


# ---------------------------------------------------------------- 参考策略
def test_reference_policy_scales_with_mass_and_quantizes():
    p = ReferencePolicy()
    assert p.decide('x', 150, CROP).force_n == 1.5            # 0.15 kg × 9.81 = 1.47 → 1.5
    assert p.decide('x', 300, CROP).force_n > p.decide('x', 150, CROP).force_n
    assert p.quantize(0.01) == 0.25 and p.quantize(99) == 7.0


def test_reference_policy_steps_up_after_failure_and_respects_damage_limit():
    p = ReferencePolicy()
    assert p.decide('x', 150, CROP, last_failed_n=1.5).force_n == 1.75
    assert p.decide('x', 590, {**CROP, 'max_force_n': 3.1}).force_n == 3.0


@pytest.mark.parametrize('bad', [0, -5, float('nan'), float('inf')])
def test_bad_mass_rejected(bad):
    with pytest.raises(ValueError):
        ReferencePolicy().decide('x', bad, CROP)


# ---------------------------------------------------------------- 运动学
def test_ik_fk_roundtrip():
    geo = ArmGeometry()
    for p in [(0.50, 0.08, 0.27), (0.34, 0.0, 0.42), (0.30, -0.30, 0.34), (0.45, 0.2, 0.15)]:
        q = inverse(geo, *p)
        assert forward(geo, q) == pytest.approx(p, abs=1e-9)
        assert q[1] + q[2] + q[3] == pytest.approx(0)             # 夹爪保持水平


@pytest.mark.parametrize('p', [(0.9, 0, 0.3), (0.05, 0, 0.3), (0.3, 0, -0.4)])
def test_ik_rejects_unreachable(p):
    with pytest.raises(UnreachableError):
        inverse(ArmGeometry(), *p)


def test_gravity_torque_opposes_forward_lean():
    geo = ArmGeometry()
    assert gravity_torque(geo, [0, 0, 0, 0]) == pytest.approx([0, 0, 0, 0], abs=1e-12)
    t = gravity_torque(geo, [0, 0.8, 0.6, -1.4])
    assert t[1] < t[2] < 0
    assert gravity_torque(geo, [0, 0.8, 0.6, -1.4], 0.3)[1] < t[1]


# ---------------------------------------------------------------- 视觉定位
def _sim(over=None):
    cfg = load_config(overrides={'mode': 'sim', **(over or {})})
    events = []
    pipe = build_pipeline(cfg, lambda s, d: events.append((s, d)), realtime=False)
    return pipe, events


def test_sim_camera_detection_localizes_within_15mm():
    pipe, _ = _sim()
    dets = pipe.detector.detect(pipe.camera.read())
    assert len(dets) == len(pipe.world.objects)
    for o in pipe.world.objects:
        best = min(dets, key=lambda d: math.dist(d.position_base, o.position))
        assert best.label == o.label
        assert math.dist(best.position_base, o.position) < 0.015
        assert best.width_m == pytest.approx(o.width_m, abs=0.008)


def test_hand_eye_roundtrip():
    he = HandEye([0.15, 0.0, 0.55], [0.0, 0.5236, 0.0])
    p = np.array([0.5, 0.08, 0.27])
    assert he.optical_to_base(he.base_to_optical(p)) == pytest.approx(p)
    assert he.base_to_optical(p)[2] > 0


def test_geometric_mass_fallback_stays_in_range():
    det = Detection('eggplant', 0.9, (0, 0, 10, 10), (0.5, 0, 0.3), 0.05, 0.17, 0.5)
    assert 80 <= geometric_estimate(det, CROP).mass_g <= 600


# ---------------------------------------------------------------- 全流程
def test_full_pipeline_picks_everything():
    pipe, events = _sim()
    results = [pipe.run_once() for _ in range(3)]
    assert [r.status for r in results] == ['success'] * 3
    assert results[0].label == 'tomato'                       # 离机械臂最近的先采
    assert all(o.state == 'in_basket' for o in pipe.world.objects)
    for r in results:
        assert r.final_force_n <= pipe.cfg['crops'][r.label]['max_force_n']
    assert pipe.run_once().status == 'no_target'


def test_retry_then_give_up_at_damage_limit():
    pipe, _ = _sim({'force_policy': {'friction_mu': 1.5}, 'crops': {'eggplant': {'max_force_n': 1.5}}})
    rs = [pipe.run_once() for _ in range(3)]
    egg = [r for r in rs if r.label == 'eggplant']
    assert egg and all(r.status in ('force_limited', 'slipped') for r in egg)
    assert all(a['force_n'] <= 1.5 for r in egg for a in r.attempts)
    tom = next(r for r in rs if r.label == 'tomato')          # μ=1.5 时起步力偏轻：加档后抓稳
    assert tom.status == 'success' and len(tom.attempts) > 1
    assert [a['outcome'] for a in tom.attempts][:-1] == ['slip'] * (len(tom.attempts) - 1)


def test_custom_policy_plugin_receives_feedback():
    seen = []

    class Fixed(ForcePolicy):
        @classmethod
        def from_config(cls, cfg, ctx):
            return cls()

        def decide(self, label, mass_g, crop, last_failed_n=None):
            return ForceDecision(3.0, True, '固定力', {'secret': 1})

        def feedback(self, label, mass_g, decision, slipped):
            seen.append((label, decision.force_n, slipped))

    import zhihui_picker.force_policy as fp
    fp.FixedForTest = Fixed
    pipe, events = _sim({'force_policy': {'plugin': 'zhihui_picker.force_policy:FixedForTest'}})
    assert pipe.run_once().status == 'success'
    assert seen == [('tomato', 3.0, False)]
    assert any(s == 'force' and d['experience_used'] for s, d in events)


def test_unknown_plugin_rejected():
    with pytest.raises(ValueError, match='无法加载插件'):
        _sim({'force_policy': {'plugin': 'no_such_pkg.mod:Cls'}})
    with pytest.raises(ValueError, match='gripper.backend'):
        build_pipeline(load_config(overrides={'mode': 'real', 'camera': {'backend': 'sim'},
                                              'gripper': {'backend': 'magic'}}), realtime=False)


def test_out_of_workspace_target_ignored():
    pipe, _ = _sim({'arm': {'workspace': {'x': [0.25, 0.30]}}})
    assert pipe.run_once().status == 'no_target'
    assert pipe.arm.read() == pytest.approx(inverse(pipe.geo, *pipe.cfg['arm']['home_xyz']))


def test_abort_stops_motion():
    pipe, _ = _sim()
    orig = pipe.gripper.grasp
    pipe.gripper.grasp = lambda w, f: (orig(w, f), pipe.abort.set())
    r = pipe.run_once()
    assert r.status == 'aborted' and not pipe.arm.enabled


def test_hardware_fault_triggers_estop():
    pipe, _ = _sim()
    n = {'i': 0}
    orig = pipe.arm.command

    def flaky(q, payload_kg=0.0):
        n['i'] += 1
        if n['i'] > 80:
            raise HardwareFault('关节2 反馈超时')
        orig(q, payload_kg)
    pipe.arm.command = flaky
    r = pipe.run_once()
    assert r.status == 'fault' and '反馈超时' in r.message and not pipe.arm.enabled


def test_joint_speed_limit_respected():
    pipe, _ = _sim()
    qs = []
    orig = pipe.arm.command
    pipe.arm.command = lambda q, payload_kg=0.0: (qs.append(list(q)), orig(q, payload_kg))
    pipe.run_once()
    vmax = max(abs(a - b) for q0, q1 in zip(qs, qs[1:]) for a, b in zip(q0, q1)) / pipe.dt
    assert vmax <= pipe.cfg['arm']['max_joint_speed'] * 1.875 * 1.5 + 1e-6


def test_mass_estimator_failure_falls_back_to_geometry():
    pipe, events = _sim()

    class Broken:
        def estimate(self, frame, det, crop):
            raise RuntimeError('网络断开')
    pipe.estimator = Broken()
    assert pipe.run_once().mass_source == 'geometry'


def test_config_validation():
    with pytest.raises(ValueError):
        load_config(overrides={'mode': 'banana'})
    with pytest.raises(ValueError):
        load_config(overrides={'pick': {'lift_m': 0.03}})
