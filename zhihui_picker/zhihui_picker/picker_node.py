"""ROS2 节点：把采摘流程接入 ROS（数字孪生显示、相机画面、状态话题、启动/急停服务）。

话题   /joint_states            sensor_msgs/JointState   机械臂 + 夹爪实时关节角
       /picker/status           std_msgs/String          每个流程事件的 JSON
       /picker/markers          visualization_msgs/MarkerArray   果实、果篮、相机视野
       /picker/camera/annotated sensor_msgs/Image        相机画面 + 识别框 + 决策面板
服务   /picker/pick_once        std_srvs/Trigger         执行一次完整采摘
       /picker/estop            std_srvs/Trigger         急停
参数   config (string)          配置文件路径（逗号分隔可叠加）
       mode (string)            覆盖配置里的 mode
       auto_cycle (bool)        启动后自动连续采摘
"""
from __future__ import annotations

import json
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .config import load_config
from .kinematics import forward
from .pipeline import build_pipeline
from .vision import SimCamera, rpy_matrix

JOINTS = ['joint1_yaw', 'joint2_shoulder', 'joint3_elbow', 'joint4_wrist',
          'gripper_left_joint', 'gripper_right_joint']
WIRE_Z = 0.60          # 仿真场景里吊蔓铁丝的高度
BOX_COLOR = (60, 230, 120)
CJK_FONTS = ['/System/Library/Fonts/Hiragino Sans GB.ttc', '/System/Library/Fonts/STHeiti Medium.ttc',
             '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
             '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc']


def _load_font(size):
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for path in CJK_FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return None


class PickerNode(Node):
    def __init__(self):
        super().__init__('zhihui_picker')
        self.declare_parameter('config', '')
        self.declare_parameter('auto_cycle', False)
        self.declare_parameter('mode', '')
        mode = self.get_parameter('mode').value
        self.cfg = load_config(self.get_parameter('config').value or None,
                               {'mode': mode} if mode else None)
        self.sim = self.cfg['mode'] == 'sim'
        self.get_logger().info(f"运行模式: {self.cfg['mode']}")
        self.pipe = build_pipeline(self.cfg, self._on_event)
        # 仿真时用独立的相机实例给画面窗口供图，不与流程线程争用
        self.view_cam = (SimCamera(self.pipe.world, self.pipe.camera.hand_eye, self.cfg['camera']['width'],
                                   self.cfg['camera']['height'], seed=1) if self.sim else None)

        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.status_pub = self.create_publisher(String, '/picker/status', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/picker/markers', 10)
        self.image_pub = self.create_publisher(Image, '/picker/camera/annotated', 2)
        cb = ReentrantCallbackGroup()
        self.create_service(Trigger, '/picker/pick_once', self._srv_pick, callback_group=cb)
        self.create_service(Trigger, '/picker/estop', self._srv_estop, callback_group=cb)
        self.create_timer(1 / 30, self._publish_state, callback_group=cb)
        self.create_timer(1 / 4, self._publish_image, callback_group=cb)

        self._busy = threading.Lock()
        self._panel = {'crop': '', 'mass': '', 'force': '', 'result': '待机', 'tries': []}
        self._font = _load_font(24)
        self._target = None
        self._round = 1
        self._stop = threading.Event()
        self._worker = None
        if self.get_parameter('auto_cycle').value:
            self._worker = threading.Thread(target=self._auto_loop, daemon=True)
            self._worker.start()

    # ---------------------------------------------------------- 流程
    def _on_event(self, stage, data):
        self.status_pub.publish(String(data=json.dumps({'stage': stage, **data}, ensure_ascii=False,
                                                       default=str)))
        # 面板只显示对外可见的结果：作物、质量、最终力、成败
        p = self._panel
        if stage == 'detected':
            self._target = data
            p.update(crop=f"{data['name_zh']}  果径 {data['width_cm']} cm", mass='', force='', result='', tries=[])
        elif stage == 'mass':
            p['mass'] = f"约 {data['mass_g']:.0f} g"
        elif stage == 'force':
            basis = '物理模型 + 抓取经验' if data['experience_used'] else '物理模型'
            p.update(force=f"{data['force_n']:.2f} N（{basis}）", result='接近并闭合夹爪…')
        elif stage == 'slip':
            p['tries'].append(f"{data['force_n']:.2f} 偏轻")
            p['result'] = '力偏轻：果实仍在枝上、未受损，小步加力'
        elif stage == 'lifted':
            p['tries'].append(f"{data['force_n']:.2f} 抓稳")
            p['result'] = '抓稳，送往果篮'
        elif stage == 'placed':
            p['result'] = '已放入果篮'
        elif stage == 'done':
            self._target = None
            if data['status'] not in ('success', 'no_target'):
                p['result'] = f"{data['status']} {data['message']}"

    def _run_once(self):
        if not self._busy.acquire(blocking=False):
            return None
        try:
            return self.pipe.run_once()
        finally:
            self._busy.release()

    def _auto_loop(self):
        while not self._stop.is_set():
            r = self._run_once()
            if r is None or r.status in ('fault', 'aborted'):
                self.get_logger().error('自动循环停止：需人工检查后重启节点')
                return
            if r.status == 'no_target':
                self._panel.update(crop='', mass='', force='', tries=[], result='视野内的果实已全部采完')
                self._stop.wait(3.0)
                if self.sim and not self._stop.is_set():
                    # 仿真演示：重新挂果再采一遍（策略积累的经验保留）
                    with self.pipe.world.lock:
                        for o in self.pipe.world.objects:
                            o.state = 'hanging'
                    self._round += 1

    def _srv_pick(self, req, resp):
        r = self._run_once()
        if r is None:
            resp.success, resp.message = False, '正在执行上一次采摘'
        else:
            resp.success = r.status == 'success'
            resp.message = f'{r.status} {r.label} {r.mass_g:.0f}g {r.final_force_n:.2f}N {r.message}'
        return resp

    def _srv_estop(self, req, resp):
        self._stop.set()
        self.pipe.abort.set()
        if not self._busy.locked():
            self.pipe.arm.estop()
        self._panel['result'] = '急停'
        resp.success, resp.message = True, '已急停'
        self.get_logger().warn('收到急停')
        return resp

    # ---------------------------------------------------------- 可视化
    def _publish_state(self):
        try:
            q = self.pipe.arm.read()
        except Exception as e:
            self.get_logger().warn(f'读取关节失败: {e}', throttle_duration_sec=2.0)
            return
        half = min(0.052, max(0.012, self.pipe.gripper.finger_gap() / 2))
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name, js.position = JOINTS, [*q, half, half]
        self.js_pub.publish(js)

        arr = MarkerArray()
        ee = forward(self.pipe.geo, q)
        b = self.cfg['arm']['basket_xyz']
        arr.markers.append(self._marker(0, Marker.CUBE, (b[0], b[1], 0.06), (0.26, 0.22, 0.12),
                                        (0.55, 0.38, 0.2, 0.75)))
        arr.markers.append(self._lines(2, self._frustum(), 0.002, (0.55, 0.75, 0.95, 0.30)))
        if self.pipe.world:                      # 仿真：按真值显示果实、果柄与吊蔓铁丝
            stems, n_in = [(0.50, -0.32, WIRE_Z), (0.50, 0.32, WIRE_Z)], 0
            for i, o in enumerate(self.pipe.world.objects):
                if o.state == 'held':
                    pos = ee
                elif o.state == 'in_basket':
                    pos = (b[0] - 0.07 + 0.07 * n_in, b[1], 0.03 + o.width_m / 2)
                    n_in += 1
                else:
                    pos = o.position
                    stems += [(pos[0], pos[1], pos[2] + o.height_m / 2), (pos[0], pos[1], WIRE_Z)]
                lying = o.state == 'in_basket'
                m = self._marker(10 + i, Marker.SPHERE, pos,
                                 (o.height_m, o.width_m, o.width_m) if lying else (o.width_m, o.width_m, o.height_m),
                                 (*[c / 255 for c in o.color], 1.0))
                arr.markers.append(m)
            arr.markers.append(self._lines(4, stems, 0.006, (0.25, 0.55, 0.25, 1.0)))
        elif self._target:                       # 实机：显示当前识别到的目标
            arr.markers.append(self._marker(10, Marker.SPHERE, self._target['position'],
                                            (0.05, 0.05, 0.05), (0.2, 0.9, 0.3, 0.9)))
        self.marker_pub.publish(arr)

    def _frustum(self, depth=0.45):
        """相机视野的四条棱线 + 远端矩形（base_link）。"""
        c = self.cfg['camera']
        he = self.cfg['hand_eye']
        R, t = rpy_matrix(*he['rpy']), np.asarray(he['xyz'], float)
        hx, hy = (c['width'] / 2) / 600.0, (c['height'] / 2) / 600.0
        corners = [R @ np.array([depth, sx * hx * depth, sy * hy * depth]) + t
                   for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        pts = []
        for i, p in enumerate(corners):
            pts += [t, p, p, corners[(i + 1) % 4]]
        return [tuple(float(v) for v in p) for p in pts]

    def _publish_image(self):
        if self.view_cam is not None:
            frame = self.view_cam.read()
            dets = self.pipe.detector.detect(frame)
        elif self.pipe.last_view is not None:
            frame, dets = self.pipe.last_view
        else:
            return
        rgb = frame.rgb.copy()
        for d in dets:
            x0, y0, x1, y1 = d.bbox
            x0, y0, x1, y1 = max(0, x0 - 4), max(0, y0 - 4), min(rgb.shape[1], x1 + 4), min(rgb.shape[0], y1 + 4)
            rgb[y0:y0 + 3, x0:x1] = BOX_COLOR
            rgb[y1 - 3:y1, x0:x1] = BOX_COLOR
            rgb[y0:y1, x0:x0 + 3] = BOX_COLOR
            rgb[y0:y1, x1 - 3:x1] = BOX_COLOR
        rgb = self._with_panel(rgb, dets)
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'
        msg.height, msg.width, msg.encoding = rgb.shape[0], rgb.shape[1], 'rgb8'
        msg.step, msg.data = rgb.shape[1] * 3, rgb.tobytes()
        self.image_pub.publish(msg)

    def _with_panel(self, rgb, dets):
        """在相机画面下方拼一块决策面板（RViz 的三维文字不支持中文，所以把文字画进图像里）。
        没装 Pillow 或找不到中文字体时原样返回相机画面。"""
        if self._font is None:
            return rgb
        from PIL import Image as PILImage, ImageDraw
        cam = PILImage.fromarray(rgb).resize((480, 360))
        draw = ImageDraw.Draw(cam)
        small = self._font.font_variant(size=18)
        for d in dets:
            name = self.cfg['crops'][d.label].get('name_zh', d.label)
            draw.text((d.bbox[0] * 0.75, max(0, d.bbox[1] * 0.75 - 26)), name, font=small, fill=BOX_COLOR)
        canvas = PILImage.new('RGB', (480, 360 + 300), (24, 30, 26))
        canvas.paste(cam, (0, 0))
        draw = ImageDraw.Draw(canvas)
        p, y = self._panel, 372
        title = f'仿真演示 · 第 {self._round} 遍' if self.sim else '智汇蔬香 · 实机'
        draw.text((14, y), title, font=small, fill=(240, 180, 90))
        rows = [('① 识别定位', p['crop']), ('② 估计质量', p['mass']), ('③ 夹持力', p['force']),
                ('④ 抓取验证', p['result']), ('本果尝试', '  →  '.join(p['tries']))]
        for i, (k, v) in enumerate(rows):
            yy = y + 34 + i * 50
            draw.text((14, yy), k, font=small, fill=(150, 165, 155))
            draw.text((14, yy + 20), v, font=self._font.font_variant(size=21), fill=(236, 240, 236))
        return np.asarray(canvas)

    def _marker(self, mid, mtype, pos, scale, rgba):
        m = Marker()
        m.header.frame_id = 'base_link'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = 'picker', mid, mtype, Marker.ADD
        m.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = (float(s) for s in scale)
        m.color.r, m.color.g, m.color.b, m.color.a = (float(c) for c in rgba)
        return m

    def _lines(self, mid, pts, width, rgba):
        m = self._marker(mid, Marker.LINE_LIST, (0, 0, 0), (width, 0.0, 0.0), rgba)
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts]
        return m

    def close(self):
        self._stop.set()
        self.pipe.abort.set()
        if self._worker:
            self._worker.join(timeout=3.0)
        self.pipe.shutdown()


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
    rclpy.init()
    node = PickerNode()
    exe = MultiThreadedExecutor(num_threads=4)
    exe.add_node(node)
    try:
        exe.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
