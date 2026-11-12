"""命令行演示（不依赖 ROS）:  zhihui-picker demo [--config a.yaml,b.yaml] [--cycles N] [--fast]"""
from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config

STAGE_ZH = {'home': '回待机位', 'detected': '视觉识别定位', 'mass': '估计质量', 'force': '决定夹持力',
            'grasp': '夹爪闭合', 'slip': '力偏轻没夹牢', 'lifted': '稳定提起', 'placed': '放入果篮',
            'no_target': '未发现目标', 'done': '本轮结束'}


def _print_event(stage, data):
    print(f"  ▸ {STAGE_ZH.get(stage, stage):<8} " + '  '.join(f'{k}={v}' for k, v in data.items()))


def cmd_demo(args):
    from .pipeline import build_pipeline
    pipe = build_pipeline(load_config(args.config, {'mode': 'sim'}), _print_event, realtime=not args.fast)
    print('=== 智汇蔬香采摘流程 · 全仿真演示（无硬件）===')
    ok = 0
    for i in range(1, args.cycles + 1):
        print(f'\n--- 第 {i} 轮 ---')
        r = pipe.run_once()
        if r.status == 'no_target':
            break
        ok += r.status == 'success'
        forces = ' → '.join(f"{a['force_n']:.2f}N({'偏轻' if a['outcome'] == 'slip' else '抓稳'})" for a in r.attempts)
        print(f'  结果: {r.status}  {r.label}  估质量 {r.mass_g:.0f} g  施力 {forces}')
    print(f'\n成功采摘 {ok} 个。')
    pipe.shutdown()


def main(argv=None):
    ap = argparse.ArgumentParser(prog='zhihui-picker', description=__doc__)
    ap.add_argument('--config', help='配置文件路径，多个用逗号分隔、后者覆盖前者')
    ap.add_argument('-v', '--verbose', action='store_true')
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('demo')
    d.add_argument('--cycles', type=int, default=4)
    d.add_argument('--fast', action='store_true', help='不按真实时间等待')
    d.set_defaults(fn=cmd_demo)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format='%(levelname)s %(name)s: %(message)s')
    args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
