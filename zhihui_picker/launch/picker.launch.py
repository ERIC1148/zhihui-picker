"""一键启动：机器人模型 + 采摘节点 + RViz。

  仿真演示: ros2 launch zhihui_picker picker.launch.py
  自定义配置: ros2 launch zhihui_picker picker.launch.py config:=/path/to/a.yaml,/path/to/b.yaml
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('zhihui_picker')
    with open(os.path.join(share, 'urdf', 'picker_arm.urdf'), encoding='utf-8') as f:
        robot_desc = f.read()
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='sim', description='sim 或 real'),
        DeclareLaunchArgument('config', default_value='', description='配置文件路径（空 = 包内默认）'),
        DeclareLaunchArgument('auto_cycle', default_value='true', description='启动后自动连续采摘'),
        DeclareLaunchArgument('rviz', default_value='true'),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_desc}], output='screen'),
        Node(package='zhihui_picker', executable='picker_node', output='screen', parameters=[{
            'mode': LaunchConfiguration('mode'),
            'config': LaunchConfiguration('config'),
            'auto_cycle': ParameterValue(LaunchConfiguration('auto_cycle'), value_type=bool),
        }]),
        Node(package='rviz2', executable='rviz2', output='screen',
             arguments=['-d', os.path.join(share, 'urdf', 'picker.rviz')],
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
