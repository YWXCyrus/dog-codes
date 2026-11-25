"""Perception + reporter on their own, for use NEXT TO any navigation core (reactive-core).

Runs the mission's own nodes from frontier-fastlio/dog_codes (perception.py, reporter.py) with its
config/params.yaml - one copy of the perception code for both cores. Nothing here moves the dog:
perception only publishes motion in mode QR_APPROACH (set by the frontier supervisor, which is
not started here), and that goes to /quad/cmd_vel, which reactive-core does not listen to.

  # after reactive-core's run.py printed "Ready" (its fresh FAST-LIO feeds /odom via the bring-up):
  ros2 launch ~/dog-codes/perception/perception.launch.py
  # test without sending anything to the scoring server:
  ros2 launch ~/dog-codes/perception/perception.launch.py dry_run:=true

With frontier-fastlio do NOT use this: its mission.launch.py already starts perception + reporter.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import LaunchConfiguration as LC

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, '..', 'frontier-fastlio', 'dog_codes'))
PARAMS = os.path.join(CODE, 'config', 'params.yaml')


def local_group_id():
    """Registered group id (group##_XXXX, with its PIN) from ~/.dog_group_id - kept OUT of the repo,
    which is public on GitHub. Placeholder if the file is missing (the nodes then log an error)."""
    try:
        with open(os.path.expanduser('~/.dog_group_id')) as f:
            gid = f.read().strip()
        return gid or 'group02_XXXX'
    except OSError:
        return 'group02_XXXX'


def node(name, extra):
    cmd = ['python3', '-u', os.path.join(CODE, name + '.py'), '--ros-args', '-r', '__node:=' + name,
           '--params-file', PARAMS]
    for k, v in extra:
        cmd += ['-p', [k + ':=', LC(v)]]
    return ExecuteProcess(cmd=cmd, output='screen', name=name, respawn=True, respawn_delay=2.0)


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('group_id', default_value=local_group_id()),
        DeclareLaunchArgument('dry_run', default_value='false',
                              description='true: reporter logs the JSON, sends nothing'),
        DeclareLaunchArgument('arena', default_value='A',
                              description='A = left arena, B = right (organisers 2026-11-25: both accepted in the competition)'),
        DeclareLaunchArgument('report_url', default_value=['http://192.168.10.141:8001/task_report_', LC('arena')]),
        DeclareLaunchArgument('scene_server', default_value='http://192.168.10.141:8000'),
        DeclareLaunchArgument('scene_enabled', default_value='true',
                              description='false: no scene-server queries (objects/survivors/rooms)'),
        DeclareLaunchArgument('start_mode', default_value='EXPLORE',
                              description='no frontier supervisor: EXPLORE enables the scene queries'),
        DeclareLaunchArgument('odom_topic', default_value='/odom'),
        DeclareLaunchArgument('image_topic', default_value='/camera/color/image_raw'),
        LogInfo(msg=['perception standalone: code ', CODE, ', group ', LC('group_id'),
                     ', dry_run ', LC('dry_run')]),
        node('perception', [('group_id', 'group_id'), ('scene_server', 'scene_server'),
                            ('scene_enabled', 'scene_enabled'), ('start_mode', 'start_mode'),
                            ('odom_topic', 'odom_topic'), ('image_topic', 'image_topic')]),
        node('reporter', [('group_id', 'group_id'), ('dry_run', 'dry_run'), ('url', 'report_url')]),
    ])
