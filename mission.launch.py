"""Start every dog_codes node.  Usage (after the TAs' bringup is running):

  # terminal 1 (TA bringup, no Nav2):
  ros2 launch lite3_launcher lite3.launch.py navigation:=False
  # terminal 2:
  ros2 launch ~/dog_codes/mission.launch.py group_id:=group02_XXXX
  # then, with the dog on the hoist and someone on STOP:
  ros2 topic pub --once /quad/arm std_msgs/msg/Bool "{data: true}"

Launch args: group_id, dry_run (true = nothing is ever sent towards the dog), final_topic.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration as LC

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = os.path.join(HERE, 'config', 'params.yaml')
NODES = ['safety_gate', 'cloud_proc', 'explorer', 'terrain', 'curtain', 'perception', 'reporter',
         'supervisor']


def generate_launch_description():
    ld = LaunchDescription([
        DeclareLaunchArgument('group_id', default_value='group02_XXXX'),
        DeclareLaunchArgument('dry_run', default_value='false'),
        DeclareLaunchArgument('final_topic', default_value='/final_cell_reached'),
        DeclareLaunchArgument('scene_server', default_value='http://192.168.10.141:8000'),
        DeclareLaunchArgument('record', default_value='true'),
        DeclareLaunchArgument('record_heavy', default_value='false'),
        DeclareLaunchArgument('report_url', default_value='http://192.168.10.141:8001/task_report'),
    ])
    for n in NODES:
        cmd = ['python3', '-u', os.path.join(HERE, n + '.py'), '--ros-args', '-r', '__node:=' + n,
               '--params-file', PARAMS]
        if n in ('perception', 'reporter'):
            cmd += ['-p', ['group_id:=', LC('group_id')]]
        if n in ('safety_gate', 'reporter'):
            cmd += ['-p', ['dry_run:=', LC('dry_run')]]
        if n == 'perception':
            cmd += ['-p', ['scene_server:=', LC('scene_server')]]
        if n == 'reporter':
            cmd += ['-p', ['url:=', LC('report_url')]]
        if n == 'supervisor':
            cmd += ['-p', ['final_topic:=', LC('final_topic')]]
        ld.add_action(ExecuteProcess(cmd=cmd, output='screen', name=n, respawn=True,
                                     respawn_delay=2.0))
    # ---- data for refining the code after a run (bag in run_logs/, also the node logs in ~/.ros/log)
    light = ('/quad/status /quad/mode /quad/limits /quad/event /quad/report /quad/supervisor '
             '/quad/explore_state /quad/curtain_info /quad/scene_state /quad/reporter_status /quad/vlm_ask '
             '/quad/vlm_answer /quad/behaviour /quad/explore_ctrl /quad/cmd_vel /nav_cmd_vel /cmd_vel '
             '/simple_cmd /odom /imu/data /joint_states /tf /tf_static /final_cell_reached')
    heavy = ' /quad/cloud /camera/color/image_raw'
    out = os.path.join(HERE, 'run_logs')
    ld.add_action(ExecuteProcess(
        cmd=['bash', '-c', 'mkdir -p "%s" && exec ros2 bag record -o "%s/bag_$(date +%%Y%%m%%d_%%H%%M%%S)" %s' % (
            out, out, light)],
        output='log', name='bag_light', condition=IfCondition(LC('record'))))
    ld.add_action(ExecuteProcess(
        cmd=['bash', '-c', 'mkdir -p "%s" && exec ros2 bag record -o "%s/bagheavy_$(date +%%Y%%m%%d_%%H%%M%%S)" %s' % (
            out, out, heavy.strip())],
        output='log', name='bag_heavy', condition=IfCondition(LC('record_heavy'))))
    return ld
