"""The feeding engine (rammp_feeding/engine.py). It registers with the orchestrator and moves the arm only in an action
the orchestrator starts. The image runs this file; a deployment runs the node's executable with its parameters.

    ros2 launch rammp_feeding feeding.launch.py                          # with the orchestrator
    ros2 launch rammp_feeding feeding.launch.py use_orchestrator:=false  # standalone
    ros2 launch rammp_feeding feeding.launch.py mock:=true               # no hardware
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = ("use_orchestrator", "true"), ("mock", "false")
    return LaunchDescription(
        [DeclareLaunchArgument(name, default_value=default) for name, default in args]
        + [Node(package="rammp_feeding", executable="feeding_node", output="screen",
                parameters=[{name: ParameterValue(LaunchConfiguration(name), value_type=bool)} for name, _ in args])])
