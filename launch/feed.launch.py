"""rammp feed under sheppy: the feeding loop, bite after bite, until the bowl is empty or the user declines. Stopping
the node (sheppy) cancels the move in flight; the arm halts where it is."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("bites", default_value="99"),
        Node(package="rammp_feeding", executable="rammp", name="rammp_feeding", output="screen",
             arguments=["feed", "--bites", LaunchConfiguration("bites")], on_exit=Shutdown()),
    ])
