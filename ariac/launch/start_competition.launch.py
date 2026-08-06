
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch.conditions import IfCondition

from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
)

# from ariac_moveit_config.parameters import generate_parameters

def launch_setup(context, *args, **kwargs):
    # Launch arguments
    trial_name = LaunchConfiguration("trial_name")
    
    # # Move Group
    # moveit = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         [FindPackageShare("ariac_moveit_config"), "/launch", "/ariac_robots_moveit.launch.py"]
    #     )
    # )

    # ARIAC_environment
    ariac_environment = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ariac_gazebo"), "/launch", "/ariac.launch.py"]
        ),
        launch_arguments={
            'trial_name': trial_name,
            'competitor_pkg': "competition_tutorials",
            'sensor_config': "sensors",
            "dev_mode": "false",

            
        }.items()
    )

    # Test Competitor node
    test_competitor = Node(
        package='competition_tutorials',
        executable='sia_interface.py',
        output="screen",
        # parameters=generate_parameters(),
    )

    nodes_to_start = [
        test_competitor,
        ariac_environment,
        # moveit
    ]

    return nodes_to_start

def generate_launch_description():
    declared_arguments = []


    declared_arguments.append(
        DeclareLaunchArgument("trial_name", default_value="kitting", description="Name of ariac trial")
    )

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
