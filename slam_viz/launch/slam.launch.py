from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os


def generate_launch_description():
    return LaunchDescription([
        # Declare arguments
        DeclareLaunchArgument(
            'data_dir',
            default_value='',
            description='Path to directory containing PLY files'
        ),
        DeclareLaunchArgument(
            'voxel_size',
            default_value='0.5',
            description='Voxel size for downsampling (meters)'
        ),
        DeclareLaunchArgument(
            'playback_rate',
            default_value='10.0',
            description='Frame processing rate (Hz)'
        ),
        
        # SLAM node
        Node(
            package='slam_viz',
            executable='slam_node',
            name='slam_node',
            output='screen',
            parameters=[{
                'data_dir': LaunchConfiguration('data_dir'),
                'voxel_size': LaunchConfiguration('voxel_size'),
                'playback_rate': LaunchConfiguration('playback_rate'),
            }]
        ),
    ])