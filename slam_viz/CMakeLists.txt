cmake_minimum_required(VERSION 3.14)
project(slam_viz)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ROS2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

# GTSAM
set(GTSAM_INCLUDE_DIR "/usr/local/include")
set(GTSAM_LIBRARY "/usr/local/lib/libgtsam.so")

set(LIDAR_PROJECT_DIR "/home/kaushik/Coding/LiDAR-ICP-Algorithm-from-scratch")

# Build the icp_lib from the cpp directory
add_library(icp_lib STATIC
    ${LIDAR_PROJECT_DIR}/cpp/src/pose_graph.cpp
)
target_include_directories(icp_lib PUBLIC
    ${LIDAR_PROJECT_DIR}/cpp/include
    ${GTSAM_INCLUDE_DIR}
)
target_link_libraries(icp_lib PUBLIC Eigen3::Eigen ${GTSAM_LIBRARY})

# SLAM visualization node
add_executable(slam_node
    src/slam_node.cpp
)
target_include_directories(slam_node PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
ament_target_dependencies(slam_node
rclcpp
sensor_msgs
nav_msgs
geometry_msgs
tf2_ros
tf2_geometry_msgs
)
target_link_libraries(slam_node icp_lib)

# Install executable
install(TARGETS slam_node
    DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
)

# Install rviz config
install(DIRECTORY rviz/
    DESTINATION share/${PROJECT_NAME}/rviz
)

ament_package()