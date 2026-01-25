cmake_minimum_required(VERSION 3.14)
project(slam_viz)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# ==============================================================================
# Find dependencies
# ==============================================================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# GTSAM (manual install location - adjust if needed)
set(GTSAM_INCLUDE_DIR "/usr/local/include")
set(GTSAM_LIBRARY "/usr/local/lib/libgtsam.so")
# TBB (required by GTSAM)
find_package(TBB REQUIRED)

# ==============================================================================
# Core library (no ROS2 dependency)
# ==============================================================================
add_library(slam_core
  src/core/pose_graph.cpp
  src/core/file_utils.cpp
)

target_include_directories(slam_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${GTSAM_INCLUDE_DIR}
)

target_link_libraries(slam_core PUBLIC
  Eigen3::Eigen
  ${GTSAM_LIBRARY}
  TBB::tbb
)

# ==============================================================================
# ROS2 node executable
# ==============================================================================
add_executable(slam_node
  src/ros/slam_node.cpp
)

target_include_directories(slam_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Use ament_target_dependencies FIRST, then link our library
ament_target_dependencies(slam_node
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf2_ros
  tf2
)

# Link our core library AFTER ament_target_dependencies
target_link_libraries(slam_node
  slam_core
)

# ==============================================================================
# Install
# ==============================================================================
install(TARGETS slam_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  OPTIONAL
)

install(DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
  OPTIONAL
)

ament_package()