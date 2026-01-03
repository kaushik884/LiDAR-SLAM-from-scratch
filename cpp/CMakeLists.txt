cmake_minimum_required(VERSION 3.14)

# ICP library (header-only for now)
add_library(icp INTERFACE)
target_include_directories(icp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(icp INTERFACE Eigen3::Eigen)

# Main executable
add_executable(run_icp src/run_icp.cpp)
target_link_libraries(run_icp PRIVATE icp)

# Tests (simple standalone test, no GTest required)
# add_executable(test_icp tests/test_icp.cpp)
# target_link_libraries(test_icp PRIVATE icp)

# LiDAR odometry executable
add_executable(lidar_odometry src/lidar_odometry.cpp)
target_link_libraries(lidar_odometry PRIVATE icp)