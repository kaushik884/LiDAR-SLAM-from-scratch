cmake_minimum_required(VERSION 3.14)

# GTSAM - manual linking since we did manual install
set(GTSAM_INCLUDE_DIR "/usr/local/include")
set(GTSAM_LIBRARY "/usr/local/lib/libgtsam.so")

# ICP library (static library now, includes pose_graph)
add_library(icp_lib STATIC
    src/pose_graph.cpp
)
target_include_directories(icp_lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GTSAM_INCLUDE_DIR}
)
target_link_libraries(icp_lib PUBLIC Eigen3::Eigen ${GTSAM_LIBRARY} tbb)

# Main executable
add_executable(run_icp apps/run_icp.cpp)
target_link_libraries(run_icp PRIVATE icp_lib)

# LiDAR odometry executable
add_executable(lidar_odometry apps/lidar_odometry.cpp)
target_link_libraries(lidar_odometry PRIVATE icp_lib)