#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <vector>
#include <string>
#include <unordered_set>
#include <memory>

// Core library includes (no ROS2 dependency)
#include "slam_viz/core/types.hpp"
#include "slam_viz/core/pose_graph.hpp"
#include "slam_viz/core/loop_closure.hpp"

namespace slam_viz {

/**
 * Configuration for the SLAM node.
 */
struct SlamConfig {
    double voxel_size = 0.5;
    int max_iterations = 50;
    double tolerance = 1e-6;
    int min_points = 1000;
};

/**
 * Configuration for occupancy grid.
 */
struct OccupancyGridConfig {
    double resolution = 0.2;
    double height_min = 0.3;
    double height_max = 2.0;
    double max_range = 40.0;
};

/**
 * Grid cell for occupancy mapping.
 */
struct GridCell {
    int x, y;
    bool operator==(const GridCell& other) const {
        return x == other.x && y == other.y;
    }
};

struct GridCellHash {
    size_t operator()(const GridCell& cell) const {
        size_t h = 0;
        h ^= std::hash<int>{}(cell.x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(cell.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};


/**
 * ROS2 SLAM Node.
 * 
 * Wraps the core SLAM algorithms and provides:
 * - Timer-driven frame processing
 * - Real-time visualization via RViz2
 * - TF broadcasting
 * 
 * Design: This is the ONLY file that includes ROS2 headers.
 * All algorithms are in the core library.
 */
class SlamNode : public rclcpp::Node {
public:
    SlamNode();

private:
    // ========================================================================
    // Callbacks
    // ========================================================================
    void timer_callback();
    
    // ========================================================================
    // Processing
    // ========================================================================
    void load_frame_list();
    void process_frame(int frame_idx);
    void run_pose_graph_optimization();
    
    // ========================================================================
    // Publishing
    // ========================================================================
    void publish_current_scan(const slam::PointCloud::Matrix& cloud);
    void publish_global_map();
    void publish_trajectory();
    void publish_current_pose();
    void publish_occupancy_grid();
    
    // ========================================================================
    // Occupancy Grid
    // ========================================================================
    void update_occupancy_grid(
        const slam::PointCloud::Matrix& cloud_world,
        const Eigen::Vector3d& sensor_position);
    void rebuild_occupancy_grid();
    void rebuild_recent_clouds();
    void build_final_global_map();
    nav_msgs::msg::OccupancyGrid cells_to_occupancy_grid_msg();
    
    // ========================================================================
    // Message Conversion
    // ========================================================================
    sensor_msgs::msg::PointCloud2 eigen_to_pointcloud2(
        const slam::PointCloud::Matrix& cloud,
        const std::string& frame_id);
    geometry_msgs::msg::PoseStamped transformation_to_pose_stamped(
        const slam::Transformation& transform,
        const std::string& frame_id);

    // ========================================================================
    // ROS2 Communication
    // ========================================================================
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr current_scan_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr global_map_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr trajectory_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr current_pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_grid_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // ========================================================================
    // Configuration
    // ========================================================================
    SlamConfig config_;
    OccupancyGridConfig grid_config_;
    std::string data_dir_;
    double playback_rate_;

    // ========================================================================
    // Frame Data
    // ========================================================================
    std::vector<std::pair<long long, std::string>> frames_;
    int current_frame_idx_;
    bool processing_complete_;

    // ========================================================================
    // SLAM State (using core library types)
    // ========================================================================
    std::vector<slam::Transformation> poses_;
    std::vector<Eigen::Vector3d> positions_;
    std::vector<slam::PointCloud::Matrix> downsampled_clouds_;
    slam::PointCloud::Matrix prev_points_;
    slam::PoseGraph pose_graph_;
    slam::LoopClosureDetector loop_detector_;
    
    // ========================================================================
    // Visualization State
    // ========================================================================
    slam::PointCloud::Matrix global_map_points_;
    std::vector<slam::PointCloud::Matrix> recent_clouds_world_;
    std::unordered_set<GridCell, GridCellHash> occupied_cells_;
    
    // ========================================================================
    // Statistics
    // ========================================================================
    int loop_closures_found_ = 0;
    bool has_loop_closure_pending_ = false;
    
    static constexpr size_t MAX_RECENT_CLOUDS = 20;
};

}  // namespace slam_viz