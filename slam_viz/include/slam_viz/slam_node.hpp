#ifndef SLAM_VIZ__SLAM_NODE_HPP_
#define SLAM_VIZ__SLAM_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <unordered_set>
#include <vector>
#include <string>
#include <Eigen/Dense>

#include "icp.hpp"
#include "pose_graph.hpp"

namespace slam_viz {

/**
 * Configuration matching your OdometryConfig
 */
struct SlamConfig {
    double voxel_size = 0.2;
    int max_iterations = 50;
    double tolerance = 1e-6;
    int min_points = 1000;
    int optimization_interval = 5;  // Run pose graph optimization every N frames
};
/**
 * Configuration for occupancy grid
 */
struct OccupancyGridConfig {
    double resolution = 0.2;      // Grid cell size in meters
    double height_min = 0.3;      // Minimum z height (above ground)
    double height_max = 2.0;      // Maximum z height (below ceiling)
    double max_range = 40.0;      // Maximum distance from sensor
};

/**
 * Grid cell for hashing
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
class SlamNode : public rclcpp::Node {
public:
    SlamNode();

private:
    // Timer callback - processes one frame per tick
    void timer_callback();
    
    // Initialization
    void load_frame_list();
    
    // Processing
    void process_frame(int frame_idx);
    void run_pose_graph_optimization();
    
    // Publishing
    void publish_current_scan(
        const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& cloud);
    void publish_global_map();
    void publish_trajectory();
    void publish_current_pose();
    
    // Message conversion helpers
    sensor_msgs::msg::PointCloud2 eigen_to_pointcloud2(
        const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& cloud,
        const std::string& frame_id);
    geometry_msgs::msg::PoseStamped transformation_to_pose_stamped(
        const icp::Transformation& transform,
        const std::string& frame_id);

    // ROS2 communication
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr current_scan_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr global_map_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr trajectory_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr current_pose_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // Configuration
    SlamConfig config_;
    std::string data_dir_;
    double playback_rate_;  // Hz

    // Frame data
    std::vector<std::pair<long long, std::string>> frames_;  // (timestamp, filepath)
    int current_frame_idx_;
    bool processing_complete_;

    // SLAM state
    std::vector<icp::Transformation> poses_;
    std::vector<Eigen::Vector3d> positions_;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> downsampled_clouds_;
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> prev_points_;
    icp::PoseGraph pose_graph_;
    
    // Global map (accumulated points in world frame)
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> global_map_points_;
    // New publisher
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_grid_pub_;

    // New config
    OccupancyGridConfig grid_config_;

    // Occupancy grid state
    std::unordered_set<GridCell, GridCellHash> occupied_cells_;

    // New methods
    void update_occupancy_grid(
        const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& cloud_world,
        const Eigen::Vector3d& sensor_position);
    void rebuild_occupancy_grid();
    void publish_occupancy_grid();
    nav_msgs::msg::OccupancyGrid cells_to_occupancy_grid_msg();
};

}  // namespace slam_viz

#endif  // SLAM_VIZ__SLAM_NODE_HPP_