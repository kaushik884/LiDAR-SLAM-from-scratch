#include "slam_viz/slam_node.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <cstring>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>

namespace fs = std::filesystem;

namespace slam_viz {

// ============================================================================
// File loading utilities (copied from your lidar_odometry.cpp)
// ============================================================================

namespace {

Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> load_ply(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    std::string line;
    int num_vertices = 0;
    bool is_binary = false;
    std::vector<std::pair<std::string, std::string>> properties;

    // Parse header
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        
        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "format") {
            std::string format_type;
            iss >> format_type;
            if (format_type == "binary_little_endian" || format_type == "binary_big_endian") {
                is_binary = true;
            }
        } else if (token == "element") {
            std::string type;
            iss >> type;
            if (type == "vertex") {
                iss >> num_vertices;
            }
        } else if (token == "property") {
            std::string dtype, name;
            iss >> dtype >> name;
            properties.emplace_back(name, dtype);
        } else if (token == "end_header") {
            break;
        }
    }

    // Calculate offsets
    size_t bytes_per_vertex = 0;
    size_t x_offset = 0, y_offset = 0, z_offset = 0;

    auto get_type_size = [](const std::string& dtype) -> size_t {
        if (dtype == "float" || dtype == "float32") return 4;
        if (dtype == "double" || dtype == "float64") return 8;
        if (dtype == "uchar" || dtype == "uint8" || dtype == "char" || dtype == "int8") return 1;
        if (dtype == "ushort" || dtype == "uint16" || dtype == "short" || dtype == "int16") return 2;
        if (dtype == "uint" || dtype == "uint32" || dtype == "int" || dtype == "int32") return 4;
        return 4;
    };

    for (const auto& [name, dtype] : properties) {
        size_t type_size = get_type_size(dtype);
        if (name == "x") x_offset = bytes_per_vertex;
        else if (name == "y") y_offset = bytes_per_vertex;
        else if (name == "z") z_offset = bytes_per_vertex;
        bytes_per_vertex += type_size;
    }

    // Read points
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> points(num_vertices, 3);

    if (is_binary) {
        std::vector<char> buffer(bytes_per_vertex);
        for (int i = 0; i < num_vertices; ++i) {
            file.read(buffer.data(), bytes_per_vertex);
            float x, y, z;
            std::memcpy(&x, buffer.data() + x_offset, sizeof(float));
            std::memcpy(&y, buffer.data() + y_offset, sizeof(float));
            std::memcpy(&z, buffer.data() + z_offset, sizeof(float));
            points(i, 0) = static_cast<double>(x);
            points(i, 1) = static_cast<double>(y);
            points(i, 2) = static_cast<double>(z);
        }
    } else {
        for (int i = 0; i < num_vertices; ++i) {
            std::getline(file, line);
            std::istringstream iss(line);
            iss >> points(i, 0) >> points(i, 1) >> points(i, 2);
        }
    }

    return points;
}


Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> voxel_downsample(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& points,
    double voxel_size
) {
    if (voxel_size <= 0) return points;

    struct VoxelKey {
        long long x, y, z;
        bool operator==(const VoxelKey& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct VoxelHash {
        size_t operator()(const VoxelKey& v) const {
            size_t h = 0;
            h ^= std::hash<long long>{}(v.x) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<long long>{}(v.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<long long>{}(v.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    std::unordered_map<VoxelKey, std::vector<int>, VoxelHash> voxel_map;

    for (int i = 0; i < points.rows(); ++i) {
        VoxelKey key;
        key.x = static_cast<long long>(std::floor(points(i, 0) / voxel_size));
        key.y = static_cast<long long>(std::floor(points(i, 1) / voxel_size));
        key.z = static_cast<long long>(std::floor(points(i, 2) / voxel_size));
        voxel_map[key].push_back(i);
    }

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> downsampled(voxel_map.size(), 3);
    int idx = 0;
    for (const auto& [voxel, indices] : voxel_map) {
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (int i : indices) {
            centroid += points.row(i).transpose();
        }
        centroid /= indices.size();
        downsampled.row(idx++) = centroid.transpose();
    }

    return downsampled;
}


long long extract_timestamp(const std::string& filename) {
    std::regex pattern(R"((\d+)\.ply)");
    std::smatch match;
    if (std::regex_search(filename, match, pattern)) {
        return std::stoll(match[1].str());
    }
    return -1;
}

}  // anonymous namespace


// ============================================================================
// SlamNode implementation
// ============================================================================

SlamNode::SlamNode() : Node("slam_node"), current_frame_idx_(0), processing_complete_(false) {
    // Declare parameters
    this->declare_parameter<std::string>("data_dir", "");
    this->declare_parameter<double>("voxel_size", 0.5);
    this->declare_parameter<double>("playback_rate", 10.0);
    this->declare_parameter<int>("optimization_interval", 10);
    this->declare_parameter<int>("max_iterations", 50);
    this->declare_parameter<double>("tolerance", 1e-6);

    // Get parameters
    data_dir_ = this->get_parameter("data_dir").as_string();
    config_.voxel_size = this->get_parameter("voxel_size").as_double();
    playback_rate_ = this->get_parameter("playback_rate").as_double();
    config_.optimization_interval = this->get_parameter("optimization_interval").as_int();
    config_.max_iterations = this->get_parameter("max_iterations").as_int();
    config_.tolerance = this->get_parameter("tolerance").as_double();
    
    // Occupancy grid parameters
    this->declare_parameter<double>("grid_resolution", 0.2);
    this->declare_parameter<double>("height_min", 0.3);
    this->declare_parameter<double>("height_max", 2.0);
    this->declare_parameter<double>("max_range", 40.0);

    // Get occupancy grid parameters
    grid_config_.resolution = this->get_parameter("grid_resolution").as_double();
    grid_config_.height_min = this->get_parameter("height_min").as_double();
    grid_config_.height_max = this->get_parameter("height_max").as_double();
    grid_config_.max_range = this->get_parameter("max_range").as_double();

    RCLCPP_INFO(this->get_logger(), "  grid_resolution: %.2f", grid_config_.resolution);
    RCLCPP_INFO(this->get_logger(), "  height_filter: [%.1f, %.1f]", grid_config_.height_min, grid_config_.height_max);

    if (data_dir_.empty()) {
        RCLCPP_ERROR(this->get_logger(), "data_dir parameter is required!");
        throw std::runtime_error("data_dir parameter not set");
    }

    RCLCPP_INFO(this->get_logger(), "Configuration:");
    RCLCPP_INFO(this->get_logger(), "  data_dir: %s", data_dir_.c_str());
    RCLCPP_INFO(this->get_logger(), "  voxel_size: %.2f", config_.voxel_size);
    RCLCPP_INFO(this->get_logger(), "  playback_rate: %.1f Hz", playback_rate_);
    RCLCPP_INFO(this->get_logger(), "  optimization_interval: %d frames", config_.optimization_interval);

    // Create publishers
    current_scan_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/slam/current_scan", 10);
    global_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/slam/global_map", 10);
    trajectory_pub_ = this->create_publisher<nav_msgs::msg::Path>(
        "/slam/trajectory", 10);
    current_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/slam/current_pose", 10);
    occupancy_grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
        "/slam/occupancy_grid", 10);

    // Create TF broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Load frame list
    load_frame_list();

    if (frames_.empty()) {
        RCLCPP_ERROR(this->get_logger(), "No PLY files found in %s", data_dir_.c_str());
        throw std::runtime_error("No PLY files found");
    }

    RCLCPP_INFO(this->get_logger(), "Found %zu frames", frames_.size());

    // Initialize SLAM state
    poses_.push_back(icp::Transformation::identity());
    positions_.push_back(Eigen::Vector3d::Zero());
    pose_graph_.addPrior(0, icp::Transformation::identity());

    // Load first frame
    RCLCPP_INFO(this->get_logger(), "Loading first frame...");
    auto first_frame_raw = load_ply(frames_[0].second);
    // raw_clouds_.push_back(first_frame_raw);
    downsampled_clouds_.push_back(prev_points_);
    prev_points_ = voxel_downsample(first_frame_raw, config_.voxel_size);
    
    // Initialize global map with first frame
    global_map_points_ = prev_points_;

    RCLCPP_INFO(this->get_logger(), "First frame: %ld raw points, %ld after downsampling",
        first_frame_raw.rows(), prev_points_.rows());

    current_frame_idx_ = 1;  // Start from second frame

    // Create timer
    auto period_ms = static_cast<int>(1000.0 / playback_rate_);
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(period_ms),
        std::bind(&SlamNode::timer_callback, this));

    RCLCPP_INFO(this->get_logger(), "SLAM node ready, starting processing...");
}


void SlamNode::load_frame_list() {
    for (const auto& entry : fs::directory_iterator(data_dir_)) {
        if (entry.path().extension() == ".ply") {
            std::string filename = entry.path().filename().string();
            long long timestamp = extract_timestamp(filename);
            if (timestamp >= 0) {
                frames_.emplace_back(timestamp, entry.path().string());
            }
        }
    }

    // Sort by timestamp
    std::sort(frames_.begin(), frames_.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
}


void SlamNode::timer_callback() {
    if (processing_complete_) {
        // Still publish the final state for visualization
        publish_global_map();
        publish_trajectory();
        publish_current_pose();
        publish_occupancy_grid();
        return;
    }

    if (current_frame_idx_ >= static_cast<int>(frames_.size())) {
        RCLCPP_INFO(this->get_logger(), "Processing complete! Processed %zu frames", frames_.size());
        processing_complete_ = true;
        
        // Run final optimization
        RCLCPP_INFO(this->get_logger(), "Running final pose graph optimization...");
        run_pose_graph_optimization();
        return;
    }

    process_frame(current_frame_idx_);
    current_frame_idx_++;

    // Periodic optimization
    if (current_frame_idx_ % config_.optimization_interval == 0) {
        run_pose_graph_optimization();
    }
}


void SlamNode::process_frame(int frame_idx) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Load and downsample
    auto curr_points_raw = load_ply(frames_[frame_idx].second);
    // raw_clouds_.push_back(curr_points_raw);
    auto curr_points = voxel_downsample(curr_points_raw, config_.voxel_size);
    downsampled_clouds_.push_back(curr_points);
    RCLCPP_INFO(this->get_logger(), "Raw: %ld, Downsampled: %ld, prev_points: %ld",
    curr_points_raw.rows(), curr_points.rows(), prev_points_.rows());
    if (curr_points.rows() < config_.min_points) {
        RCLCPP_WARN(this->get_logger(), "Frame %d: Skipped (only %ld points)", 
            frame_idx, curr_points.rows());
        poses_.push_back(poses_.back());
        positions_.push_back(positions_.back());
        prev_points_ = curr_points;
        return;
    }

    // Run ICP
    icp::PointCloud source(curr_points);
    icp::PointCloud target(prev_points_);

    icp::ICPConfig icp_config;
    icp_config.max_iterations = config_.max_iterations;
    icp_config.tolerance = config_.tolerance;

    icp::ICPResult result = icp::icp_point_to_plane(source, target, icp_config);

    // Get transform
    icp::Transformation delta_transform;
    if (!result.converged || result.final_error > 1.0) {
        RCLCPP_WARN(this->get_logger(), "Frame %d: ICP failed (error=%.4f)", 
            frame_idx, result.final_error);
        delta_transform = icp::Transformation::identity();
    } else {
        delta_transform = result.transformation;
    }

    // Accumulate pose
    icp::Transformation new_pose = poses_.back() * delta_transform;
    poses_.push_back(new_pose);
    positions_.push_back(new_pose.t());

    // Add to pose graph
    pose_graph_.addOdometryFactor(
        poses_.size() - 2,
        poses_.size() - 1,
        delta_transform,
        result.final_error
    );

    // Transform current scan to world frame and add to global map
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> curr_world = 
        (curr_points * new_pose.R().transpose()).rowwise() + new_pose.t().transpose();
    
    // Append to global map
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> new_global_map(
        global_map_points_.rows() + curr_world.rows(), 3);
    new_global_map << global_map_points_, curr_world;
    global_map_points_ = new_global_map;

    // Update previous points
    prev_points_ = curr_points;
    // Update occupancy grid with new points
    update_occupancy_grid(curr_world, new_pose.t());

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    // Publish visualizations
    publish_current_scan(curr_world);
    if(current_frame_idx_ % 5 == 0){
        publish_global_map();
        publish_occupancy_grid();
    }
    publish_trajectory();
    publish_current_pose();

    if (frame_idx % 10 == 0) {
        RCLCPP_INFO(this->get_logger(), 
            "Frame %d/%zu: %ld pts, error=%.2e, iters=%d, time=%.1fms, map=%ld pts",
            frame_idx, frames_.size() - 1, curr_points.rows(), 
            result.final_error, result.num_iterations, elapsed_ms,
            global_map_points_.rows());
    }
}


void SlamNode::run_pose_graph_optimization() {
    RCLCPP_INFO(this->get_logger(), "Running pose graph optimization...");
    
    if (pose_graph_.optimize()) {
        RCLCPP_INFO(this->get_logger(), "Optimization converged, error: %.2f", 
            pose_graph_.getFinalError());
        
        // Update poses
        std::vector<icp::Transformation> optimized_poses = pose_graph_.getAllPoses();
        poses_ = optimized_poses;
        
        // Update positions
        positions_.clear();
        for (const auto& pose : poses_) {
            positions_.push_back(pose.t());
        }
        
        // Rebuild global map with optimized poses
        global_map_points_.resize(0, 3);
        for (size_t i = 0; i < downsampled_clouds_.size() && i < poses_.size(); ++i) {
            // auto downsampled = voxel_downsample(raw_clouds_[i], config_.voxel_size);
            Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> cloud_world = 
                (downsampled_clouds_[i] * poses_[i].R().transpose()).rowwise() + poses_[i].t().transpose();
            
            Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> new_global_map(
                global_map_points_.rows() + cloud_world.rows(), 3);
            new_global_map << global_map_points_, cloud_world;
            global_map_points_ = new_global_map;
        }
        rebuild_occupancy_grid();
        
        RCLCPP_INFO(this->get_logger(), "Rebuilt global map with %ld points", 
            global_map_points_.rows());
    } else {
        RCLCPP_WARN(this->get_logger(), "Optimization failed");
    }
}


void SlamNode::publish_current_scan(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& cloud) {
    auto msg = eigen_to_pointcloud2(cloud, "map");
    current_scan_pub_->publish(msg);
}


void SlamNode::publish_global_map() {
    // Downsample global map for visualization to reduce bandwidth
    auto downsampled = voxel_downsample(global_map_points_, config_.voxel_size * 2);
    auto msg = eigen_to_pointcloud2(downsampled, "map");
    global_map_pub_->publish(msg);
}


void SlamNode::publish_trajectory() {
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = this->now();
    path_msg.header.frame_id = "map";

    for (size_t i = 0; i < poses_.size(); ++i) {
        auto pose_stamped = transformation_to_pose_stamped(poses_[i], "map");
        path_msg.poses.push_back(pose_stamped);
    }

    trajectory_pub_->publish(path_msg);
}


void SlamNode::publish_current_pose() {
    if (poses_.empty()) return;

    auto pose_msg = transformation_to_pose_stamped(poses_.back(), "map");
    current_pose_pub_->publish(pose_msg);

    // Also broadcast TF
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = this->now();
    tf_msg.header.frame_id = "map";
    tf_msg.child_frame_id = "base_link";
    tf_msg.transform.translation.x = poses_.back().t().x();
    tf_msg.transform.translation.y = poses_.back().t().y();
    tf_msg.transform.translation.z = poses_.back().t().z();

    // Convert rotation matrix to quaternion
    Eigen::Quaterniond q(poses_.back().R());
    tf_msg.transform.rotation.x = q.x();
    tf_msg.transform.rotation.y = q.y();
    tf_msg.transform.rotation.z = q.z();
    tf_msg.transform.rotation.w = q.w();

    tf_broadcaster_->sendTransform(tf_msg);
}

void SlamNode::update_occupancy_grid(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& cloud_world,
    const Eigen::Vector3d& sensor_position) {
    
    for (int i = 0; i < cloud_world.rows(); ++i) {
        double x = cloud_world(i, 0);
        double y = cloud_world(i, 1);
        double z = cloud_world(i, 2);
        
        // Height filter
        if (z < grid_config_.height_min || z > grid_config_.height_max) {
            continue;
        }
        
        // Range filter (distance from sensor)
        double dx = x - sensor_position.x();
        double dy = y - sensor_position.y();
        double range = std::sqrt(dx*dx + dy*dy);
        if (range > grid_config_.max_range || range < 0.5) {
            continue;
        }
        
        // Add to grid
        GridCell cell;
        cell.x = static_cast<int>(std::floor(x / grid_config_.resolution));
        cell.y = static_cast<int>(std::floor(y / grid_config_.resolution));
        occupied_cells_.insert(cell);
    }
}


void SlamNode::rebuild_occupancy_grid() {
    RCLCPP_INFO(this->get_logger(), "Rebuilding occupancy grid...");
    
    occupied_cells_.clear();
    
    for (size_t i = 0; i < downsampled_clouds_.size() && i < poses_.size(); ++i) {
        const auto& cloud = downsampled_clouds_[i];
        const auto& pose = poses_[i];
        
        // Transform to world frame
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> cloud_world = 
            (cloud * pose.R().transpose()).rowwise() + pose.t().transpose();
        
        update_occupancy_grid(cloud_world, pose.t());
    }
    
    RCLCPP_INFO(this->get_logger(), "Occupancy grid rebuilt with %zu cells", occupied_cells_.size());
}


void SlamNode::publish_occupancy_grid() {
    auto msg = cells_to_occupancy_grid_msg();
    occupancy_grid_pub_->publish(msg);
}


nav_msgs::msg::OccupancyGrid SlamNode::cells_to_occupancy_grid_msg() {
    nav_msgs::msg::OccupancyGrid msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = "map";
    
    if (occupied_cells_.empty()) {
        return msg;
    }
    
    // Find grid bounds
    int min_x = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int min_y = std::numeric_limits<int>::max();
    int max_y = std::numeric_limits<int>::min();
    
    for (const auto& cell : occupied_cells_) {
        min_x = std::min(min_x, cell.x);
        max_x = std::max(max_x, cell.x);
        min_y = std::min(min_y, cell.y);
        max_y = std::max(max_y, cell.y);
    }
    
    // Add padding
    int padding = 5;
    min_x -= padding;
    min_y -= padding;
    max_x += padding;
    max_y += padding;
    
    int width = max_x - min_x + 1;
    int height = max_y - min_y + 1;
    
    // Set metadata
    msg.info.resolution = grid_config_.resolution;
    msg.info.width = width;
    msg.info.height = height;
    msg.info.origin.position.x = min_x * grid_config_.resolution;
    msg.info.origin.position.y = min_y * grid_config_.resolution;
    msg.info.origin.position.z = 0.0;
    msg.info.origin.orientation.w = 1.0;
    
    // Initialize all cells as unknown (-1) or free (0)
    msg.data.resize(width * height, 0);  // 0 = free
    
    // Mark occupied cells
    for (const auto& cell : occupied_cells_) {
        int grid_x = cell.x - min_x;
        int grid_y = cell.y - min_y;
        int index = grid_y * width + grid_x;
        msg.data[index] = 100;  // 100 = occupied
    }
    
    return msg;
}

sensor_msgs::msg::PointCloud2 SlamNode::eigen_to_pointcloud2(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& cloud,
    const std::string& frame_id) {
    
    sensor_msgs::msg::PointCloud2 msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = frame_id;
    
    msg.height = 1;
    msg.width = cloud.rows();
    msg.is_dense = true;
    msg.is_bigendian = false;

    // Define fields (x, y, z as float32)
    sensor_msgs::msg::PointField field_x, field_y, field_z;
    field_x.name = "x";
    field_x.offset = 0;
    field_x.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_x.count = 1;

    field_y.name = "y";
    field_y.offset = 4;
    field_y.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_y.count = 1;

    field_z.name = "z";
    field_z.offset = 8;
    field_z.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_z.count = 1;

    msg.fields = {field_x, field_y, field_z};
    msg.point_step = 12;  // 3 floats * 4 bytes
    msg.row_step = msg.point_step * msg.width;

    // Fill data
    msg.data.resize(msg.row_step * msg.height);
    float* data_ptr = reinterpret_cast<float*>(msg.data.data());

    for (int i = 0; i < cloud.rows(); ++i) {
        data_ptr[i * 3 + 0] = static_cast<float>(cloud(i, 0));
        data_ptr[i * 3 + 1] = static_cast<float>(cloud(i, 1));
        data_ptr[i * 3 + 2] = static_cast<float>(cloud(i, 2));
    }

    return msg;
}


geometry_msgs::msg::PoseStamped SlamNode::transformation_to_pose_stamped(
    const icp::Transformation& transform,
    const std::string& frame_id) {
    
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = this->now();
    pose_msg.header.frame_id = frame_id;

    pose_msg.pose.position.x = transform.t().x();
    pose_msg.pose.position.y = transform.t().y();
    pose_msg.pose.position.z = transform.t().z();

    Eigen::Quaterniond q(transform.R());
    pose_msg.pose.orientation.x = q.x();
    pose_msg.pose.orientation.y = q.y();
    pose_msg.pose.orientation.z = q.z();
    pose_msg.pose.orientation.w = q.w();

    return pose_msg;
}

}  // namespace slam_viz


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    try {
        rclcpp::spin(std::make_shared<slam_viz::SlamNode>());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("slam_node"), "Error: %s", e.what());
    }
    
    rclcpp::shutdown();
    return 0;
}