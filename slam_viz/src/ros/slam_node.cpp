#include "slam_viz/ros/slam_node.hpp"
#include "slam_viz/core/file_utils.hpp"
#include "slam_viz/core/icp.hpp"

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <chrono>

namespace slam_viz {

SlamNode::SlamNode() 
    : Node("slam_node")
    , current_frame_idx_(0)
    , processing_complete_(false) 
{
    // Declare parameters
    this->declare_parameter<std::string>("data_dir", "");
    this->declare_parameter<double>("voxel_size", 0.5);
    this->declare_parameter<double>("playback_rate", 10.0);
    this->declare_parameter<int>("max_iterations", 50);
    this->declare_parameter<double>("tolerance", 1e-6);
    this->declare_parameter<double>("grid_resolution", 0.2);
    this->declare_parameter<double>("height_min", 0.3);
    this->declare_parameter<double>("height_max", 2.0);
    this->declare_parameter<double>("max_range", 40.0);

    data_dir_ = this->get_parameter("data_dir").as_string();
    config_.voxel_size = this->get_parameter("voxel_size").as_double();
    playback_rate_ = this->get_parameter("playback_rate").as_double();
    config_.max_iterations = this->get_parameter("max_iterations").as_int();
    config_.tolerance = this->get_parameter("tolerance").as_double();
    grid_config_.resolution = this->get_parameter("grid_resolution").as_double();
    grid_config_.height_min = this->get_parameter("height_min").as_double();
    grid_config_.height_max = this->get_parameter("height_max").as_double();
    grid_config_.max_range = this->get_parameter("max_range").as_double();

    if (data_dir_.empty()) {
        RCLCPP_ERROR(this->get_logger(), "data_dir parameter is required!");
        throw std::runtime_error("data_dir parameter not set");
    }

    RCLCPP_INFO(this->get_logger(), "Configuration:");
    RCLCPP_INFO(this->get_logger(), "  data_dir: %s", data_dir_.c_str());
    RCLCPP_INFO(this->get_logger(), "  voxel_size: %.2f", config_.voxel_size);
    RCLCPP_INFO(this->get_logger(), "  playback_rate: %.1f Hz", playback_rate_);

    // Create publishers
    current_scan_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/slam/current_scan", 10);
    global_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/slam/global_map", 10);
    trajectory_pub_ = this->create_publisher<nav_msgs::msg::Path>("/slam/trajectory", 10);
    current_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/slam/current_pose", 10);
    occupancy_grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/slam/occupancy_grid", 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Load frames
    load_frame_list();
    if (frames_.empty()) {
        RCLCPP_ERROR(this->get_logger(), "No frames found in %s", data_dir_.c_str());
        throw std::runtime_error("No frames found");
    }
    RCLCPP_INFO(this->get_logger(), "Found %zu frames", frames_.size());

    // Initialize SLAM
    poses_.push_back(slam::Transformation::identity());
    positions_.push_back(Eigen::Vector3d::Zero());
    pose_graph_.addPrior(0, slam::Transformation::identity());

    // Load first frame
    auto first_frame_raw = slam::load_ply(frames_[0].second);
    prev_points_ = slam::voxel_downsample(first_frame_raw, config_.voxel_size);
    downsampled_clouds_.push_back(prev_points_);
    global_map_points_ = prev_points_;
    RCLCPP_INFO(this->get_logger(), "First frame: %ld raw, %ld downsampled", 
        first_frame_raw.rows(), prev_points_.rows());

    // Initialize loop closure
    slam::LoopClosureConfig lc_config;
    lc_config.frame_gap = 50;
    lc_config.sc_distance_threshold = 0.2;
    lc_config.icp_fitness_threshold = 0.3;
    loop_detector_ = slam::LoopClosureDetector(lc_config);

    // Start timer
    current_frame_idx_ = 1;
    auto period_ms = static_cast<int>(1000.0 / playback_rate_);
    timer_ = this->create_wall_timer(std::chrono::milliseconds(period_ms),
        std::bind(&SlamNode::timer_callback, this));
    RCLCPP_INFO(this->get_logger(), "SLAM node ready");
}

void SlamNode::load_frame_list() {
    frames_ = slam::discover_frames(data_dir_);
}

void SlamNode::timer_callback() {
    if (processing_complete_) {
        publish_global_map();
        publish_trajectory();
        publish_current_pose();
        publish_occupancy_grid();
        return;
    }
    if (current_frame_idx_ >= static_cast<int>(frames_.size())) {
        processing_complete_ = true;
        RCLCPP_INFO(this->get_logger(), "Processing complete!");
        run_pose_graph_optimization();
        build_final_global_map();
        return;
    }
    process_frame(current_frame_idx_);
    current_frame_idx_++;
    if (has_loop_closure_pending_) {
        run_pose_graph_optimization();
        has_loop_closure_pending_ = false;
    }
}

void SlamNode::process_frame(int frame_idx) {
    auto start = std::chrono::high_resolution_clock::now();

    auto raw = slam::load_ply(frames_[frame_idx].second);
    auto curr = slam::voxel_downsample(raw, config_.voxel_size);
    downsampled_clouds_.push_back(curr);

    if (curr.rows() < config_.min_points) {
        poses_.push_back(poses_.back());
        positions_.push_back(positions_.back());
        prev_points_ = curr;
        return;
    }

    slam::PointCloud source(curr);
    slam::PointCloud target(prev_points_);
    slam::ICPConfig icp_cfg;
    icp_cfg.max_iterations = config_.max_iterations;
    icp_cfg.tolerance = config_.tolerance;

    auto result = slam::icp_point_to_plane(source, target, icp_cfg);
    auto delta = (!result.converged || result.final_error > 1.0) 
        ? slam::Transformation::identity() : result.transformation;

    auto new_pose = poses_.back() * delta;
    poses_.push_back(new_pose);
    positions_.push_back(new_pose.t());
    pose_graph_.addOdometryFactor(poses_.size()-2, poses_.size()-1, delta, result.final_error);

    slam::PointCloud::Matrix world = (curr * new_pose.R().transpose()).rowwise() + new_pose.t().transpose();
    recent_clouds_world_.push_back(world);
    if (recent_clouds_world_.size() > MAX_RECENT_CLOUDS) recent_clouds_world_.erase(recent_clouds_world_.begin());

    prev_points_ = curr;
    update_occupancy_grid(world, new_pose.t());

    publish_current_scan(world);
    if (frame_idx % 5 == 0) publish_global_map();
    publish_trajectory();
    publish_current_pose();

    loop_detector_.addFrame(curr, frame_idx);
    if (frame_idx % 10 == 0 && frame_idx > 50) {
        for (const auto& lc : loop_detector_.detect()) {
            RCLCPP_INFO(this->get_logger(), "Loop: %d<->%d", lc.query_frame, lc.match_frame);
            pose_graph_.addLoopClosure(lc.match_frame, lc.query_frame, lc.transform);
            loop_closures_found_++;
            has_loop_closure_pending_ = true;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    if (frame_idx % 10 == 0) {
        double ms = std::chrono::duration<double,std::milli>(end-start).count();
        RCLCPP_INFO(this->get_logger(), "Frame %d/%zu: %.1fms, err=%.2e, loops=%d",
            frame_idx, frames_.size()-1, ms, result.final_error, loop_closures_found_);
    }
}

void SlamNode::run_pose_graph_optimization() {
    if (pose_graph_.optimize()) {
        poses_ = pose_graph_.getAllPoses();
        positions_.clear();
        for (const auto& p : poses_) positions_.push_back(p.t());
        rebuild_recent_clouds();
        RCLCPP_INFO(this->get_logger(), "Optimized, error=%.2f", pose_graph_.getFinalError());
    }
}

void SlamNode::rebuild_recent_clouds() {
    recent_clouds_world_.clear();
    size_t start = downsampled_clouds_.size() > MAX_RECENT_CLOUDS ? downsampled_clouds_.size() - MAX_RECENT_CLOUDS : 0;
    for (size_t i = start; i < downsampled_clouds_.size() && i < poses_.size(); ++i) {
        slam::PointCloud::Matrix w = (downsampled_clouds_[i] * poses_[i].R().transpose()).rowwise() + poses_[i].t().transpose();
        recent_clouds_world_.push_back(w);
    }
}

void SlamNode::build_final_global_map() {
    size_t total = 0;
    for (const auto& c : downsampled_clouds_) total += c.rows();
    global_map_points_.resize(total, 3);
    size_t row = 0;
    for (size_t i = 0; i < downsampled_clouds_.size() && i < poses_.size(); ++i) {
        for (int j = 0; j < downsampled_clouds_[i].rows(); ++j) {
            Eigen::Vector3d pt = poses_[i].R() * downsampled_clouds_[i].row(j).transpose() + poses_[i].t();
            global_map_points_.row(row++) = pt.transpose();
        }
    }
    rebuild_occupancy_grid();
    RCLCPP_INFO(this->get_logger(), "Built global map: %ld points", global_map_points_.rows());
}

void SlamNode::update_occupancy_grid(const slam::PointCloud::Matrix& cloud, const Eigen::Vector3d& sensor) {
    for (int i = 0; i < cloud.rows(); ++i) {
        double x = cloud(i,0), y = cloud(i,1), z = cloud(i,2);
        if (z < grid_config_.height_min || z > grid_config_.height_max) continue;
        double r = std::sqrt((x-sensor.x())*(x-sensor.x()) + (y-sensor.y())*(y-sensor.y()));
        if (r > grid_config_.max_range || r < 0.5) continue;
        GridCell cell{static_cast<int>(std::floor(x/grid_config_.resolution)), 
                      static_cast<int>(std::floor(y/grid_config_.resolution))};
        occupied_cells_.insert(cell);
    }
}

void SlamNode::rebuild_occupancy_grid() {
    occupied_cells_.clear();
    for (size_t i = 0; i < downsampled_clouds_.size() && i < poses_.size(); ++i) {
        slam::PointCloud::Matrix w = (downsampled_clouds_[i] * poses_[i].R().transpose()).rowwise() + poses_[i].t().transpose();
        update_occupancy_grid(w, poses_[i].t());
    }
}

void SlamNode::publish_current_scan(const slam::PointCloud::Matrix& cloud) {
    current_scan_pub_->publish(eigen_to_pointcloud2(cloud, "map"));
}

void SlamNode::publish_global_map() {
    if (processing_complete_) {
        auto ds = slam::voxel_downsample(global_map_points_, config_.voxel_size * 2);
        global_map_pub_->publish(eigen_to_pointcloud2(ds, "map"));
    } else if (!recent_clouds_world_.empty()) {
        size_t total = 0;
        for (const auto& c : recent_clouds_world_) total += c.rows();
        slam::PointCloud::Matrix m(total, 3);
        size_t row = 0;
        for (const auto& c : recent_clouds_world_) { m.middleRows(row, c.rows()) = c; row += c.rows(); }
        global_map_pub_->publish(eigen_to_pointcloud2(m, "map"));
    }
}

void SlamNode::publish_trajectory() {
    nav_msgs::msg::Path msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = "map";
    for (const auto& p : poses_) msg.poses.push_back(transformation_to_pose_stamped(p, "map"));
    trajectory_pub_->publish(msg);
}

void SlamNode::publish_current_pose() {
    if (poses_.empty()) return;
    current_pose_pub_->publish(transformation_to_pose_stamped(poses_.back(), "map"));
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = this->now();
    tf.header.frame_id = "map";
    tf.child_frame_id = "base_link";
    tf.transform.translation.x = poses_.back().t().x();
    tf.transform.translation.y = poses_.back().t().y();
    tf.transform.translation.z = poses_.back().t().z();
    Eigen::Quaterniond q(poses_.back().R());
    tf.transform.rotation.x = q.x();
    tf.transform.rotation.y = q.y();
    tf.transform.rotation.z = q.z();
    tf.transform.rotation.w = q.w();
    tf_broadcaster_->sendTransform(tf);
}

void SlamNode::publish_occupancy_grid() {
    occupancy_grid_pub_->publish(cells_to_occupancy_grid_msg());
}

nav_msgs::msg::OccupancyGrid SlamNode::cells_to_occupancy_grid_msg() {
    nav_msgs::msg::OccupancyGrid msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = "map";
    if (occupied_cells_.empty()) return msg;
    int minx=INT_MAX, maxx=INT_MIN, miny=INT_MAX, maxy=INT_MIN;
    for (const auto& c : occupied_cells_) { minx=std::min(minx,c.x); maxx=std::max(maxx,c.x); miny=std::min(miny,c.y); maxy=std::max(maxy,c.y); }
    minx-=5; miny-=5; maxx+=5; maxy+=5;
    int w = maxx-minx+1, h = maxy-miny+1;
    msg.info.resolution = grid_config_.resolution;
    msg.info.width = w;
    msg.info.height = h;
    msg.info.origin.position.x = minx * grid_config_.resolution;
    msg.info.origin.position.y = miny * grid_config_.resolution;
    msg.info.origin.orientation.w = 1.0;
    msg.data.resize(w*h, 0);
    for (const auto& c : occupied_cells_) msg.data[(c.y-miny)*w + (c.x-minx)] = 100;
    return msg;
}

sensor_msgs::msg::PointCloud2 SlamNode::eigen_to_pointcloud2(const slam::PointCloud::Matrix& cloud, const std::string& frame_id) {
    sensor_msgs::msg::PointCloud2 msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = frame_id;
    msg.height = 1;
    msg.width = cloud.rows();
    msg.is_dense = true;
    msg.is_bigendian = false;
    sensor_msgs::msg::PointField fx, fy, fz;
    fx.name="x"; fx.offset=0; fx.datatype=sensor_msgs::msg::PointField::FLOAT32; fx.count=1;
    fy.name="y"; fy.offset=4; fy.datatype=sensor_msgs::msg::PointField::FLOAT32; fy.count=1;
    fz.name="z"; fz.offset=8; fz.datatype=sensor_msgs::msg::PointField::FLOAT32; fz.count=1;
    msg.fields = {fx, fy, fz};
    msg.point_step = 12;
    msg.row_step = 12 * msg.width;
    msg.data.resize(msg.row_step);
    float* ptr = reinterpret_cast<float*>(msg.data.data());
    for (int i = 0; i < cloud.rows(); ++i) {
        ptr[i*3+0] = static_cast<float>(cloud(i,0));
        ptr[i*3+1] = static_cast<float>(cloud(i,1));
        ptr[i*3+2] = static_cast<float>(cloud(i,2));
    }
    return msg;
}

geometry_msgs::msg::PoseStamped SlamNode::transformation_to_pose_stamped(const slam::Transformation& t, const std::string& frame_id) {
    geometry_msgs::msg::PoseStamped msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = frame_id;
    msg.pose.position.x = t.t().x();
    msg.pose.position.y = t.t().y();
    msg.pose.position.z = t.t().z();
    Eigen::Quaterniond q(t.R());
    msg.pose.orientation.x = q.x();
    msg.pose.orientation.y = q.y();
    msg.pose.orientation.z = q.z();
    msg.pose.orientation.w = q.w();
    return msg;
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