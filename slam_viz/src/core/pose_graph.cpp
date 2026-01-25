#include "slam_viz/core/pose_graph.hpp"

#include <iostream>
#include <stdexcept>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/inference/Symbol.h>

namespace slam {

// Use 'X' as the character for pose symbols: X(0), X(1), X(2), ...
using gtsam::symbol_shorthand::X;

// ============================================================================
// Constructor / Destructor
// ============================================================================

PoseGraph::PoseGraph(const PoseGraphConfig& config)
    : config_(config)
    , graph_(std::make_unique<gtsam::NonlinearFactorGraph>())
    , initial_estimates_(std::make_unique<gtsam::Values>())
    , optimized_estimates_(std::make_unique<gtsam::Values>())
{
}

PoseGraph::~PoseGraph() = default;

PoseGraph::PoseGraph(PoseGraph&&) noexcept = default;
PoseGraph& PoseGraph::operator=(PoseGraph&&) noexcept = default;

// ============================================================================
// Conversion helpers
// ============================================================================

gtsam::Pose3 PoseGraph::toGtsamPose(const Transformation& t) {
    gtsam::Rot3 rotation(t.R());
    gtsam::Point3 translation(t.t());
    return gtsam::Pose3(rotation, translation);
}

Transformation PoseGraph::fromGtsamPose(const gtsam::Pose3& p) {
    Eigen::Matrix3d R = p.rotation().matrix();
    Eigen::Vector3d t = p.translation();
    return Transformation::from_rt(R, t);
}

// ============================================================================
// Factor addition
// ============================================================================

void PoseGraph::addPrior(size_t index, const Transformation& pose) {
    // Create noise model for prior (very tight)
    auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 
            config_.prior_rotation_sigma,
            config_.prior_rotation_sigma,
            config_.prior_rotation_sigma,
            config_.prior_translation_sigma,
            config_.prior_translation_sigma,
            config_.prior_translation_sigma
        ).finished()
    );
    
    gtsam::Pose3 gtsam_pose = toGtsamPose(pose);
    graph_->addPrior(X(index), gtsam_pose, prior_noise);
    
    // Also add to initial estimates if not present
    if (!initial_estimates_->exists(X(index))) {
        initial_estimates_->insert(X(index), gtsam_pose);
        num_poses_ = std::max(num_poses_, index + 1);
    }
}

void PoseGraph::addOdometryFactor(
    size_t from_idx,
    size_t to_idx,
    const Transformation& relative_transform,
    double fitness_score
) {
    // Scale noise based on fitness score (higher error = more uncertainty)
    double scale = 1.0 + fitness_score * 10.0;  // Heuristic scaling
    
    auto odom_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 
            config_.odom_rotation_sigma * scale,
            config_.odom_rotation_sigma * scale,
            config_.odom_rotation_sigma * scale,
            config_.odom_translation_sigma * scale,
            config_.odom_translation_sigma * scale,
            config_.odom_translation_sigma * scale
        ).finished()
    );
    
    gtsam::Pose3 relative_pose = toGtsamPose(relative_transform);
    graph_->emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        X(from_idx), X(to_idx), relative_pose, odom_noise
    );
    
    // Add initial estimate for 'to' pose if not present
    if (!initial_estimates_->exists(X(to_idx))) {
        // Propagate from 'from' pose
        gtsam::Pose3 from_pose = initial_estimates_->at<gtsam::Pose3>(X(from_idx));
        gtsam::Pose3 to_pose = from_pose * relative_pose;
        initial_estimates_->insert(X(to_idx), to_pose);
        num_poses_ = std::max(num_poses_, to_idx + 1);
    }
    
    optimized_ = false;  // Need to re-optimize
}

void PoseGraph::addLoopClosure(
    size_t from_idx,
    size_t to_idx,
    const Transformation& relative_transform
) {
    auto loop_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 
            config_.loop_rotation_sigma,
            config_.loop_rotation_sigma,
            config_.loop_rotation_sigma,
            config_.loop_translation_sigma,
            config_.loop_translation_sigma,
            config_.loop_translation_sigma
        ).finished()
    );
    
    gtsam::Pose3 relative_pose = toGtsamPose(relative_transform);
    graph_->emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        X(from_idx), X(to_idx), relative_pose, loop_noise
    );
    
    num_loop_closures_++;
    optimized_ = false;
}

// ============================================================================
// Optimization
// ============================================================================

bool PoseGraph::optimize() {
    if (num_poses_ == 0) {
        return false;
    }
    
    try {
        gtsam::LevenbergMarquardtParams params;
        params.setVerbosityLM("SILENT");
        params.setMaxIterations(config_.max_iterations);
        params.setRelativeErrorTol(config_.relative_error_tol);
        params.setAbsoluteErrorTol(config_.absolute_error_tol);
        
        gtsam::LevenbergMarquardtOptimizer optimizer(*graph_, *initial_estimates_, params);
        *optimized_estimates_ = optimizer.optimize();
        
        final_error_ = optimizer.error();
        iterations_ = optimizer.iterations();
        optimized_ = true;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Pose graph optimization failed: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// Pose retrieval
// ============================================================================

Transformation PoseGraph::getPose(size_t index) const {
    const gtsam::Values& values = optimized_ ? *optimized_estimates_ : *initial_estimates_;
    
    if (!values.exists(X(index))) {
        throw std::out_of_range("Pose index " + std::to_string(index) + " not found");
    }
    
    return fromGtsamPose(values.at<gtsam::Pose3>(X(index)));
}

std::vector<Transformation> PoseGraph::getAllPoses() const {
    std::vector<Transformation> poses;
    poses.reserve(num_poses_);
    
    const gtsam::Values& values = optimized_ ? *optimized_estimates_ : *initial_estimates_;
    
    for (size_t i = 0; i < num_poses_; ++i) {
        if (values.exists(X(i))) {
            poses.push_back(fromGtsamPose(values.at<gtsam::Pose3>(X(i))));
        }
    }
    
    return poses;
}

}  // namespace slam