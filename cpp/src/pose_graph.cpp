// File: cpp/src/pose_graph.cpp

#include "pose_graph.hpp"
#include "icp/loop_closure.hpp"
#include "icp/point_to_point.hpp"
#include <iostream>
#include <stdexcept>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/inference/Symbol.h>

namespace icp {

// We use 'X' as the character for pose symbols: X(0), X(1), X(2), ...
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

// ============================================================================
// Conversion helpers
// ============================================================================

gtsam::Pose3 PoseGraph::toGtsamPose(const Transformation& t) {
    // GTSAM Rot3 can be constructed directly from a 3x3 rotation matrix
    gtsam::Rot3 rotation(t.R());
    
    // GTSAM Point3 is the translation vector
    gtsam::Point3 translation(t.t());
    
    return gtsam::Pose3(rotation, translation);
}

Transformation PoseGraph::fromGtsamPose(const gtsam::Pose3& p) {
    // Extract rotation matrix and translation vector
    Eigen::Matrix3d R = p.rotation().matrix();
    Eigen::Vector3d t = p.translation();
    
    return Transformation::from_rt(R, t);
}

// ============================================================================
// Factor graph construction
// ============================================================================

void PoseGraph::addPrior(size_t index, const Transformation& pose) {
    // Create noise model for prior (very tight - we're confident about starting pose)
    // Order is: rotation (rx, ry, rz), translation (tx, ty, tz)
    auto noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 
            config_.prior_rotation_sigma,
            config_.prior_rotation_sigma,
            config_.prior_rotation_sigma,
            config_.prior_translation_sigma,
            config_.prior_translation_sigma,
            config_.prior_translation_sigma
        ).finished()
    );
    
    // Add the prior factor
    graph_->addPrior(X(index), toGtsamPose(pose), noise);
    
    // Also set as initial estimate if not already set
    if (!initial_estimates_->exists(X(index))) {
        initial_estimates_->insert(X(index), toGtsamPose(pose));
        num_poses_ = std::max(num_poses_, index + 1);
    }
    
    optimized_ = false;
}
void PoseGraph::addSkipConnections(
    const std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>& clouds,
    int skip_step
) {
    std::cout << "Adding skip connections (step=" << skip_step << ")...\n";
    
    int connections_added = 0;
    int connections_failed = 0;
    
    // Create noise model for skip connections
    auto noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 
            config_.loop_rotation_sigma,
            config_.loop_rotation_sigma,
            config_.loop_rotation_sigma,
            config_.loop_translation_sigma,
            config_.loop_translation_sigma,
            config_.loop_translation_sigma
        ).finished()
    );
    
    for (size_t i = 0; i + skip_step < clouds.size(); i += skip_step) {
        size_t j = i + skip_step;
        
        gtsam::Pose3 pose_i = initial_estimates_->at<gtsam::Pose3>(X(i));
        gtsam::Pose3 pose_j = initial_estimates_->at<gtsam::Pose3>(X(j));
        gtsam::Pose3 relative_guess = pose_i.inverse() * pose_j;
        // Create point clouds for ICP
        icp::PointCloud source(clouds[i]);
        icp::PointCloud target(clouds[j]);
        
        // Run ICP
        icp::ICPConfig icp_config;
        icp_config.max_iterations = 50;
        icp_config.tolerance = 1e-6;
        icp_config.initial_transform = fromGtsamPose(relative_guess);
        
        icp::ICPResult result = icp::icp_point_to_point(source, target, icp_config);
        // // Debug output
        // Eigen::Vector3d guess_t = fromGtsamPose(relative_guess).t();
        // Eigen::Vector3d result_t = result.transformation.t();
        // std::cout << "  Skip " << i << "->" << j 
        //         << ": guess=(" << guess_t.x() << "," << guess_t.y() << "," << guess_t.z() << ")"
        //         << " result=(" << result_t.x() << "," << result_t.y() << "," << result_t.z() << ")"
        //         << " error=" << result.final_error << "\n";
        if (result.converged && result.final_error < 1.0) {
            // Add factor: constraint from pose i to pose j
            // Use inverse to match GTSAM convention
            graph_->emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                X(i),
                X(j),
                toGtsamPose(result.transformation.inverse()),
                noise
            );
            connections_added++;
        } else {
            connections_failed++;
        }
    }
    
    std::cout << "  Added: " << connections_added << " skip connections\n";
    std::cout << "  Failed: " << connections_failed << "\n";
    
    optimized_ = false;
}
void PoseGraph::addOdometryFactor(
    size_t from_index,
    size_t to_index,
    const Transformation& relative_pose,
    double fitness
) {
    // Scale noise based on ICP fitness if provided
    // Higher fitness (worse ICP) = larger sigma = less trust
    double rotation_sigma = config_.odom_rotation_sigma;
    double translation_sigma = config_.odom_translation_sigma;
    
    // Create noise model
    auto noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 
            rotation_sigma,
            rotation_sigma,
            rotation_sigma,
            translation_sigma,
            translation_sigma,
            translation_sigma
        ).finished()
    );
    
    // Add between factor: constraint between pose[from] and pose[to]
    graph_->emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        X(from_index),
        X(to_index),
        toGtsamPose(relative_pose),
        noise
    );
    
    // Update initial estimate for 'to' pose based on 'from' pose and relative transform
    if (initial_estimates_->exists(X(from_index)) && !initial_estimates_->exists(X(to_index))) {
        gtsam::Pose3 from_pose = initial_estimates_->at<gtsam::Pose3>(X(from_index));
        gtsam::Pose3 to_pose = from_pose * toGtsamPose(relative_pose);
        initial_estimates_->insert(X(to_index), to_pose);
    }
    
    num_poses_ = std::max(num_poses_, std::max(from_index, to_index) + 1);
    optimized_ = false;
}

void PoseGraph::addLoopClosure(
    size_t from_index,
    size_t to_index,
    const Transformation& relative_pose
) {
    // Loop closures typically have tighter noise than odometry
    // because we've verified this is a good match
    auto noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 
            config_.loop_rotation_sigma,
            config_.loop_rotation_sigma,
            config_.loop_rotation_sigma,
            config_.loop_translation_sigma,
            config_.loop_translation_sigma,
            config_.loop_translation_sigma
        ).finished()
    );
    
    // Add between factor
    graph_->emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        X(from_index),
        X(to_index),
        toGtsamPose(relative_pose),
        noise
    );
    
    optimized_ = false;
}

void PoseGraph::addLoopClosureEdge(
    int from_frame,
    int to_frame,
    const Transformation& transform,
    double confidence
) {
    // The noise model can be tuned - loop closures are often
    // less certain than odometry, so we use higher variance
    auto noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 
            0.1 * confidence,   // roll
            0.1 * confidence,   // pitch
            0.1 * confidence,   // yaw
            0.2 * confidence,   // x
            0.2 * confidence,   // y
            0.1 * confidence    // z
        ).finished()
    );
    
    // Add between factor
    graph_->emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        X(from_frame),
        X(to_frame),
        toGtsamPose(transform),
        noise
    );
    
    // loop_closure_count_++;
}
void PoseGraph::setInitialEstimate(size_t index, const Transformation& pose) {
    gtsam::Pose3 gtsam_pose = toGtsamPose(pose);
    
    if (initial_estimates_->exists(X(index))) {
        initial_estimates_->update(X(index), gtsam_pose);
    } else {
        initial_estimates_->insert(X(index), gtsam_pose);
    }
    
    num_poses_ = std::max(num_poses_, index + 1);
    optimized_ = false;
}

// ============================================================================
// Optimization
// ============================================================================

bool PoseGraph::optimize() {
    if (num_poses_ == 0) {
        std::cerr << "PoseGraph::optimize() - No poses to optimize\n";
        return false;
    }
    
    // Configure optimizer
    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = config_.max_iterations;
    params.relativeErrorTol = config_.relative_error_tol;
    params.absoluteErrorTol = config_.absolute_error_tol;
    params.verbosityLM = gtsam::LevenbergMarquardtParams::SILENT;
    
    try {
        // Run optimization
        gtsam::LevenbergMarquardtOptimizer optimizer(*graph_, *initial_estimates_, params);
        *optimized_estimates_ = optimizer.optimize();
        
        // Store statistics
        final_error_ = graph_->error(*optimized_estimates_);
        iterations_ = optimizer.iterations();
        optimized_ = true;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "PoseGraph::optimize() failed: " << e.what() << "\n";
        return false;
    }
}

// ============================================================================
// Pose retrieval
// ============================================================================

Transformation PoseGraph::getPose(size_t index) const {
    // Use optimized if available, otherwise initial
    const gtsam::Values& values = optimized_ ? *optimized_estimates_ : *initial_estimates_;
    
    if (!values.exists(X(index))) {
        throw std::out_of_range("PoseGraph::getPose() - Index " + std::to_string(index) + " not found");
    }
    
    return fromGtsamPose(values.at<gtsam::Pose3>(X(index)));
}

std::vector<Transformation> PoseGraph::getAllPoses() const {
    std::vector<Transformation> poses;
    poses.reserve(num_poses_);
    
    for (size_t i = 0; i < num_poses_; ++i) {
        poses.push_back(getPose(i));
    }
    
    return poses;
}

}  // namespace icp