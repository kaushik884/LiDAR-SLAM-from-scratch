// File: cpp/include/pose_graph.hpp
#pragma once

#include <vector>
#include <memory>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include "icp/types.hpp"
#include "icp/loop_closure.hpp"

namespace icp {

/**
 * Configuration for pose graph noise models.
 * 
 * These values represent standard deviations (sigmas) for each DOF.
 * Lower values = higher confidence = tighter constraints.
 */
struct PoseGraphConfig {
    // Odometry noise (from ICP)
    // Format: [roll, pitch, yaw, x, y, z]
    double odom_rotation_sigma = 1e-6;    // radians (~0.5 degrees)
    double odom_translation_sigma = 1e-4; // meters
    
    // Prior noise (anchor first pose)
    double prior_rotation_sigma = 1e-6;  // very tight
    double prior_translation_sigma = 1e-6;
    
    // Loop closure noise (typically tighter than odometry)
    double loop_rotation_sigma = 1e-6;
    double loop_translation_sigma = 1e-4;
    
    // Optimizer settings
    int max_iterations = 100;
    double relative_error_tol = 1e-5;
    double absolute_error_tol = 1e-5;
};

/**
 * Pose graph optimizer using GTSAM.
 * 
 * Accumulates odometry factors from ICP and optimizes
 * all poses globally to reduce drift.
 */
class PoseGraph {
public:
    explicit PoseGraph(const PoseGraphConfig& config = PoseGraphConfig());
    ~PoseGraph();
    
    /**
     * Add a prior factor to anchor a pose.
     * Typically called once for pose 0 at the origin.
     * 
     * @param index Pose index (usually 0)
     * @param pose The pose to anchor (usually identity)
     */
    void addPrior(size_t index, const Transformation& pose);
    
    /**
     * Add an odometry factor between consecutive poses.
     * 
     * @param from_index Source pose index
     * @param to_index Target pose index (usually from_index + 1)
     * @param relative_pose The ICP result: transform from 'from' to 'to'
     * @param fitness Optional ICP fitness score to scale noise (lower = better)
     */
    void addOdometryFactor(
        size_t from_index,
        size_t to_index,
        const Transformation& relative_pose,
        double fitness = -1.0
    );
    /**
     * Add skip connection factors by running ICP between non-consecutive frames.
     * This adds constraints like 0->2, 2->4, etc.
     * 
     * @param clouds Vector of point clouds (one per frame)
     * @param skip_step Step size (2 means connect frame i to frame i+2)
     */
    void addSkipConnections(
        const std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>& clouds,
        int skip_step = 2
    );
    /**
     * Add a loop closure factor between non-consecutive poses.
     * 
     * @param from_index First pose index
     * @param to_index Second pose index
     * @param relative_pose Transform from 'from' to 'to'
     */
    void addLoopClosure(
        size_t from_index,
        size_t to_index,
        const Transformation& relative_pose
    );
    /**
     * Add a loop closure constraint between two frames.
     * 
     * @param from_frame Source frame index
     * @param to_frame Target frame index  
     * @param transform Relative transform from source to target
     * @param confidence Optional weight (lower = trust less)
     */
    void addLoopClosureEdge(
        int from_frame,
        int to_frame,
        const Transformation& transform,
        double confidence = 1.0
    ); 
    
    /**
     * Set initial estimate for a pose.
     * Called automatically by addOdometryFactor, but can be set manually.
     * 
     * @param index Pose index
     * @param pose Initial pose estimate
     */
    void setInitialEstimate(size_t index, const Transformation& pose);
    
    /**
     * Run optimization on the factor graph.
     * 
     * @return true if optimization converged
     */
    bool optimize();
    
    /**
     * Get optimized pose for a given index.
     * Must call optimize() first.
     * 
     * @param index Pose index
     * @return Optimized pose
     */
    Transformation getPose(size_t index) const;
    
    /**
     * Get all optimized poses.
     * 
     * @return Vector of optimized poses in order
     */
    std::vector<Transformation> getAllPoses() const;
    
    /**
     * Get number of poses in the graph.
     */
    size_t size() const { return num_poses_; }
    
    /**
     * Get optimization statistics.
     */
    double getFinalError() const { return final_error_; }
    int getIterations() const { return iterations_; }
    int getLoopClosureCount() const { return loop_closure_count_; }
    
private:
    // Convert between our Transformation type and GTSAM Pose3
    static gtsam::Pose3 toGtsamPose(const Transformation& t);
    static Transformation fromGtsamPose(const gtsam::Pose3& p);
    
    PoseGraphConfig config_;
    
    // GTSAM internals (using pimpl-like approach with unique_ptr)
    std::unique_ptr<gtsam::NonlinearFactorGraph> graph_;
    std::unique_ptr<gtsam::Values> initial_estimates_;
    std::unique_ptr<gtsam::Values> optimized_estimates_;
    
    size_t num_poses_ = 0;
    bool optimized_ = false;
    double final_error_ = 0.0;
    int iterations_ = 0;
    int loop_closure_count_ = 0;
};

}  // namespace icp