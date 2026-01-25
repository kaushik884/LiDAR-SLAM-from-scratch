#pragma once

#include "types.hpp"
#include <vector>
#include <memory>

// Forward declarations (avoid including heavy GTSAM headers here)
namespace gtsam {
    class NonlinearFactorGraph;
    class Values;
    class Pose3;
}

namespace slam {

/**
 * Configuration for pose graph noise models.
 * 
 * These values represent standard deviations (sigmas) for each DOF.
 * Lower values = higher confidence = tighter constraints.
 */
struct PoseGraphConfig {
    // Odometry noise (from ICP)
    // Format: [roll, pitch, yaw, x, y, z]
    double odom_rotation_sigma = 0.01;    // radians (~0.5 degrees)
    double odom_translation_sigma = 0.05; // meters
    
    // Prior noise (anchor first pose)
    double prior_rotation_sigma = 0.001;  // very tight
    double prior_translation_sigma = 0.001;
    
    // Loop closure noise (typically tighter than odometry)
    double loop_rotation_sigma = 0.005;
    double loop_translation_sigma = 0.025;
    
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
    
    // Prevent copying (GTSAM objects are heavy)
    PoseGraph(const PoseGraph&) = delete;
    PoseGraph& operator=(const PoseGraph&) = delete;
    
    // Allow moving
    PoseGraph(PoseGraph&&) noexcept;
    PoseGraph& operator=(PoseGraph&&) noexcept;
    
    /**
     * Add a prior factor to anchor a pose.
     * Typically called once for pose 0 at the origin.
     */
    void addPrior(size_t index, const Transformation& pose);
    
    /**
     * Add an odometry factor between consecutive poses.
     * 
     * @param from_idx Source pose index
     * @param to_idx Target pose index
     * @param relative_transform Transform from 'from' to 'to'
     * @param fitness_score Optional ICP fitness (higher = more uncertain)
     */
    void addOdometryFactor(
        size_t from_idx,
        size_t to_idx,
        const Transformation& relative_transform,
        double fitness_score = 0.0
    );
    
    /**
     * Add a loop closure constraint.
     * 
     * @param from_idx Earlier pose index (the "seen before" location)
     * @param to_idx Later pose index (current pose that detected loop)
     * @param relative_transform Transform from 'from' to 'to'
     */
    void addLoopClosure(
        size_t from_idx,
        size_t to_idx,
        const Transformation& relative_transform
    );
    
    /**
     * Run optimization on all poses.
     * 
     * @return true if optimization converged
     */
    bool optimize();
    
    /**
     * Get optimized pose for a given index.
     * Must call optimize() first.
     */
    Transformation getPose(size_t index) const;
    
    /**
     * Get all optimized poses.
     */
    std::vector<Transformation> getAllPoses() const;
    
    /**
     * Get number of poses in the graph.
     */
    size_t size() const { return num_poses_; }
    
    /**
     * Get number of loop closure constraints.
     */
    size_t loopClosureCount() const { return num_loop_closures_; }
    
    /**
     * Get optimization statistics.
     */
    double getFinalError() const { return final_error_; }
    int getIterations() const { return iterations_; }

private:
    // Conversion helpers
    static gtsam::Pose3 toGtsamPose(const Transformation& t);
    static Transformation fromGtsamPose(const gtsam::Pose3& p);
    
    PoseGraphConfig config_;
    
    // GTSAM internals (pimpl pattern to avoid heavy headers)
    std::unique_ptr<gtsam::NonlinearFactorGraph> graph_;
    std::unique_ptr<gtsam::Values> initial_estimates_;
    std::unique_ptr<gtsam::Values> optimized_estimates_;
    
    size_t num_poses_ = 0;
    size_t num_loop_closures_ = 0;
    bool optimized_ = false;
    double final_error_ = 0.0;
    int iterations_ = 0;
};

}  // namespace slam