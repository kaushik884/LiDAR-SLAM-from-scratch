#pragma once

#include "types.hpp"
#include "scan_context.hpp"
#include "icp.hpp"
#include <vector>
#include <deque>

namespace slam {

/**
 * Configuration for loop closure detection.
 */
struct LoopClosureConfig {
    int frame_gap = 50;              // Minimum frames between loop candidates
    double sc_distance_threshold = 0.25;  // Scan context similarity threshold
    double icp_fitness_threshold = 0.3;   // ICP error threshold for verification
    int max_candidates = 3;          // Max candidates to verify per frame
};


/**
 * Result of a detected loop closure.
 */
struct LoopClosureResult {
    int query_frame;                 // Current frame index
    int match_frame;                 // Matched historical frame index
    Transformation transform;        // Relative transform from match to query
    double scan_context_distance;    // SC similarity score
    double icp_fitness;              // ICP error after verification
};


/**
 * Loop closure detector using Scan Context + ICP verification.
 * 
 * Two-stage pipeline:
 * 1. Scan Context: Fast place recognition to find candidates
 * 2. ICP: Geometric verification to confirm match and compute transform
 */
class LoopClosureDetector {
public:
    explicit LoopClosureDetector(const LoopClosureConfig& config = LoopClosureConfig())
        : config_(config) {}
    
    /**
     * Add a new frame to the database.
     * Call this for every processed frame.
     * 
     * @param cloud Downsampled point cloud
     * @param frame_idx Frame index
     */
    void addFrame(const PointCloud::Matrix& cloud, int frame_idx) {
        ScanContext sc(cloud);
        descriptors_.push_back(sc);
        clouds_.push_back(cloud);
        frame_indices_.push_back(frame_idx);
        latest_frame_idx_ = frame_idx;
    }
    
    /**
     * Detect loop closures for the most recently added frame.
     * 
     * @return Vector of detected loop closures (may be empty)
     */
    std::vector<LoopClosureResult> detect() {
        std::vector<LoopClosureResult> results;
        
        if (descriptors_.size() < 2) return results;
        
        int query_idx = descriptors_.size() - 1;
        const ScanContext& query_sc = descriptors_.back();
        const PointCloud::Matrix& query_cloud = clouds_.back();
        
        // Find candidates using Scan Context
        std::vector<std::pair<double, int>> candidates;
        
        for (size_t i = 0; i < descriptors_.size() - 1; ++i) {
            // Check frame gap
            int frame_diff = frame_indices_[query_idx] - frame_indices_[i];
            if (frame_diff < config_.frame_gap) continue;
            
            // Compute SC distance
            double dist = query_sc.distance(descriptors_[i]);
            
            if (dist < config_.sc_distance_threshold) {
                candidates.push_back({dist, static_cast<int>(i)});
            }
        }
        
        // Sort by distance (best first)
        std::sort(candidates.begin(), candidates.end());
        
        // Verify top candidates with ICP
        int verified = 0;
        for (const auto& [sc_dist, candidate_idx] : candidates) {
            if (verified >= config_.max_candidates) break;
            
            const PointCloud::Matrix& candidate_cloud = clouds_[candidate_idx];
            
            // Run ICP for verification
            PointCloud source(query_cloud);
            PointCloud target(candidate_cloud);
            
            ICPConfig icp_config;
            icp_config.max_iterations = 30;
            icp_config.tolerance = 1e-6;
            
            ICPResult icp_result = icp_point_to_plane(source, target, icp_config);
            
            // Check ICP fitness
            if (icp_result.converged && icp_result.final_error < config_.icp_fitness_threshold) {
                LoopClosureResult lc;
                lc.query_frame = frame_indices_[query_idx];
                lc.match_frame = frame_indices_[candidate_idx];
                lc.transform = icp_result.transformation;
                lc.scan_context_distance = sc_dist;
                lc.icp_fitness = icp_result.final_error;
                
                results.push_back(lc);
                verified++;
            }
        }
        
        return results;
    }
    
    /**
     * Get number of frames in database.
     */
    size_t size() const { return descriptors_.size(); }
    
    /**
     * Clear all stored data.
     */
    void clear() {
        descriptors_.clear();
        clouds_.clear();
        frame_indices_.clear();
        latest_frame_idx_ = -1;
    }

private:
    LoopClosureConfig config_;
    std::vector<ScanContext> descriptors_;
    std::vector<PointCloud::Matrix> clouds_;
    std::vector<int> frame_indices_;
    int latest_frame_idx_ = -1;
};

}  // namespace slam