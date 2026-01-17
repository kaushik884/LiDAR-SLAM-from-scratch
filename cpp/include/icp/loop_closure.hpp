#pragma once

#include "scan_context.hpp"
#include "point_to_plane.hpp"
#include <vector>
#include <optional>

namespace icp {

/**
 * Result of a detected loop closure.
 */
struct LoopClosureResult {
    int query_frame;              // The current frame
    int match_frame;              // The matched previous frame
    double scan_context_distance; // Descriptor similarity (lower = better)
    double icp_fitness;           // ICP alignment error (lower = better)
    Transformation transform;     // Transform from query to match frame
};

/**
 * Configuration for loop closure detection.
 */
struct LoopClosureConfig {
    // Scan context settings
    ScanContextConfig sc_config;
    
    // Search parameters
    int frame_gap = 50;                    // Minimum frame gap to consider
    double sc_distance_threshold = 0.25;  // Max scan context distance for candidates
    double ring_key_threshold = 0.15;     // Pre-filter threshold for ring keys
    int max_candidates = 5;               // Max candidates to verify per frame
    
    // ICP verification
    double icp_fitness_threshold = 0.3;   // Max ICP error to accept loop
    int icp_max_iterations = 30;          // ICP iterations for verification
};

/**
 * Loop closure detection using Scan Context.
 * 
 * Maintains a database of scan context descriptors and detects
 * when the robot revisits a previous location.
 */
class LoopClosureDetector {
public:
    explicit LoopClosureDetector(const LoopClosureConfig& config = LoopClosureConfig{})
        : config_(config), scan_context_(config.sc_config) {}
    
    /**
     * Add a new frame to the database.
     * 
     * Call this for every frame, even if you don't search for loops.
     * 
     * @param points Point cloud for this frame
     * @param frame_id Unique identifier (typically frame index)
     */
    void addFrame(
        const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& points,
        int frame_id
    ) {
        FrameData data;
        data.frame_id = frame_id;
        data.descriptor = scan_context_.build(points);
        data.ring_key = scan_context_.compute_ring_key(data.descriptor);
        data.points = points;  // Store for ICP verification
        
        frames_.push_back(std::move(data));
    }
    
    /**
     * Detect loop closures for the most recent frame.
     * 
     * Searches all frames older than frame_gap, finds candidates
     * by scan context similarity, and verifies with ICP.
     * 
     * @return Vector of verified loop closures (may be empty)
     */
    std::vector<LoopClosureResult> detect() {
        std::vector<LoopClosureResult> results;
        
        if (frames_.size() < static_cast<size_t>(config_.frame_gap + 1)) {
            return results;  // Not enough frames yet
        }
        
        const FrameData& query = frames_.back();
        int query_idx = static_cast<int>(frames_.size()) - 1;
        
        // Find candidates using ring key pre-filtering
        std::vector<std::pair<int, double>> candidates;
        candidates = findCandidates(query, query_idx);
        
        // Verify each candidate with ICP
        for (const auto& [candidate_idx, sc_distance] : candidates) {
            auto verification = verifyWithICP(query, frames_[candidate_idx]);
            
            if (verification.has_value()) {
                LoopClosureResult result;
                result.query_frame = query.frame_id;
                result.match_frame = frames_[candidate_idx].frame_id;
                result.scan_context_distance = sc_distance;
                result.icp_fitness = verification->first;
                result.transform = verification->second;
                
                results.push_back(result);
            }
        }
        
        return results;
    }
    
    /**
     * Get number of frames in database.
     */
    size_t size() const { return frames_.size(); }
    
    const LoopClosureConfig& config() const { return config_; }

private:
    struct FrameData {
        int frame_id;
        ScanContext::Descriptor descriptor;
        Eigen::VectorXd ring_key;
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> points;
    };
    
    LoopClosureConfig config_;
    ScanContext scan_context_;
    std::vector<FrameData> frames_;
    
    /**
     * Find candidate matches using two-stage search.
     */
    std::vector<std::pair<int, double>> findCandidates(
        const FrameData& query,
        int query_idx
    ) {
        std::vector<std::pair<int, double>> candidates;
        
        int search_end = query_idx - config_.frame_gap;
        
        for (int i = 0; i < search_end; ++i) {
            // Stage 1: Ring key pre-filter
            double ring_distance = computeRingKeyDistance(
                query.ring_key, 
                frames_[i].ring_key
            );
            
            if (ring_distance > config_.ring_key_threshold) {
                continue;  // Skip, not similar enough
            }
            
            // Stage 2: Full scan context comparison
            double sc_distance = scan_context_.compare(
                query.descriptor, 
                frames_[i].descriptor
            );
            
            if (sc_distance < config_.sc_distance_threshold) {
                candidates.emplace_back(i, sc_distance);
            }
        }
        
        // Sort by distance and keep top candidates
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        
        if (candidates.size() > static_cast<size_t>(config_.max_candidates)) {
            candidates.resize(config_.max_candidates);
        }
        
        return candidates;
    }
    
    /**
     * Compute L1 distance between ring keys.
     */
    double computeRingKeyDistance(
        const Eigen::VectorXd& key1,
        const Eigen::VectorXd& key2
    ) {
        return (key1 - key2).lpNorm<1>() / key1.size();
    }
    
    /**
     * Verify candidate match using ICP.
     * 
     * @return Pair of (fitness score, transform) if verified, nullopt otherwise
     */
    std::optional<std::pair<double, Transformation>> verifyWithICP(
        const FrameData& query,
        const FrameData& candidate
    ) {
        // Configure ICP for verification
        ICPConfig icp_config;
        icp_config.max_iterations = config_.icp_max_iterations;
        icp_config.tolerance = 1e-4;
        
        // Run point-to-plane ICP
        ICPResult result = icp_point_to_plane(
            PointCloud(query.points),
            PointCloud(candidate.points),
            icp_config
        );
        
        // Check if alignment is good enough
        if (result.final_error < config_.icp_fitness_threshold && result.converged) {
            return std::make_pair(result.final_error, result.transformation);
        }
        
        return std::nullopt;
    }
};

}  // namespace icp