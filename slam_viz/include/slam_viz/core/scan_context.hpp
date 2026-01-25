#pragma once

#include "types.hpp"
#include <Eigen/Dense>
#include <vector>
#include <cmath>

namespace slam {

/**
 * Scan Context descriptor for place recognition.
 * 
 * Converts a 3D point cloud into a 2D descriptor by:
 * 1. Project points to polar coordinates (range, azimuth)
 * 2. Divide into ring (range) and sector (angle) bins
 * 3. Store max height in each bin
 * 
 * This creates a rotation-invariant-ish descriptor that can be
 * efficiently compared to find similar places.
 * 
 * Reference: "Scan Context: Egocentric Spatial Descriptor for 
 *            Place Recognition within 3D Point Cloud Map"
 */
class ScanContext {
public:
    // Default parameters (tuned for outdoor LiDAR)
    static constexpr int NUM_RINGS = 20;      // Radial divisions
    static constexpr int NUM_SECTORS = 60;    // Angular divisions
    static constexpr double MAX_RANGE = 80.0; // Maximum range in meters
    
    ScanContext() : descriptor_(NUM_RINGS, NUM_SECTORS) {
        descriptor_.setZero();
    }
    
    explicit ScanContext(const PointCloud::Matrix& cloud) 
        : descriptor_(NUM_RINGS, NUM_SECTORS) 
    {
        compute(cloud);
    }
    
    /**
     * Compute descriptor from point cloud.
     */
    void compute(const PointCloud::Matrix& cloud) {
        descriptor_.setConstant(-std::numeric_limits<double>::max());
        
        double ring_size = MAX_RANGE / NUM_RINGS;
        double sector_size = 2.0 * M_PI / NUM_SECTORS;
        
        for (int i = 0; i < cloud.rows(); ++i) {
            double x = cloud(i, 0);
            double y = cloud(i, 1);
            double z = cloud(i, 2);
            
            // Compute range and angle
            double range = std::sqrt(x * x + y * y);
            double angle = std::atan2(y, x) + M_PI;  // [0, 2π]
            
            if (range > MAX_RANGE || range < 0.1) continue;
            
            // Compute bin indices
            int ring_idx = static_cast<int>(range / ring_size);
            int sector_idx = static_cast<int>(angle / sector_size);
            
            ring_idx = std::clamp(ring_idx, 0, NUM_RINGS - 1);
            sector_idx = std::clamp(sector_idx, 0, NUM_SECTORS - 1);
            
            // Store max height in bin
            if (z > descriptor_(ring_idx, sector_idx)) {
                descriptor_(ring_idx, sector_idx) = z;
            }
        }
        
        // Replace -inf with 0 for empty bins
        for (int i = 0; i < NUM_RINGS; ++i) {
            for (int j = 0; j < NUM_SECTORS; ++j) {
                if (descriptor_(i, j) < -1000) {
                    descriptor_(i, j) = 0;
                }
            }
        }
    }
    
    /**
     * Compute distance to another scan context.
     * Uses column-shifted cosine distance for rotation invariance.
     * 
     * Returns value in [0, 1] where 0 = identical, 1 = completely different.
     */
    double distance(const ScanContext& other) const {
        double min_distance = std::numeric_limits<double>::max();
        
        // Try all column shifts (rotation invariance)
        for (int shift = 0; shift < NUM_SECTORS; ++shift) {
            double dist = column_shifted_distance(other, shift);
            if (dist < min_distance) {
                min_distance = dist;
            }
        }
        
        return min_distance;
    }
    
    /**
     * Get the ring key (column-wise mean) for fast filtering.
     */
    Eigen::VectorXd ring_key() const {
        return descriptor_.rowwise().mean();
    }
    
    /**
     * Get the sector key (row-wise mean) for fast filtering.
     */
    Eigen::VectorXd sector_key() const {
        return descriptor_.colwise().mean().transpose();
    }
    
    const Eigen::MatrixXd& descriptor() const { return descriptor_; }

private:
    double column_shifted_distance(const ScanContext& other, int shift) const {
        double sum_ab = 0;
        double sum_aa = 0;
        double sum_bb = 0;
        
        for (int i = 0; i < NUM_RINGS; ++i) {
            for (int j = 0; j < NUM_SECTORS; ++j) {
                double a = descriptor_(i, j);
                double b = other.descriptor_(i, (j + shift) % NUM_SECTORS);
                
                sum_ab += a * b;
                sum_aa += a * a;
                sum_bb += b * b;
            }
        }
        
        double norm = std::sqrt(sum_aa) * std::sqrt(sum_bb);
        if (norm < 1e-10) return 1.0;
        
        double cosine_sim = sum_ab / norm;
        return 1.0 - cosine_sim;  // Convert similarity to distance
    }
    
    Eigen::MatrixXd descriptor_;  // NUM_RINGS x NUM_SECTORS
};

}  // namespace slam