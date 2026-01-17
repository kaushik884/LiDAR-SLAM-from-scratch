#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>

namespace icp {

/**
 * Configuration for Scan Context descriptor.
 */
struct ScanContextConfig {
    int num_rings = 20;           // Radial divisions
    int num_sectors = 60;         // Angular divisions  
    double max_range = 80.0;      // Maximum range to consider (meters)
    double min_range = 2.0;       // Ignore points closer than this
    double min_height = -2.0;     // Ignore points below this z
    double max_height = 30.0;     // Ignore points above this z
};

/**
 * Scan Context descriptor for place recognition.
 * 
 * Encodes a point cloud as a 2D matrix (rings x sectors) where each
 * cell contains the maximum point height in that polar bin.
 */
class ScanContext {
public:
    using Descriptor = Eigen::MatrixXd;  // (num_rings x num_sectors)
    
    explicit ScanContext(const ScanContextConfig& config = ScanContextConfig{})
        : config_(config) {}
    
    /**
     * Build descriptor from point cloud.
     * 
     * @param points Nx3 matrix of points (x, y, z)
     * @return Descriptor matrix (rings x sectors)
     */
    Descriptor build(const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& points) const {
        Descriptor desc = Descriptor::Zero(config_.num_rings, config_.num_sectors);
        
        const double ring_width = config_.max_range / config_.num_rings;
        const double sector_width = 2.0 * M_PI / config_.num_sectors;
        
        for (int i = 0; i < points.rows(); ++i) {
            const double x = points(i, 0);
            const double y = points(i, 1);
            const double z = points(i, 2);
            
            // Height filter
            if (z < config_.min_height || z > config_.max_height) {
                continue;
            }
            
            // Compute range and angle
            const double range = std::sqrt(x * x + y * y);
            
            // Range filter
            if (range < config_.min_range || range >= config_.max_range) {
                continue;
            }
            
            // Angle in [0, 2*pi)
            double angle = std::atan2(y, x);
            if (angle < 0) {
                angle += 2.0 * M_PI;
            }
            
            // Compute bin indices
            int ring_idx = static_cast<int>(range / ring_width);
            int sector_idx = static_cast<int>(angle / sector_width);
            
            // Clamp to valid range (handles edge cases)
            ring_idx = std::min(ring_idx, config_.num_rings - 1);
            sector_idx = std::min(sector_idx, config_.num_sectors - 1);
            
            // Update with max height
            desc(ring_idx, sector_idx) = std::max(desc(ring_idx, sector_idx), z);
        }
        
        return desc;
    }
    
    /**
     * Compare two descriptors with rotation alignment.
     * 
     * Tries all column shifts to find best alignment, handling
     * the case where scans are from the same place but different orientations.
     * 
     * @param desc1 First descriptor
     * @param desc2 Second descriptor
     * @return Distance in [0, 1], lower means more similar
     */
    double compare(const Descriptor& desc1, const Descriptor& desc2) const {
        double min_distance = std::numeric_limits<double>::max();
        
        // Try all column shifts
        for (int shift = 0; shift < config_.num_sectors; ++shift) {
            double dist = compute_distance_at_shift(desc1, desc2, shift);
            min_distance = std::min(min_distance, dist);
        }
        
        return min_distance;
    }
    
    /**
     * Fast candidate search using ring key.
     * 
     * The ring key is the mean height per ring (1D vector).
     * Used for fast pre-filtering before full comparison.
     */
    Eigen::VectorXd compute_ring_key(const Descriptor& desc) const {
        Eigen::VectorXd key(config_.num_rings);
        for (int r = 0; r < config_.num_rings; ++r) {
            double sum = 0;
            int count = 0;
            for (int s = 0; s < config_.num_sectors; ++s) {
                if (desc(r, s) > 0) {
                    sum += desc(r, s);
                    count++;
                }
            }
            key(r) = (count > 0) ? sum / count : 0;
        }
        return key;
    }
    
    const ScanContextConfig& config() const { return config_; }

private:
    ScanContextConfig config_;
    
    /**
     * Compute distance between descriptors at a specific column shift.
     */
    double compute_distance_at_shift(
        const Descriptor& desc1, 
        const Descriptor& desc2, 
        int shift
    ) const {
        double total_distance = 0;
        int valid_columns = 0;
        
        for (int s = 0; s < config_.num_sectors; ++s) {
            int s_shifted = (s + shift) % config_.num_sectors;
            
            Eigen::VectorXd col1 = desc1.col(s);
            Eigen::VectorXd col2 = desc2.col(s_shifted);
            
            // Skip if either column is empty
            double norm1 = col1.norm();
            double norm2 = col2.norm();
            
            if (norm1 < 1e-6 || norm2 < 1e-6) {
                continue;
            }
            
            // Cosine distance: 1 - cos(theta)
            double cosine_sim = col1.dot(col2) / (norm1 * norm2);
            double cosine_dist = 1.0 - cosine_sim;
            
            total_distance += cosine_dist;
            valid_columns++;
        }
        
        if (valid_columns == 0) {
            return 1.0;  // Maximum distance if no valid columns
        }
        
        return total_distance / valid_columns;
    }
};

}  // namespace icp