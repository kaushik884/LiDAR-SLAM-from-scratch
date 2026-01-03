#pragma once

#include "types.hpp"
#include "kdtree.hpp"
#include "svd.hpp"
#include <cmath>

namespace icp {

/**
 * Point-to-Point ICP configuration.
 */
struct ICPConfig {
    int max_iterations = 50;
    double tolerance = 1e-6;      // Convergence threshold for error change
    double min_error = 1e-9;      // Stop if error falls below this
};


/**
 * Point-to-Point ICP registration.
 * 
 * Iteratively finds correspondences and computes optimal rigid
 * transformation to align source to target.
 * 
 * @param source Source point cloud to align
 * @param target Target point cloud (reference)
 * @param config Algorithm parameters
 * @return ICPResult with transformation and convergence info
 */
inline ICPResult icp_point_to_point(
    const PointCloud& source,
    const PointCloud& target,
    const ICPConfig& config = ICPConfig{}
) {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

    // Build KD-tree on target for fast nearest neighbor
    NearestNeighborSearch nn_search(target.points());

    // Work on a copy so we don't modify original
    Matrix current_source = source.points();

    // Accumulated transformation (source -> target)
    Transformation total_transform = Transformation::identity();

    ICPResult result;
    double prev_error = std::numeric_limits<double>::infinity();

    for (int iter = 0; iter < config.max_iterations; ++iter) {
        // Step 1: Find correspondences
        Matrix target_matched;
        Eigen::VectorXd distances;
        nn_search.find_correspondences(current_source, target_matched, distances);

        // Step 2: Compute error
        double error = std::sqrt(distances.array().square().mean());
        result.error_history.push_back(error);

        // Step 3: Check convergence
        double error_change = std::abs(prev_error - error);

        if (error < config.min_error) {
            result.converged = true;
            break;
        }

        if (error_change < config.tolerance) {
            result.converged = true;
            break;
        }

        // Step 4: Estimate transformation from correspondences
        Transformation delta_transform = estimate_transformation(current_source, target_matched);

        // Step 5: Apply to current source
        current_source = (current_source * delta_transform.R().transpose()).rowwise() 
                        + delta_transform.t().transpose();

        // Step 6: Accumulate transformation
        total_transform = delta_transform * total_transform;

        prev_error = error;
    }

    // Final error after last transform
    Matrix target_matched;
    Eigen::VectorXd final_distances;
    nn_search.find_correspondences(current_source, target_matched, final_distances);
    result.final_error = std::sqrt(final_distances.array().square().mean());
    result.error_history.push_back(result.final_error);

    result.transformation = total_transform;
    result.num_iterations = static_cast<int>(result.error_history.size()) - 1;

    return result;
}

} // namespace icp