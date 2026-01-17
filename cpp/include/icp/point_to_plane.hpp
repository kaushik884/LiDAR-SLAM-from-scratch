#pragma once

#include "types.hpp"
#include "kdtree.hpp"
#include "normal_estimation.hpp"
#include <Eigen/Dense>

namespace icp {

/**
 * Solve one iteration of point-to-plane ICP.
 * 
 * Solves the linearized system to find optimal (rotation, translation)
 * that minimizes point-to-plane distances.
 * 
 * @param source_points Nx3 source points (already transformed)
 * @param target_points Nx3 corresponding target points
 * @param target_normals Nx3 normals at target points
 * @return Transformation for this iteration
 */
inline Transformation solve_point_to_plane(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& source_points,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& target_points,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& target_normals
) {
    const int n = source_points.rows();
    
    // Build the linear system: Jx = r
    // J is Nx6, x is 6x1, r is Nx1
    Eigen::MatrixXd J(n, 6);
    Eigen::VectorXd r(n);
    
    for (int i = 0; i < n; ++i) {
        Eigen::Vector3d p = source_points.row(i).transpose();
        Eigen::Vector3d q = target_points.row(i).transpose();
        Eigen::Vector3d normal = target_normals.row(i).transpose();
        
        // Cross product: p × n
        Eigen::Vector3d p_cross_n = p.cross(normal);
        
        // Jacobian row: [p × n, n]
        J(i, 0) = p_cross_n.x();
        J(i, 1) = p_cross_n.y();
        J(i, 2) = p_cross_n.z();
        J(i, 3) = normal.x();
        J(i, 4) = normal.y();
        J(i, 5) = normal.z();
        
        // Residual: (q - p) · n
        r(i) = (q - p).dot(normal);
    }
    
    // Solve normal equations: (JᵀJ)x = Jᵀr
    Eigen::Matrix<double, 6, 6> JtJ = J.transpose() * J;
    Eigen::Matrix<double, 6, 1> Jtr = J.transpose() * r;
    
    // Solve using Cholesky (fast for symmetric positive definite)
    Eigen::Matrix<double, 6, 1> x = JtJ.ldlt().solve(Jtr);
    
    // Extract rotation vector and translation
    Eigen::Vector3d rotation_vec(x(0), x(1), x(2));
    Eigen::Vector3d translation(x(3), x(4), x(5));
    
    // Convert rotation vector to rotation matrix using Rodrigues' formula
    double angle = rotation_vec.norm();
    Eigen::Matrix3d R;
    
    if (angle < 1e-10) {
        R = Eigen::Matrix3d::Identity();
    } else {
        Eigen::Vector3d axis = rotation_vec / angle;
        Eigen::Matrix3d K;  // Skew-symmetric matrix
        K <<     0,    -axis.z(),  axis.y(),
             axis.z(),     0,     -axis.x(),
            -axis.y(),  axis.x(),     0;
        
        R = Eigen::Matrix3d::Identity() 
            + std::sin(angle) * K 
            + (1 - std::cos(angle)) * K * K;
    }
    
    return Transformation::from_rt(R, translation);
}


/**
 * Point-to-Plane ICP registration.
 * 
 * Uses linearized optimization for faster convergence on smooth surfaces.
 * 
 * @param source Source point cloud to align
 * @param target Target point cloud (reference)
 * @param config Algorithm parameters
 * @return ICPResult with transformation and convergence info
 */
inline ICPResult icp_point_to_plane(
    const PointCloud& source,
    const PointCloud& target,
    const ICPConfig& config = ICPConfig{}
) {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

    // Precompute normals on target cloud
    Matrix target_normals = estimate_normals(target.points(), 20);
    
    // Build KD-tree on target for correspondence search
    NearestNeighborSearch nn_search(target.points());

    // Work on a copy
    Matrix current_source = source.points();

    // Apply initial transform if provided
    Transformation total_transform = config.initial_transform;
    current_source = (current_source * total_transform.R().transpose()).rowwise() 
                    + total_transform.t().transpose();

    ICPResult result;
    double prev_error = std::numeric_limits<double>::infinity();

    for (int iter = 0; iter < config.max_iterations; ++iter) {
        // Step 1: Find correspondences
        Matrix target_matched;
        std::vector<int> indices;
        Eigen::VectorXd distances;
        nn_search.find_correspondences_with_indices(current_source, target_matched, indices, distances);

        // Get normals for matched target points
        // We need indices to look up the correct normals
        // std::vector<int> indices;
        // std::vector<double> distances_sq;
        // KDTree tree(target.points());
        // tree.nearest_batch(current_source, indices, distances_sq);
        
        Matrix matched_normals(current_source.rows(), 3);
        for (int i = 0; i < current_source.rows(); ++i) {
            matched_normals.row(i) = target_normals.row(indices[i]);
        }

        // Step 2: Compute point-to-plane error
        double error = 0;
        for (int i = 0; i < current_source.rows(); ++i) {
            Eigen::Vector3d diff = target_matched.row(i).transpose() 
                                 - current_source.row(i).transpose();
            Eigen::Vector3d n = matched_normals.row(i).transpose();
            double plane_dist = diff.dot(n);
            error += plane_dist * plane_dist;
        }
        error = std::sqrt(error / current_source.rows());
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

        // Step 4: Solve for transformation
        Transformation delta_transform = solve_point_to_plane(
            current_source, target_matched, matched_normals
        );

        // Step 5: Apply to current source
        current_source = (current_source * delta_transform.R().transpose()).rowwise() 
                        + delta_transform.t().transpose();

        // Step 6: Accumulate transformation
        total_transform = delta_transform * total_transform;

        prev_error = error;
    }

    // Final error computation (reuse same nn_search)
    Matrix target_matched;
    std::vector<int> indices;
    Eigen::VectorXd distances;
    nn_search.find_correspondences_with_indices(current_source, target_matched, indices, distances);
    
    double final_error = 0;
    
    for (int i = 0; i < current_source.rows(); ++i) {
        Eigen::Vector3d diff = target_matched.row(i).transpose() 
                             - current_source.row(i).transpose();
        Eigen::Vector3d n = target_normals.row(indices[i]).transpose();
        double plane_dist = diff.dot(n);
        final_error += plane_dist * plane_dist;
    }
    result.final_error = std::sqrt(final_error / current_source.rows());
    result.error_history.push_back(result.final_error);

    result.transformation = total_transform;
    result.num_iterations = static_cast<int>(result.error_history.size()) - 1;

    return result;
}

} // namespace icp