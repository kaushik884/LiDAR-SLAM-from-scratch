#pragma once

#include "types.hpp"
#include "kdtree.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace slam {

// ============================================================================
// Normal Estimation
// ============================================================================

/**
 * Estimate surface normals using PCA on k-nearest neighbors.
 * 
 * For each point, we:
 * 1. Find k nearest neighbors
 * 2. Build covariance matrix of neighbors
 * 3. Compute eigenvectors - smallest eigenvalue's eigenvector is normal
 * 4. Orient normal consistently (pointing "up" in z by convention)
 */
inline PointCloud::Matrix estimate_normals(
    const PointCloud::Matrix& points,
    const KDTree& tree,
    int k = 20
) {
    PointCloud::Matrix normals(points.rows(), 3);
    
    for (int i = 0; i < points.rows(); ++i) {
        Eigen::Vector3d query = points.row(i).transpose();
        std::vector<int> neighbors = tree.k_nearest(query, k);
        
        if (neighbors.size() < 3) {
            normals.row(i) = Eigen::Vector3d(0, 0, 1).transpose();
            continue;
        }
        
        // Compute centroid of neighbors
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (int idx : neighbors) {
            centroid += points.row(idx).transpose();
        }
        centroid /= neighbors.size();
        
        // Build covariance matrix
        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
        for (int idx : neighbors) {
            Eigen::Vector3d diff = points.row(idx).transpose() - centroid;
            cov += diff * diff.transpose();
        }
        cov /= neighbors.size();
        
        // Eigen decomposition - smallest eigenvector is normal
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
        Eigen::Vector3d normal = solver.eigenvectors().col(0);  // Smallest eigenvalue
        
        // Orient normal consistently (convention: positive z component)
        if (normal.z() < 0) {
            normal = -normal;
        }
        
        normals.row(i) = normal.normalized().transpose();
    }
    
    return normals;
}


// ============================================================================
// Point-to-Plane ICP Solver
// ============================================================================

/**
 * Solve one iteration of point-to-plane ICP.
 * 
 * Minimizes: sum_i [(R*pi + t - qi) · ni]^2
 * 
 * Using small-angle approximation for rotation:
 *   R ≈ I + [r]_x  where r = (rx, ry, rz) is rotation vector
 * 
 * This becomes linear in unknowns x = [rx, ry, rz, tx, ty, tz]:
 *   J*x = b
 * 
 * where for each correspondence:
 *   J_row = [(p × n), n]  (6 elements)
 *   b_val = (q - p) · n
 */
inline Transformation solve_point_to_plane(
    const PointCloud::Matrix& source,
    const PointCloud::Matrix& target,
    const PointCloud::Matrix& normals
) {
    int n = source.rows();
    
    Eigen::MatrixXd J(n, 6);
    Eigen::VectorXd b(n);
    
    for (int i = 0; i < n; ++i) {
        Eigen::Vector3d p = source.row(i).transpose();
        Eigen::Vector3d q = target.row(i).transpose();
        Eigen::Vector3d normal = normals.row(i).transpose();
        
        // Cross product p × n
        Eigen::Vector3d p_cross_n = p.cross(normal);
        
        // Jacobian row: [p × n, n]
        J(i, 0) = p_cross_n.x();
        J(i, 1) = p_cross_n.y();
        J(i, 2) = p_cross_n.z();
        J(i, 3) = normal.x();
        J(i, 4) = normal.y();
        J(i, 5) = normal.z();
        
        // Residual: (q - p) · n
        b(i) = (q - p).dot(normal);
    }
    
    // Solve normal equations: (J^T J) x = J^T b
    Eigen::VectorXd x = (J.transpose() * J).ldlt().solve(J.transpose() * b);
    
    // Extract rotation vector and translation
    Eigen::Vector3d rotation_vec(x(0), x(1), x(2));
    Eigen::Vector3d translation(x(3), x(4), x(5));
    
    // Convert rotation vector to matrix using Rodrigues' formula
    double angle = rotation_vec.norm();
    Eigen::Matrix3d R;
    
    if (angle < 1e-10) {
        R = Eigen::Matrix3d::Identity();
    } else {
        Eigen::Vector3d axis = rotation_vec / angle;
        Eigen::Matrix3d K;  // Skew-symmetric matrix
        K << 0, -axis.z(), axis.y(),
             axis.z(), 0, -axis.x(),
             -axis.y(), axis.x(), 0;
        R = Eigen::Matrix3d::Identity() 
            + std::sin(angle) * K 
            + (1 - std::cos(angle)) * K * K;
    }
    
    return Transformation::from_rt(R, translation);
}


// ============================================================================
// Main ICP Functions
// ============================================================================

/**
 * Point-to-Plane ICP registration.
 * 
 * Aligns source point cloud to target point cloud.
 * Returns transformation T such that T(source) ≈ target.
 */
inline ICPResult icp_point_to_plane(
    const PointCloud& source,
    const PointCloud& target,
    const ICPConfig& config = ICPConfig()
) {
    ICPResult result;
    result.transformation = config.initial_transform;
    
    // Build KD-tree on target
    NearestNeighborSearch nn_search(target);
    
    // Estimate normals on target (once, before iteration loop)
    PointCloud::Matrix target_normals = estimate_normals(
        target.points(), nn_search.tree(), 20
    );
    
    // Apply initial transform to source
    PointCloud::Matrix current_source = 
        (source.points() * config.initial_transform.R().transpose()).rowwise() 
        + config.initial_transform.t().transpose();
    
    Transformation total_transform = config.initial_transform;
    double prev_error = std::numeric_limits<double>::max();
    
    for (int iter = 0; iter < config.max_iterations; ++iter) {
        // Step 1: Find correspondences
        PointCloud::Matrix target_matched;
        Eigen::VectorXd distances;
        nn_search.find_correspondences(current_source, target_matched, distances);
        
        // Step 2: Get normals for matched points
        std::vector<int> indices;
        std::vector<double> distances_sq;
        nn_search.tree().nearest_batch(current_source, indices, distances_sq);
        
        PointCloud::Matrix matched_normals(current_source.rows(), 3);
        for (int i = 0; i < current_source.rows(); ++i) {
            matched_normals.row(i) = target_normals.row(indices[i]);
        }
        
        // Step 3: Compute error
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
        
        // Step 4: Check convergence
        if (error < config.min_error) {
            result.converged = true;
            break;
        }
        if (std::abs(prev_error - error) < config.tolerance) {
            result.converged = true;
            break;
        }
        
        // Step 5: Solve for transformation update
        Transformation delta_transform = solve_point_to_plane(
            current_source, target_matched, matched_normals
        );
        
        // Step 6: Apply to current source
        current_source = (current_source * delta_transform.R().transpose()).rowwise() 
                        + delta_transform.t().transpose();
        
        // Step 7: Accumulate transformation
        total_transform = delta_transform * total_transform;
        
        prev_error = error;
    }
    
    // Final error
    PointCloud::Matrix target_matched;
    Eigen::VectorXd final_distances;
    nn_search.find_correspondences(current_source, target_matched, final_distances);
    
    std::vector<int> indices;
    std::vector<double> distances_sq;
    nn_search.tree().nearest_batch(current_source, indices, distances_sq);
    
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

}  // namespace slam