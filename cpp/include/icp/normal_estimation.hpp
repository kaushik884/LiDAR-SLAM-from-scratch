#pragma once

#include "kdtree.hpp"
#include <Eigen/Eigenvalues>

namespace icp {

/**
 * Estimate surface normals for a point cloud using PCA.
 * 
 * For each point, finds k nearest neighbors and computes the
 * normal as the eigenvector with smallest eigenvalue of the
 * local covariance matrix.
 * 
 * @param points Nx3 matrix of points
 * @param k Number of neighbors for local surface estimation
 * @return Nx3 matrix of unit normals
 */
inline Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> estimate_normals(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& points,
    int k = 20
) {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
    using Vector3 = Eigen::Vector3d;
    using Matrix3 = Eigen::Matrix3d;
    
    const int n = points.rows();
    Matrix normals(n, 3);
    
    // Build KD-tree for neighbor queries
    KDTree tree(points);
    
    for (int i = 0; i < n; ++i) {
        Vector3 query = points.row(i).transpose();
        
        // Find k nearest neighbors (includes the point itself)
        auto neighbors = tree.k_nearest(query, k);
        
        // Compute centroid of neighbors
        Vector3 centroid = Vector3::Zero();
        for (const auto& [idx, dist_sq] : neighbors) {
            centroid += points.row(idx).transpose();
        }
        centroid /= neighbors.size();
        
        // Build covariance matrix
        Matrix3 covariance = Matrix3::Zero();
        for (const auto& [idx, dist_sq] : neighbors) {
            Vector3 centered = points.row(idx).transpose() - centroid;
            covariance += centered * centered.transpose();
        }
        covariance /= neighbors.size();
        
        // Eigendecomposition - Eigen sorts eigenvalues in increasing order
        Eigen::SelfAdjointEigenSolver<Matrix3> solver(covariance);
        
        // Smallest eigenvalue is first, its eigenvector is the normal
        Vector3 normal = solver.eigenvectors().col(0);
        
        // Ensure consistent orientation (optional: orient toward origin)
        // For now, just ensure z-component convention
        if (normal.z() < 0) {
            normal = -normal;
        }
        
        normals.row(i) = normal.transpose();
    }
    
    return normals;
}

} // namespace icp