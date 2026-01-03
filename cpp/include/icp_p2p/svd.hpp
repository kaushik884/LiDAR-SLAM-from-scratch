#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include "types.hpp"

namespace icp {

/**
 * Compute optimal rigid transformation aligning source to target.
 * 
 * Uses SVD closed-form solution:
 *   1. Center both point sets
 *   2. Compute cross-covariance H = Σ (p_i - p̄)(q_i - q̄)ᵀ
 *   3. SVD: H = UΣVᵀ
 *   4. R = VUᵀ (with reflection correction)
 *   5. t = q̄ - R * p̄
 * 
 * @param source_points Nx3 matrix of source points
 * @param target_points Nx3 matrix of corresponding target points
 * @return Transformation aligning source to target
 */
inline Transformation estimate_transformation(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& source_points,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& target_points
) {
    using Vector3 = Eigen::Vector3d;
    using Matrix3 = Eigen::Matrix3d;

    const int n = source_points.rows();
    assert(n == target_points.rows() && "Point sets must have same size");
    assert(n >= 3 && "Need at least 3 points");

    // Step 1: Compute centroids
    Vector3 p_centroid = source_points.colwise().mean().transpose();
    Vector3 q_centroid = target_points.colwise().mean().transpose();

    // Step 2: Center the points
    auto p_centered = source_points.rowwise() - p_centroid.transpose();
    auto q_centered = target_points.rowwise() - q_centroid.transpose();

    // Step 3: Cross-covariance matrix H = P^T * Q
    Matrix3 H = p_centered.transpose() * q_centered;

    // Step 4: SVD decomposition
    Eigen::JacobiSVD<Matrix3> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Matrix3 U = svd.matrixU();
    Matrix3 V = svd.matrixV();

    // Step 5: Rotation matrix R = V * U^T
    Matrix3 R = V * U.transpose();

    // Handle reflection case (det = -1)
    if (R.determinant() < 0) {
        V.col(2) *= -1;
        R = V * U.transpose();
    }

    // Step 6: Translation t = q_centroid - R * p_centroid
    Vector3 t = q_centroid - R * p_centroid;

    return Transformation::from_rt(R, t);
}


/**
 * Compute RMS error between point sets.
 * Optionally applies transformation to source first.
 */
inline double compute_rms_error(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& source_points,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& target_points,
    const Transformation* transform = nullptr
) {
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> src = source_points;
    
    if (transform) {
        // Apply transformation: src = src * R^T + t^T (row-wise)
        src = (source_points * transform->R().transpose()).rowwise() 
              + transform->t().transpose();
    }

    // Compute squared distances
    auto diff = src - target_points;
    double mse = diff.rowwise().squaredNorm().mean();
    return std::sqrt(mse);
}

} // namespace icp