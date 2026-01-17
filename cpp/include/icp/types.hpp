#pragma once

#include <Eigen/Dense>
#include <vector>
#include <stdexcept>

namespace icp {
/**
 * Point cloud represented as Nx3 Eigen matrix.
 * Each row is a point [x, y, z].
 */
class PointCloud {
public:
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
    using Vector3 = Eigen::Vector3d;

    PointCloud() = default;

    explicit PointCloud(const Matrix& points) : points_(points) {}

    explicit PointCloud(Matrix&& points) : points_(std::move(points)) {}

    // Construct from vector of points
    explicit PointCloud(const std::vector<Vector3>& points) {
        points_.resize(points.size(), 3);
        for (size_t i = 0; i < points.size(); ++i) {
            points_.row(i) = points[i].transpose();
        }
    }

    // Access
    const Matrix& points() const { return points_; }
    Matrix& points() { return points_; }

    size_t size() const { return static_cast<size_t>(points_.rows()); }
    bool empty() const { return points_.rows() == 0; }

    // Row access
    auto row(int i) const { return points_.row(i); }
    auto row(int i) { return points_.row(i); }

    // Compute centroid
    Vector3 centroid() const {
        return points_.colwise().mean().transpose();
    }

    // Return centered copy (centroid at origin)
    PointCloud centered() const {
        Matrix centered_pts = points_.rowwise() - points_.colwise().mean();
        return PointCloud(std::move(centered_pts));
    }

    // Deep copy
    PointCloud copy() const {
        return PointCloud(Matrix(points_));
    }

private:
    Matrix points_;
};


/**
 * Rigid transformation in 3D, stored as 4x4 homogeneous matrix.
 */
class Transformation {
public:
    using Matrix4 = Eigen::Matrix4d;
    using Matrix3 = Eigen::Matrix3d;
    using Vector3 = Eigen::Vector3d;

    Transformation() : matrix_(Matrix4::Identity()) {}

    explicit Transformation(const Matrix4& matrix) : matrix_(matrix) {}

    // Construct from R and t
    static Transformation from_rt(const Matrix3& R, const Vector3& t) {
        Matrix4 mat = Matrix4::Identity();
        mat.block<3, 3>(0, 0) = R;
        mat.block<3, 1>(0, 3) = t;
        return Transformation(mat);
    }

    static Transformation identity() {
        return Transformation();
    }

    // Access
    const Matrix4& matrix() const { return matrix_; }

    Matrix3 R() const { return matrix_.block<3, 3>(0, 0); }
    Vector3 t() const { return matrix_.block<3, 1>(0, 3); }

    // Apply to single point
    Vector3 apply(const Vector3& p) const {
        return R() * p + t();
    }

    // Apply to point cloud
    PointCloud apply(const PointCloud& cloud) const {
        // P_transformed = (R * P^T)^T + t = P * R^T + t^T (row-wise)
        PointCloud::Matrix transformed = 
            (cloud.points() * R().transpose()).rowwise() + t().transpose();
        return PointCloud(std::move(transformed));
    }

    // Compose: this @ other (this applied after other)
    Transformation compose(const Transformation& other) const {
        return Transformation(matrix_ * other.matrix_);
    }

    // Operator for composition
    Transformation operator*(const Transformation& other) const {
        return compose(other);
    }

    // Inverse (efficient for rigid transforms)
    Transformation inverse() const {
        Matrix3 R_inv = R().transpose();
        Vector3 t_inv = -R_inv * t();
        return from_rt(R_inv, t_inv);
    }

private:
    Matrix4 matrix_;
};


/**
 * Result of ICP registration.
 */
struct ICPResult {
    Transformation transformation;
    bool converged = false;
    int num_iterations = 0;
    std::vector<double> error_history;
    double final_error = 0.0;

    // Helper to check success
    bool success() const { return converged && final_error < 0.01; }
};
/**
 * Point-to-Point ICP configuration.
 */
struct ICPConfig {
    int max_iterations = 50;
    double tolerance = 1e-6;      // Convergence threshold for error change
    double min_error = 1e-9;      // Stop if error falls below this
    Transformation initial_transform = Transformation::identity();
};
} // namespace icp