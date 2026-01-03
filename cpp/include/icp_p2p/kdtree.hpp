#pragma once

#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <memory>
#include <limits>
#include <queue>

namespace icp {

/**
 * Simple KD-Tree for 3D nearest neighbor search.
 * 
 * This is a from-scratch implementation for educational purposes.
 * For production, consider nanoflann.
 */
class KDTree {
public:
    using Vector3 = Eigen::Vector3d;

    KDTree() = default;

    /**
     * Build tree from Nx3 matrix of points.
     */
    explicit KDTree(const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& points) {
        const int n = points.rows();
        if (n == 0) return;

        // Store points and create index array
        points_.resize(n);
        std::vector<int> indices(n);
        for (int i = 0; i < n; ++i) {
            points_[i] = points.row(i).transpose();
            indices[i] = i;
        }

        // Build tree recursively
        root_ = build(indices, 0, n, 0);
    }

    /**
     * Find nearest neighbor for a query point.
     * Returns (index, squared_distance).
     */
    std::pair<int, double> nearest(const Vector3& query) const {
        if (!root_) return {-1, std::numeric_limits<double>::max()};

        int best_idx = -1;
        double best_dist_sq = std::numeric_limits<double>::max();
        search_nearest(root_.get(), query, 0, best_idx, best_dist_sq);
        return {best_idx, best_dist_sq};
    }

    /**
     * Find nearest neighbors for multiple query points.
     * Returns vectors of (indices, squared_distances).
     */
    void nearest_batch(
        const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& queries,
        std::vector<int>& indices,
        std::vector<double>& distances_sq
    ) const {
        const int n = queries.rows();
        indices.resize(n);
        distances_sq.resize(n);

        for (int i = 0; i < n; ++i) {
            Vector3 q = queries.row(i).transpose();
            auto [idx, dist_sq] = nearest(q);
            indices[i] = idx;
            distances_sq[i] = dist_sq;
        }
    }

    /**
     * Get the stored point at given index.
     */
    const Vector3& point(int idx) const { return points_[idx]; }

    size_t size() const { return points_.size(); }

private:
    struct Node {
        int point_idx;          // Index into points_ array
        int split_dim;          // Dimension to split on (0, 1, or 2)
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;

        Node(int idx, int dim) : point_idx(idx), split_dim(dim) {}
    };

    std::vector<Vector3> points_;
    std::unique_ptr<Node> root_;

    /**
     * Recursively build the tree.
     */
    std::unique_ptr<Node> build(std::vector<int>& indices, int start, int end, int depth) {
        if (start >= end) return nullptr;

        int dim = depth % 3;  // Cycle through x, y, z

        // Sort by current dimension and pick median
        int mid = start + (end - start) / 2;
        std::nth_element(
            indices.begin() + start,
            indices.begin() + mid,
            indices.begin() + end,
            [this, dim](int a, int b) {
                return points_[a](dim) < points_[b](dim);
            }
        );

        auto node = std::make_unique<Node>(indices[mid], dim);
        node->left = build(indices, start, mid, depth + 1);
        node->right = build(indices, mid + 1, end, depth + 1);

        return node;
    }

    /**
     * Recursive nearest neighbor search.
     */
    void search_nearest(
        const Node* node,
        const Vector3& query,
        int depth,
        int& best_idx,
        double& best_dist_sq
    ) const {
        if (!node) return;

        // Check current node
        const Vector3& p = points_[node->point_idx];
        double dist_sq = (p - query).squaredNorm();
        if (dist_sq < best_dist_sq) {
            best_dist_sq = dist_sq;
            best_idx = node->point_idx;
        }

        // Determine which subtree to search first
        int dim = node->split_dim;
        double diff = query(dim) - p(dim);
        
        Node* first = (diff < 0) ? node->left.get() : node->right.get();
        Node* second = (diff < 0) ? node->right.get() : node->left.get();

        // Search near subtree first
        search_nearest(first, query, depth + 1, best_idx, best_dist_sq);

        // Check if we need to search far subtree
        // Only if the splitting plane is closer than current best
        if (diff * diff < best_dist_sq) {
            search_nearest(second, query, depth + 1, best_idx, best_dist_sq);
        }
    }
};


/**
 * Higher-level wrapper matching Python API.
 */
class NearestNeighborSearch {
public:
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
    using Vector3 = Eigen::Vector3d;

    explicit NearestNeighborSearch(const Matrix& target_points) 
        : tree_(target_points), target_points_(target_points) {}

    /**
     * Find correspondences between source and target.
     * Returns matched target points and distances.
     */
    void find_correspondences(
        const Matrix& source_points,
        Matrix& target_matched,
        Eigen::VectorXd& distances
    ) const {
        const int n = source_points.rows();
        target_matched.resize(n, 3);
        distances.resize(n);

        std::vector<int> indices;
        std::vector<double> distances_sq;
        tree_.nearest_batch(source_points, indices, distances_sq);

        for (int i = 0; i < n; ++i) {
            target_matched.row(i) = target_points_.row(indices[i]);
            distances(i) = std::sqrt(distances_sq[i]);
        }
    }

private:
    KDTree tree_;
    Matrix target_points_;
};

} // namespace icp