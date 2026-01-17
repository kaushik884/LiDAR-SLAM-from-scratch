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
     * Find k nearest neighbors for a query point.
     * Returns vector of (index, squared_distance) pairs, sorted by distance.
     */
    std::vector<std::pair<int, double>> k_nearest(const Vector3& query, int k) const {
        if (!root_ || k <= 0) return {};

        // Max-heap: keeps track of k closest points
        // We use max-heap so we can quickly reject points farther than current worst
        std::priority_queue<std::pair<double, int>> heap;  // (dist_sq, index)
        
        search_k_nearest(root_.get(), query, k, heap);

        // Convert heap to sorted vector
        std::vector<std::pair<int, double>> results;
        results.reserve(heap.size());
        while (!heap.empty()) {
            auto [dist_sq, idx] = heap.top();
            heap.pop();
            results.push_back({idx, dist_sq});
        }
        
        // Reverse to get closest first
        std::reverse(results.begin(), results.end());
        return results;
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
     * Recursive k-nearest neighbor search.
     */
    void search_k_nearest(
        const Node* node,
        const Vector3& query,
        int k,
        std::priority_queue<std::pair<double, int>>& heap
    ) const {
        if (!node) return;

        // Check current node
        const Vector3& p = points_[node->point_idx];
        double dist_sq = (p - query).squaredNorm();

        if (heap.size() < static_cast<size_t>(k)) {
            heap.push({dist_sq, node->point_idx});
        } else if (dist_sq < heap.top().first) {
            heap.pop();
            heap.push({dist_sq, node->point_idx});
        }

        // Determine which subtree to search first
        int dim = node->split_dim;
        double diff = query(dim) - p(dim);
        
        Node* first = (diff < 0) ? node->left.get() : node->right.get();
        Node* second = (diff < 0) ? node->right.get() : node->left.get();

        // Search near subtree first
        search_k_nearest(first, query, k, heap);

        // Check if we need to search far subtree
        // Only if heap isn't full OR splitting plane is closer than current worst
        if (heap.size() < static_cast<size_t>(k) || diff * diff < heap.top().first) {
            search_k_nearest(second, query, k, heap);
        }
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
    /**
     * Find correspondences and return indices.
     * Returns matched target points, their indices, and distances.
     */
    void find_correspondences_with_indices(
        const Matrix& source_points,
        Matrix& target_matched,
        std::vector<int>& indices,
        Eigen::VectorXd& distances
    ) const {
        const int n = source_points.rows();
        target_matched.resize(n, 3);
        distances.resize(n);
        indices.resize(n);

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