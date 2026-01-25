#pragma once

#include "types.hpp"
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>

namespace slam {

/**
 * KD-Tree for fast nearest neighbor search.
 * 
 * Design decision: Custom implementation for educational purposes.
 * In production, you'd use nanoflann for better performance.
 * This implementation is simple but correct.
 */
class KDTree {
public:
    explicit KDTree(const PointCloud::Matrix& points) : points_(points) {
        indices_.resize(points_.rows());
        for (int i = 0; i < points_.rows(); ++i) {
            indices_[i] = i;
        }
        root_ = build(0, points_.rows(), 0);
    }

    /**
     * Find nearest neighbor for a query point.
     * Returns index into original point array.
     */
    int nearest(const Eigen::Vector3d& query) const {
        int best_idx = -1;
        double best_dist_sq = std::numeric_limits<double>::max();
        search_nearest(root_, query, 0, best_idx, best_dist_sq);
        return best_idx;
    }

    /**
     * Find nearest neighbors for multiple query points.
     * More efficient than calling nearest() in a loop.
     */
    void nearest_batch(
        const PointCloud::Matrix& queries,
        std::vector<int>& indices,
        std::vector<double>& distances_sq
    ) const {
        indices.resize(queries.rows());
        distances_sq.resize(queries.rows());
        
        for (int i = 0; i < queries.rows(); ++i) {
            Eigen::Vector3d q = queries.row(i).transpose();
            int best_idx = -1;
            double best_dist_sq = std::numeric_limits<double>::max();
            search_nearest(root_, q, 0, best_idx, best_dist_sq);
            indices[i] = best_idx;
            distances_sq[i] = best_dist_sq;
        }
    }

    /**
     * Find k nearest neighbors for a query point.
     * Returns indices sorted by distance (closest first).
     */
    std::vector<int> k_nearest(const Eigen::Vector3d& query, int k) const {
        // Max-heap: largest distance at top
        std::priority_queue<std::pair<double, int>> heap;
        search_k_nearest(root_, query, 0, k, heap);
        
        std::vector<int> result;
        result.reserve(heap.size());
        while (!heap.empty()) {
            result.push_back(heap.top().second);
            heap.pop();
        }
        std::reverse(result.begin(), result.end());
        return result;
    }

private:
    struct Node {
        int index;          // Index into points array
        int left = -1;      // Index into nodes array
        int right = -1;     // Index into nodes array
    };

    int build(int start, int end, int depth) {
        if (start >= end) return -1;
        
        int axis = depth % 3;
        int mid = (start + end) / 2;
        
        // Partial sort to find median
        std::nth_element(
            indices_.begin() + start,
            indices_.begin() + mid,
            indices_.begin() + end,
            [this, axis](int a, int b) {
                return points_(a, axis) < points_(b, axis);
            }
        );
        
        Node node;
        node.index = indices_[mid];
        node.left = build(start, mid, depth + 1);
        node.right = build(mid + 1, end, depth + 1);
        
        nodes_.push_back(node);
        return static_cast<int>(nodes_.size()) - 1;
    }

    void search_nearest(
        int node_idx,
        const Eigen::Vector3d& query,
        int depth,
        int& best_idx,
        double& best_dist_sq
    ) const {
        if (node_idx < 0) return;
        
        const Node& node = nodes_[node_idx];
        Eigen::Vector3d point = points_.row(node.index).transpose();
        
        double dist_sq = (point - query).squaredNorm();
        if (dist_sq < best_dist_sq) {
            best_dist_sq = dist_sq;
            best_idx = node.index;
        }
        
        int axis = depth % 3;
        double diff = query(axis) - point(axis);
        
        int first = diff < 0 ? node.left : node.right;
        int second = diff < 0 ? node.right : node.left;
        
        search_nearest(first, query, depth + 1, best_idx, best_dist_sq);
        
        // Only search other branch if it could contain closer points
        if (diff * diff < best_dist_sq) {
            search_nearest(second, query, depth + 1, best_idx, best_dist_sq);
        }
    }

    void search_k_nearest(
        int node_idx,
        const Eigen::Vector3d& query,
        int depth,
        int k,
        std::priority_queue<std::pair<double, int>>& heap
    ) const {
        if (node_idx < 0) return;
        
        const Node& node = nodes_[node_idx];
        Eigen::Vector3d point = points_.row(node.index).transpose();
        
        double dist_sq = (point - query).squaredNorm();
        
        if (static_cast<int>(heap.size()) < k) {
            heap.push({dist_sq, node.index});
        } else if (dist_sq < heap.top().first) {
            heap.pop();
            heap.push({dist_sq, node.index});
        }
        
        int axis = depth % 3;
        double diff = query(axis) - point(axis);
        
        int first = diff < 0 ? node.left : node.right;
        int second = diff < 0 ? node.right : node.left;
        
        search_k_nearest(first, query, depth + 1, k, heap);
        
        double threshold = static_cast<int>(heap.size()) < k 
            ? std::numeric_limits<double>::max() 
            : heap.top().first;
            
        if (diff * diff < threshold) {
            search_k_nearest(second, query, depth + 1, k, heap);
        }
    }

    PointCloud::Matrix points_;
    std::vector<int> indices_;
    std::vector<Node> nodes_;
    int root_ = -1;
};


/**
 * Nearest neighbor search helper for ICP.
 * Wraps KDTree and provides correspondence-finding interface.
 */
class NearestNeighborSearch {
public:
    explicit NearestNeighborSearch(const PointCloud& target) 
        : target_(target.points()), tree_(target.points()) {}

    void find_correspondences(
        const PointCloud::Matrix& source,
        PointCloud::Matrix& matched_target,
        Eigen::VectorXd& distances
    ) const {
        std::vector<int> indices;
        std::vector<double> distances_sq;
        tree_.nearest_batch(source, indices, distances_sq);
        
        matched_target.resize(source.rows(), 3);
        distances.resize(source.rows());
        
        for (int i = 0; i < source.rows(); ++i) {
            matched_target.row(i) = target_.row(indices[i]);
            distances(i) = std::sqrt(distances_sq[i]);
        }
    }

    const KDTree& tree() const { return tree_; }

private:
    PointCloud::Matrix target_;
    KDTree tree_;
};

}  // namespace slam