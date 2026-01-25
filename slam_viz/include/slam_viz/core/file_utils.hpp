#pragma once

#include "types.hpp"
#include <string>
#include <vector>
#include <utility>

namespace slam {

/**
 * Load point cloud from PLY file.
 * Supports both ASCII and binary PLY formats.
 * 
 * @param filepath Path to .ply file
 * @return Nx3 matrix of points
 * @throws std::runtime_error if file cannot be opened or parsed
 */
PointCloud::Matrix load_ply(const std::string& filepath);


/**
 * Load point cloud from KITTI binary format (.bin).
 * 
 * @param filepath Path to .bin file
 * @return Nx3 matrix of points (intensity channel discarded)
 * @throws std::runtime_error if file cannot be opened
 */
PointCloud::Matrix load_bin(const std::string& filepath);


/**
 * Voxel grid downsampling.
 * 
 * Groups points into voxels and replaces each group with its centroid.
 * This reduces point count while preserving overall structure.
 * 
 * @param points Input point cloud
 * @param voxel_size Size of each voxel cube in meters
 * @return Downsampled point cloud
 */
PointCloud::Matrix voxel_downsample(
    const PointCloud::Matrix& points,
    double voxel_size
);


/**
 * Extract timestamp from filename.
 * Assumes format: "1234567890.ply" where digits are the timestamp.
 * 
 * @param filename Filename (not full path)
 * @return Timestamp as long long, or -1 if parsing fails
 */
long long extract_timestamp(const std::string& filename);


/**
 * Discover and sort frame files in a directory.
 * 
 * @param data_dir Path to directory containing .ply files
 * @return Vector of (timestamp, filepath) pairs, sorted by timestamp
 */
std::vector<std::pair<long long, std::string>> discover_frames(
    const std::string& data_dir
);

}  // namespace slam