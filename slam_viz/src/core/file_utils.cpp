#include "slam_viz/core/file_utils.hpp"

#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

namespace slam {

// ============================================================================
// PLY Loading
// ============================================================================

PointCloud::Matrix load_ply(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    std::string line;
    int num_vertices = 0;
    bool is_binary = false;
    std::vector<std::pair<std::string, std::string>> properties;

    // Parse header
    while (std::getline(file, line)) {
        // Handle Windows line endings
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        
        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "format") {
            std::string format_type;
            iss >> format_type;
            if (format_type == "binary_little_endian" || format_type == "binary_big_endian") {
                is_binary = true;
            }
        } else if (token == "element") {
            std::string type;
            iss >> type;
            if (type == "vertex") {
                iss >> num_vertices;
            }
        } else if (token == "property") {
            std::string dtype, name;
            iss >> dtype >> name;
            properties.emplace_back(name, dtype);
        } else if (token == "end_header") {
            break;
        }
    }

    // Calculate byte offsets for x, y, z
    auto get_type_size = [](const std::string& dtype) -> size_t {
        if (dtype == "float" || dtype == "float32") return 4;
        if (dtype == "double" || dtype == "float64") return 8;
        if (dtype == "uchar" || dtype == "uint8" || dtype == "char" || dtype == "int8") return 1;
        if (dtype == "ushort" || dtype == "uint16" || dtype == "short" || dtype == "int16") return 2;
        if (dtype == "uint" || dtype == "uint32" || dtype == "int" || dtype == "int32") return 4;
        return 4;  // Default to float size
    };

    size_t bytes_per_vertex = 0;
    size_t x_offset = 0, y_offset = 0, z_offset = 0;

    for (const auto& [name, dtype] : properties) {
        size_t type_size = get_type_size(dtype);
        if (name == "x") x_offset = bytes_per_vertex;
        else if (name == "y") y_offset = bytes_per_vertex;
        else if (name == "z") z_offset = bytes_per_vertex;
        bytes_per_vertex += type_size;
    }

    // Read points
    PointCloud::Matrix points(num_vertices, 3);

    if (is_binary) {
        std::vector<char> buffer(bytes_per_vertex);
        for (int i = 0; i < num_vertices; ++i) {
            file.read(buffer.data(), bytes_per_vertex);
            float x, y, z;
            std::memcpy(&x, buffer.data() + x_offset, sizeof(float));
            std::memcpy(&y, buffer.data() + y_offset, sizeof(float));
            std::memcpy(&z, buffer.data() + z_offset, sizeof(float));
            points(i, 0) = static_cast<double>(x);
            points(i, 1) = static_cast<double>(y);
            points(i, 2) = static_cast<double>(z);
        }
    } else {
        for (int i = 0; i < num_vertices; ++i) {
            std::getline(file, line);
            std::istringstream iss(line);
            iss >> points(i, 0) >> points(i, 1) >> points(i, 2);
        }
    }

    return points;
}


// ============================================================================
// BIN Loading (KITTI format)
// ============================================================================

PointCloud::Matrix load_bin(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    // Get file size to determine point count
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // KITTI format: x, y, z, intensity (4 floats per point)
    size_t num_points = file_size / (4 * sizeof(float));
    
    PointCloud::Matrix points(num_points, 3);
    std::vector<float> buffer(4);

    for (size_t i = 0; i < num_points; ++i) {
        file.read(reinterpret_cast<char*>(buffer.data()), 4 * sizeof(float));
        points(i, 0) = static_cast<double>(buffer[0]);
        points(i, 1) = static_cast<double>(buffer[1]);
        points(i, 2) = static_cast<double>(buffer[2]);
        // Discard intensity (buffer[3])
    }

    return points;
}


// ============================================================================
// Voxel Downsampling
// ============================================================================

PointCloud::Matrix voxel_downsample(
    const PointCloud::Matrix& points,
    double voxel_size
) {
    if (voxel_size <= 0) return points;

    // Voxel key for hashing
    struct VoxelKey {
        long long x, y, z;
        bool operator==(const VoxelKey& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct VoxelHash {
        size_t operator()(const VoxelKey& v) const {
            size_t h = 0;
            h ^= std::hash<long long>{}(v.x) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<long long>{}(v.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<long long>{}(v.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    // Group points by voxel
    std::unordered_map<VoxelKey, std::vector<int>, VoxelHash> voxel_map;

    for (int i = 0; i < points.rows(); ++i) {
        VoxelKey key;
        key.x = static_cast<long long>(std::floor(points(i, 0) / voxel_size));
        key.y = static_cast<long long>(std::floor(points(i, 1) / voxel_size));
        key.z = static_cast<long long>(std::floor(points(i, 2) / voxel_size));
        voxel_map[key].push_back(i);
    }

    // Compute centroid of each voxel
    PointCloud::Matrix downsampled(voxel_map.size(), 3);
    int idx = 0;
    for (const auto& [voxel, indices] : voxel_map) {
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (int i : indices) {
            centroid += points.row(i).transpose();
        }
        centroid /= indices.size();
        downsampled.row(idx++) = centroid.transpose();
    }

    return downsampled;
}


// ============================================================================
// Timestamp extraction
// ============================================================================

long long extract_timestamp(const std::string& filename) {
    std::regex pattern(R"((\d+)\.ply)");
    std::smatch match;
    if (std::regex_search(filename, match, pattern)) {
        return std::stoll(match[1].str());
    }
    return -1;
}


// ============================================================================
// Frame discovery
// ============================================================================

std::vector<std::pair<long long, std::string>> discover_frames(
    const std::string& data_dir
) {
    std::vector<std::pair<long long, std::string>> frames;

    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.path().extension() == ".ply") {
            std::string filename = entry.path().filename().string();
            long long timestamp = extract_timestamp(filename);
            if (timestamp >= 0) {
                frames.emplace_back(timestamp, entry.path().string());
            }
        } else if (entry.path().extension() == ".bin") {
            // For KITTI format, use filename order
            std::string filename = entry.path().filename().string();
            // Extract numeric part
            std::regex pattern(R"((\d+)\.bin)");
            std::smatch match;
            if (std::regex_search(filename, match, pattern)) {
                long long idx = std::stoll(match[1].str());
                frames.emplace_back(idx, entry.path().string());
            }
        }
    }

    // Sort by timestamp/index
    std::sort(frames.begin(), frames.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    return frames;
}

}  // namespace slam