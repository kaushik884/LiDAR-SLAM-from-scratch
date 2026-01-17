#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <regex>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <set>

#include "icp.hpp"
#include "pose_graph.hpp"
#include "icp/loop_closure.hpp"

namespace fs = std::filesystem;

namespace {

/**
 * Configuration for LiDAR odometry.
 */
struct OdometryConfig {
    double voxel_size = 0.5;        // Downsampling voxel size (meters)
    int max_iterations = 50;         // ICP max iterations
    double tolerance = 1e-6;         // ICP convergence tolerance
    int min_points = 1000;           // Minimum points after downsampling
};


/**
 * Configuration for occupancy grid building.
 */
struct OccupancyGridConfig {
    double resolution = 0.5;         // Grid cell size in meters
    double height_min = 0.3;         // Minimum z height (above ground)
    double height_max = 2.0;         // Maximum z height (below ceiling)
    double max_range = 40.0;         // Maximum distance from sensor
    int subsample_rate = 2;          // Use every Nth frame
};


/**
 * Simple 2D grid cell for hashing.
 */
struct GridCell {
    int x, y;
    
    bool operator==(const GridCell& other) const {
        return x == other.x && y == other.y;
    }
};

struct GridCellHash {
    size_t operator()(const GridCell& cell) const {
        size_t h = 0;
        h ^= std::hash<int>{}(cell.x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(cell.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};


/**
 * Load point cloud from PLY file (handles both ASCII and binary formats).
 */
Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> load_ply(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    std::string line;
    int num_vertices = 0;
    bool is_binary = false;
    bool is_little_endian = true;
    
    // Track property order and types
    std::vector<std::pair<std::string, std::string>> properties; // (name, type)

    // Parse header
    while (std::getline(file, line)) {
        // Remove carriage return if present (Windows line endings)
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        
        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "format") {
            std::string format_type;
            iss >> format_type;
            if (format_type == "binary_little_endian") {
                is_binary = true;
                is_little_endian = true;
            } else if (format_type == "binary_big_endian") {
                is_binary = true;
                is_little_endian = false;
            }
            // else ASCII format (default)
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

    // Calculate bytes per vertex and find x,y,z offsets
    size_t bytes_per_vertex = 0;
    size_t x_offset = 0, y_offset = 0, z_offset = 0;
    bool x_found = false, y_found = false, z_found = false;
    
    auto get_type_size = [](const std::string& dtype) -> size_t {
        if (dtype == "float" || dtype == "float32") return 4;
        if (dtype == "double" || dtype == "float64") return 8;
        if (dtype == "uchar" || dtype == "uint8" || dtype == "char" || dtype == "int8") return 1;
        if (dtype == "ushort" || dtype == "uint16" || dtype == "short" || dtype == "int16") return 2;
        if (dtype == "uint" || dtype == "uint32" || dtype == "int" || dtype == "int32") return 4;
        return 4; // default
    };

    for (const auto& [name, dtype] : properties) {
        size_t type_size = get_type_size(dtype);
        if (name == "x") { x_offset = bytes_per_vertex; x_found = true; }
        else if (name == "y") { y_offset = bytes_per_vertex; y_found = true; }
        else if (name == "z") { z_offset = bytes_per_vertex; z_found = true; }
        bytes_per_vertex += type_size;
    }

    if (!x_found || !y_found || !z_found) {
        throw std::runtime_error("PLY file missing x, y, or z properties");
    }

    // Read points
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> points(num_vertices, 3);

    if (is_binary) {
        // Binary format
        std::vector<char> buffer(bytes_per_vertex);
        
        for (int i = 0; i < num_vertices; ++i) {
            file.read(buffer.data(), bytes_per_vertex);
            
            // Extract x, y, z (assuming float type for coordinates)
            float x, y, z;
            std::memcpy(&x, buffer.data() + x_offset, sizeof(float));
            std::memcpy(&y, buffer.data() + y_offset, sizeof(float));
            std::memcpy(&z, buffer.data() + z_offset, sizeof(float));
            
            points(i, 0) = static_cast<double>(x);
            points(i, 1) = static_cast<double>(y);
            points(i, 2) = static_cast<double>(z);
        }
    } else {
        // ASCII format
        for (int i = 0; i < num_vertices; ++i) {
            std::getline(file, line);
            std::istringstream iss(line);
            iss >> points(i, 0) >> points(i, 1) >> points(i, 2);
        }
    }

    return points;
}


/**
 * Voxel grid downsampling.
 * Each voxel keeps the centroid of all points within it.
 */
Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> voxel_downsample(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& points,
    double voxel_size
) {
    if (voxel_size <= 0) return points;

    // Use long long for voxel indices to handle large coordinates
    struct VoxelKey {
        long long x, y, z;
        
        bool operator==(const VoxelKey& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    // Better hash function using prime multipliers
    struct VoxelHash {
        size_t operator()(const VoxelKey& v) const {
            // Large primes for better distribution
            size_t h = 0;
            h ^= std::hash<long long>{}(v.x) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<long long>{}(v.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<long long>{}(v.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    // Accumulate points per voxel
    std::unordered_map<VoxelKey, std::vector<int>, VoxelHash> voxel_map;

    for (int i = 0; i < points.rows(); ++i) {
        VoxelKey key;
        key.x = static_cast<long long>(std::floor(points(i, 0) / voxel_size));
        key.y = static_cast<long long>(std::floor(points(i, 1) / voxel_size));
        key.z = static_cast<long long>(std::floor(points(i, 2) / voxel_size));
        voxel_map[key].push_back(i);
    }

    // Compute centroids
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> downsampled(voxel_map.size(), 3);
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


/**
 * Extract timestamp from filename like "PC_3159677985.ply"
 */
long long extract_timestamp(const std::string& filename) {
    std::regex pattern(R"((\d+)\.ply)");
    std::smatch match;
    if (std::regex_search(filename, match, pattern)) {
        return std::stoll(match[1].str());
    }
    return -1;
}


/**
 * Get sorted list of frame files.
 */
std::vector<std::pair<long long, std::string>> get_sorted_frames(const std::string& data_dir) {
    std::vector<std::pair<long long, std::string>> frames;

    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.path().extension() == ".ply") {
            std::string filename = entry.path().filename().string();
            long long timestamp = extract_timestamp(filename);
            if (timestamp >= 0) {
                frames.emplace_back(timestamp, entry.path().string());
            }
        }
    }

    // Sort by timestamp
    std::sort(frames.begin(), frames.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    return frames;
}


/**
 * Save trajectory to CSV file.
 */
void save_trajectory(const std::vector<Eigen::Vector3d>& positions, const std::string& filepath) {
    std::ofstream file(filepath);
    file << "x,y,z\n";
    file << std::fixed << std::setprecision(6);
    for (const auto& pos : positions) {
        file << pos.x() << "," << pos.y() << "," << pos.z() << "\n";
    }
    std::cout << "Saved trajectory to " << filepath << "\n";
}


/**
 * Save trajectory as simple gnuplot-compatible format.
 */
void save_trajectory_gnuplot(const std::vector<Eigen::Vector3d>& positions, const std::string& filepath) {
    std::ofstream file(filepath);
    file << "# x y z\n";
    file << std::fixed << std::setprecision(6);
    for (const auto& pos : positions) {
        file << pos.x() << " " << pos.y() << " " << pos.z() << "\n";
    }
    std::cout << "Saved gnuplot trajectory to " << filepath << "\n";
}

/**
 * Transform a point cloud from vehicle frame to world frame.
 */
Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> transform_cloud(
    const icp::Transformation& pose,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& cloud
) {
    // Apply transformation: cloud_world = cloud * R^T + t^T
    return (cloud * pose.R().transpose()).rowwise() + pose.t().transpose();
}


/**
 * Build a 2D occupancy grid from accumulated point clouds.
 */
std::unordered_set<GridCell, GridCellHash> build_occupancy_grid(
    const std::vector<icp::Transformation>& poses,
    const std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>& clouds,
    const OccupancyGridConfig& config
) {
    std::unordered_set<GridCell, GridCellHash> occupied_cells;
    
    std::cout << "\nBuilding occupancy grid...\n";
    std::cout << "  Resolution: " << config.resolution << "m\n";
    std::cout << "  Height filter: [" << config.height_min << ", " << config.height_max << "]m\n";
    std::cout << "  Max range: " << config.max_range << "m\n";
    std::cout << "  Subsample rate: " << config.subsample_rate << "\n";
    
    int frames_used = 0;
    
    for (size_t i = 0; i < clouds.size(); ++i) {
        // Subsample frames
        if (i % config.subsample_rate != 0) continue;
        if (i >= poses.size()) continue;
        
        const auto& cloud = clouds[i];
        const auto& pose = poses[i];
        
        // Filter by range (distance from sensor origin)
        std::vector<int> valid_indices;
        for (int j = 0; j < cloud.rows(); ++j) {
            double dist = cloud.row(j).norm();
            if (dist < config.max_range && dist > 0.5) {  // Also filter very close points
                valid_indices.push_back(j);
            }
        }
        
        if (valid_indices.empty()) continue;
        
        // Extract valid points
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> cloud_filtered(valid_indices.size(), 3);
        for (size_t j = 0; j < valid_indices.size(); ++j) {
            cloud_filtered.row(j) = cloud.row(valid_indices[j]);
        }
        
        // Transform to world frame
        auto cloud_world = transform_cloud(pose, cloud_filtered);
        
        // Filter by height and add to grid
        for (int j = 0; j < cloud_world.rows(); ++j) {
            double z = cloud_world(j, 2);
            if (z > config.height_min && z < config.height_max) {
                GridCell cell;
                cell.x = static_cast<int>(std::floor(cloud_world(j, 0) / config.resolution));
                cell.y = static_cast<int>(std::floor(cloud_world(j, 1) / config.resolution));
                occupied_cells.insert(cell);
            }
        }
        
        frames_used++;
    }
    
    std::cout << "  Frames used: " << frames_used << "\n";
    std::cout << "  Occupied cells: " << occupied_cells.size() << "\n";
    
    return occupied_cells;
}


/**
 * Save occupancy grid to CSV file.
 */
void save_occupancy_grid(
    const std::unordered_set<GridCell, GridCellHash>& grid,
    double resolution,
    const std::string& filepath
) {
    std::ofstream file(filepath);
    file << "x,y\n";
    file << std::fixed << std::setprecision(2);
    
    for (const auto& cell : grid) {
        double x = cell.x * resolution;
        double y = cell.y * resolution;
        file << x << "," << y << "\n";
    }
    
    std::cout << "Saved occupancy grid to " << filepath << "\n";
}

/**
 * Run odometry on sequential frames.
 */
void run_odometry(
    const std::string& data_dir,
    const std::string& output_dir,
    const OdometryConfig& config,
    const OccupancyGridConfig& grid_config,
    int max_frames = -1,
    bool build_map = true
) {
    // Get sorted frames
    auto frames = get_sorted_frames(data_dir);
    
    if (frames.empty()) {
        throw std::runtime_error("No PLY files found in " + data_dir);
    }

    int n_frames = (max_frames > 0) ? std::min(max_frames, static_cast<int>(frames.size())) 
                                     : static_cast<int>(frames.size());

    std::cout << "Found " << frames.size() << " frames, processing " << n_frames << "\n";
    std::cout << "Voxel size: " << config.voxel_size << "m\n";
    std::cout << "Build occupancy map: " << (build_map ? "yes" : "no") << "\n";
    std::cout << "Processing...\n\n";

    // Initialize loop closure detector
    icp::LoopClosureConfig lc_config;
    lc_config.frame_gap = 50;
    lc_config.sc_distance_threshold = 0.15;
    lc_config.icp_fitness_threshold = 0.15;
    icp::LoopClosureDetector loop_detector(lc_config);

    // Storage
    std::vector<Eigen::Vector3d> positions;
    std::vector<icp::Transformation> poses;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> raw_clouds;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> downsampled_clouds;
    
    positions.push_back(Eigen::Vector3d::Zero());
    poses.push_back(icp::Transformation::identity());

    // Initialize pose graph
    icp::PoseGraph pose_graph;
    pose_graph.addPrior(0, icp::Transformation::identity());

    // Load first frame
    auto prev_points_raw = load_ply(frames[0].second);
    
    std::cout << "First frame statistics:\n";
    std::cout << "  Raw points: " << prev_points_raw.rows() << "\n";
    std::cout << "  X range: [" << prev_points_raw.col(0).minCoeff() << ", " << prev_points_raw.col(0).maxCoeff() << "]\n";
    std::cout << "  Y range: [" << prev_points_raw.col(1).minCoeff() << ", " << prev_points_raw.col(1).maxCoeff() << "]\n";
    std::cout << "  Z range: [" << prev_points_raw.col(2).minCoeff() << ", " << prev_points_raw.col(2).maxCoeff() << "]\n";
    
    if (build_map) {
        raw_clouds.push_back(prev_points_raw);
    }
    
    auto prev_points = voxel_downsample(prev_points_raw, config.voxel_size);
    std::cout << "  After downsampling: " << prev_points.rows() << " points\n\n";

    // Add first frame to loop closure detector
    downsampled_clouds.push_back(prev_points);
    loop_detector.addFrame(prev_points, 0);

    // Process frames
    double total_time = 0;
    int failed_frames = 0;
    int total_loop_closures = 0;

    for (int i = 1; i < n_frames; ++i) {
        // Load current frame
        auto curr_points_raw = load_ply(frames[i].second);
        
        if (build_map) {
            raw_clouds.push_back(curr_points_raw);
        }
        
        auto curr_points = voxel_downsample(curr_points_raw, config.voxel_size);

        if (curr_points.rows() < config.min_points) {
            std::cout << "Frame " << i << ": Skipped (only " << curr_points.rows() << " points)\n";
            failed_frames++;
            poses.push_back(poses.back());
            positions.push_back(positions.back());
            downsampled_clouds.push_back(curr_points);
            loop_detector.addFrame(curr_points, i);
            prev_points = curr_points;
            continue;
        }

        // Create point clouds
        icp::PointCloud source(curr_points);
        icp::PointCloud target(prev_points);

        // Run ICP
        auto start = std::chrono::high_resolution_clock::now();
        
        icp::ICPConfig icp_config;
        icp_config.max_iterations = config.max_iterations;
        icp_config.tolerance = config.tolerance;
        
        icp::ICPResult result = icp::icp_point_to_plane(source, target, icp_config);
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        total_time += elapsed;

        // Get delta transform
        icp::Transformation delta_transform;
        if (!result.converged || result.final_error > 1.0) {
            std::cout << "Frame " << i << ": ICP failed (error=" << result.final_error << ")\n";
            failed_frames++;
            delta_transform = icp::Transformation::identity();
        } else {
            delta_transform = result.transformation;
        }

        // Accumulate pose
        icp::Transformation new_pose = poses.back() * delta_transform.inverse();
        poses.push_back(new_pose);
        positions.push_back(new_pose.t());

        // Add odometry factor to pose graph
        pose_graph.addOdometryFactor(
            poses.size() - 2,
            poses.size() - 1,
            delta_transform.inverse(),
            result.final_error
        );

        // Store cloud and add to loop closure detector
        // downsampled_clouds.push_back(curr_points);
        // loop_detector.addFrame(curr_points, i);

        // // Detect loop closures
        // auto loop_closures = loop_detector.detect();
        
        // for (const auto& lc : loop_closures) {
        //     std::cout << "  Loop closure: frame " << lc.query_frame 
        //               << " -> frame " << lc.match_frame
        //               << " (SC: " << std::fixed << std::setprecision(3) << lc.scan_context_distance 
        //               << ", ICP: " << lc.icp_fitness << ")\n";
            
        //     pose_graph.addLoopClosureEdge(
        //         lc.match_frame,
        //         lc.query_frame,
        //         lc.transform
        //     );
        //     total_loop_closures++;
        // }

        // Update previous frame
        prev_points = curr_points;

        // Progress output
        if (i % 10 == 0 || i == n_frames - 1) {
            std::cout << "Frame " << i << "/" << n_frames - 1 
                      << ": " << curr_points.rows() << " pts"
                      << ", error=" << std::scientific << std::setprecision(2) << result.final_error
                      << ", iters=" << result.num_iterations
                      << ", time=" << std::fixed << std::setprecision(1) << elapsed << "ms\n";
        }
    }

    // Summary before optimization
    std::cout << "\n=== Loop Closure Summary ===\n";
    std::cout << "Total loop closures detected: " << total_loop_closures << "\n";

    // Add skip connections and optimize
    std::cout << "\nOptimizing pose graph...\n";
    pose_graph.addSkipConnections(downsampled_clouds, 2);

    if (pose_graph.optimize()) {
        std::cout << "  Optimization converged in " << pose_graph.getIterations() << " iterations\n";
        std::cout << "  Final error: " << pose_graph.getFinalError() << "\n";
        
        std::vector<icp::Transformation> optimized_poses = pose_graph.getAllPoses();
        poses = optimized_poses;
        
        positions.clear();
        for (const auto& pose : poses) {
            positions.push_back(pose.t());
        }
        
        std::cout << "  Replaced " << poses.size() << " poses with optimized versions\n";
    } else {
        std::cout << "  Optimization failed, using raw odometry poses\n";
    }
    // Summary
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "Odometry Done!\n";
    std::cout << "Total frames processed: " << positions.size() << "\n";
    std::cout << "Failed frames: " << failed_frames << "\n";
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time / 1000.0 << "s\n";
    std::cout << "Avg time per frame: " << total_time / (n_frames - 1) << "ms\n";

    // Compute trajectory stats
    double total_distance = 0;
    for (size_t i = 1; i < positions.size(); ++i) {
        total_distance += (positions[i] - positions[i-1]).norm();
    }
    std::cout << "Total distance traveled: " << total_distance << "m\n";
    std::cout << "============================================================\n";

    // Create output directory
    fs::create_directories(output_dir);

    // Save trajectory
    save_trajectory(positions, output_dir + "/trajectory.csv");
    save_trajectory_gnuplot(positions, output_dir + "/trajectory.dat");

    // Build and save occupancy grid
    if (build_map && !raw_clouds.empty()) {
        auto occupancy_grid = build_occupancy_grid(poses, raw_clouds, grid_config);
        save_occupancy_grid(occupancy_grid, grid_config.resolution, output_dir + "/occupancy_grid.csv");
    }
}

}  // namespace


int main(int argc, char* argv[]) {
    std::cout << "============================================================\n";
    std::cout << "LiDAR Odometry using Point-to-Point ICP (C++)\n";
    std::cout << "============================================================\n\n";

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data_dir> [options]\n";
        std::cerr << "\nOptions:\n";
        std::cerr << "  --voxel-size <float>     Voxel size for downsampling (default: 0.2)\n";
        std::cerr << "  --max-frames <int>       Maximum frames to process (default: all)\n";
        std::cerr << "  --output-dir <path>      Output directory (default: <data_dir>/odometry_results)\n";
        std::cerr << "  --no-map                 Skip occupancy grid building\n";
        std::cerr << "  --grid-resolution <float> Occupancy grid cell size (default: 0.5)\n";
        std::cerr << "  --height-min <float>     Min height filter for obstacles (default: 0.3)\n";
        std::cerr << "  --height-max <float>     Max height filter for obstacles (default: 2.0)\n";
        std::cerr << "  --max-range <float>      Max range from sensor (default: 40.0)\n";
        std::cerr << "  --subsample <int>        Use every Nth frame for map (default: 2)\n";
        std::cerr << "\nExample:\n";
        std::cerr << "  " << argv[0] << " ./lidar_data --voxel-size 0.3 --max-frames 50\n";
        return 1;
    }

    std::string data_dir = argv[1];
    std::string output_dir = "/home/kaushik/Coding/LiDAR-ICP-Algorithm-from-scratch/python/odometry_results";
    OdometryConfig config;
    OccupancyGridConfig grid_config;
    int max_frames = -1;
    bool build_map = true;

    // Parse arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--voxel-size" && i + 1 < argc) {
            config.voxel_size = std::stod(argv[++i]);
        } else if (arg == "--max-frames" && i + 1 < argc) {
            max_frames = std::stoi(argv[++i]);
        } else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--no-map") {
            build_map = false;
        } else if (arg == "--grid-resolution" && i + 1 < argc) {
            grid_config.resolution = std::stod(argv[++i]);
        } else if (arg == "--height-min" && i + 1 < argc) {
            grid_config.height_min = std::stod(argv[++i]);
        } else if (arg == "--height-max" && i + 1 < argc) {
            grid_config.height_max = std::stod(argv[++i]);
        } else if (arg == "--max-range" && i + 1 < argc) {
            grid_config.max_range = std::stod(argv[++i]);
        } else if (arg == "--subsample" && i + 1 < argc) {
            grid_config.subsample_rate = std::stoi(argv[++i]);
        }
    }

    std::cout << "Data directory: " << data_dir << "\n";
    std::cout << "Output directory: " << output_dir << "\n\n";

    try {
        run_odometry(data_dir, output_dir, config, grid_config, max_frames, build_map);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}