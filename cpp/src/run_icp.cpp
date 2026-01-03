#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>

#include "icp_p2p/icp.hpp"

namespace {

/**
 * Load point cloud from PLY file (ASCII format).
 */
icp::PointCloud load_ply(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    std::string line;
    int num_vertices = 0;
    bool in_header = true;

    // Parse header
    while (in_header && std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "element") {
            std::string type;
            iss >> type >> num_vertices;
        } else if (token == "end_header") {
            in_header = false;
        }
    }

    // Read points
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> points(num_vertices, 3);
    for (int i = 0; i < num_vertices; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);
        iss >> points(i, 0) >> points(i, 1) >> points(i, 2);
    }

    return icp::PointCloud(std::move(points));
}


/**
 * Load ground truth transformation from NPZ-like format.
 * (Simple text format since we can't read NPZ directly)
 */
struct GroundTruth {
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
};

GroundTruth load_ground_truth(const std::string& filepath) {
    // We'll create a simple text format since NPZ is complex
    // For now, hardcode the values we used in Python
    GroundTruth gt;
    
    if (filepath.find("bunny") != std::string::npos) {
        double angle_z = 0.15, angle_x = 0.1;
        Eigen::Matrix3d Rz, Rx;
        Rz << cos(angle_z), -sin(angle_z), 0,
              sin(angle_z),  cos(angle_z), 0,
              0, 0, 1;
        Rx << 1, 0, 0,
              0, cos(angle_x), -sin(angle_x),
              0, sin(angle_x),  cos(angle_x);
        gt.R = Rz * Rx;
        gt.t << 0.2, -0.1, 0.15;
    } else if (filepath.find("wave") != std::string::npos) {
        double angle_z = 0.2;
        gt.R << cos(angle_z), -sin(angle_z), 0,
                sin(angle_z),  cos(angle_z), 0,
                0, 0, 1;
        gt.t << 0.3, 0.1, 0.05;
    } else if (filepath.find("room") != std::string::npos) {
        double angle_z = 0.1, angle_y = 0.08;
        Eigen::Matrix3d Rz, Ry;
        Rz << cos(angle_z), -sin(angle_z), 0,
              sin(angle_z),  cos(angle_z), 0,
              0, 0, 1;
        Ry << cos(angle_y), 0, sin(angle_y),
              0, 1, 0,
              -sin(angle_y), 0, cos(angle_y);
        gt.R = Rz * Ry;
        gt.t << 0.15, 0.1, 0.05;
    } else {
        gt.R = Eigen::Matrix3d::Identity();
        gt.t = Eigen::Vector3d::Zero();
    }
    
    return gt;
}


/**
 * Compute rotation error in degrees.
 */
double rotation_error_deg(const Eigen::Matrix3d& R_est, const Eigen::Matrix3d& R_gt) {
    Eigen::Matrix3d R_diff = R_est.transpose() * R_gt;
    double trace = R_diff.trace();
    double cos_angle = (trace - 1.0) / 2.0;
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle));  // Clamp for numerical stability
    return std::acos(cos_angle) * 180.0 / M_PI;
}


/**
 * Run ICP and report results.
 */
void run_test(const std::string& name, const std::string& data_dir) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Testing: " << name << "\n";
    std::cout << std::string(60, '=') << "\n";

    // Load data
    std::string source_path = data_dir + "/" + name + "_source.ply";
    std::string target_path = data_dir + "/" + name + "_target.ply";

    icp::PointCloud source = load_ply(source_path);
    icp::PointCloud target = load_ply(target_path);

    std::cout << "Source points: " << source.size() << "\n";
    std::cout << "Target points: " << target.size() << "\n";

    // Load ground truth
    GroundTruth gt = load_ground_truth(name);
    double gt_angle = std::acos((gt.R.trace() - 1.0) / 2.0) * 180.0 / M_PI;
    std::cout << "Ground truth rotation: " << std::fixed << std::setprecision(2) 
              << gt_angle << " degrees\n";
    std::cout << "Ground truth translation: [" << gt.t.transpose() << "]\n";

    // Run ICP
    std::cout << "\nRunning ICP...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    icp::ICPConfig config;
    config.max_iterations = 100;
    icp::ICPResult result = icp::icp_point_to_point(source, target, config);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Report results
    std::cout << "\nResults:\n";
    std::cout << "  Converged: " << (result.converged ? "true" : "false") << "\n";
    std::cout << "  Iterations: " << result.num_iterations << "\n";
    std::cout << "  Final error: " << std::scientific << std::setprecision(6) 
              << result.final_error << "\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(2) 
              << duration.count() / 1000.0 << " ms\n";

    // Accuracy
    double rot_err = rotation_error_deg(result.transformation.R(), gt.R);
    double trans_err = (result.transformation.t() - gt.t).norm();

    std::cout << "\nAccuracy:\n";
    std::cout << "  Rotation error: " << std::fixed << std::setprecision(4) 
              << rot_err << " degrees\n";
    std::cout << "  Translation error: " << std::scientific << std::setprecision(6) 
              << trans_err << "\n";
}

}  // namespace


int main(int argc, char* argv[]) {
    std::string data_dir = "../data";
    
    if (argc > 1) {
        data_dir = argv[1];
    }

    std::cout << "ICP Point-to-Point Registration (C++ Implementation)\n";
    std::cout << "Data directory: " << data_dir << "\n";

    try {
        run_test("bunny", data_dir);
        run_test("wave", data_dir);
        run_test("room", data_dir);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "All tests complete!\n";
    std::cout << std::string(60, '=') << "\n";

    return 0;
}