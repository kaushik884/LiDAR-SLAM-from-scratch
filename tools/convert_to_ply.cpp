#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cstring>

namespace fs = std::filesystem;

struct Point {
    float x, y, z, intensity;
};

bool convertFile(const std::string& inputPath, const std::string& outputPath) {
    // Open input binary file
    std::ifstream inFile(inputPath, std::ios::binary | std::ios::ate);
    if (!inFile.is_open()) {
        std::cerr << "Error: Cannot open input file: " << inputPath << std::endl;
        return false;
    }

    // Get file size and calculate number of points
    std::streamsize fileSize = inFile.tellg();
    inFile.seekg(0, std::ios::beg);

    // KITTI format: 4 floats per point (x, y, z, intensity)
    size_t numPoints = fileSize / (4 * sizeof(float));
    
    if (fileSize % (4 * sizeof(float)) != 0) {
        std::cerr << "Warning: File size not divisible by point size, truncating" << std::endl;
    }

    // Read all points
    std::vector<Point> points(numPoints);
    inFile.read(reinterpret_cast<char*>(points.data()), numPoints * sizeof(Point));
    inFile.close();

    // Write PLY file
    std::ofstream outFile(outputPath, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "Error: Cannot open output file: " << outputPath << std::endl;
        return false;
    }

    // Write PLY header (ASCII)
    std::string header = 
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex " + std::to_string(numPoints) + "\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float intensity\n"
        "end_header\n";
    
    outFile.write(header.c_str(), header.size());

    // Write binary point data
    // Memory layout is already correct (4 consecutive floats per point)
    outFile.write(reinterpret_cast<const char*>(points.data()), numPoints * sizeof(Point));
    outFile.close();

    std::cout << "Converted: " << inputPath << " -> " << outputPath 
              << " (" << numPoints << " points)" << std::endl;

    return true;
}

bool convertDirectory(const std::string& inputDir, const std::string& outputDir) {
    // Create output directory if it doesn't exist
    if (!fs::exists(outputDir)) {
        fs::create_directories(outputDir);
    }

    int converted = 0;
    int failed = 0;

    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".bin") {
            std::string inputPath = entry.path().string();
            std::string outputPath = outputDir + "/" + entry.path().stem().string() + ".ply";

            if (convertFile(inputPath, outputPath)) {
                converted++;
            } else {
                failed++;
            }
        }
    }

    std::cout << "\nConversion complete: " << converted << " succeeded, " 
              << failed << " failed" << std::endl;

    return failed == 0;
}

void printUsage(const char* programName) {
    std::cout << "KITTI .bin to .ply Converter\n"
              << "Usage:\n"
              << "  " << programName << " <input.bin> <output.ply>      Convert single file\n"
              << "  " << programName << " -d <input_dir> <output_dir>   Convert all .bin files in directory\n"
              << "\nExample:\n"
              << "  " << programName << " 000000.bin 000000.ply\n"
              << "  " << programName << " -d velodyne/ ply_output/\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    // Directory mode
    if (std::strcmp(argv[1], "-d") == 0) {
        if (argc < 4) {
            printUsage(argv[0]);
            return 1;
        }
        return convertDirectory(argv[2], argv[3]) ? 0 : 1;
    }

    // Single file mode
    return convertFile(argv[1], argv[2]) ? 0 : 1;
}