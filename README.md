# LiDAR SLAM from Scratch

A complete LiDAR SLAM (Simultaneous Localization and Mapping) system built from the ground up in C++. This project demonstrates deep understanding of robotics fundamentals by implementing core algorithms without relying on existing SLAM libraries.

## Features

- **Point-to-Plane ICP**: Frame-to-frame odometry with surface normal estimation
- **Scan Context**: Rotation-invariant place recognition descriptors
- **Loop Closure Detection**: Two-stage detection (ring key filtering + ICP verification)
- **Pose Graph Optimization**: Global optimization using GTSAM
- **Real-time Visualization**: Live trajectory and map visualization in RViz2
- **Occupancy Grid Mapping**: 2D grid map generation from 3D point clouds

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                │
├─────────────────────────────────────────────────────────────────┤
│  Point Cloud ──→ Downsample ──→ Point-to-Plane ICP ──→ Odometry │
│                                        │                        │
│                               Scan Context Descriptor           │
│                                        │                        │
│                          Loop Closure Detection                 │
│                         (Ring Key + ICP Verification)           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         BACKEND                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    Odometry Factors ──→ Pose Graph ←── Loop Closure Factors     │
│                             │                                   │
│                      GTSAM Optimizer                            │
│                             │                                   │
│                      Optimized Poses                            │
│                             │                                   │
│                   Global Map & Trajectory                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
LiDAR-SLAM/
├── cpp/
│   ├── include/
│   │   ├── icp/
│   │   │   ├── kdtree.hpp
│   │   │   ├── loop_closure.hpp         
│   │   │   ├── normal_estimation.hpp    
|   |   |   ├── point_to_plane.hpp       
|   |   |   ├── point_to_point.hpp
|   |   |   ├── scan_context.hpp
|   |   |   ├── svd.hpp
|   |   |   ├── types.hpp
│   │   ├── pose_graph.hpp            
│   │   └── icp.hpp
│   ├── src/
│   │   └── pose_graph.cpp
│   ├── apps/
│   │   └── lidar_odometry.cpp        
│   └── CMakeLists.txt
├── slam_viz/                         
│   ├── include/slam_viz/
│   │   └── slam_node.hpp
│   ├── src/
│   │   └── slam_node.cpp
│   ├── CMakeLists.txt
│   └── package.xml
├── tools/
│   └── convert_to_ply.cpp         
├── data/
|   ├──your_lidar_data(.ply format)
└── README.md
```

## Prerequisites

### System Requirements
- Ubuntu 22.04 (tested)
- C++17 compatible compiler (GCC 9+ or Clang 10+)

### Dependencies

#### 1. Eigen3
```bash
sudo apt install libeigen3-dev
```

#### 2. GTSAM (Georgia Tech Smoothing and Mapping)
```bash
# Install dependencies
sudo apt install libboost-all-dev cmake

# Build from source
git clone https://github.com/borglab/gtsam.git
cd gtsam
mkdir build && cd build
cmake .. -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_BUILD_TESTS=OFF
make -j$(nproc)
sudo make install
```

#### 3. ROS2 Humble (for visualization)
Follow the official installation guide: https://docs.ros.org/en/humble/Installation.html

```bash
# Install ROS2 Humble (Ubuntu 22.04)
sudo apt install ros-humble-desktop

# Install additional ROS2 packages
sudo apt install ros-humble-tf2-ros ros-humble-tf2-geometry-msgs
```

## Building

### Build Core Library (without ROS2)

```bash
cd LiDAR-SLAM
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Build with ROS2 Visualization

```bash
# Create a ROS2 workspace (if you don't have one)
mkdir -p ~/ros2_ws/src

# Symlink the slam_viz package
ln -s /path/to/LiDAR-SLAM/slam_viz ~/ros2_ws/src/slam_viz

# Build
cd ~/ros2_ws
colcon build --packages-select slam_viz --cmake-args -DCMAKE_BUILD_TYPE=Release

source install/setup.bash
```

## Usage

### Dataset Preparation

This system works with PLY point cloud files. For KITTI dataset:

1. Download KITTI odometry velodyne data from: https://www.cvlibs.net/datasets/kitti/eval_odometry.php

2. Convert .bin files to .ply:
```bash
# Build the converter
cd tools
g++ -o convert_to_ply convert_to_ply.cpp

# Convert entire sequence
./convert_to_ply -d /path/to/downloaded_dataset/ /path/to/data/your_lidar_data
```

### Running SLAM

```bash
# Terminal 1: Launch SLAM node
source ~/ros2_ws/install/setup.bash
ros2 run slam_viz slam_node --ros-args -p data_dir:=/path/to/data/your_lidar_data/


# Terminal 2: Launch RViz2 (if not included in launch file)
rviz2
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `voxel_size` | 0.5 | Downsampling voxel size (meters) |
| `max_iterations` | 50 | Maximum ICP iterations |
| `tolerance` | 1e-6 | ICP convergence tolerance |
| `frame_gap` | 50 | Minimum frames between loop candidates |
| `sc_distance_threshold` | 0.2 | Scan Context similarity threshold |
| `icp_fitness_threshold` | 0.3 | ICP verification threshold |

## Algorithm Details

### Point-to-Plane ICP

Minimizes the point-to-plane error metric:

```
E = Σ [(R·pᵢ + t - qᵢ) · nᵢ]²
```

Where `nᵢ` is the surface normal at target point `qᵢ`. Solved using Gauss-Newton optimization with a linearized rotation parameterization.

### Scan Context

Encodes point clouds as 2D descriptors (rings × sectors) where each cell contains the maximum point height. Rotation invariance achieved by testing all column shifts during comparison.

### Loop Closure Detection

Two-stage process:
1. **Candidate Selection**: Fast filtering using ring key (1D descriptor) with L1 distance
2. **Verification**: Full Scan Context comparison + ICP alignment check

### Pose Graph Optimization

Uses GTSAM's Levenberg-Marquardt optimizer with:
- Prior factor on first pose
- Between factors for odometry constraints
- Between factors for loop closure constraints

## Results

Tested on:
- KITTI Odometry Sequence 00 (with loop closures)
- Custom Ouster OS1-64 datasets

The system successfully detects loop closures and corrects trajectory drift through pose graph optimization.

## Troubleshooting

### GTSAM not found
```bash
# Make sure GTSAM is installed and ldconfig is updated
sudo ldconfig
# Or set library path manually
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

### ROS2 package not found
```bash
# Make sure you've sourced the workspace
source ~/ros2_ws/install/setup.bash
```

### Out of memory with large datasets
- Increase `voxel_size` for more aggressive downsampling
- Reduce the number of stored point clouds

## Future Improvements

- [ ] GPU acceleration for ICP (CUDA)
- [ ] Incremental optimization with iSAM2
- [ ] Ground segmentation for improved odometry
- [ ] Multi-session mapping and localization
- [ ] Integration with IMU data

## References

- [LOAM: Lidar Odometry and Mapping in Real-time](https://www.ri.cmu.edu/pub_files/2014/7/Ji_LidarMapping_RSS2014_v8.pdf)
- [Scan Context: Egocentric Spatial Descriptor for Place Recognition](https://irap.kaist.ac.kr/publications/gkim-2018-iros.pdf)
- [GTSAM: Georgia Tech Smoothing and Mapping](https://gtsam.org/)

---

*Built from scratch to demonstrate understanding of SLAM fundamentals. No existing SLAM libraries (e.g., Cartographer, LOAM, ORB-SLAM) were used for the core algorithms.*