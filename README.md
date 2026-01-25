# LiDAR SLAM from Scratch

A complete LiDAR SLAM (Simultaneous Localization and Mapping) system built from the ground up in C++. This project demonstrates deep understanding of robotics fundamentals by implementing core algorithms without relying on existing SLAM libraries.

## Features

- **Point-to-Plane ICP**: Frame-to-frame odometry with surface normal estimation
- **Scan Context**: Rotation-invariant place recognition descriptors
- **Loop Closure Detection**: Two-stage detection (Scan Context filtering + ICP verification)
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
│                         (Scan Context + ICP Verification)       │
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
LiDAR-SLAM-from-scratch/
├── slam_viz/                           # ROS2 package (main codebase)
│   ├── CMakeLists.txt
│   ├── package.xml
│   │
│   ├── include/slam_viz/
│   │   ├── core/                       # Pure C++ algorithms (no ROS2 dependency)
│   │   │   ├── types.hpp               # PointCloud, Transformation, ICPResult
│   │   │   ├── kdtree.hpp              # KD-tree for nearest neighbor search
│   │   │   ├── icp.hpp                 # Point-to-Plane ICP with normal estimation
│   │   │   ├── scan_context.hpp        # Place recognition descriptor
│   │   │   ├── loop_closure.hpp        # Loop closure detection pipeline
│   │   │   ├── pose_graph.hpp          # GTSAM wrapper declaration
│   │   │   └── file_utils.hpp          # PLY/BIN loading, voxel downsampling
│   │   │
│   │   └── ros/
│   │       └── slam_node.hpp           # ROS2 node declaration
│   │
│   ├── src/
│   │   ├── core/                       # Algorithm implementations
│   │   │   ├── pose_graph.cpp          # GTSAM factor graph implementation
│   │   │   └── file_utils.cpp          # File I/O implementation
│   │   │
│   │   └── ros/
│   │       └── slam_node.cpp           # ROS2 node + main()
│   │
│   ├── launch/
│   │   └── slam.launch.py              # ROS2 launch file
│   │
│   └── rviz/
│       └── slam_config.rviz            # RViz2 visualization config
│
├── tools/
│   └── convert_to_ply.cpp              # KITTI .bin to .ply converter
│
├── data/                               # Your LiDAR datasets
│   ├── kitti/
│   └── your_lidar_data/
│
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

#### 2. Intel TBB (Threading Building Blocks)
```bash
sudo apt install libtbb-dev
```

#### 3. GTSAM (Georgia Tech Smoothing and Mapping)
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
sudo ldconfig
```

#### 4. ROS2 Humble
Follow the official installation guide: https://docs.ros.org/en/humble/Installation.html

```bash
# Install ROS2 Humble (Ubuntu 22.04)
sudo apt install ros-humble-desktop

# Install additional ROS2 packages
sudo apt install ros-humble-tf2-ros ros-humble-tf2-geometry-msgs
```

## Building

### Build in a ROS2 Workspace

```bash
# Create a ROS2 workspace (if you don't have one)
mkdir -p ~/ros2_ws/src

# Symlink the slam_viz package
ln -s /path/to/LiDAR-SLAM-from-scratch/slam_viz ~/ros2_ws/src/slam_viz

# Build
cd ~/ros2_ws
colcon build --packages-select slam_viz --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source the workspace
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
./convert_to_ply -d /path/to/kitti/sequences/00/velodyne/ /path/to/data/kitti_00/
```

### Running SLAM

**Terminal 1: Launch SLAM node**
```bash
source ~/ros2_ws/install/setup.bash

# Using ros2 run
ros2 run slam_viz slam_node --ros-args \
    -p data_dir:=/path/to/data/your_lidar_data \
    -p voxel_size:=0.5 \
    -p playback_rate:=10.0

# Or using launch file
ros2 launch slam_viz slam.launch.py data_dir:=/path/to/data/your_lidar_data
```

**Terminal 2: Launch RViz2**
```bash
source ~/ros2_ws/install/setup.bash
rviz2 -d ~/ros2_ws/install/slam_viz/share/slam_viz/rviz/slam_config.rviz
```

### ROS2 Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/slam/current_scan` | `sensor_msgs/PointCloud2` | Current LiDAR frame (world frame) |
| `/slam/global_map` | `sensor_msgs/PointCloud2` | Accumulated point cloud map |
| `/slam/trajectory` | `nav_msgs/Path` | Robot trajectory |
| `/slam/current_pose` | `geometry_msgs/PoseStamped` | Current robot pose |
| `/slam/occupancy_grid` | `nav_msgs/OccupancyGrid` | 2D occupancy grid map |

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_dir` | (required) | Path to directory containing PLY files |
| `voxel_size` | 0.5 | Downsampling voxel size (meters) |
| `playback_rate` | 10.0 | Frame processing rate (Hz) |
| `max_iterations` | 50 | Maximum ICP iterations |
| `tolerance` | 1e-6 | ICP convergence tolerance |
| `grid_resolution` | 0.2 | Occupancy grid cell size (meters) |
| `height_min` | 0.3 | Minimum height for occupancy grid (meters) |
| `height_max` | 2.0 | Maximum height for occupancy grid (meters) |
| `max_range` | 40.0 | Maximum range for occupancy grid (meters) |

## Algorithm Details

### Point-to-Plane ICP

Minimizes the point-to-plane error metric:

```
E = Σ [(R·pᵢ + t - qᵢ) · nᵢ]²
```

Where `nᵢ` is the surface normal at target point `qᵢ`. Solved using Gauss-Newton optimization with linearized rotation (Rodrigues' formula).

**Key implementation details:**
- Normals estimated via PCA on k-nearest neighbors (k=20)
- LDLT decomposition for solving normal equations
- Convergence based on error change tolerance

### Scan Context

Encodes point clouds as 2D descriptors (20 rings × 60 sectors) where each cell contains the maximum point height. Rotation invariance achieved by testing all column shifts during comparison.

```
Distance = 1 - cos_similarity(SC₁, shift(SC₂, θ))
```

### Loop Closure Detection

Two-stage pipeline:
1. **Candidate Selection**: Scan Context distance < 0.25 (configurable)
2. **Verification**: ICP alignment with fitness threshold < 0.3

Only frames separated by at least 50 frames (configurable `frame_gap`) are considered as loop candidates.

### Pose Graph Optimization

Uses GTSAM's Levenberg-Marquardt optimizer with:
- **Prior factor** on first pose (anchors trajectory)
- **Between factors** for odometry constraints (noise scaled by ICP fitness)
- **Between factors** for loop closure constraints (tighter noise model)

**Key optimization:** Pose graph optimization only runs when loop closures are detected, avoiding unnecessary computation.

## Results

Tested on:
- KITTI Odometry Sequence 00 (4541 frames, multiple loop closures)
- Custom Ouster OS1-64 datasets

The system successfully:
- Detects loop closures using Scan Context
- Corrects trajectory drift through pose graph optimization
- Generates consistent occupancy grid maps

## Troubleshooting

### GTSAM not found
```bash
# Make sure GTSAM is installed and ldconfig is updated
sudo ldconfig

# Or set library path manually
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

### TBB linking error
```bash
# Install TBB development package
sudo apt install libtbb-dev
```

### ROS2 package not found
```bash
# Make sure you've sourced the workspace
source ~/ros2_ws/install/setup.bash
```

### Out of memory with large datasets
- Increase `voxel_size` for more aggressive downsampling
- The system uses a sliding window (20 frames) for visualization during SLAM
- Full map is only built after processing completes

### Slow performance
- Make sure you built with Release mode: `--cmake-args -DCMAKE_BUILD_TYPE=Release`
- Debug builds can be 20-30x slower

## Future Improvements

- [ ] GPU acceleration for ICP (CUDA)
- [ ] Incremental optimization with iSAM2
- [ ] Ground segmentation for cleaner occupancy maps
- [ ] Motion planning integration
- [ ] Multi-session mapping and localization
- [ ] Integration with IMU data

## References

- [LOAM: Lidar Odometry and Mapping in Real-time](https://www.ri.cmu.edu/pub_files/2014/7/Ji_LidarMapping_RSS2014_v8.pdf)
- [Scan Context: Egocentric Spatial Descriptor for Place Recognition](https://irap.kaist.ac.kr/publications/gkim-2018-iros.pdf)
- [GTSAM: Georgia Tech Smoothing and Mapping](https://gtsam.org/)

## License

MIT License

---

*Built from scratch to demonstrate understanding of SLAM fundamentals. No existing SLAM libraries (e.g., Cartographer, LOAM, ORB-SLAM) were used for the core algorithms.*