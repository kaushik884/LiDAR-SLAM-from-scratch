#!/usr/bin/env python3
"""Plot occupancy grid with trajectory overlay."""

import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = "/home/kaushik/Coding/LiDAR-ICP-Algorithm-from-scratch/python/odometry_results"

# Load occupancy grid
grid_path = os.path.join(script_dir, "occupancy_grid.csv")
grid_data = np.genfromtxt(grid_path, delimiter=',', skip_header=1)
grid_x, grid_y = grid_data[:, 0], grid_data[:, 1]

# Load trajectory
traj_path = os.path.join(script_dir, "trajectory.csv")
traj_data = np.genfromtxt(traj_path, delimiter=',', skip_header=1)
traj_x, traj_y = traj_data[:, 0], traj_data[:, 1]

# Create figure
plt.figure(figsize=(16, 12))

# Plot occupancy grid
plt.scatter(grid_x, grid_y, c='darkblue', s=2, alpha=0.6, marker='s', label='Obstacles')

# Plot trajectory
plt.plot(traj_x, traj_y, 'r-', linewidth=2, label='Vehicle Path', alpha=0.8)
plt.scatter(traj_x[0], traj_y[0], c='green', s=200, marker='o', 
            edgecolors='white', linewidths=2, label='Start', zorder=10)
plt.scatter(traj_x[-1], traj_y[-1], c='red', s=200, marker='o',
            edgecolors='white', linewidths=2, label='End', zorder=10)

# Add direction arrows
for i in range(0, len(traj_x)-10, max(1, len(traj_x)//10)):
    dx = traj_x[min(i+10, len(traj_x)-1)] - traj_x[i]
    dy = traj_y[min(i+10, len(traj_y)-1)] - traj_y[i]
    if abs(dx) > 0.1 or abs(dy) > 0.1:
        plt.arrow(traj_x[i], traj_y[i], dx*0.5, dy*0.5,
                 head_width=1, head_length=0.5, fc='orange', ec='orange', alpha=0.7)

plt.xlabel('X (meters)', fontsize=14)
plt.ylabel('Y (meters)', fontsize=14)
plt.title('2D Occupancy Grid with Vehicle Trajectory', fontsize=16, fontweight='bold')
plt.axis('equal')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=12, loc='best')
plt.tight_layout()

# Save and show
plt.savefig(os.path.join(script_dir, 'occupancy_map.png'), dpi=150, bbox_inches='tight')
print("Saved occupancy_map.png")

# Print statistics
print(f"\n=== Map Statistics ===")
print(f"Occupied cells: {len(grid_x)}")
print(f"Map bounds:")
print(f"  X: [{grid_x.min():.1f}, {grid_x.max():.1f}] m")
print(f"  Y: [{grid_y.min():.1f}, {grid_y.max():.1f}] m")
print(f"  Width: {grid_x.max() - grid_x.min():.1f} m")
print(f"  Height: {grid_y.max() - grid_y.min():.1f} m")
print(f"Trajectory length: {np.sum(np.linalg.norm(np.diff(traj_data[:,:2], axis=0), axis=1)):.1f} m")

plt.show()