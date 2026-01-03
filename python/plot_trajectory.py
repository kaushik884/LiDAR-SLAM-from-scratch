#!/usr/bin/env python3
"""Plot trajectory from CSV file."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Load trajectory
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "trajectory.csv")

data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
x, y, z = data[:, 0], data[:, 1], data[:, 2]

fig = plt.figure(figsize=(15, 5))

# 3D view
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(x, y, z, 'b-', linewidth=1)
ax1.scatter(x[0], y[0], z[0], c='green', s=100, marker='o', label='Start')
ax1.scatter(x[-1], y[-1], z[-1], c='red', s=100, marker='x', label='End')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.set_title('3D Trajectory')
ax1.legend()

# Top-down (X-Y)
ax2 = fig.add_subplot(132)
ax2.plot(x, y, 'b-', linewidth=1)
ax2.scatter(x[0], y[0], c='green', s=100, marker='o', label='Start')
ax2.scatter(x[-1], y[-1], c='red', s=100, marker='x', label='End')
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_title('Top-Down View')
ax2.axis('equal')
ax2.grid(True)
ax2.legend()

# Side view (X-Z)
ax3 = fig.add_subplot(133)
ax3.plot(x, z, 'b-', linewidth=1)
ax3.scatter(x[0], z[0], c='green', s=100, marker='o', label='Start')
ax3.scatter(x[-1], z[-1], c='red', s=100, marker='x', label='End')
ax3.set_xlabel('X (m)')
ax3.set_ylabel('Z (m)')
ax3.set_title('Side View')
ax3.axis('equal')
ax3.grid(True)
ax3.legend()

plt.suptitle(f'LiDAR Odometry Trajectory ({len(x)} frames)')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'trajectory.png'), dpi=150, bbox_inches='tight')
print(f"Saved trajectory.png")
plt.show()