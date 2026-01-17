import os
import struct
import numpy as np
from rosbags.rosbag1 import Reader

# List your bag files in order
bag_files = [
    "/home/kaushik/Coding/LiDAR-ICP-Algorithm-from-scratch/data/rosbag1.bag",
    "/home/kaushik/Coding/LiDAR-ICP-Algorithm-from-scratch/data/rosbag2.bag",
    "/home/kaushik/Coding/LiDAR-ICP-Algorithm-from-scratch/data/rosbag3.bag",
    "/home/kaushik/Coding/LiDAR-ICP-Algorithm-from-scratch/data/rosbag4.bag",
    "/home/kaushik/Coding/LiDAR-ICP-Algorithm-from-scratch/data/rosbag5.bag",
]

# Output directory
output_dir = "extracted_frames"
os.makedirs(output_dir, exist_ok=True)

# Topic name
topic = "/ouster/points"

def parse_pointcloud2_raw(rawdata):
    """Parse PointCloud2 message directly from raw bytes."""
    offset = 0
    
    # Skip header (seq, stamp, frame_id)
    seq = struct.unpack('<I', rawdata[offset:offset+4])[0]
    offset += 4
    stamp_sec = struct.unpack('<I', rawdata[offset:offset+4])[0]
    offset += 4
    stamp_nsec = struct.unpack('<I', rawdata[offset:offset+4])[0]
    offset += 4
    frame_id_len = struct.unpack('<I', rawdata[offset:offset+4])[0]
    offset += 4
    frame_id = rawdata[offset:offset+frame_id_len].decode('utf-8')
    offset += frame_id_len
    
    # PointCloud2 fields
    height = struct.unpack('<I', rawdata[offset:offset+4])[0]
    offset += 4
    width = struct.unpack('<I', rawdata[offset:offset+4])[0]
    offset += 4
    
    # Parse fields array
    num_fields = struct.unpack('<I', rawdata[offset:offset+4])[0]
    offset += 4
    
    fields = {}
    for _ in range(num_fields):
        name_len = struct.unpack('<I', rawdata[offset:offset+4])[0]
        offset += 4
        name = rawdata[offset:offset+name_len].decode('utf-8')
        offset += name_len
        field_offset = struct.unpack('<I', rawdata[offset:offset+4])[0]
        offset += 4
        datatype = struct.unpack('<B', rawdata[offset:offset+1])[0]
        offset += 1
        count = struct.unpack('<I', rawdata[offset:offset+4])[0]
        offset += 4
        fields[name] = {'offset': field_offset, 'datatype': datatype, 'count': count}
    
    is_bigendian = struct.unpack('<?', rawdata[offset:offset+1])[0]
    offset += 1
    point_step = struct.unpack('<I', rawdata[offset:offset+4])[0]
    offset += 4
    row_step = struct.unpack('<I', rawdata[offset:offset+4])[0]
    offset += 4
    
    data_len = struct.unpack('<I', rawdata[offset:offset+4])[0]
    offset += 4
    data = rawdata[offset:offset+data_len]
    
    # Debug: print field info on first call
    if not hasattr(parse_pointcloud2_raw, 'printed'):
        print(f"  Height: {height}, Width: {width}")
        print(f"  Point step: {point_step} bytes")
        print(f"  Fields:")
        for name, info in fields.items():
            print(f"    {name}: offset={info['offset']}, datatype={info['datatype']}, count={info['count']}")
        parse_pointcloud2_raw.printed = True
    
    # Extract XYZ - need to handle different datatypes
    # Datatype 7 = FLOAT32, Datatype 8 = FLOAT64
    num_points = height * width
    points = np.zeros((num_points, 3), dtype=np.float64)
    
    x_info = fields['x']
    y_info = fields['y']
    z_info = fields['z']
    
    # Determine format based on datatype
    # 7 = FLOAT32 (4 bytes), 8 = FLOAT64 (8 bytes)
    if x_info['datatype'] == 7:
        dtype_char = '<f'
        dtype_size = 4
    elif x_info['datatype'] == 8:
        dtype_char = '<d'
        dtype_size = 8
    else:
        raise ValueError(f"Unexpected datatype: {x_info['datatype']}")
    
    x_off = x_info['offset']
    y_off = y_info['offset']
    z_off = z_info['offset']
    
    # Vectorized extraction for speed
    data_array = np.frombuffer(data, dtype=np.uint8)
    
    for i in range(num_points):
        base = i * point_step
        points[i, 0] = struct.unpack(dtype_char, data[base + x_off:base + x_off + dtype_size])[0]
        points[i, 1] = struct.unpack(dtype_char, data[base + y_off:base + y_off + dtype_size])[0]
        points[i, 2] = struct.unpack(dtype_char, data[base + z_off:base + z_off + dtype_size])[0]
    
    # Filter invalid points
    valid = np.isfinite(points).all(axis=1)
    valid &= (np.abs(points) < 1000).all(axis=1)  # Reasonable range filter
    valid &= (np.abs(points) > 1e-6).any(axis=1)  # Remove origin points
    
    return points[valid]

def save_ply_binary(filepath, points):
    """Save points as binary PLY file."""
    header = f"""ply
format binary_little_endian 1.0
element vertex {len(points)}
property float x
property float y
property float z
end_header
"""
    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))
        points.astype(np.float32).tofile(f)

# Process all bags
frame_count = 0

for bag_path in bag_files:
    if not os.path.exists(bag_path):
        print(f"Warning: {bag_path} not found, skipping")
        continue
    
    print(f"\nProcessing {bag_path}...")
    
    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        
        if not connections:
            print(f"  No {topic} messages found")
            continue
        
        bag_frames = 0
        for conn, timestamp, rawdata in reader.messages(connections=connections):
            try:
                points = parse_pointcloud2_raw(rawdata)
                
                output_path = os.path.join(output_dir, f"{frame_count:06d}.ply")
                save_ply_binary(output_path, points)
                
                if frame_count % 50 == 0:
                    print(f"  Frame {frame_count}: {len(points)} points")
                
                frame_count += 1
                bag_frames += 1
            except Exception as e:
                print(f"  Error on frame {frame_count}: {e}")
                continue
        
        print(f"  Extracted {bag_frames} frames from {bag_path}")

print(f"\nDone! Total: {frame_count} frames in {output_dir}/")