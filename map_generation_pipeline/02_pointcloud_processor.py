#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import struct
import math
from pathlib import Path
from pyproj import Proj
import json

# Native ROS 2 Imports
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# ==========================================
# Configuration
# ==========================================
BAG_PATH = 'carla_raw_data/phase1_raw_data_0.mcap'
LIDAR_TOPIC = '/carla/hero/lidar/point_cloud'
GNSS_TOPIC = '/carla/hero/gnss'
OUTPUT_PCD = 'phase2_road_surface.pcd'

SKIP_FRAMES = 100  # Skip first few seconds to let CARLA autopilot accelerate
MAX_FRAMES = 200   # Extract a specific window of driving data

def parse_pointcloud2(msg):
    """
    Deserializes a ROS 2 PointCloud2 binary payload into a NumPy array (N, 3).
    Assumes standard float32 X, Y, Z fields at the start of the point step.
    """
    num_points = msg.width * msg.height
    
    dt = np.dtype([
        ('x', np.float32), 
        ('y', np.float32), 
        ('z', np.float32), 
        ('padding', np.uint8, msg.point_step - 12) 
    ])
    
    # Convert native ROS 2 array.array or sequence to bytes for numpy
    cloud_data = np.frombuffer(bytes(msg.data), dtype=dt)
    points = np.vstack([cloud_data['x'], cloud_data['y'], cloud_data['z']]).T
    return points

def get_rotation_matrix_z(yaw):
    """Returns a 3x3 rotation matrix for a given yaw angle (around the Z axis)."""
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    return np.array([
        [cos_y, -sin_y, 0],
        [sin_y,  cos_y, 0],
        [0,      0,     1]
    ])

def create_native_reader(bag_path):
    """Helper to initialize the native rosbag2_py reader for MCAP."""
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id='mcap'
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader

def main():
    print(f"Opening native ROS 2 bag: {BAG_PATH}")
    
    gnss_records = []
    
    # ---------------------------------------------------------
    # PASS 1: Extract all GNSS data for Time-Syncing
    # ---------------------------------------------------------
    reader = create_native_reader(BAG_PATH)
    
    # Get topic types to dynamically deserialize
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        
        if topic == GNSS_TOPIC:
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            
            # Timestamp in nanoseconds
            t_nano = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
            gnss_records.append((t_nano, msg.latitude, msg.longitude, msg.altitude))

    if not gnss_records:
        raise ValueError("No GNSS data found in the bag!")
        
    print(f"Extracted {len(gnss_records)} GNSS frames. Establishing global origin...")

    gnss_records.sort(key=lambda x: x[0])
    gnss_times = np.array([r[0] for r in gnss_records])

    # ---------------------------------------------------------
    # GNSS to Cartesian Projection (Transverse Mercator)
    # ---------------------------------------------------------
    origin_lat = gnss_records[0][1]
    origin_lon = gnss_records[0][2]
    origin_alt = gnss_records[0][3]
    
    proj = Proj(proj='tmerc', lat_0=origin_lat, lon_0=origin_lon, ellps='WGS84')

    # ---------------------------------------------------------
    # PASS 2: Extract LiDAR, Time-Sync, Transform, and Aggregate
    # ---------------------------------------------------------
    # Re-initialize reader to scan from the beginning for LiDAR
    reader = create_native_reader(BAG_PATH)
    
    master_point_cloud = []
    trajectory_path = []
    
    total_lidar_frames = 0
    extracted_count = 0
    
    prev_x, prev_y = 0.0, 0.0
    start_x, start_y, start_z = 0.0, 0.0, 0.0
    current_yaw = 0.0

    print("Beginning LiDAR extraction and coordinate transformation...")
    
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        
        if topic == LIDAR_TOPIC:
            total_lidar_frames += 1
            
            # Skip the warm-up phase
            if total_lidar_frames <= SKIP_FRAMES:
                continue
                
            # Stop once we have our desired window
            if extracted_count >= MAX_FRAMES:
                break
                
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            
            lidar_t = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
            
            # Find closest GNSS timestamp
            closest_idx = np.abs(gnss_times - lidar_t).argmin()
            _, lat, lon, alt = gnss_records[closest_idx]
            
            # Project Lat/Lon to local Metric (X, Y)
            x, y = proj(lon, lat)
            z = alt - origin_alt
            
            # Set the origin to the first *extracted* frame to properly anchor the translation
            if extracted_count == 0:
                start_x, start_y, start_z = x, y, z
                prev_x, prev_y = x, y
            
            # Calculate cumulative distance traveled from the start of the extraction window
            dx_moved = x - start_x
            dy_moved = y - start_y
            dz_moved = z - start_z

            trajectory_path.append([float(dx_moved), float(dy_moved)])
            
            # Calculate Heading (Yaw) using the trajectory vector between consecutive readings
            dx_step = x - prev_x
            dy_step = y - prev_y
            
            if math.hypot(dx_step, dy_step) > 0.1:
                current_yaw = math.atan2(dy_step, dx_step)
                
            prev_x, prev_y = x, y
            
            # Deserialize local LiDAR points
            local_points = parse_pointcloud2(msg)
            
            # Apply Rigid Body Transformation: Rotation (Yaw) + Translation from extracted origin
            R = get_rotation_matrix_z(current_yaw)
            translation = np.array([dx_moved, dy_moved, dz_moved])
            
            global_points = (local_points @ R.T) + translation
            
            # Accumulate
            master_point_cloud.append(global_points)
            extracted_count += 1
            
            # Debug logging
            print(f"Frame {extracted_count}: Car moved {dx_moved:.2f}m laterally, {dy_moved:.2f}m longitudinally")

    # ---------------------------------------------------------
    # Aggregation & Filtering (Open3D)
    # ---------------------------------------------------------
    if not master_point_cloud:
        print("Error: No LiDAR frames were extracted. Check your SKIP_FRAMES threshold and bag length.")
        return

    print("Stacking transformed arrays into massive point cloud...")
    stacked_points = np.vstack(master_point_cloud)
    
    print(f"Total raw points gathered: {stacked_points.shape[0]:,}")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(stacked_points)
    
    print("Applying Voxel Down-sampling (0.2m resolution)...")
    pcd_down = pcd.voxel_down_sample(voxel_size=0.2)
    
    print(f"Points after down-sampling: {len(pcd_down.points):,}")
    
    print("Applying RANSAC to segment the flat road surface...")
    plane_model, inliers = pcd_down.segment_plane(distance_threshold=0.05, 
                                                  ransac_n=3, 
                                                  num_iterations=1000)
    
    print(f"Road surface model [a, b, c, d]: {plane_model}")
    road_surface_cloud = pcd_down.select_by_index(inliers)
    
    # ---------------------------------------------------------
    # Export
    # ---------------------------------------------------------
    print(f"Exporting isolated road surface ({len(road_surface_cloud.points):,} points) to {OUTPUT_PCD}...")
    o3d.io.write_point_cloud(OUTPUT_PCD, road_surface_cloud)

    # <-- ADD THIS BLOCK
    print("Exporting trajectory and origin metadata to phase2_metadata.json...")
    metadata = {
        "origin_lat": float(origin_lat),
        "origin_lon": float(origin_lon),
        "trajectory": trajectory_path
    }
    with open("phase2_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("Pipeline complete.")

if __name__ == '__main__':
    main()