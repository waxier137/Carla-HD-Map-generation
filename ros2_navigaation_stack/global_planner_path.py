#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
import xml.etree.ElementTree as ET
import math

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class GlobalPlanner(Node):
    def __init__(self):
        super().__init__('global_planner')
        
        # ROS 2 Publisher and Timer
        self.path_pub = self.create_publisher(Path, '/global_plan', 10)
        self.timer = self.create_timer(1.0, self.publish_path)  # 1 Hz frequency
        
        # Load and parse the Lanelet2 OSM map
        self.map_file = 'phase3_lanelet2_map.osm'
        self.centerline_points = []
        self.global_path = Path()
        
        self.get_logger().info(f"Initializing Global Planner. Loading map: {self.map_file}")
        self.load_map_and_compute_centerline()

    def load_map_and_compute_centerline(self):
        if not os.path.isfile(self.map_file):
            self.get_logger().error(f"Map file '{self.map_file}' not found! Ensure it is in the current working directory.")
            return

        try:
            tree = ET.parse(self.map_file)
            root = tree.getroot()
        except ET.ParseError as e:
            self.get_logger().error(f"Failed to parse XML: {e}")
            return

        # 1. Extract all nodes -> {node_id: (x, y, z)}
        nodes = {}
        EARTH_RADIUS_EQUATORIAL = 6378137.0  # CARLA's default WGS84 earth radius in meters

        for node in root.findall('node'):
            node_id = node.get('id')
            
            # Extract geographic coordinates in degrees
            lon_deg = float(node.get('lon', 0.0))
            lat_deg = float(node.get('lat', 0.0))
            z = float(node.get('ele', 0.0))
            
            # Convert degrees to radians
            lat_rad = math.radians(lat_deg)
            lon_rad = math.radians(lon_deg)
            
            # Convert to local Cartesian metric coordinates (meters)
            metric_x = EARTH_RADIUS_EQUATORIAL * lon_rad * math.cos(lat_rad)
            metric_y = EARTH_RADIUS_EQUATORIAL * lat_rad
                    
            # Apply the ROS 2 Right-Handed conversion (invert Y)
            x = metric_x
            y = -metric_y 
            
            nodes[node_id] = (x, y, z)

        # 2. Extract all ways -> {way_id: [node_id_1, node_id_2, ...]}
        ways = {}
        for way in root.findall('way'):
            way_id = way.get('id')
            ways[way_id] = [nd.get('ref') for nd in way.findall('nd')]

        # 3. Find the lanelet relation and extract left/right ways
        left_way_id = None
        right_way_id = None
        
        for rel in root.findall('relation'):
            is_lanelet = any(tag.get('k') == 'type' and tag.get('v') == 'lanelet' for tag in rel.findall('tag'))
            if is_lanelet:
                for member in rel.findall('member'):
                    if member.get('type') == 'way':
                        if member.get('role') == 'left':
                            left_way_id = member.get('ref')
                        elif member.get('role') == 'right':
                            right_way_id = member.get('ref')
                break  # Stop after finding the first lanelet relation

        if not left_way_id or not right_way_id:
            self.get_logger().error("Could not find a valid lanelet relation with both 'left' and 'right' roles.")
            return

        # 4. Extract node IDs for the left and right ways
        left_nodes = ways.get(left_way_id, [])
        right_nodes = ways.get(right_way_id, [])

        if not left_nodes or not right_nodes:
            self.get_logger().error("Left or right way references missing nodes.")
            return

        # 5. Calculate Centerline Midpoints
        # Iterating simultaneously; zip handles arrays of equal length. 
        # (If unequal, it truncates to the shortest length).
        for left_id, right_id in zip(left_nodes, right_nodes):
            if left_id in nodes and right_id in nodes:
                lx, ly, lz = nodes[left_id]
                rx, ry, rz = nodes[right_id]
                
                mid_x = (lx + rx) / 2.0
                mid_y = (ly + ry) / 2.0
                mid_z = (lz + rz) / 2.0
                
                self.centerline_points.append((mid_x, mid_y, mid_z))

        self.get_logger().info(f"Successfully loaded map and generated {len(self.centerline_points)} centerline waypoints.")
        
        # 6. Pre-construct the static Path message
        self.global_path.header.frame_id = 'map'
        
        for p_idx, (x, y, z) in enumerate(self.centerline_points):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            
            # Set spatial positions
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            
            # Identity quaternion (default orientation, as requested)
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            
            self.global_path.poses.append(pose)

    def publish_path(self):
        if not self.global_path.poses:
            return
            
        # Update the timestamp dynamically to current ROS time
        current_time = self.get_clock().now().to_msg()
        self.global_path.header.stamp = current_time
        
        for pose in self.global_path.poses:
            pose.header.stamp = current_time
            
        self.path_pub.publish(self.global_path)


def main(args=None):
    rclpy.init(args=args)
    node = GlobalPlanner()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down Global Planner.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()