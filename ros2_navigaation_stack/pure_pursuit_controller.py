#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import numpy as np

from nav_msgs.msg import Path
from geometry_msgs.msg import Pose2D
from carla_msgs.msg import CarlaEgoVehicleControl
from rclpy.qos import qos_profile_sensor_data 

class PurePursuitController(Node):
    def __init__(self):
        super().__init__('pure_pursuit_controller')

        # --- Parameters ---
        self.lookahead_distance = 5.0  # L_d in meters
        self.wheelbase = 2.9           # L in meters (Lincoln MKZ)
        self.target_speed = 5.0        # m/s
        self.k_p = 0.5                 # Proportional gain for throttle
        self.max_steer_angle = 1.22    # Radians (~70 degrees max steering for normalization)

        # --- State Variables ---
        self.path_points = None
        self.current_pose = None       # Will hold [x, y, yaw]
        self.last_pose = None
        self.last_time = self.get_clock().now()
        self.current_speed = 0.0

        # --- Subscribers & Publishers ---
        self.path_sub = self.create_subscription(Path, '/global_plan', self.path_callback, 10)
        self.state_sub = self.create_subscription(Pose2D, '/vehicle_state_2d', self.pose_callback, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(CarlaEgoVehicleControl, '/carla/hero/vehicle_control_cmd', 10)

        # --- Control Loop Timer (20 Hz) ---
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info("Pure Pursuit Local Controller Initialized.")

    def path_callback(self, msg):
        """Extract X and Y coordinates from the Path message into a NumPy array."""
        if not msg.poses:
            self.get_logger().warn("Received empty path!", throttle_duration_sec=2.0)
            self.path_points = None
            return

        points = []
        for pose_stamped in msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            points.append([x, y])
        
        self.path_points = np.array(points)

    def pose_callback(self, msg):
        """Update vehicle state from Pose2D and estimate current speed."""
        now = self.get_clock().now()

        # Directly read 2D spatial and heading values
        x = msg.x
        y = msg.y
        yaw = msg.theta

        self.current_pose = np.array([x, y, yaw])

        # Calculate speed by tracking positional translation over time
        if self.last_pose is not None:
            dt = (now - self.last_time).nanoseconds / 1e9
            if dt > 0.001:
                dx = x - self.last_pose[0]
                dy = y - self.last_pose[1]
                self.current_speed = math.hypot(dx, dy) / dt

        self.last_pose = np.array([x, y, yaw])
        self.last_time = now

    def stop_vehicle(self):
        """Safety fallback: apply brakes and stop steering."""
        cmd = CarlaEgoVehicleControl()
        cmd.throttle = 0.0
        cmd.brake = 1.0 
        cmd.steer = 0.0
        self.cmd_pub.publish(cmd)

    def control_loop(self):
        """Main 20 Hz control loop for Pure Pursuit and P-control."""
        # SAFETY FALLBACK: Log warnings if data is missing
        if self.path_points is None or self.current_pose is None:
            self.stop_vehicle()
            if self.path_points is None:
                self.get_logger().warn("Waiting for /global_plan...", throttle_duration_sec=2.0)
            if self.current_pose is None:
                self.get_logger().warn("Waiting for /vehicle_state_2d...", throttle_duration_sec=2.0)
            return

        # Extract parsed pose data
        veh_x = self.current_pose[0]
        veh_y = self.current_pose[1]
        veh_yaw = self.current_pose[2]

        # 1. Find the closest point on the path
        veh_pos = np.array([veh_x, veh_y])
        distances = np.linalg.norm(self.path_points - veh_pos, axis=1)
        closest_idx = np.argmin(distances)

        # 2. Find the look-ahead point
        lookahead_pt = None
        for i in range(closest_idx, len(self.path_points)):
            dist = np.linalg.norm(self.path_points[i] - veh_pos)
            if dist >= self.lookahead_distance:
                lookahead_pt = self.path_points[i]
                break

        # Fallback if the end of the path is reached
        if lookahead_pt is None:
            if distances[closest_idx] < 1.5:  # Within 1.5m of the final point
                self.get_logger().info("End of path reached. Stopping.", throttle_duration_sec=2.0)
                self.stop_vehicle()
                self.path_points = None # Clear path to prevent looping
                return
            lookahead_pt = self.path_points[-1]

        # 3. Transform look-ahead point to vehicle's local coordinate frame
        dx = lookahead_pt[0] - veh_x
        dy = lookahead_pt[1] - veh_y

        # 2D Rotation matrix to shift global to local frame
        local_x = math.cos(veh_yaw) * dx + math.sin(veh_yaw) * dy
        local_y = -math.sin(veh_yaw) * dx + math.cos(veh_yaw) * dy

        # 4. Calculate steering angle using Pure Pursuit formula
        alpha = math.atan2(local_y, local_x)
        delta = math.atan2(2.0 * self.wheelbase * math.sin(alpha), self.lookahead_distance)

        # Normalize steering output to [-1.0, 1.0] for CARLA
        steer_norm = np.clip(delta / self.max_steer_angle, -1.0, 1.0)

        cmd = CarlaEgoVehicleControl()

        # 5. Longitudinal Math (Speed) using a Proportional (P) Controller
        speed_error = self.target_speed - self.current_speed
        accel_cmd = self.k_p * speed_error

        # Pack into CarlaEgoVehicleControl message
        cmd = CarlaEgoVehicleControl()
        cmd.gear = 1  # Explicitly shift into Drive
        
        if accel_cmd > 0:
            cmd.throttle = float(np.clip(accel_cmd, 0.0, 1.0))
            cmd.brake = 0.0
        else:
            cmd.throttle = 0.0
            cmd.brake = float(np.clip(-accel_cmd, 0.0, 1.0))

        # Pack normalized steering
        cmd.steer = float(-steer_norm)
        
        # Publish the command
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Pure Pursuit Controller...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()