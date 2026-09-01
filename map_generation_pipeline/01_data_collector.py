import carla
import random
import math
import rclpy
from nav_msgs.msg import Odometry
# Remove: from geometry_msgs.msg import Twist
from carla_msgs.msg import CarlaEgoVehicleControl
from rclpy.qos import qos_profile_sensor_data

# Helper function to convert CARLA's Left-Handed Euler angles to ROS 2 Quaternions
def euler_to_quaternion(roll, pitch, yaw):
    qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    return qx, qy, qz, qw

def main():
    # --- ROS 2 INITIALIZATION ---
    rclpy.init()
    ros_node = rclpy.create_node('carla_odometry_publisher')
    
    # Odometry Publisher
    odom_publisher = ros_node.create_publisher(Odometry, '/carla/hero/odometry', qos_profile_sensor_data)
    # ----------------------------

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Save original settings to restore on exit
    original_settings = world.get_settings()
    
    # 1. Enable Synchronous Mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)

    # Set up the TM in synchronous mode
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    # -----------------------

    actor_list = []
    ego_vehicle = None # Initialize ego_vehicle outside try block so callback can access it

    # --- ROS 2 VEHICLE CONTROL SUBSCRIBER ---
    def control_cmd_callback(msg):
        """ Passes CarlaEgoVehicleControl commands to the CARLA API """
        if ego_vehicle is not None and ego_vehicle.is_alive:
            control = carla.VehicleControl()
            
            # Map fields directly from the incoming CarlaEgoVehicleControl message
            control.throttle = msg.throttle
            control.brake = msg.brake
            control.steer = msg.steer
            control.gear = msg.gear
            control.hand_brake = msg.hand_brake
            control.reverse = msg.reverse
            
            ego_vehicle.apply_control(control)

    ros_node.create_subscription(
        CarlaEgoVehicleControl,
        '/carla/hero/vehicle_control_cmd',
        control_cmd_callback,
        10
    )
    # ----------------------------------------

    try:
        blueprint_library = world.get_blueprint_library()

        # 2. Spawn Ego Vehicle
        vehicle_bp = blueprint_library.filter('vehicle.lincoln.mkz_2020')[0]
        vehicle_bp.set_attribute('ros_name', 'hero') 
        vehicle_bp.set_attribute('role_name', 'hero')

        from global_planner_path import GlobalPlanner

        # Instantiate planner to parse the map 
        planner = GlobalPlanner()
        p1 = planner.centerline_points[0]
        p2 = planner.centerline_points[1]

        # --- NEW: Draw the path in the CARLA world ---
        for pt in planner.centerline_points:
            # Convert ROS 2 right-handed to CARLA left-handed
            loc = carla.Location(x=pt[0], y=-pt[1], z=pt[2] + 0.5)
            # Draw a red dot at each waypoint that lasts forever (life_time=0.0)
            world.debug.draw_point(loc, size=0.1, color=carla.Color(255, 0, 0), life_time=0.0)
        # ---------------------------------------------
        
        # Destroy the temporary node since we only needed the parsed map data
        planner.destroy_node() 
        
        # Convert ROS 2 (Right-Handed) coordinates BACK to CARLA (Left-Handed)
        carla_x = p1[0]
        carla_y = -p1[1]  # Re-invert Y for CARLA
        carla_z = p1[2] + 3.0    
        
        # Calculate heading (yaw) to point the car exactly down the path
        dy = p2[1] - p1[1]
        dx = p2[0] - p1[0]
        ros_yaw = math.atan2(dy, dx)
        carla_yaw = math.degrees(-ros_yaw) # Convert to degrees and invert
        
        # Construct the exact starting transform
        spawn_transform = carla.Transform(
            carla.Location(x=carla_x, y=carla_y, z=carla_z),
            carla.Rotation(pitch=0.0, yaw=carla_yaw, roll=0.0)
        )
        
        ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
        
        if ego_vehicle is None:
            raise RuntimeError("[ERROR] Failed to spawn vehicle at the first path coordinate.")
            
        actor_list.append(ego_vehicle)
        world.tick()
        print(f"[INFO] Successfully spawned ego vehicle at Path Start: {spawn_transform.location}")
        # --------------------------------------------
        # 3. Spawn 32-Channel LiDAR
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('range', '50.0')
        lidar_bp.set_attribute('rotation_frequency', '20.0') 
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_bp.set_attribute('ros_name', 'lidar')
        lidar_bp.set_attribute('role_name', 'lidar')

        lidar_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.4))
        lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
        actor_list.append(lidar_sensor)
        
        lidar_sensor.listen(lambda data: None) 

        if hasattr(lidar_sensor, 'enable_for_ros'):
            lidar_sensor.enable_for_ros()
        
        print("[INFO] Spawned LiDAR. Data stream forcefully activated.")

        # 3.5 Spawn GNSS Sensor
        gnss_bp = blueprint_library.find('sensor.other.gnss')
        gnss_bp.set_attribute('ros_name', 'gnss')
        gnss_bp.set_attribute('role_name', 'gnss')
        
        gnss_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.4))
        gnss_sensor = world.spawn_actor(gnss_bp, gnss_transform, attach_to=ego_vehicle)
        actor_list.append(gnss_sensor)
        
        gnss_sensor.listen(lambda data: None)
        
        if hasattr(gnss_sensor, 'enable_for_ros'):
            gnss_sensor.enable_for_ros()
            
        print("[INFO] Spawned GNSS Sensor. Native ROS 2 streaming enabled.")

        # 4. Set Autopilot (DISABLED FOR ROS 2 CONTROL)
        # ego_vehicle.set_autopilot(True, traffic_manager.get_port())
        print("[INFO] Autopilot disabled. Vehicle is awaiting ROS 2 commands.")

        # 5. Grab Spectator Camera
        spectator = world.get_spectator()

        print("\n[SUCCESS] Simulation ticking. Odometry data is broadcasting.")
        print("-> Run 'python3 state_estimator.py' in a new terminal!")

        # 6. Synchronous Tick Loop
        while True:
            # Step the simulation forward explicitly
            world.tick()
            
            if not ego_vehicle.is_alive:
                print("\n[WARNING] Ego vehicle was destroyed by the server.")
                break 
            
            # --- ODOMETRY PUBLISHING LOGIC ---
            transform = ego_vehicle.get_transform()
            
            # Convert CARLA (Left-Handed) to ROS 2 (Right-Handed)
            ros_x = transform.location.x
            ros_y = -transform.location.y
            ros_z = transform.location.z
            
            roll = math.radians(transform.rotation.roll)
            pitch = math.radians(transform.rotation.pitch)
            yaw = math.radians(-transform.rotation.yaw)
            
            qx, qy, qz, qw = euler_to_quaternion(roll, pitch, yaw)
            
            odom_msg = Odometry()
            odom_msg.header.stamp = ros_node.get_clock().now().to_msg()
            odom_msg.header.frame_id = "map"
            odom_msg.child_frame_id = "hero"
            
            odom_msg.pose.pose.position.x = ros_x
            odom_msg.pose.pose.position.y = ros_y
            odom_msg.pose.pose.position.z = ros_z
            
            odom_msg.pose.pose.orientation.x = qx
            odom_msg.pose.pose.orientation.y = qy
            odom_msg.pose.pose.orientation.z = qz
            odom_msg.pose.pose.orientation.w = qw
            
            odom_publisher.publish(odom_msg)
            
            # This spin_once handles checking for new vehicle_control_cmd messages
            rclpy.spin_once(ros_node, timeout_sec=0.0) 
            # ---------------------------------

            # Make the CARLA server camera smoothly follow behind the ego vehicle
            spectator.set_transform(carla.Transform(
                transform.location + carla.Location(z=6, x=-8),
                carla.Rotation(pitch=-20, yaw=transform.rotation.yaw)
            ))

    except KeyboardInterrupt:
        print("\n[INFO] Stop requested. Shutting down cleanly...")
        
    finally:
        world.apply_settings(original_settings)
        for actor in actor_list:
            if actor.is_alive:
                actor.destroy()
        print("[INFO] Cleanup complete. Actors destroyed.")
        
        # --- SHUTDOWN ROS 2 ---
        ros_node.destroy_node()
        rclpy.shutdown()
        # ----------------------

if __name__ == '__main__':
    main()