import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D
from rclpy.qos import qos_profile_sensor_data 

class StateEstimatorNode(Node):
    def __init__(self):
        super().__init__('state_estimator')
        
        # 1. Subscriber Setup
        self.subscription = self.create_subscription(
            Odometry,
            '/carla/hero/odometry',
            self.odom_callback,
            qos_profile_sensor_data
        )
        
        # 2. Publisher Setup
        self.publisher = self.create_publisher(
            Pose2D,
            '/vehicle_state_2d',
            10
        )
        
        # --- NEW: Coordinate Offset Variables ---
        self.spawn_x = None
        self.spawn_y = None
        
        self.log_counter = 0
        self.get_logger().info("State Estimator initialized. Waiting for odometry...")

    def euler_from_quaternion(self, x, y, z, w):
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t3, t4)

    def odom_callback(self, msg: Odometry):
        # Directly use the absolute map coordinates
        local_x = msg.pose.pose.position.x
        local_y = msg.pose.pose.position.y
        
        # Extract Orientation
        q = msg.pose.pose.orientation
        yaw = self.euler_from_quaternion(q.x, q.y, q.z, q.w)
        
        # Publish Absolute State
        state_msg = Pose2D()
        state_msg.x = local_x
        state_msg.y = local_y
        state_msg.theta = yaw
        
        self.publisher.publish(state_msg)

def main(args=None):
    rclpy.init(args=args)
    node = StateEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()