# CARLA ROS 2 Autonomous Navigation Stack

An end-to-end autonomous vehicle software stack bridging the CARLA simulator with native ROS 2. This project features an offline High-Definition (HD) vector mapping pipeline and a real-time autonomous navigation stack using a Pure Pursuit controller.

**Author:** Lucas Nguyen

## Requirements
- **OS:** Ubuntu 24.04 LTS
- **Simulator:** CARLA 0.9.16
- **Middleware:** ROS 2
- **Python Packages:** `open3d`, `numpy`, `scipy`, `pyproj`
- **ROS 2 Packages:** `carla-msgs`, `nav-msgs`, `geometry-msgs`, `rosbag2_py`

## System Architecture

The project is divided into two primary workflows:

### Phase 1: Offline HD Map Generation Pipeline
Generates a standard Lanelet2 `.osm` vector map from raw simulator sensor data.

1. **`01_data_collector.py`**: Spawns an ego vehicle equipped with a 32-channel LiDAR and GNSS in CARLA. Records world data to `.mcap` bags and streams 3D odometry.
2. **`02_pointcloud_processor.py`**: Parses ROS 2 bags natively, synchronizes GNSS and LiDAR, and projects data into a unified Cartesian frame. Uses Open3D voxel downsampling and RANSAC planar segmentation to isolate the drivable road surface.
3. **`03_lanelet2_generator.py`**: Extracts the point cloud boundary crust via a 2D Concave Hull (Alpha Shape). Bisects boundaries using the vehicle's trajectory, applies B-spline smoothing to eliminate steering wobbles, and exports a structured Lanelet2 `.osm` XML map.

### Phase 2: Real-Time Autonomous Navigation
Executes real-time autonomous driving using the generated vector map.

1. **`global_planner_path.py`**: Parses the `.osm` map to compute the geometric centerline of the road, publishing it as a static `nav_msgs/Path`.
2. **`state_estimator.py`**: Subscribes to 3D odometry and simplifies quaternion orientations into a standard 2D spatial state (X, Y, Yaw) for planar tracking.
3. **`pure_pursuit_controller.py`**: A 20 Hz local controller implementing a Pure Pursuit geometric algorithm for steering (via dynamic look-ahead) and a Proportional (P) controller for smooth longitudinal acceleration and braking.

## Usage

1. **Start CARLA Server**: Launch your CARLA 0.9.16 instance.
2. **Data Collection**: Run `python3 01_data_collector.py` to record a driving session.
3. **Process Map**: 
   - Execute `python3 02_pointcloud_processor.py` to process the bag.
   - Execute `python3 03_lanelet2_generator.py` to generate `phase3_lanelet2_map.osm`.
4. **Navigate**: 
   - Launch `python3 01_data_collector.py` to spawn the vehicle.
   - Run `python3 state_estimator.py` to broadcast 2D state.
   - Run `python3 global_planner_path.py` to publish the path.
   - Start `python3 pure_pursuit_controller.py` to begin autonomous navigation.
