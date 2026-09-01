# HD vector mapping and Ros2 Navigation in CARLA Simulation 
## Overview
This repo contains a closed-loop autonomous driving architechture built from scratch. 
The system extracts raw LiDAR and GNSS data from CARLA Simulator, processed the point cloud into a 'Lanelet2' XML vector map, and uses Ros2 navigation stack to drive the vehicle along the generated boundaries using a Pure Pursuit controller.

## System Architecture
The system is divided into 2 distinct module: Map Generation and Real-Time control
### 1. Map generation pipeling (ETL and Spatial processing)
* **'01_data_collector'**: ingest raw '.mcap' odometry and LiDAR data from the CARLA ego-vehicle
* **'02_pointcloud_processor.py'**: Utilizes 
