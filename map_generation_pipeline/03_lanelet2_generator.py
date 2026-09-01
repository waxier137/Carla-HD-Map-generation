import os
os.environ["WAYLAND_DISPLAY"] = ""
import json
import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay, cKDTree
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pyproj import Proj
from scipy.interpolate import splprep, splev


def smooth_boundary_spline(points, smooth_factor=100.0, num_nodes=None):
    """
    Fits a mathematically smooth B-spline through the boundary points, 
    ironing out autopilot swerves and bridging over intersection gaps.
    """
    if len(points) < 4:
        return points # Not enough points to spline
        
    # Extract X, Y, Z
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Fit a B-Spline. 
    # 's' is the smoothing factor. The higher 's' is, the stiffer the line.
    # It will resist bending into intersections or following sudden autopilot wobbles.
    tck, u = splprep([x, y, z], s=smooth_factor, k=3)
    
    # Determine how many nodes we want in the final map (default to 1 node per meter)
    if num_nodes is None:
        path_length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        num_nodes = max(int(path_length), 10)
        
    # Evaluate the spline at evenly spaced intervals
    u_new = np.linspace(0, 1.0, num_nodes)
    new_x, new_y, new_z = splev(u_new, tck)
    
    return np.vstack((new_x, new_y, new_z)).T

def load_metadata(metadata_file="phase2_metadata.json"):
    """Loads the GNSS anchor and the exact driven trajectory."""
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"[!] Metadata file '{metadata_file}' missing. Run Phase 2 first.")
    
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
        
    return metadata["origin_lat"], metadata["origin_lon"], np.array(metadata["trajectory"])

def extract_alpha_shape_boundary(points, alpha=2.5):
    """
    Extracts the outer crust of the entire point cloud using a 2D Concave Hull.
    """
    points_2d = points[:, :2]
    tri = Delaunay(points_2d)
    edge_counts = {}
    
    for ia, ib, ic in tri.simplices:
        pa, pb, pc = points_2d[ia], points_2d[ib], points_2d[ic]
        
        a = np.linalg.norm(pa - pb)
        b = np.linalg.norm(pb - pc)
        c = np.linalg.norm(pc - pa)
        
        s = (a + b + c) / 2.0
        area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
        
        circum_r = a * b * c / (4.0 * area) if area > 0 else np.inf
        
        if circum_r < alpha:
            edges = [(ia, ib), (ib, ic), (ic, ia)]
            for edge in edges:
                sorted_edge = tuple(sorted(edge))
                edge_counts[sorted_edge] = edge_counts.get(sorted_edge, 0) + 1

    boundary_indices = set()
    for edge, count in edge_counts.items():
        if count == 1:
            boundary_indices.add(edge[0])
            boundary_indices.add(edge[1])
            
    return points[list(boundary_indices)]

def process_boundaries_via_trajectory(points, trajectory, min_lateral=1.0, max_lateral=15.0):
    """
    Bisects the road using robust look-ahead vectors, bins points by trajectory step, 
    and averages them to create a single, mathematically perfectly thin line per side.
    """
    traj_tree = cKDTree(trajectory)
    _, s_indices = traj_tree.query(points[:, :2])
    
    left_bins = {}
    right_bins = {}
    
    for i, pt in enumerate(points):
        idx = s_indices[i]
        
        T_current = trajectory[idx]
        
        # --- THE FIX: Look ahead for a valid vector ---
        ahead_idx = idx + 1
        # Keep searching forward until we find a GNSS point at least 0.5m away
        while ahead_idx < len(trajectory) and np.linalg.norm(trajectory[ahead_idx] - T_current) < 0.5:
            ahead_idx += 1
            
        if ahead_idx < len(trajectory):
            heading_vec = trajectory[ahead_idx] - T_current
        else:
            # If we are at the very end of the map, look backwards instead
            back_idx = idx - 1
            while back_idx >= 0 and np.linalg.norm(T_current - trajectory[back_idx]) < 0.5:
                back_idx -= 1
            if back_idx >= 0:
                heading_vec = T_current - trajectory[back_idx]
            else:
                heading_vec = np.array([1e-6, 1e-6]) # Fallback failsafe
        # ----------------------------------------------
        
        point_vec = pt[:2] - T_current
        
        # 2D Cross Product
        cross_prod = heading_vec[0] * point_vec[1] - heading_vec[1] * point_vec[0]
        lateral_dist = cross_prod / (np.linalg.norm(heading_vec) + 1e-6)
        
        # Filter and Bin
        if min_lateral < lateral_dist < max_lateral:
            if idx not in left_bins:
                left_bins[idx] = []
            left_bins[idx].append(pt)
            
        elif -max_lateral < lateral_dist < -min_lateral:
            if idx not in right_bins:
                right_bins[idx] = []
            right_bins[idx].append(pt)
            
    # Average the bins to collapse the spiderweb into a single node per step
    sorted_left = []
    for idx in sorted(left_bins.keys()):
        pts = np.array(left_bins[idx])
        sorted_left.append(np.mean(pts, axis=0)) 
        
    sorted_right = []
    for idx in sorted(right_bins.keys()):
        pts = np.array(right_bins[idx])
        sorted_right.append(np.mean(pts, axis=0))
        
    return np.array(sorted_left), np.array(sorted_right)
def resample_trajectory(trajectory, step=1.0):
    """
    Cleans the trajectory by finding total cumulative distance 
    and generating evenly spaced points along the path.
    """
    traj_array = np.array(trajectory)
    if len(traj_array) < 2:
        return traj_array
        
    # Calculate cumulative distance along the trajectory
    diffs = np.diff(traj_array, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.insert(np.cumsum(dists), 0, 0)
    
    total_length = cum_dists[-1]
    print(f"      -> Total trajectory path length: {total_length:.2f} meters")
    
    if total_length == 0:
        raise ValueError("Trajectory length is 0! The car never moved according to the metadata.")
        
    # Create new evenly spaced targets every 1.0 meter
    new_dists = np.arange(0, total_length, step)
    
    new_x = np.interp(new_dists, cum_dists, traj_array[:, 0])
    new_y = np.interp(new_dists, cum_dists, traj_array[:, 1])
    
    return np.vstack((new_x, new_y)).T
def prettify_xml(elem):
    """Return a pretty-printed XML string."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def main():
    input_file = "phase2_road_surface.pcd"
    output_file = "phase3_lanelet2_map.osm"
    
    print("[1/6] Loading PCD data and JSON metadata...")
    pcd = o3d.io.read_point_cloud(input_file)
    points = np.asarray(pcd.points)
    
    # Change "trajectory" to "raw_trajectory" here
    origin_lat, origin_lon, raw_trajectory = load_metadata() 
    
    print(f"      -> Loaded {len(points)} points.")
    print(f"      -> Loaded {len(raw_trajectory)} raw trajectory steps.")
    
    # Now this line will work perfectly
    trajectory = resample_trajectory(raw_trajectory, step=1.0)
    print(f"      -> Resampled trajectory to {len(trajectory)} smooth 1m steps.")
    
    print("[2/6] Extracting outer boundary crust (Alpha Shape)...")
    boundary_crust = extract_alpha_shape_boundary(points, alpha=2.0)
    
    print("[3/6] Applying Vector Logic to Bisect and Sort...")
    sorted_left, sorted_right = process_boundaries_via_trajectory(
        boundary_crust, trajectory, min_lateral=1.0, max_lateral=12.0
    )
    print(f"      -> Filtered Left lane nodes: {len(sorted_left)}")
    print(f"      -> Filtered Right lane nodes: {len(sorted_right)}")

    # --- ADD THIS: The Mathematical Iron ---
    print("[3.5/6] Applying B-Spline to smooth autopilot wobbles...")
    sorted_left = smooth_boundary_spline(sorted_left, smooth_factor=50.0)
    sorted_right = smooth_boundary_spline(sorted_right, smooth_factor=50.0)
    
    print(f"      -> Smoothed Left lane nodes: {len(sorted_left)}")
    print(f"      -> Smoothed Right lane nodes: {len(sorted_right)}")

    print("[4/6] Constructing XML Nodes (WGS84 Inverse Projection)...")
    root = ET.Element("osm", version="0.6")
    node_id_counter = -1
    left_node_ids = []
    right_node_ids = []
    
    proj = Proj(proj='tmerc', lat_0=origin_lat, lon_0=origin_lon, ellps='WGS84')
    
    for pt in sorted_left:
        lon, lat = proj(pt[0], pt[1], inverse=True)
        node = ET.SubElement(root, "node", id=str(node_id_counter), lat=str(lat), lon=str(lon))
        ET.SubElement(node, "tag", k="ele", v=str(pt[2]))
        left_node_ids.append(node_id_counter)
        node_id_counter -= 1
        
    for pt in sorted_right:
        lon, lat = proj(pt[0], pt[1], inverse=True)
        node = ET.SubElement(root, "node", id=str(node_id_counter), lat=str(lat), lon=str(lon))
        ET.SubElement(node, "tag", k="ele", v=str(pt[2]))
        right_node_ids.append(node_id_counter)
        node_id_counter -= 1
        
    print("[5/6] Constructing XML Ways and Relations...")
    left_way_id = node_id_counter - 1
    left_way = ET.SubElement(root, "way", id=str(left_way_id))
    for nid in left_node_ids:
        ET.SubElement(left_way, "nd", ref=str(nid))
    ET.SubElement(left_way, "tag", k="type", v="line_thin")
    ET.SubElement(left_way, "tag", k="subtype", v="solid")
    
    right_way_id = left_way_id - 1
    right_way = ET.SubElement(root, "way", id=str(right_way_id))
    for nid in right_node_ids:
        ET.SubElement(right_way, "nd", ref=str(nid))
    ET.SubElement(right_way, "tag", k="type", v="line_thin")
    ET.SubElement(right_way, "tag", k="subtype", v="solid")
    
    relation_id = right_way_id - 1
    relation = ET.SubElement(root, "relation", id=str(relation_id))
    ET.SubElement(relation, "member", type="way", ref=str(left_way_id), role="left")
    ET.SubElement(relation, "member", type="way", ref=str(right_way_id), role="right")
    ET.SubElement(relation, "tag", k="type", v="lanelet")
    ET.SubElement(relation, "tag", k="subtype", v="road")
    
    print("[6/6] Exporting map...")
    xml_str = prettify_xml(root)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(xml_str)
        
    print(f"SUCCESS: Pipeline complete! Map saved to '{output_file}'.")

    # ---------------------------------------------------------
    # The Overlay Test: Visualize the map over the point cloud
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # The Overlay Test: Bird's-Eye View PNG Export
    # ---------------------------------------------------------
    print("[7/6] Generating top-down map verification image...")
    import matplotlib.pyplot as plt

    # Create a large, high-res plot
    plt.figure(figsize=(15, 15), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')

    # 1. Plot the raw point cloud (Gray dots)
    plt.scatter(points[:, 0], points[:, 1], c='gray', s=1, alpha=0.3, label="Raw PCD")

    # 2. Plot the Left Lane Boundary (Red line)
    plt.plot(sorted_left[:, 0], sorted_left[:, 1], c='red', linewidth=2, label="Left Boundary")

    # 3. Plot the Right Lane Boundary (Cyan line)
    plt.plot(sorted_right[:, 0], sorted_right[:, 1], c='cyan', linewidth=2, label="Right Boundary")

    # Lock the aspect ratio so the road doesn't look stretched or warped
    plt.axis('equal')
    plt.legend(loc="upper right", facecolor='white')
    plt.title("HD Map Overlay Verification", color='white')

    # Export to a PNG file instead of opening a window
    image_filename = "map_verification_overlay.png"
    plt.savefig(image_filename, dpi=300, bbox_inches='tight')
    print(f"      -> SUCCESS: Verification image saved to '{image_filename}'.")

if __name__ == "__main__":
    main()