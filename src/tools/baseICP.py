import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1, :] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    """

    # assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()




def icp(src_tm: "<class 'trimesh'>", dst_tm: "<class 'trimesh'>",
        init_pose=None, max_iterations=20, tolerance=None, samplerate=1):
    """
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
        samplerate: subsampling rate
    Output:
        T: final homogeneous transformation that maps A on to B
        MeanError: list, report each iteration's distance mean error
    """

    # get vertices and their normals
    src_pts = np.array(src_tm.vertices)
    dst_pts = np.array(dst_tm.vertices)
    src_pt_normals = np.array(src_tm.vertex_normals)
    dst_pt_normals = np.array(dst_tm.vertex_normals)

    # subsampling
    ids = np.random.uniform(0, 1, size=src_pts.shape[0])
    A = src_pts[ids < samplerate, :]
    A_normals = src_pt_normals[ids < samplerate, :]
    ids = np.random.uniform(0, 1, size=dst_pts.shape[0])
    B = dst_pts[ids < 1, :]
    B_normals = dst_pt_normals[ids < 1, :]

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1, A.shape[0]))
    dst = np.ones((m+1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0
    MeanError = []

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # match each point of source-set to closest point of destination-set,
        matched_src_pts = src[:m, :].T.copy()
        matched_dst_pts = dst[:m, indices].T

        # compute angle between 2 matched vertexs' normals
        matched_src_pt_normals = A_normals.copy()
        matched_dst_pt_normals = B_normals[indices, :]
        angles = np.zeros(matched_src_pt_normals.shape[0])
        for k in range(matched_src_pt_normals.shape[0]):
            v1 = matched_src_pt_normals[k, :]
            v2 = matched_dst_pt_normals[k, :]
            cos_angle = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angles[k] = np.arccos(cos_angle) / np.pi * 180

        # and reject the bad corresponding
        dist_threshold = np.inf
        dist_bool_flag = (distances < dist_threshold)
        angle_threshold = 20
        angle_bool_flag = (angles < angle_threshold)
        reject_part_flag = dist_bool_flag * angle_bool_flag

        matched_src_pts = matched_src_pts[reject_part_flag, :]
        matched_dst_pts = matched_dst_pts[reject_part_flag, :]

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(matched_src_pts, matched_dst_pts)

        # update the current source
        src = np.dot(T, src)

        # print iteration
        print('\ricp iteration: %d/%d ...' % (i+1, max_iterations), end='', flush=True)

        # check error
        mean_error = np.mean(distances[reject_part_flag])
        MeanError.append(mean_error)
        if tolerance is not None:
            if np.abs(prev_error - mean_error) < tolerance:
                print('\nbreak iteration, the distance between two adjacent iterations '
                      'is lower than tolerance (%.f < %f)'
                      % (np.abs(prev_error - mean_error), tolerance))
                break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T)
    print()

    return T, MeanError


import open3d as o3d
import numpy as np
import cv2
import os
import os
import numpy as np
import open3d as o3d
import cv2
import imageio

def icp_with_open3d_animation(src_tm, dst_tm, init_pose=None, max_iterations=20, tolerance=None, samplerate=0.1):
    # Subsample the points for clarity
    src_pts = np.array(src_tm.vertices)[::int(1/samplerate)]
    dst_pts = np.array(dst_tm.vertices)[::int(1/samplerate)]
    
    # Initialize point clouds for Open3D visualization
    src_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_pts))
    dst_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst_pts))
    
    # Coloring the destination (reference) points red and source points blue
    dst_o3d.paint_uniform_color([1, 0, 0])  # Red
    src_o3d.paint_uniform_color([0, 0, 1])  # Blue

    # Visualization setup
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(dst_o3d)
    vis.add_geometry(src_o3d)

    # Prepare to save frames for GIF creation
    frames = []

    prev_error = 0
    for i in range(max_iterations):
        # ICP step
        distances, indices = nearest_neighbor(src_pts, dst_pts)
        matched_dst_pts = dst_pts[indices]
        T, _, _ = best_fit_transform(src_pts, matched_dst_pts)
        src_pts = (T[:3, :3] @ src_pts.T + T[:3, 3:4]).T

        # Update source point cloud and visualize the transformation
        src_o3d.points = o3d.utility.Vector3dVector(src_pts)
        vis.update_geometry(src_o3d)
        vis.poll_events()
        vis.update_renderer()

        # Capture frame for GIF
        frame = np.asarray(vis.capture_screen_float_buffer(False))
        frame = (frame * 255).astype(np.uint8)  # Convert from float to uint8
        frames.append(frame)  # Save the captured frame

        # Error check for convergence
        mean_error = np.mean(distances)
        if tolerance is not None and abs(prev_error - mean_error) < tolerance:
            print(f"Converged at iteration {i + 1}")
            break
        prev_error = mean_error

    vis.destroy_window()

    # Export as a GIF using imageio
    imageio.mimsave('icp_animation.gif', frames, duration=0.9)  # Adjust duration as needed

    print("Animation saved as icp_animation.gif.")
