import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors

# Functions for best-fit transform and nearest neighbor
def best_fit_transform_3d(A, B):
    # Calculates the least-squares best-fit transform for 3D
    assert A.shape == B.shape
    m = A.shape[1]
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t
    return T, R, t

def nearest_neighbor_3d(src, dst):
    # Find nearest neighbor for each point in 3D
    neigh = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

# Generate a 3D star shape
def generate_star_shape_3d(num_points, radius_inner, radius_outer):
    """Generate a 3D star shape."""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    z_levels = np.linspace(-1, 1, num_points)  # Varying z-values for depth
    points = []
    for i, angle in enumerate(angles):
        radius = radius_outer if i % 2 == 0 else radius_inner
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = z_levels[i]  # Assign z-coordinate
        points.append((x, y, z))
    return np.array(points)

def icp_3d_animation():
    # Create two star shapes in 3D, one as the source and one as the slightly shifted/rotated destination
    num_points = 30  # Increased number of points for the star shape
    src_shape = generate_star_shape_3d(num_points=num_points, radius_inner=3, radius_outer=5)
    dst_shape = generate_star_shape_3d(num_points=num_points, radius_inner=3, radius_outer=5)

    # Slightly rotate the destination shape
    angle_offset = np.pi / 4  # initial rotation
    rotation_matrix = np.array([[np.cos(angle_offset), -np.sin(angle_offset), 0],
                                 [np.sin(angle_offset), np.cos(angle_offset), 0],
                                 [0, 0, 1]])
    dst_shape = (rotation_matrix @ dst_shape.T).T

    max_iterations = 100
    tolerance = 1e-8

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('ICP Alignment of 3D Shapes')
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_zlim(-7, 7)

    # Initial plot setup using scatter for better visibility
    src_scatter = ax.scatter(src_shape[:, 0], src_shape[:, 1], src_shape[:, 2], color='blue', s=100, label='Source Shape')
    dst_scatter = ax.scatter(dst_shape[:, 0], dst_shape[:, 1], dst_shape[:, 2], color='red', s=100, label='Destination Shape')
    error_text = ax.text2D(0.05, 0.9, '', transform=ax.transAxes)

    def icp_step(i):
        nonlocal src_shape, dst_shape
        # Rotate the destination shape incrementally
        angle_increment = np.pi / (2 * max_iterations)  # Adjust the rotation speed
        rotation_matrix = np.array([[np.cos(angle_increment), -np.sin(angle_increment), 0],
                                     [np.sin(angle_increment), np.cos(angle_increment), 0],
                                     [0, 0, 1]])
        dst_shape = (rotation_matrix @ dst_shape.T).T  # Update destination shape with rotation

        T, _, _ = best_fit_transform_3d(src_shape, dst_shape)
        src_shape = (T @ np.vstack((src_shape.T, np.ones((1, src_shape.shape[0])))))[:3].T
        src_scatter._offsets3d = (src_shape[:, 0], src_shape[:, 1], src_shape[:, 2])

        # Update color to transition from blue to green as it converges
        src_scatter.set_color((0, 0.5 + 0.5 * (i / max_iterations), 1 - 0.5 * (i / max_iterations)))

        # Calculate mean error for current frame and update text
        distances, _ = nearest_neighbor_3d(src_shape, dst_shape)
        mean_error = np.mean(distances)
        error_text.set_text(f'Mean Error: {mean_error:.4f}')

    # Animation function
    ani = animation.FuncAnimation(fig, icp_step, frames=max_iterations, repeat=False)
    ani.save('icp_3d_star_shape_animation.gif', writer='imagemagick', fps=5)
    plt.show()

# Run the modified animation
icp_3d_animation()
