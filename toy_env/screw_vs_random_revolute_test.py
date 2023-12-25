import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm, expm
import matplotlib.pyplot as plt

# Function to update the plot with a new line
def update_plot(ax, q, endpoint, new_traj_pts, q_target, endpoint_target, target_traj_pts, pause_time):
    ax.clear()

    ax.set_xlim([0, 12])
    ax.set_ylim([0, 12])
    ax.set_zlim([0, 12])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    
    ax.plot([q_target[0], endpoint_target[0]], [q_target[1], endpoint_target[1]], [q_target[2], endpoint_target[2]], color='g')
    for pt in target_traj_pts:
        ax.scatter(pt[0][3], pt[1][3], pt[2][3], color='g', marker='o')
    
    if q is not None:
        ax.plot([q[0], endpoint[0]], [q[1], endpoint[1]], [q[2], endpoint[2]], color='m')
    for pt in new_traj_pts:
        ax.scatter(pt[0][3], pt[1][3], pt[2][3], color='m', marker='o')

    plt.draw()
    plt.pause(1.5)

def update_plot_pts(ax, new_traj_pts, q_target, endpoint_target, target_traj_pts, pause_time):
    ax.clear()

    ax.set_xlim([0, 12])
    ax.set_ylim([0, 12])
    ax.set_zlim([0, 12])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    
    ax.plot([q_target[0], endpoint_target[0]], [q_target[1], endpoint_target[1]], [q_target[2], endpoint_target[2]], color='g')
    for pt in target_traj_pts:
        ax.scatter(pt[0], pt[1], pt[2], color='g', marker='o')
    
    for i in range(len(new_traj_pts)):
        pt = new_traj_pts[i]
        ax.scatter(pt[0], pt[1], pt[2], color='m', marker='o')
        if i+1 < len(new_traj_pts):
            pt2 = new_traj_pts[i+1]
            ax.plot([pt[0], pt2[0]], [pt[1], pt2[1]], [pt[2], pt2[2]], color='m')

    plt.draw()
    plt.pause(0.5)


def calc_dist(target_traj_pts, start_traj_pts):
    target_traj_pts = np.array(target_traj_pts)
    start_traj_pts = np.array(start_traj_pts)
    # print(target_traj_pts.shape, start_traj_pts.shape)
    dist = np.linalg.norm(target_traj_pts - start_traj_pts, axis=1).sum()
    # print(dist)
    return dist

def get_points(start_pt, direction_vec, len_org_line, sample_pts=10):
    uniform_lens = np.linspace(0, len_org_line, sample_pts)
    pts = []
    for len in uniform_lens:
        endpoint = start_pt + len * direction_vec
        pts.append(endpoint)
    return pts

def get_points_revolute(s_hat, q, initial_pose, angle_moved=1.57):
    # Calculate the twist vector
    h = 0 # pure rotation
    w = s_hat
    v = -np.cross(s_hat, q)
    twist = np.concatenate((w,v)) 
    # print("twist: ", twist)

    # Calculate the matrix form of the twist vector
    w = twist[:3]
    # w_matrix = R.from_rotvec(w).as_matrix()
    w_matrix = [
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0],
    ]
    # print("w_matrix: ", w_matrix)

    S = [
        [w_matrix[0][0], w_matrix[0][1], w_matrix[0][2], twist[3]],
        [w_matrix[1][0], w_matrix[1][1], w_matrix[1][2], twist[4]],
        [w_matrix[2][0], w_matrix[2][1], w_matrix[2][2], twist[5]],
        [0, 0, 0, 0]
    ]

    final_points = []
    # Calculate the transformation of the point when moved by theta along the screw axis
    for theta in np.arange(0, angle_moved, 0.1):
        S_theta = theta * np.array(S)

        # T1 = np.dot(T0, expm(S_theta))
        T1 = np.dot(expm(S_theta), T0)
        final_points.append(T1)
        # ax.scatter(T1[0][3], T1[1][3], T1[2][3], color='g', marker='o')
    
    return final_points



np.random.seed(1)
# Create a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([0, 12])
ax.set_ylim([0, 12])
ax.set_zlim([0, 12])
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# Initial pose
T0 = np.array([[1, 0, 0, 4],
    [0, 1, 0, 4],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])

# Target_axis
s_hat_target = np.array([1.0, 0.0, 0.0])
q_target = np.array([1.0, 1.0, 0.0])
# Plot the screw axis
length = 6
endpoint_target = q_target + length * s_hat_target
ax.plot([q_target[0], endpoint_target[0]], [q_target[1], endpoint_target[1]], [q_target[2], endpoint_target[2]], color='g')
target_traj_pts = get_points_revolute(q=q_target, s_hat=s_hat_target, initial_pose=T0)

# Incorrect start axis
noise = np.random.uniform(low=-0.15, high=0.15, size=(3,))
print(noise)
s_start = s_hat_target + noise
s_hat_start = s_start / np.linalg.norm(s_start)
noise = np.random.uniform(low=-0.15, high=0.15, size=(3,))
q_start = q_target + noise
# Plot the screw axis
length = 6
endpoint_start = q_start + length * s_hat_start
ax.plot([q_start[0], endpoint_start[0]], [q_start[1], endpoint_start[1]], [q_start[2], endpoint_start[2]], color='r')
start_traj_pts = get_points_revolute(q=q_start, s_hat=s_hat_start, initial_pose=T0)

dist = calc_dist(target_traj_pts, start_traj_pts)
print("Initial traj dist: ", dist)
# display the two trajs
for pt in target_traj_pts:
    ax.scatter(pt[0][3], pt[1][3], pt[2][3], color='g', marker='o')
for pt in start_traj_pts:
    ax.scatter(pt[0][3], pt[1][3], pt[2][3], color='r', marker='o')
plt.show()

# # --------- joint paramerization space exploration ----------------
# # Create a 3D axis
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlim([0, 12])
# ax.set_ylim([0, 12])
# ax.set_zlim([0, 12])
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')

# min_dist = 1000
# final_axis = None
# final_traj = None
# for counter in range(30):
#     # Sample in the screw axis parameter space
#     noise = np.random.uniform(low=-0.15, high=0.15, size=(3,))
#     s_new = s_hat_start + noise
#     s_hat_new = s_new / np.linalg.norm(s_new)
#     noise = np.random.uniform(low=-0.15, high=0.15, size=(3,))
#     q_new = q_start + noise
#     # Plot the screw axis
#     length = 6
#     endpoint_new = q_new + length * s_hat_new
#     # confirm the length of the line
#     line_norm = np.linalg.norm(q_new - endpoint_new)
#     # print("norm of the new line: ", line_norm)
#     new_traj_pts = get_points_revolute(q=q_new, s_hat=s_hat_new, initial_pose=T0)

#     dist = calc_dist(target_traj_pts, new_traj_pts)
#     if dist < min_dist:
#         min_dist = dist
#         final_axis = (q_new, endpoint_new)
#         final_traj_pts = new_traj_pts
#         pause_time = 5
#     print("Current traj dist: ", dist)
#     update_plot(ax, q_new, endpoint_new, new_traj_pts, q_target, endpoint_target, target_traj_pts, pause_time=pause_time)
#     pause_time = 1

# plt.show()


# # Show target, start and final axis
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlim([0, 12])
# ax.set_ylim([0, 12])
# ax.set_zlim([0, 12])
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')
# print("min_dist: ", min_dist)
# # target 
# ax.plot([q_target[0], endpoint_target[0]], [q_target[1], endpoint_target[1]], [q_target[2], endpoint_target[2]], color='g')
# for pt in target_traj_pts:
#     ax.scatter(pt[0][3], pt[1][3], pt[2][3], color='g', marker='o')
# # start
# ax.plot([q_start[0], endpoint_start[0]], [q_start[1], endpoint_start[1]], [q_start[2], endpoint_start[2]], color='r')
# for pt in start_traj_pts:
#     ax.scatter(pt[0][3], pt[1][3], pt[2][3], color='r', marker='o')
# # final
# q_final, endpoint_final = final_axis[0], final_axis[1]
# ax.plot([q_final[0], endpoint_final[0]], [q_final[1], endpoint_final[1]], [q_final[2], endpoint_final[2]], color='m')
# for pt in final_traj_pts:
#     ax.scatter(pt[0][3], pt[1][3], pt[2][3], color='m', marker='o')
# plt.show()

# ---------- 3 dof space exploration -----------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([0, 12])
ax.set_ylim([0, 12])
ax.set_zlim([0, 12])
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

pause_time = 1
min_dist = 1000
final_traj = None
for counter in range(30):
    # Sample in the 6dof  space
    noise = np.random.uniform(low=-0.45, high=0.45, size=(len(start_traj_pts),3))
    # noise = np.array([[0,0,0]]*10)
    # print()
    new_traj_pts = []
    start_traj_pts_copy = start_traj_pts.copy()
    for i, pt in enumerate(start_traj_pts_copy):
        new_pt = pt.copy()
        if i == 0:
            new_traj_pts.append(new_pt)
            continue
        new_pt[0][3] = new_pt[0][3] + noise[i][0]
        new_pt[1][3] = new_pt[1][3] + noise[i][1]
        new_pt[2][3] = new_pt[2][3] + noise[i][2]
        new_traj_pts.append(new_pt)

    dist = calc_dist(target_traj_pts, new_traj_pts)
    print("dist: ", dist)
    if dist < min_dist:
        min_dist = dist
        final_traj_pts = new_traj_pts
        pause_time = 5
    update_plot(ax, None, None, new_traj_pts, q_target, endpoint_target, target_traj_pts, pause_time=pause_time)
    pause_time = 1


print("min_dist in 6dof space: ", min_dist)
plt.show()

# Show target, start and final traj
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([0, 12])
ax.set_ylim([0, 12])
ax.set_zlim([0, 12])
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
print("min_dist: ", min_dist)
# target 
ax.plot([q_target[0], endpoint_target[0]], [q_target[1], endpoint_target[1]], [q_target[2], endpoint_target[2]], color='g')
for pt in target_traj_pts:
    ax.scatter(pt[0][3], pt[1][3], pt[2][3], color='g', marker='o')
# start
for pt in start_traj_pts:
    ax.scatter(pt[0][3], pt[1][3], pt[2][3], color='r', marker='o')
# final
for pt in final_traj_pts:
    ax.scatter(pt[0][3], pt[1][3], pt[2][3], color='m', marker='o')
plt.show()