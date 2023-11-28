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
        ax.scatter(pt[0], pt[1], pt[2], color='g', marker='o')
    
    ax.plot([q[0], endpoint[0]], [q[1], endpoint[1]], [q[2], endpoint[2]], color='m')
    for pt in new_traj_pts:
        ax.scatter(pt[0], pt[1], pt[2], color='m', marker='o')

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

# Target_axis
s_hat_target = np.array([1.0, 0.0, 0.0])
q_target = np.array([1.0, 1.0, 0.0])
# Plot the screw axis
length = 10
endpoint_target = q_target + length * s_hat_target
ax.plot([q_target[0], endpoint_target[0]], [q_target[1], endpoint_target[1]], [q_target[2], endpoint_target[2]], color='g')
target_traj_pts = get_points(start_pt=q_target, direction_vec=s_hat_target, len_org_line=length, sample_pts=10)

# Incorrect start axis
noise = np.random.uniform(low=-0.15, high=0.15, size=(3,))
print(noise)
s_start = s_hat_target + noise
s_hat_start = s_start / np.linalg.norm(s_start)
noise = np.random.uniform(low=-0.15, high=0.15, size=(3,))
q_start = q_target + noise
# Plot the screw axis
length = 10
endpoint_start = q_start + length * s_hat_start
ax.plot([q_start[0], endpoint_start[0]], [q_start[1], endpoint_start[1]], [q_start[2], endpoint_start[2]], color='r')
start_traj_pts = get_points(start_pt=q_start, direction_vec=s_hat_start, len_org_line=length, sample_pts=10)
dist = calc_dist(target_traj_pts, start_traj_pts)
print("Initial traj dist: ", dist)
# display the two trajs
for pt in target_traj_pts:
    ax.scatter(pt[0], pt[1], pt[2], color='g', marker='o')
for pt in start_traj_pts:
    ax.scatter(pt[0], pt[1], pt[2], color='r', marker='o')
# plt.show()

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
#     length = 10
#     endpoint_new = q_new + length * s_hat_new
#     # confirm the length of the line
#     line_norm = np.linalg.norm(q_new - endpoint_new)
#     # print("norm of the new line: ", line_norm)
#     new_traj_pts = get_points(start_pt=q_new, direction_vec=s_hat_new, len_org_line=length, sample_pts=10)
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
#     ax.scatter(pt[0], pt[1], pt[2], color='g', marker='o')
# # start
# ax.plot([q_start[0], endpoint_start[0]], [q_start[1], endpoint_start[1]], [q_start[2], endpoint_start[2]], color='r')
# for pt in start_traj_pts:
#     ax.scatter(pt[0], pt[1], pt[2], color='r', marker='o')
# # final
# q_final, endpoint_final = final_axis[0], final_axis[1]
# ax.plot([q_final[0], endpoint_final[0]], [q_final[1], endpoint_final[1]], [q_final[2], endpoint_final[2]], color='m')
# for pt in final_traj_pts:
#     ax.scatter(pt[0], pt[1], pt[2], color='m', marker='o')
# plt.show()

# With points
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
    noise = np.random.uniform(low=-0.35, high=0.35, size=(10,3))
    # noise = np.array([[0,0,0]]*10)
    # print()
    new_traj_pts = start_traj_pts + noise
    dist = calc_dist(target_traj_pts, new_traj_pts)
    print("dist: ", dist)
    if dist < min_dist:
        min_dist = dist
        final_traj_pts = new_traj_pts
        pause_time = 5
    update_plot_pts(ax, new_traj_pts, q_target, endpoint_target, target_traj_pts, pause_time=pause_time)
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
    ax.scatter(pt[0], pt[1], pt[2], color='g', marker='o')
# start
ax.plot([q_start[0], endpoint_start[0]], [q_start[1], endpoint_start[1]], [q_start[2], endpoint_start[2]], color='r')
for pt in start_traj_pts:
    ax.scatter(pt[0], pt[1], pt[2], color='r', marker='o')
# final
for i in range(len(final_traj_pts)):
    pt = final_traj_pts[i]
    ax.scatter(pt[0], pt[1], pt[2], color='m', marker='o')
    if i+1 < len(final_traj_pts):
        pt2 = final_traj_pts[i+1]
        ax.plot([pt[0], pt2[0]], [pt[1], pt2[1]], [pt[2], pt2[2]], color='m')
plt.show()