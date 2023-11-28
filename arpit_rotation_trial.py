import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math

# # Define the point
# point = np.array([1, 0, 0])

# # Define the unit vector
# unit_vector = np.array([1.5, 0.7, 1]) / np.linalg.norm([1.5, 0.7, 1])

# # Define the length of the line
# length = 5

# # Calculate the endpoint of the line
# endpoint = point + length * unit_vector

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the line
# ax.plot([point[0], endpoint[0]], [point[1], endpoint[1]], [point[2], endpoint[2]], color='b')

# # # Plot the point
# # ax.scatter(point[0], point[1], point[2], color='r', marker='o')

# pt1 = [3, 1, 0]
# # angle_of_rotation = np.pi / 4  # 45 degrees
# axis_of_rotation = unit_vector
# angle = math.radians(1)
# rotation_R = Rotation.from_rotvec(angle * axis_of_rotation)
# rotation_matrix_pre = Rotation.as_matrix(rotation_R)

# ax.scatter(pt1[0], pt1[1], pt1[2], color='r', marker='o')
# for i in range(50, 360, 50):
#     angle = math.radians(i)
#     rotation_R = Rotation.from_rotvec(angle * axis_of_rotation)
#     rotation_matrix_post = Rotation.as_matrix(rotation_R)
#     delta_rot = np.dot(rotation_matrix_pre, np.linalg.inv(rotation_matrix_post))
    
#     temp = Rotation.from_matrix(delta_rot)
#     temp_axis_angle = Rotation.as_rotvec(temp)
#     print("temp_axis_angle: ", np.array(temp_axis_angle) / np.linalg.norm(temp_axis_angle))

#     rotated_point_P = np.dot(rotation_matrix_post, pt1)
#     # rotated_point_P = rotation_R.apply(pt1)
#     ax.scatter(rotated_point_P[0], rotated_point_P[1], rotated_point_P[2], color='g', marker='o')
#     rotation_matrix_pre = rotation_matrix_post

# pred_axis = np.array(temp_axis_angle) / np.linalg.norm(temp_axis_angle)
# dummy_pt = [0,0,0]
# # Define the length of the line
# length = 5
# # Calculate the endpoint of the line
# endpoint = dummy_pt + length * pred_axis
# # Plot the line
# ax.plot([dummy_pt[0], endpoint[0]], [dummy_pt[1], endpoint[1]], [dummy_pt[2], endpoint[2]], color='r')

# # Set labels
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # Show the plot
# plt.show()


# ----------------------------------------------------------

# Function to plot the axis lines with arrows
def plot_axis(ax, color, vector, label, pos=[0,0,0]):
    ax.quiver(pos[0], pos[1], pos[2], vector[0], vector[1], vector[2], color=color, linewidth=2, length=0.1, normalize=True)

# Create a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# pt0 
pt0 = [3, 1, 0]
# pt0_orn = [[1,0,0],[0,1,0],[0,0,1]]
axis_of_rotation = np.array([1, 0, 0]) / np.linalg.norm([1, 0, 0])
print("SELECTED AXIS OF ROTATION: ", axis_of_rotation)
# plot
plot_axis(ax, 'red', [1, 0, 0], 'X', pt0)
plot_axis(ax, 'green', [0, 1, 0], 'Y', pt0)
plot_axis(ax, 'blue', [0, 0, 1], 'Z', pt0)
counter = 1

T0 = [
    [1, 0, 0, 0], 
    [0, 1, 0, 0], 
    [0, 0, 1, 0], 
    [0, 0, 0, 1], 
]

delta_Ts = []
# pt1 20 degrees for pos, 40 degrees for orn
for i in range(30, 360, 30):
    angle = i
    angle = math.radians(angle)
    rotation_R = R.from_rotvec(angle * axis_of_rotation)
    pt1_pos = rotation_R.apply(pt0)
    # # when no pos change
    # pt1_pos = pt0
    ax.scatter(pt1_pos[0], pt1_pos[1], pt1_pos[2], c=[[counter*0.05,0,0]], marker='o')
    print("pos of point: ", counter, pt1_pos)
    
    # angle = 20*counter
    angle = i
    angle = math.radians(angle)
    rotation_R = R.from_rotvec(angle * axis_of_rotation)
    pt1_rot_matrix = R.as_matrix(rotation_R)
    T1 = [
        [pt1_rot_matrix[0][0], pt1_rot_matrix[0][1], pt1_rot_matrix[0][2], pt1_pos[0]],
        [pt1_rot_matrix[1][0], pt1_rot_matrix[1][1], pt1_rot_matrix[1][2], pt1_pos[1]],
        [pt1_rot_matrix[2][0], pt1_rot_matrix[2][1], pt1_rot_matrix[2][2], pt1_pos[2]],
        [0, 0, 0, 1]
    ]
    transformed_x = np.dot(pt1_rot_matrix, [1, 0, 0])
    transformed_y = np.dot(pt1_rot_matrix, [0, 1, 0])
    transformed_z = np.dot(pt1_rot_matrix, [0, 0, 1])
    # plot
    plot_axis(ax, 'red', transformed_x, 'X', pt1_pos)
    plot_axis(ax, 'green', transformed_y, 'Y', pt1_pos)
    plot_axis(ax, 'blue', transformed_z, 'Z', pt1_pos)
    counter += 1
    delta_T = np.dot(T1, np.linalg.inv(T0))
    delta_Ts.append(delta_T)
    T0 = T1


from scipy.linalg import logm, expm
import numpy as np

# Example transformation matrix
# T = np.array([[0.707, -0.707, 0, 1],
#               [0.707, 0.707, 0, 2],
#               [0, 0, 1, 3],
#               [0, 0, 0, 1]])

print("delta_Ts: ", np.array(delta_Ts).shape)

for counter, delta_T in enumerate(delta_Ts):
    print(f"----- {counter} -----")
    if counter == 0:
        continue
    # Compute the matrix logarithm
    log_T = logm(delta_T)
    print(log_T)

    # Extract linear velocities
    linear_velocities = log_T[:3, 3]

    # Extract skew-symmetric matrix (angular velocities)
    S = log_T[:3, :3]

    # Calculate angular velocities from the skew-symmetric matrix
    angular_velocities = np.array([S[2, 1] - S[1, 2], S[0, 2] - S[2, 0], S[1, 0] - S[0, 1]]) / 2.0

    # Combine linear and angular velocities to get the twist
    twist = np.concatenate((linear_velocities, angular_velocities))

    print("Twist vector:", twist)

    screw_axis = angular_velocities / np.linalg.norm(angular_velocities)
    theta = np.linalg.norm(angular_velocities)
    q = np.cross(screw_axis, linear_velocities) / theta
    print("screw_axis: ", screw_axis)
    print("q: ", q)

    # Define the length of the line
    length = 2

    # Calculate the endpoint of the line
    endpoint = q + length * screw_axis

    # Plot the line
    ax.plot([q[0], endpoint[0]], [q[1], endpoint[1]], [q[2], endpoint[2]], c=[counter*0.05,0,0])

    # Set labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    
# Show the plot
plt.show()
