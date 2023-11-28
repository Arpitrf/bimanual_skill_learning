import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm, expm
import matplotlib.pyplot as plt

# Create a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

s_hat = np.array([1, 0, 0])
q = np.array([1, 1, 0])

# Plot the screw axis
length = 10
# Calculate the endpoint of the line
endpoint = q + length * s_hat
# print(q, endpoint)
ax.plot([q[0], endpoint[0]], [q[1], endpoint[1]], [q[2], endpoint[2]], color='m')


# Calculate the twist vector
h = np.inf # pure translation
w = np.array([0, 0, 0])
v = -np.cross(s_hat, q)
twist = np.concatenate((w,v)) 
print("twist: ", twist)

twist = [0, 0, 0, 1, 0, 0]
w = twist[:3]
w_matrix = [
    [0, -w[2], w[1]],
    [w[2], 0, -w[0]],
    [-w[1], w[0], 0],
]
print("w_matrix: ", w_matrix)

S = [
    [w_matrix[0][0], w_matrix[0][1], w_matrix[0][2], twist[3]],
    [w_matrix[1][0], w_matrix[1][1], w_matrix[1][2], twist[4]],
    [w_matrix[2][0], w_matrix[2][1], w_matrix[2][2], twist[5]],
    [0, 0, 0, 0]
]

# initial pose
T0 = np.array([[1, 0, 0, 4],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])
ax.scatter(T0[0][3], T0[1][3], T0[2][3], color='r', marker='o')

for theta in np.arange(0.1, 3.2, 0.1):
    S_theta = theta * np.array(S)

    T1 = np.dot(T0, expm(S_theta))
    # T1 = np.dot(expm(S_theta), T0)
    ax.scatter(T1[0][3], T1[1][3], T1[2][3], color='g', marker='o')

 # Set labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

plt.show()

# ---------------------------------------------------------

# Create a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

s_hat = np.array([1, 0, 0])
q = np.array([1, 1, 0])

# Plot the screw axis
length = 10
# Calculate the endpoint of the line
endpoint = q + length * s_hat
ax.plot([q[0], endpoint[0]], [q[1], endpoint[1]], [q[2], endpoint[2]], color='m')

# Calculate the twist vector
h = 0 # pure rotation
w = s_hat
v = -np.cross(s_hat, q)
twist = np.concatenate((w,v)) 
print("twist: ", twist)

# Calculate the matrix form of the twist vector
w = twist[:3]
# w_matrix = R.from_rotvec(w).as_matrix()
w_matrix = [
    [0, -w[2], w[1]],
    [w[2], 0, -w[0]],
    [-w[1], w[0], 0],
]
print("w_matrix: ", w_matrix)

S = [
    [w_matrix[0][0], w_matrix[0][1], w_matrix[0][2], twist[3]],
    [w_matrix[1][0], w_matrix[1][1], w_matrix[1][2], twist[4]],
    [w_matrix[2][0], w_matrix[2][1], w_matrix[2][2], twist[5]],
    [0, 0, 0, 0]
]

# Initial pose
T0 = np.array([[1, 0, 0, 4],
    [0, 1, 0, 2],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])
ax.scatter(T0[0][3], T0[1][3], T0[2][3], color='r', marker='o')

# Calculate the transformation of the point when moved by theta along the screw axis
for theta in np.arange(0.1, 6.4, 0.1):
    S_theta = theta * np.array(S)

    # T1 = np.dot(T0, expm(S_theta))
    T1 = np.dot(expm(S_theta), T0)
    ax.scatter(T1[0][3], T1[1][3], T1[2][3], color='g', marker='o')

 # Set labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

plt.show()
