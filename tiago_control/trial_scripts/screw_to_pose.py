import time
import rospy
import numpy as np
from tiago_gym import TiagoGym, Listener
from control_msgs.msg  import JointTrajectoryControllerState
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm, expm
import pickle

def get_points_prismatic(s_hat, q, intial_pose, dist_moved=0.2):
    # Calculate the twist vector
    h = np.inf # pure translation
    w = np.array([0, 0, 0])
    v = np.array(s_hat)
    twist = np.concatenate((w,v)) 
    # twist = [0, 0, 0, 1, 0, 0]
    print("twist: ", twist)

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

    final_points = []
    # Calculate the transformation of the point when moved by theta along the screw axis
    for theta in np.arange(0, dist_moved, 0.1):
        S_theta = theta * np.array(S)

        T1 = np.dot(T0, expm(S_theta))
        final_points.append(T1)
        # T1 = np.dot(expm(S_theta), T0)

    return final_points


def get_points_revolute(s_hat, q, initial_pose, angle_moved=3.14):
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

        # printing
        T1_mat = T1[:3,:3]
        T1_euler = R.from_matrix(T1_mat).as_euler('XYZ')
        print("T1 POS: ", T1[0:3,3])
        # print("T1 ROT: ", T1_euler)
        # print("------")
        # ax.scatter(T1[0][3], T1[1][3], T1[2][3], color='g', marker='o')
    
    return final_points

def wait_right(desired_pos, env):
    start_time = time.time()
    while True:
        right_eef_pose = env._observation()['right_eef']
        # print("right_eef_pos: ", right_eef_pose[:3])
        error = abs(desired_pos - right_eef_pose[:3])
        current_time = time.time()
        time_diff = current_time - start_time
        if (error < 0.01).all() or time_diff > 30:
            break

def wait_left(desired_pos, env):
    start_time = time.time()
    while True:
        left_eef_pose = env._observation()['left_eef']
        error = abs(desired_pos - left_eef_pose[:3])
        current_time = time.time()
        time_diff = current_time - start_time
        if (error < 0.01).all() or time_diff > 30:
            break

def home_right_hand(env):
    pos = np.array([-0.02927815, -0.44696712,  1.11367162])
    quat = np.array([-0.62445749, -0.55287778, -0.39586573,  0.38427767])
    
    # New Home Pose: [ 0.4383153  -0.47736873  1.26197116  0.6875126   0.26625633 -0.45903672 0.49570079]
    pos = np.array([0.4383153,  -0.47736873,  1.26197116])
    quat = np.array([0.6875126,  0.26625633, -0.45903672, 0.49570079])
    pose_command = env.tiago.create_pose_command(pos, quat)
    env.tiago.tiago_pose_writer['right'].write(pose_command)
    wait_right(pos, env)
    right_eef_pose = env._observation()['right_eef']
    print("Actual right_eef_pos: ", right_eef_pose[:3])
    print("error: ", abs(pos - right_eef_pose[:3]))

def home_left_hand(env):
    # pos = np.array([-0.02635916,  0.45657024,  1.32772584])
    # quat = np.array([0.38418642,  0.39585825,  0.55286772, -0.62452728])
    
    # New Home Pose: [ 0.45367934  0.4771977   1.04679635  0.45480808 -0.15021598  0.17083846 -0.86104529]
    pos = np.array([0.50367934,  0.4771977,   1.04679635])
    quat = np.array([0.45480808, -0.15021598,  0.17083846, -0.86104529])
    pose_command = env.tiago.create_pose_command(pos, quat)
    env.tiago.tiago_pose_writer['left'].write(pose_command)
    wait_left(pos, env)
    left_eef_pose = env._observation()['left_eef']
    print("Actual left_eef_pos: ", left_eef_pose[:3])
    print("error: ", abs(pos - left_eef_pose[:3]))

# Initializations
rospy.init_node('tiago_test')
env = TiagoGym(frequency=10, right_arm_enabled=True, left_arm_enabled=True, right_gripper_type='robotiq', left_gripper_type='pal')

# This is an example axis
q = np.array([ 0.67766165, -0.09062318,  0.88022864])
s_hat = np.array([0, 0, 1])

# Obtain axis from perception
with open('/home/pal/arpit/tiago_teleop/tiago_control/data/tiago_full_pipeline_2/extrinsic.pickle', 'rb') as handle:
    extr = pickle.load(handle)
with open('/home/pal/arpit/tiago_teleop/tiago_control/data/tiago_full_pipeline_2/axis.pickle', 'rb') as handle:
    axis = pickle.load(handle)

q = np.array(axis['q'])
s_hat = np.array(axis['s_hat'])
# q_homo = np.append(q, 1.0)
# s_hat_homo = np.append(s_hat, 1.0)
# print("---", q_homo.shape, s_hat_homo.shape)
# q_final = np.dot(extr, q_homo) / 1000
# q_final = q_final[:3]
# s_hat_final = np.dot(extr, s_hat_homo)
# s_hat_final = s_hat_final[:3] 
print("s_hat, q: ", s_hat, q)

# Left hand: Set initial pose to a default value
# bottle_right_pose
left_eef_pos_pre = np.array([ 0.54033371,  0.10095334,  0.79674643])
left_eef_quat_pre = np.array([ 0.43567574, -0.54655992,  0.44715073, -0.5581354])
left_eef_pos = np.array([ 0.54033371,  0.03095334,  0.79674643])
left_eef_quat= np.array([ 0.43567574, -0.54655992,  0.44715073, -0.5581354])

# # bottle middle pose
# left_eef_pos_pre = np.array([ 0.61691971,  0.25651524,  0.78789944])
# left_eef_quat_pre = np.array([ 0.5509148,  -0.47211483, 0.39456799, -0.56384091])
# left_eef_pos = np.array([ 0.61691971,  0.17651524,  0.78789944])
# left_eef_quat= np.array([ 0.5509148,  -0.47211483, 0.39456799, -0.56384091])


# Right hand: Set initial pose to a default value
# bottle right 1
right_eef_pos = np.array([ 0.56027766, -0.19231454,  1.1236653])
right_eef_quat= np.array([ 0.69611122, -0.02994237, -0.71498465, -0.05770246])
# bottle middle pose - bad
# right_eef_pos = np.array([ 0.52753508, -0.03749103,  1.08782991])
# right_eef_quat= np.array([ 0.64279886,  0.4249718, -0.30581209, 0.55918472])
# bottle middle pose - good
# right_eef_pos = np.array([ 0.63405218, -0.05241308,  1.112755642])
# right_eef_quat= np.array([ 0.43944521,  0.5750991,  -0.31518151, 0.61384815])


# Initial pose
right_eef_mat = R.from_quat(right_eef_quat).as_matrix()
T0 = [
        [right_eef_mat[0][0], right_eef_mat[0][1], right_eef_mat[0][2], right_eef_pos[0]],
        [right_eef_mat[1][0], right_eef_mat[1][1], right_eef_mat[1][2], right_eef_pos[1]],
        [right_eef_mat[2][0], right_eef_mat[2][1], right_eef_mat[2][2], right_eef_pos[2]],
        [0, 0, 0, 1]
    ]
T0 = np.array(T0)

final_points = get_points_revolute(s_hat, q, T0)

print("STARTING ROBOT MOVEMENT")
# open both grippers
rospy.sleep(1)
env.tiago.gripper['left'].write(1)
env.tiago.gripper['right'].write(0)
rospy.sleep(1)
home_left_hand(env)
home_right_hand(env)

# # Make left gripper go to pre grasp pose
# rospy.sleep(1)
# pose_command = env.tiago.create_pose_command(left_eef_pos_pre, left_eef_quat_pre)
# env.tiago.tiago_pose_writer['left'].write(pose_command)
# wait_left(left_eef_pos, env)
# left_eef_pose = env._observation()['left_eef']
# print("Desired and Actual left_eef_pos: ", left_eef_pos, left_eef_pose[:3])
# print("error: ", abs(left_eef_pos - left_eef_pose[:3]))


# # Make left gripper go to grasp pose
# rospy.sleep(1)
# pose_command = env.tiago.create_pose_command(left_eef_pos, left_eef_quat)
# env.tiago.tiago_pose_writer['left'].write(pose_command)
# wait_left(left_eef_pos, env)
# left_eef_pose = env._observation()['left_eef']
# print("Desired and Actual left_eef_pos: ", left_eef_pos, left_eef_pose[:3])
# print("error: ", abs(left_eef_pos - left_eef_pose[:3]))
# # close gripper
# rospy.sleep(1)
# env.tiago.gripper['left'].write(0.3)


# Make right gripper go to grasp pose
rospy.sleep(1)
pose_command = env.tiago.create_pose_command(right_eef_pos, right_eef_quat)
env.tiago.tiago_pose_writer['right'].write(pose_command)
wait_right(right_eef_pos, env)
right_eef_pose = env._observation()['right_eef']
print("Desired and Actual right_eef_pos: ", right_eef_pos, right_eef_pose[:3])
print("error: ", abs(right_eef_pos - right_eef_pose[:3]))
# close gripper
rospy.sleep(1)
env.tiago.gripper['right'].write(0.7)

for i in range(12):
    input("Press Enter")
    pt = final_points[i]
    new_pos = np.array(pt[0:3,3])
    new_rot_mat = np.array(pt[:3,:3])
    new_quat = R.from_matrix(new_rot_mat).as_quat()
    
    pose_command = env.tiago.create_pose_command(new_pos, new_quat)
    env.tiago.tiago_pose_writer['right'].write(pose_command)
    rospy.sleep(1.5)
    right_eef_pose = env._observation()['right_eef']
    print("desied, actual right_eef_pos: ", new_pos, right_eef_pose[:3])
    print("error: ", abs(new_pos - right_eef_pose[:3]))