import time
import rospy
import numpy as np
from tiago_gym import TiagoGym, Listener
from control_msgs.msg  import JointTrajectoryControllerState
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm, expm

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
    for theta in np.arange(0, angle_moved, 0.2):
        S_theta = theta * np.array(S)

        # T1 = np.dot(T0, expm(S_theta))
        T1 = np.dot(expm(S_theta), T0)
        final_points.append(T1)

        # printing
        T1_mat = T1[:3,:3]
        T1_euler = R.from_matrix(T1_mat).as_euler('XYZ')
        print("T1: ", T1[0:3,3])
        print("T1 ROT: ", T1_euler)
        print("------")
        # ax.scatter(T1[0][3], T1[1][3], T1[2][3], color='g', marker='o')
    
    return final_points

def wait_right(desired_pos, env):
    start_time = time.time()
    while True:
        right_eef_pose = env._observation()['right_eef']
        # print("right_eef_pos: ", right_eef_pose[:3])
        # TODO: change the error calculation to account for orientation as well 
        error = abs(desired_pos - right_eef_pose[:3])
        current_time = time.time()
        time_diff = current_time - start_time
        if (error < 0.01).all() or time_diff > 30:
            break

def home_right_hand(env):
    # home
    pos = np.array([-0.02927815, -0.44696712,  1.11367162])
    quat = np.array([-0.62445749, -0.55287778, -0.39586573,  0.38427767])
    pose_command = env.tiago.create_pose_command(pos, quat)
    env.tiago.tiago_pose_writer['right'].write(pose_command)
    rospy.sleep(40)
    # right_eef_pose = env._observation()['right_eef']
    # print("Actual right_eef_pos: ", right_eef_pose[:3])
    # print("error: ", abs(pos - right_eef_pose[:3]))

# Initializations
np.random.seed(2)
rospy.init_node('tiago_test')
env = TiagoGym(frequency=10, right_arm_enabled=True, left_arm_enabled=True, right_gripper_type='robotiq', left_gripper_type='pal')

for _ in range(5):

    # Open gripper and go home
    rospy.sleep(2)
    env.tiago.gripper['right'].write(0)
    rospy.sleep(2)
    home_right_hand(env)


    # Make right gripper go to grasp pose and grasp
    rospy.sleep(2)
    pos = np.array([ 0.67766165, -0.09062318,  0.88022864])
    quat= np.array([ 0.70426831, -0.00769203, -0.70922737, -0.03071679])
    pose_command = env.tiago.create_pose_command(pos, quat)
    env.tiago.tiago_pose_writer['right'].write(pose_command)

    wait_right(pos, env)

    # Maybe this rospy.sleep makes the "obtained pose" correct?
    rospy.sleep(2)
    right_eef_pose = env._observation()['right_eef']
    print("Desired and Actual right_eef_pos: ", pos, right_eef_pose[:3])
    print("error: ", abs(pos - right_eef_pose[:3]))
    right_eef_pos, right_eef_quat = right_eef_pose[:3], right_eef_pose[3:]
    right_eef_mat = R.from_quat(right_eef_quat).as_matrix()
    print("right_eef_mat: ", right_eef_mat)

    # Grasp
    env.tiago.gripper['right'].write(0.7)

    # Initial pose
    T0 = [
            [right_eef_mat[0][0], right_eef_mat[0][1], right_eef_mat[0][2], right_eef_pos[0]],
            [right_eef_mat[1][0], right_eef_mat[1][1], right_eef_mat[1][2], right_eef_pos[1]],
            [right_eef_mat[2][0], right_eef_mat[2][1], right_eef_mat[2][2], right_eef_pos[2]],
            [0, 0, 0, 1]
        ]
    T0 = np.array(T0)
    print(T0)

    # This is the target screw axis
    s_hat_target = np.array([0, 0, 1])
    q_target = right_eef_pos

    # Adding noise to s_hat. This noise can have a larger variance.
    noise = np.random.uniform(low=-0.4, high=0.4, size=(3,))
    noise = np.random.normal(0, 0.4, size=(3,))
    # remove later
    # noise = np.array([0.0, 0.0, 0.0])
    print("noise for s_hat: ", noise)
    s_start = s_hat_target + noise
    s_hat_start = s_start / np.linalg.norm(s_start)
    # Adding noise to q. This noise should have a smaller variance.
    noise = np.random.uniform(low=-0.1, high=0.1, size=(3,))
    noise = np.random.normal(0, 0.1, size=(3,))
    # remove later
    # noise = np.array([0.0, 0.0, 0.15])
    print("noise for q: ", noise)
    q_start = q_target + noise

    print("s_hat_start, q_start: ", s_hat_start, q_start)

    final_points = get_points_revolute(s_hat_start, q_start, T0)

    # input("STARTING ROBOT MOVEMENT. PRESS ENTER")
    print("STARTING ROBOT MOVEMENT")
    rospy.sleep(3)
    for i in range(len(final_points)):
        pt = final_points[i]
        new_pos = np.array(pt[0:3,3])
        new_rot_mat = np.array(pt[:3,:3])
        new_quat = R.from_matrix(new_rot_mat).as_quat()
        
        pose_command = env.tiago.create_pose_command(new_pos, new_quat)
        env.tiago.tiago_pose_writer['right'].write(pose_command)
        rospy.sleep(3)
        # input("press enter")
        right_eef_pose = env._observation()['right_eef']
        print("desied, actual right_eef_pos: ", pos, right_eef_pose[:3])
        print("error: ", abs(pos - right_eef_pose[:3]))