import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm, expm

def get_points_prismatic(s_hat, initial_pose, dist_moved=0.2):
    twist = [0, 0, 0] + s_hat.tolist()
    w = twist[:3]
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

    # initial pose
    T0 = np.array([[1, 0, 0, initial_pose[0,3]],
        [0, 1, 0, initial_pose[1,3]],
        [0, 0, 1, initial_pose[2,3]],
        [0, 0, 0, 1]])
    # T0 = start_T
    # ax.scatter(T0[0][3], T0[1][3], T0[2][3], color='r', marker='o')
    
    final_points = []
    for theta in np.arange(0.015, 0.28, 0.015):
        S_theta = theta * np.array(S)

        T1 = np.dot(T0, expm(S_theta))
        T1 = np.array([
            [initial_pose[0][0], initial_pose[0][1], initial_pose[0][2], T1[0,3]],
            [initial_pose[1][0], initial_pose[1][1], initial_pose[1][2], T1[1,3]],
            [initial_pose[2][0], initial_pose[2][1], initial_pose[2][2], T1[2,3]],
            [0, 0, 0, 1]
        ])
        final_points.append(T1)
        # print("T1 POS: ", T1[0:3,3])

        # T1 = np.dot(expm(S_theta), T0)
        # ax.scatter(T1[0][3], T1[1][3], T1[2][3], color='g', marker='o')
        
    return final_points

def compute_trajectory_score(original_Ts, computed_Ts, log=False):
    original_Ts = np.array(original_Ts)
    computed_Ts = np.array(computed_Ts)
    trajectory_pos_dist = 0
    trajectory_orn_dist = 0
    trajectory_orn_dist2 = 0
    for i in range(len(original_Ts)):
        original_pos = original_Ts[i][:3, 3]
        computed_pos = computed_Ts[i][:3, 3]
        point_dist = np.linalg.norm(original_pos-computed_pos)
        if log:
            print(f"index {i}: ", original_pos, computed_pos, point_dist)
        trajectory_pos_dist += point_dist

        original_rot = original_Ts[i][:3, :3]
        computed_rot = computed_Ts[i][:3, :3]
        original_quat = R.from_matrix(original_rot).as_quat()
        original_quat = original_quat / np.linalg.norm(original_quat)
        computed_quat = R.from_matrix(computed_rot).as_quat()
        computed_quat = computed_quat / np.linalg.norm(computed_quat)
        orn_dist = np.linalg.norm(original_quat-computed_quat)
        orn_dist2 = 1 - np.dot(original_quat, computed_quat)**2
        trajectory_orn_dist += orn_dist
        trajectory_orn_dist2 += orn_dist2

    return trajectory_pos_dist, trajectory_orn_dist, trajectory_orn_dist2

def get_points_revolute(s_hat, q, initial_pose, angle_moved=4.0, log=False):
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
        # print("S_theta: ", S_theta)

        # T1 = np.dot(T0, expm(S_theta))
        T1 = np.dot(expm(S_theta), T0)
        final_points.append(T1)

        # printing
        T1_mat = T1[:3,:3]
        T1_euler = R.from_matrix(T1_mat).as_euler('XYZ', degrees=True)
        if log:
            print("T1 POS: ", T1[0:3,3])
        # print("T1 ROT: ", T1_euler)
        # print("------")
        # ax.scatter(T1[0][3], T1[1][3], T1[2][3], color='g', marker='o')
    
    return final_points

def updateGaussianParams(lis):
    """update parameters mu, sigma"""
    mu_x = lis[:, 0].mean()
    sigma_x = lis[:, 0].std()
    mu_y = lis[:, 1].mean()
    sigma_y = lis[:, 1].std()
    mu_z = lis[:, 2].mean()
    sigma_z = lis[:, 2].std()
    # print("----", mu_x, mu_y, mu_z)
    return mu_x, sigma_x, mu_y, sigma_y, mu_z, sigma_z

def updateUniformParams(lis):
    """update parameters mu, sigma"""
    min_x = np.amin(lis[:, 0])
    max_x = np.amax(lis[:, 0])
    min_y = np.amin(lis[:, 1])
    max_y = np.amax(lis[:, 1])
    min_z = np.amin(lis[:, 2])
    max_z = np.amax(lis[:, 2])
    return min_x, max_x, min_y, max_y, min_z, max_z

seed = 1
np.random.seed(seed)
with open('/home/pal/arpit/bimanual_skill_learning/data/tiago_jacket_1/axis.pickle', 'rb') as handle:
    axis = pickle.load(handle)
q_perception = np.array(axis['q'])
s_hat_perception = np.array(axis['s_hat'])
s_hat_perception = np.array([ 0.40045825, -0.91595374,  0.02572816])
print("------- Starting axis parameters (s_hat_perception): ", s_hat_perception)

right_eef_pos = np.array([ 0.56752687, 0.02694048,  1.00036204])
right_eef_quat= np.array([ 0.60564992,  0.40772559, -0.4305285, 0.53065358])
right_eef_mat = R.from_quat(right_eef_quat).as_matrix()

tooltip_ee_offset = np.array([0.26, 0, 0])
tooltip_ee_offset_world = np.dot(right_eef_mat, tooltip_ee_offset)
right_eef_pos = right_eef_pos + tooltip_ee_offset_world[:3]
# print("1right_eef_pos: ", temp_pos)

T0 = [
        [right_eef_mat[0][0], right_eef_mat[0][1], right_eef_mat[0][2], right_eef_pos[0]],
        [right_eef_mat[1][0], right_eef_mat[1][1], right_eef_mat[1][2], right_eef_pos[1]],
        [right_eef_mat[2][0], right_eef_mat[2][1], right_eef_mat[2][2], right_eef_pos[2]],
        [0, 0, 0, 1]
    ]
T0 = np.array(T0)

temp_lis = []
s_hat_ideal = np.array([0.0, -1.0, 0.0])
q_ideal = right_eef_pos
# ideal_traj = get_points_revolute(s_hat_ideal, q_ideal, T0)
ideal_traj = get_points_prismatic(s_hat_ideal, T0)
# trajectory_pos_dist, trajectory_orn_dist, trajectory_orn_dist2 = compute_trajectory_score(ideal_traj, ideal_traj)
# print("trajectory_pos_dist, trajectory_orn_dist, trajectory_orn_dist2: ", trajectory_pos_dist, trajectory_orn_dist, trajectory_orn_dist2)
mu_x, sigma_x = 0, 0.1
mu_y, sigma_y = 0, 0.1
mu_z, sigma_z = 0, 0.1
min_x, max_x = -0.3, 0.3
min_y, max_y = -0.3, 0.3
min_z, max_z = -0.3, 0.3

for epoch in range(5):
    # Take all traj where score < 1.5 or first 10
    trajs = []
    elite_trajs = []

    for traj_number in range(10):
        # print(f"================{traj_number}=================")   
        # Adding noise to s_hat. This noise can have a larger variance.
        if epoch == 0:
            noise_s_hat_x = np.random.uniform(low=min_x, high=max_x)
            noise_s_hat_y = np.random.uniform(low=min_y, high=max_y)
            noise_s_hat_z = np.random.uniform(low=min_z, high=max_z)
            # remove later
            noise_s_hat_z = 0.0
            noise_s_hat = np.array([noise_s_hat_x, noise_s_hat_y, noise_s_hat_z])
        else:
            s_hat_perception = s_hat_new
            noise_s_hat_x = np.random.normal(mu_x, sigma_x)
            noise_s_hat_y = np.random.normal(mu_y, sigma_y)
            noise_s_hat_z = 0.0
            noise_s_hat = np.array([noise_s_hat_x, noise_s_hat_y, noise_s_hat_z])

        s = s_hat_perception + noise_s_hat
        s_hat = s / np.linalg.norm(s)

        # if (epoch == 4 or epoch == 0) and traj_number % 5 == 0:
        # if epoch == 4: 
        #     print(f"================{epoch}: {traj_number}=================")   
        #     test_traj = get_points_revolute(s_hat, q, T0, log=True)
        #     print("new s_hat, q: ", s_hat, q)
        # else:
        #     test_traj = get_points_revolute(s_hat, q, T0)
        test_traj = get_points_prismatic(s_hat, T0)

        trajectory_pos_dist, trajectory_orn_dist, trajectory_orn_dist2 = compute_trajectory_score(ideal_traj, test_traj)
        # print("new s_hat, q: ", s_hat, q)
        # if (epoch == 4 or epoch == 0) and traj_number % 5 == 0:
        # if epoch == 4: 
        #     print("trajectory_pos_dist, trajectory_orn_dist, trajectory_orn_dist2: ", trajectory_pos_dist, trajectory_orn_dist, trajectory_orn_dist2)
        
        # compute score of a trajectory
        traj_score = (trajectory_pos_dist, trajectory_orn_dist2)
        # traj_score = trajectory_pos_dist + trajectory_orn_dist2
        # temp_lis.append((epoch, traj_number, trajectory_pos_dist, trajectory_orn_dist2))
        # if trajectory_pos_dist < 1.5 and trajectory_orn_dist2 < 0.2:
        print("traj dist: ", epoch, traj_number, s_hat, trajectory_pos_dist, trajectory_orn_dist2)

        test_traj = np.array(test_traj)
        # print("test_traj: ", np.array(test_traj).shape, traj_score)
        trajs.append((test_traj, traj_score, noise_s_hat, s_hat))

    # first epoch set normal distribution parameters
    if epoch == 0:
        # trajs = sorted(trajs, key=lambda x: x[1])
        noise_s_hat_lis = []
        s_hats = []
        for t in trajs:
            if t[1][0] < 0.5:
                elite_trajs.append(t)
                s_hats.append(t[3])
                noise_s_hat_lis.append(t[2])
        noise_s_hat_lis = np.array(noise_s_hat_lis)
        s_hats = np.array(s_hats)
        s_hat_new = np.mean(s_hats, axis=0)
        print("s_hat_new: ", s_hat_new)
        print("len(elite_trajs): ", len(elite_trajs))
    
    if epoch > 0:
        noise_s_hat_lis = []
        s_hats = []
        for t in trajs:
            # print(t[1])
            if t[1][0] < 0.5:
                elite_trajs.append(t)
                s_hats.append(t[3])
                noise_s_hat_lis.append(t[2])
        print("len(elite_trajs): ", len(elite_trajs))
        noise_s_hat_lis = np.array(noise_s_hat_lis)
        if len(noise_s_hat_lis) != 0:
            mu_x, sigma_x, mu_y, sigma_y, mu_z, sigma_z = updateGaussianParams(noise_s_hat_lis)
            print("mu_x, sigma_x, mu_y, sigma_y, mu_z, sigma_z: ", mu_x, sigma_x, mu_y, sigma_y, mu_z, sigma_z)

        

    # print("noise_q_lis: ", noise_q_lis.shape)
    # mu_x, sigma_x, mu_y, sigma_y, mu_z, sigma_z = updateGaussianParams(noise_q_lis)
    
    # if len(noise_s_hat_lis) != 0:
    #     min_x, max_x, min_y, max_y, min_z, max_z = updateUniformParams(noise_s_hat_lis)
    # if len(noise_q_lis) != 0:
    #     min_x_q, max_x_q, min_y_q, max_y_q, min_z_q, max_z_q = updateUniformParams(noise_q_lis)
    # print("mu_x, sigma_x, mu_y, sigma_y, mu_z, sigma_z: ", mu_x, sigma_x, mu_y, sigma_y, mu_z, sigma_z)
    # print("min_x_q, max_x_q, min_y_q, max_y_q, min_z_q, max_z_q: ", min_x_q, max_x_q, min_y_q, max_y_q, min_z_q, max_z_q )

