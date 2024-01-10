import sys
import os
parent_dir = os.path.dirname('/home/pal/arpit/bimanual_skill_learning/tiago_control')
sys.path.append(parent_dir)
from utils.camera_utils import Camera

import time
import rospy
import numpy as np
from tiago_control.tiago_gym import TiagoGym, Listener
from control_msgs.msg  import JointTrajectoryControllerState
from scipy.spatial.transform import Rotation as R
import pickle
import matplotlib.pyplot as plt
import cv2

rospy.init_node('tiago_test')
env = TiagoGym(frequency=10, right_arm_enabled=True, left_arm_enabled=True, right_gripper_type='robotiq', left_gripper_type='robotiq')

# -------------- ---- Print the current pose --------------------
right_eef_pose, left_eef_pose = None, None
while right_eef_pose is None or left_eef_pose is None:
    right_eef_pose = env._observation()['right_eef']
    left_eef_pose = env._observation()['left_eef']
    right_gripper_state = env._observation()['right_gripper']
    left_gripper_state = env._observation()['left_gripper']
    # print("right_gripper, left_gripper: ", right_gripper_state, left_gripper_state)

right_eef_pos, right_eef_orn_quat = right_eef_pose[:3], right_eef_pose[3:]
print("right pose: ", right_eef_pose)
right_eef_orn_euler = R.from_quat(right_eef_orn_quat).as_euler('XYZ', degrees=True)
print("right_eef_orn_euler: ", right_eef_orn_euler)

left_eef_pos, left_eef_orn_quat = left_eef_pose[:3], left_eef_pose[3:]
print("left pose: ", left_eef_pose)
left_eef_orn_euler = R.from_quat(left_eef_orn_quat).as_euler('XYZ', degrees=True)
print("left_eef_orn_euler: ", left_eef_orn_euler)
# ----------------------------------------------------------------

# for _ in range(1000):
#     # print("proprio values: ", env._observation())
#     print("------------")
#     right_eef_pose = env._observation()['right_eef']
#     left_eef_pose = env._observation()['left_eef']
#     right_gripper_state = env._observation()['right_gripper']
#     left_gripper_state = env._observation()['left_gripper']
#     print("right_gripper, left_gripper: ", right_gripper_state, left_gripper_state)
#     if right_eef_pose is not None:
#         right_eef_pos, right_eef_orn_quat = right_eef_pose[:3], right_eef_pose[3:]
#         print("right pose: ", right_eef_pose)
#         # right_eef_orn_euler = R.from_quat(right_eef_orn_quat).as_euler('XYZ', degrees=True)
#         # print("right_eef_orn_euler: ", right_eef_orn_euler)
#     if left_eef_pose is not None:
#         left_eef_pos, left_eef_orn_quat = left_eef_pose[:3], left_eef_pose[3:]
#         print("left pose: ", left_eef_pose)

def get_XYZ(depth_Z, pix_x, pix_y):
    # # for realsense camera
    # intr = np.array([
    #         [606.76220703,   0,         308.31533813],
    #         [  0,         606.91583252, 255.4833374 ],
    #         [  0,           0,           1        ]
    #     ])
    # for tiago's camera:
    intr = np.array([
        [523.9963414139355, 0.0, 328.83202929614686],
        [0.0, 524.4907272320442, 237.83703502879925],
        [0.0, 0.0, 1.0]
    ])

    click_z = depth_Z
    # click_z *= depth_scale
    click_x = (pix_x-intr[0, 2]) * \
        click_z/intr[0, 0]
    click_y = (pix_y-intr[1, 2]) * \
        click_z/intr[1, 1]
    if click_z == 0:
        raise Exception('Invalid pick point')
    # 3d point in camera coordinates
    point_3d = np.asarray([click_x, click_y, click_z])
    # print("point3d: ", point_3d)
    return point_3d

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

def wait_left(desired_pos, env, pose_command_right=None):
    start_time = time.time()
    while True:
        if pose_command_right is not None:
            # print("111")
            env.tiago.tiago_pose_writer['right'].write(pose_command_right)
        left_eef_pose = env._observation()['left_eef']
        # print("left_eef_pos: ", left_eef_pose[:3])
        error = abs(desired_pos - left_eef_pose[:3])
        current_time = time.time()
        time_diff = current_time - start_time
        if (error < 0.01).all() or time_diff > 30:
            break

def home_left_hand(env):
    # pos = np.array([-0.02635916,  0.45657024,  1.32772584])
    # quat = np.array([0.38418642,  0.39585825,  0.55286772, -0.62452728])
    
    # New Home Pose: [ 0.45367934  0.4771977   1.04679635  0.45480808 -0.15021598  0.17083846 -0.86104529]o
    # Home Pose for grasp test: [ 0.58049715  0.52146272  1.04852311 -0.799941    0.29258536  0.46524071 -0.24091343]
    pos = np.array([0.58049715,  0.52146272,  1.04852311])
    quat = np.array([-0.799941,    0.29258536,  0.46524071, -0.24091343])
    pose_command = env.tiago.create_pose_command(pos, quat)
    env.tiago.tiago_pose_writer['left'].write(pose_command)
    rospy.sleep(20)
    left_eef_pose = env._observation()['left_eef']
    print("Actual left_eef_pos: ", left_eef_pose[:3])
    print("error: ", abs(pos - left_eef_pose[:3]))


def home_right_hand(env):
    pos = np.array([-0.02927815, -0.44696712,  1.11367162])
    quat = np.array([-0.62445749, -0.55287778, -0.39586573,  0.38427767])
    
    # New Home Pose: [ 0.4383153  -0.47736873  1.26197116  0.6875126   0.26625633 -0.45903672 0.49570079]
    pos = np.array([0.4383153,  -0.47736873,  1.26197116])
    quat = np.array([0.6875126,  0.26625633, -0.45903672, 0.49570079])
    pose_command = env.tiago.create_pose_command(pos, quat)
    env.tiago.tiago_pose_writer['right'].write(pose_command)
    rospy.sleep(20)
    right_eef_pose = env._observation()['right_eef']
    print("Actual right_eef_pos: ", right_eef_pose[:3])
    print("error: ", abs(pos - right_eef_pose[:3]))


# # --------------- tesing hand movements ------------------
# # -------- reset hands ----------
# rospy.sleep(2)
# env.tiago.gripper['left'].write(1)
# env.tiago.gripper['right'].write(0)
# rospy.sleep(2)
# # go to home
# home_left_hand(env)
# home_right_hand(env)


# # ----------- left hand ----------
# # target left hand
# # tissue roll example [ 0.67208884  0.1838634   0.82299341 -0.71609082 -0.01201578  0.6965159 0.04399038]
# # bottle example [ 0.54033371  0.03095334  0.79674643  0.43567574 -0.54655992  0.44715073 -0.5581354 ]
# pos = np.array([ 0.54033371,  0.03095334,  0.79674643])
# quat= np.array([ 0.43567574, -0.54655992,  0.44715073, -0.5581354])
# pose_command = env.tiago.create_pose_command(pos, quat)

# # input("Left Hand Going To Pose. Press Enter")
# print("Left Hand Going To Pose")
# env.tiago.tiago_pose_writer['left'].write(pose_command)

# wait_left(pos, env)

# rospy.sleep(1)
# left_eef_pose = env._observation()['left_eef']
# print("Desired and Actual left_eef_pos: ", pos, left_eef_pose[:3])
# print("error in left hand position: ", abs(pos - left_eef_pose[:3]))

# rospy.sleep(2)
# env.tiago.gripper['left'].write(0)

# # ------------------------ right hand ------------------------
# # target right hand
# # tissue example: [ 0.67766165 -0.11062318  0.88022864  0.70426831 -0.00769203 -0.70922737 -0.03071679]
# # bottle example: [ 0.53027766 -0.19231454  1.1336653   0.69611122 -0.02994237 -0.71498465 -0.05770246]
# rospy.sleep(2)
# pos = np.array([ 0.53027766, -0.19231454,  1.1236653])
# quat= np.array([ 0.69611122, -0.02994237, -0.71498465, -0.05770246])
# pose_command = env.tiago.create_pose_command(pos, quat)
# env.tiago.tiago_pose_writer['right'].write(pose_command)

# wait_right(pos, env)

# # rospy.sleep(30)
# right_eef_pose = env._observation()['right_eef']
# print("Desired and Actual right_eef_pos: ", pos, right_eef_pose[:3])
# print("error: ", abs(pos - right_eef_pose[:3]))

# rospy.sleep(2)
# env.tiago.gripper['right'].write(0.7)

# Moving the right hand right side
# rospy.sleep(2)
# for counter in range(5):
#     print("============= step: {counter} ===========")
#     pos = pos - [0.0, 0.05, 0.0]
#     pose_command = env.tiago.create_pose_command(pos, quat)
#     env.tiago.tiago_pose_writer['right'].write(pose_command)
#     rospy.sleep(3)
#     right_eef_pose = env._observation()['right_eef']
#     print("desied, actual right_eef_pos: ", pos, right_eef_pose[:3])
#     print("error: ", abs(pos - right_eef_pose[:3]))

# # open gripper
# rospy.sleep(2)
# env.tiago.gripper['right'].write(0)

# --------------------------------------------------------------------------------------------

# # ---------- Testing human hand pose to ee mapping -----------
# rospy.sleep(2)
# print("env.tiago: ", env.tiago)
# pos, extr_quat = env.tiago.get_camera_extrinsic
# extr = R.from_quat(extr_quat).as_matrix()
# print("extr: ", extr)

# # R_hand_ee = [
# #     [0, -1, 0],
# #     [0, 0, 1],
# #     [-1, 0, 0]
# # ]
# R_hand_ee = [
#     [0, 0, -1],
#     [-1, 0, 0],
#     [0, 1, 0]
# ]
# R_world_cam = extr

# # move along global x axis
# R_cam_hands = [
# [[ 0.44026749, -0.31444456,  0.84100485],
#  [-0.53230671, -0.84571866, -0.03754353],
#  [ 0.72305885, -0.43114333, -0.53972338]],


# [[ 0.32567143, -0.69160702,  0.64468431],
#  [-0.67122209, -0.64933754, -0.35752156],
#  [ 0.66588214, -0.31629179, -0.67569259]],


# [[ 0.26233554, -0.87323824,  0.41065197],
#  [-0.70265923, -0.46454112, -0.53895413],
#  [ 0.66140008, -0.14716158, -0.73545456]],


# [[ 0.15345003, -0.97401559,  0.16657345],
#  [-0.79262049, -0.22198761, -0.56786817],
#  [ 0.59008969, -0.04489014, -0.80608872]],


# [[ 0.00138759, -0.99711187, -0.0759342 ],
#  [-0.74208591,  0.04987231, -0.6684469 ],
#  [ 0.67030335,  0.05727723, -0.73987346]]

# ]

# # Some more random hand poses
# R_cam_hands = [
#     [[ 0.89225018,  0.10529978, -0.43909177],
#     [ 0.31388471, -0.84369379,  0.43549648],
#     [-0.32460131, -0.526396,   -0.78583792]],


#     [[ 0.23736915, -0.53060471,  0.8137042 ],
#     [-0.94136813, -0.33238103,  0.05786969],
#     [ 0.23975391, -0.77973168, -0.57839136]],


#     [[ 0.33213141, -0.57428495,  0.74825499],
#     [-0.94312984, -0.21393436,  0.254437  ],
#     [ 0.01395811, -0.79020813, -0.61267959]]
# ]

# # Z-axis movement
# R_cam_hands = [
#     [[ 0.44456521, -0.20509596,  0.87195036],
#     [-0.4552,     -0.89009931,  0.02271954],
#     [ 0.77146272, -0.40701212, -0.48906686]],

#     [[ 0.90112709, -0.17763888,  0.39549259],
#     [-0.30110559, -0.91274558,  0.2760995 ],
#     [ 0.31193811, -0.36788577, -0.87598783]],


#     [[ 0.99625304, -0.08522186,  0.01473502],
#     [-0.08050937, -0.85160699,  0.51796116],
#     [-0.03159317, -0.51720669, -0.85527721]],


#     [[ 0.97685647,  0.20285977, -0.06781847],
#     [ 0.21370995, -0.91243397,  0.34898755],
#     [ 0.00891566, -0.35540422, -0.93467018]]
# ]

# # z axis movement 2
# R_cam_hands = [
#      [[ 0.61735501,  0.25018046, -0.7458435 ],
#     [ 0.6851102,  -0.63695547,  0.35342856],
#     [-0.38664818, -0.72917588, -0.56462883]],


#     [[ 0.94117948,  0.14881601, -0.30337269],
#     [ 0.25644221, -0.89920903,  0.35448627],
#     [-0.22004223, -0.41143276, -0.88447979]],


#     [[ 0.8374966,  -0.08337134,  0.54004505],
#     [-0.25908739, -0.93072828,  0.25810577],
#     [ 0.48111658, -0.35608157, -0.80108224]]
# ]

# print("-------------------------")
# for R_cam_hand in R_cam_hands:

#     R_temp = np.dot(R_cam_hand, R_hand_ee)
#     R_world_ee = np.dot(R_world_cam, R_temp)

#     temp = R.from_matrix(R_world_ee).as_euler('XYZ', degrees=True)
#     print(temp)

#     final_quat = R.from_matrix(R_world_ee).as_quat()
#     final_pos = np.array([0.5,  0.3,  1])
#     pose_command = env.tiago.create_pose_command(final_pos, final_quat)
#     rospy.sleep(2)
#     print("Left Hand Going To Pose")
#     env.tiago.tiago_pose_writer['left'].write(pose_command)

#     rospy.sleep(10)
#     # input("Press Enter")
#     # wait_left(final_pos, env)

# # R_cam_hand = [
# #     [ 0.40161787, -0.11069187,  0.90909317],
# #     [-0.45887813, -0.88338856,  0.09516046],
# #     [ 0.79254902, -0.45538112, -0.40557871],
# # ]

# # R_cam_hand = [
# #     [ 0.44026749, -0.31444456,  0.84100485],
# #     [-0.53230671, -0.84571866, -0.03754353],
# #     [ 0.72305885, -0.43114333, -0.53972338]
# # ]

# # R_temp = np.dot(R_cam_hand, R_hand_ee)
# # R_world_ee = np.dot(R_world_cam, R_temp)

# # temp = R.from_matrix(R_world_ee).as_euler('XYZ', degrees=True)
# # print(temp)

# # final_quat = R.from_matrix(R_world_ee).as_quat()
# # final_pos = np.array([0.5,  0.3,  1])
# # pose_command = env.tiago.create_pose_command(final_pos, final_quat)
# # rospy.sleep(2)
# # print("Left Hand Going To Pose")
# # env.tiago.tiago_pose_writer['left'].write(pose_command)

# # wait_left(pos, env)


# # -------------------------------------------------------


# # ----------------------------- Testing orientations -------------------
# right_eef_pose, left_eef_pose = None, None
# while right_eef_pose is None or left_eef_pose is None:
#     right_eef_pose = env._observation()['right_eef']
#     left_eef_pose = env._observation()['left_eef']
#     right_gripper_state = env._observation()['right_gripper']
#     left_gripper_state = env._observation()['left_gripper']
#     # print("right_gripper, left_gripper: ", right_gripper_state, left_gripper_state)

# right_eef_pos, right_eef_orn_quat = right_eef_pose[:3], right_eef_pose[3:]
# print("right pose: ", right_eef_pose)
# right_eef_orn_euler = R.from_quat(right_eef_orn_quat).as_euler('XYZ', degrees=True)
# print("right_eef_orn_euler: ", right_eef_orn_euler)

# left_eef_pos, left_eef_orn_quat = left_eef_pose[:3], left_eef_pose[3:]
# print("left pose: ", left_eef_pose)
# left_eef_orn_euler = R.from_quat(left_eef_orn_quat).as_euler('XYZ', degrees=True)
# print("left_eef_orn_euler: ", left_eef_orn_euler)
# left_eef_orn_euler_extr = R.from_quat(left_eef_orn_quat).as_euler('xyz', degrees=True)
# print("left_eef_orn_euler: ", left_eef_orn_euler_extr)


# # # [ 0.42796682  0.46058356  0.99149777 -0.71347113 -0.05168758  0.69694477 -0.05055018]
# # pos = np.array([0.42796682,  0.46058356,  0.99149777])
# # quat = np.array([ -0.71347113, -0.05168758,  0.69694477, -0.05055018])
# # pose_command = env.tiago.create_pose_command(pos, quat)
# # rospy.sleep(2)
# # print("Left Hand Going To Pose")
# # env.tiago.tiago_pose_writer['left'].write(pose_command)

# # wait_left(pos, env)

# # rospy.sleep(2)
# # left_eef_pos, left_eef_orn_quat = left_eef_pose[:3], left_eef_pose[3:]
# # print("left pose: ", left_eef_pose)
# # left_eef_orn_euler = R.from_quat(left_eef_orn_quat).as_euler('XYZ', degrees=True)
# # print("left_eef_orn_euler: ", left_eef_orn_euler)

# rospy.sleep(2)
# home_left_hand(env)

# # covert hand to robot ee
# hand_orns = np.array([
#     [-126, 0, -73],
#     [-139, 0, -75],
#     [-150, 0, -80],
#     [-166, 0, -84],
#     [-180, 0, -90]
# ])
# ee_orns = hand_orns + np.array([180, 90, 180])


# temp_quat = R.from_euler('xyz', [90, 90, 90], degrees=True).as_quat()
# temp_pos = np.array([0.54107395,  0.26952911,  0.98696835 ])
# # print("left quat: ", left_eef_orn_quat)
# print("temp_quat: ", temp_quat)

# pose_command = env.tiago.create_pose_command(temp_pos, temp_quat)
# rospy.sleep(2)
# print("Left Hand Going To Pose")
# env.tiago.tiago_pose_writer['left'].write(pose_command)

# wait_left(temp_pos, env)

# # -----------------------------------------------------------------
        

# ----------- Checking difference between two topics ---------------
# gripper_reader = Listener(input_topic_name=f'/parallel_gripper_right_controller/state', input_message_type=JointTrajectoryControllerState, post_process_func=process_gripper)
# msg1 = gripper_reader.get_most_recent_msg()
# gripper_val_left = (msg1 - self.gripper_min)/(self.gripper_max - self.gripper_min)
# print("msg1: ", msg1)

# gripper_max = 0.6939130434782608
# gripper_min = 0.009130434782608695
# gripper_reader = Listener(input_topic_name=f'/gripper_right_controller/state', input_message_type=JointTrajectoryControllerState, post_process_func=process_gripper)
# msg = gripper_reader.get_most_recent_msg()
# gripper_val_right = 1- (msg - gripper_min)/(gripper_max - gripper_min)
# print("gripper_value_right: ", gripper_val_right)
# -------------------------------------------------------------------


# # ----------------------------- Testing Hand Grasp Pose -> 6-DoF ee Grasp Pose -------------------------------
# rospy.sleep(2)
# home_left_hand(env)
# print("-----------GOING HOME COMPLETE-----------")
# rospy.sleep(2)
# extr_position, extr_quat = env.tiago.get_camera_extrinsic
# extr_rotation = R.from_quat(extr_quat).as_matrix()
# print("extr: ", extr_rotation)
# print("pos: ", extr_position)

# R_hand_ee = [
#     [0, 0, -1],
#     [-1, 0, 0],
#     [0, 1, 0]
# ]
# R_world_cam = extr_rotation

# T_hand_ee = [
#     [0, 0, -1, 0],
#     [-1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, 0, 1]
# ]
# T_world_cam = [
#     [R_world_cam[0][0], R_world_cam[0][1], R_world_cam[0][2], extr_position[0]],
#     [R_world_cam[1][0], R_world_cam[1][1], R_world_cam[1][2], extr_position[1]],
#     [R_world_cam[2][0], R_world_cam[2][1], R_world_cam[2][2], extr_position[2]],
#     [0, 0, 0, 1]
# ]

# with open('/home/pal/arpit/tiago_teleop/tiago_control/data/tiago_grasp_test_2/hand_poses.pickle', 'rb') as handle:
#     hand_poses = pickle.load(handle)

# print("hand_poses keys:", hand_poses.keys())
# k = next(iter(hand_poses))
# k = './sample_data/tiago_grasp_test_2/color_img/0005.jpg'
# print(hand_poses[k].keys())
# fig, ax = plt.subplots(1,2)
# ax[0].imshow(hand_poses[k]['color_img'])
# ax[1].imshow(hand_poses[k]['depth_img'])
# plt.show()

# left_hand_pos_wrt_cam, left_hand_orn_wrt_cam = np.array(hand_poses[k]['left_hand_pos'])/1000, np.array(hand_poses[k]['left_hand_orn'])
# print("left_hand_pos, left_hand_orn: ", left_hand_pos_wrt_cam, left_hand_orn_wrt_cam)
# left_hand_orn_wrt_cam_matrix = R.from_rotvec(left_hand_orn_wrt_cam).as_matrix()
# # Trying a fix for wrist and ee discrepancy
# temp = np.dot(left_hand_orn_wrt_cam_matrix, np.array([0.1, 0, 0]))
# left_hand_pos_wrt_cam += temp
# print("New left_hand_pos: ", left_hand_pos_wrt_cam)

# # # Trying to go to a specific positon (of a bottle)
# # pix_x, pix_y = 270, 366
# # depth_val = hand_poses[k]['depth_img'][pix_x, pix_y]
# # left_hand_new_pos = get_XYZ(depth_val, pix_x, pix_y)
# # left_hand_new_pos /= 1000
# # print("left_hand_new_pos: ", left_hand_new_pos)


# T_cam_hand = [
#     [left_hand_orn_wrt_cam_matrix[0][0], left_hand_orn_wrt_cam_matrix[0][1], left_hand_orn_wrt_cam_matrix[0][2], left_hand_pos_wrt_cam[0]],
#     [left_hand_orn_wrt_cam_matrix[1][0], left_hand_orn_wrt_cam_matrix[1][1], left_hand_orn_wrt_cam_matrix[1][2], left_hand_pos_wrt_cam[1]],
#     [left_hand_orn_wrt_cam_matrix[2][0], left_hand_orn_wrt_cam_matrix[2][1], left_hand_orn_wrt_cam_matrix[2][2], left_hand_pos_wrt_cam[2]],
#     [0, 0, 0, 1]
# ]

# temp_matrix = np.dot(T_cam_hand, T_hand_ee)
# T_world_ee = np.dot(T_world_cam, temp_matrix)
# P_world_ee = T_world_ee[:3, 3]
# # print("T_world_ee pos: ", P_world_ee)
# R_world_ee = T_world_ee[:3, :3]
# R_world_ee_euler = R.from_matrix(R_world_ee).as_euler('XYZ', degrees=True)
# # print("R_world_ee_euler: ", R_world_ee_euler)

# final_quat = R.from_matrix(R_world_ee).as_quat()
# final_pos = np.array(P_world_ee) 
# print("------final_pos:", final_pos)
# final_pos += np.array([0, 0, 0.05])

# # # Move the arm
# # pose_command = env.tiago.create_pose_command(final_pos, final_quat)
# # rospy.sleep(2)
# # print("Left Hand Going To Pose")
# # env.tiago.tiago_pose_writer['left'].write(pose_command)

# # wait_left(final_pos, env)

# # left_eef_pose = env._observation()['left_eef']
# # print("Actual left_eef_pos: ", left_eef_pose[:3])
# # print("error: ", abs(final_pos - left_eef_pose[:3]))

# # ------------------------------------------------------------------------------------------------------------


# # --------------- Testing depth values ---------------
# with open('/home/pal/arpit/bimanual_skill_learning/data/tiago_tissue_roll_2/depth/0010.pickle', 'rb') as handle:
#     depth_img = pickle.load(handle)

# plt.imshow(depth_img)
# plt.show()

# rospy.sleep(2)

# # home_left_hand(env)

# pix_x, pix_y = 270, 366
# # pix_x, pix_y = 185, 115

# depth_val = depth_img[pix_x, pix_y]
# pos = get_XYZ(depth_val, pix_y, pix_x)
# print("pos: ", pos)
# pos /= 1000
# pos_homo = np.append(pos, 1.0)

# # rgb
# extr_position, extr_quat = env.tiago.get_camera_extrinsic
# extr_rotation = R.from_quat(extr_quat).as_matrix()

# R_world_cam = extr_rotation
# T_world_cam = np.array([
#     [R_world_cam[0][0], R_world_cam[0][1], R_world_cam[0][2], extr_position[0]],
#     [R_world_cam[1][0], R_world_cam[1][1], R_world_cam[1][2], extr_position[1]],
#     [R_world_cam[2][0], R_world_cam[2][1], R_world_cam[2][2], extr_position[2]],
#     [0, 0, 0, 1]
# ])
# print("T_world_cam: ", T_world_cam)
# temp_matrix = np.linalg.inv(T_world_cam)

# final_pos = np.dot(T_world_cam, pos_homo)[:3]
# # final_pos += np.array([0, 0, 0.1])
# print("final_pos: ", final_pos)

# # # Depth
# # extr_position, extr_quat = env.tiago.get_depth_camera_extrinsic
# # extr_rotation = R.from_quat(extr_quat).as_matrix()

# # R_world_cam = extr_rotation
# # T_world_cam = np.array([
# #     [R_world_cam[0][0], R_world_cam[0][1], R_world_cam[0][2], extr_position[0]],
# #     [R_world_cam[1][0], R_world_cam[1][1], R_world_cam[1][2], extr_position[1]],
# #     [R_world_cam[2][0], R_world_cam[2][1], R_world_cam[2][2], extr_position[2]],
# #     [0, 0, 0, 1]
# # ])

# # final_pos = np.dot(T_world_cam, pos_homo)[:3]
# # final_pos += np.array([0, 0, 0.1])
# # print("final_pos: ", final_pos)

# # Fixed orientation: 0.60716551 -0.19219885  0.13620635 -0.75885274
# final_quat = np.array([0.60716551, -0.19219885,  0.13620635, -0.75885274])

# ee_orn = R.from_quat(final_quat).as_matrix()
# dist_vec = np.dot(ee_orn, np.array([-0.225, 0, 0]))
# final_pos += dist_vec
# print("---final_pos: ", final_pos)

# # # Move the arm
# # pose_command = env.tiago.create_pose_command(final_pos, final_quat)
# # rospy.sleep(2)
# # print("Left Hand Going To Pose")
# # env.tiago.tiago_pose_writer['left'].write(pose_command)

# # wait_left(final_pos, env)

# # left_eef_pose = env._observation()['left_eef']
# # print("Actual left_eef_pos: ", left_eef_pose[:3])
# # print("error: ", abs(final_pos - left_eef_pose[:3]))
# # ----------------------------------------------------

# from tiago_gym import ForceTorqueSensor
# ft = ForceTorqueSensor()
# rospy.sleep(1)
# val = ft.ft_reader.get_most_recent_msg()
# f = val.wrench.force
# t = val.wrench.torque
# print("f, torque, sum(force), sum(torque): ", [f.x, f.y, f.z], [t.x, t.y, t.z], abs(f.x) + abs(f.y) + abs(f.z), abs(t.x) + abs(t.y) + abs(t.z))

# # ------------------------------ Random testing -------------------------
# with open('/home/pal/arpit/bimanual_skill_learning/output/bottle2/seed_5/1_0.pickle', 'rb') as handle:
#     traj = pickle.load(handle)
# print(traj.keys())
# forces_sum = []
# torques_sum = []
# forces = traj['forces']
# torques = traj['torques']
# forces = np.array(traj['forces'])
# torques = np.array(traj['torques'])
# print("len(forces), len(torques): ", len(forces), len(torques))
# print("left_grasp_lost, right_grasp_lost: ", traj['left_grasp_lost'], traj['right_grasp_lost'])

# # target_s_hat = np.array([0.0, 0.0, 1.0])
# # target_q = np.array([0.55027766, -0.18231454,  1.1236653])
# # s_hat = traj['s_hat']
# # q = traj['q']
# # s_hat_dist = np.dot(s_hat, target_s_hat)
# # q_dist = np.linalg.norm(q - target_q)
# # print("target_s_hat, s_hat, s_hat_dist: ", target_s_hat, s_hat,  s_hat_dist)

# for i in range(len(forces)):
#     # print(forces[i])
#     forces_sum.append(abs(forces[i,0]) + abs(forces[i,1]) + abs(forces[i,2]))
#     torques_sum.append(abs(torques[i,0]) + abs(torques[i,1]) + abs(torques[i,2]))
# fig, ax = plt.subplots(1,2)
# ax[0].set_ylim([0, 70])
# ax[1].set_ylim([0, 15])
# ax[0].plot(np.arange(len(forces_sum)), forces_sum)  # Plot the chart 
# ax[1].plot(np.arange(len(torques_sum)), torques_sum)  # Plot the chart 
# plt.show()

# --------------------------- testing camera -----------------------------
# top_down_cam = Camera(img_topic="/top_down/color/image_raw", depth_topic="/top_down/aligned_depth_to_color/image_raw")
# obs = top_down_cam.get_camera_obs()
# print("obs: ", obs['image'].shape, obs['depth'].shape)
# rgb = obs['image'].astype(np.uint8)
# rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
# plt.imshow(rgb)
# plt.show()

# side_cam = Camera(img_topic="/side_1/color/image_raw", depth_topic="/side_1/aligned_depth_to_color/image_raw")
# obs = side_cam.get_camera_obs()
# print("obs: ", obs['color'].shape, obs['depth'].shape)
# rgb = obs['color'].astype(np.uint8)
# rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
# plt.imshow(rgb)
# plt.show()

# ego_cam = Camera(img_topic="/xtion/rgb/image_raw", depth_topic="/xtion/depth/image_raw")
# obs = ego_cam.get_camera_obs()
# print("obs: ", obs['color'].shape, obs['depth'].shape)
# rgb = obs['color'].astype(np.uint8)
# rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
# plt.imshow(rgb)
# plt.show()
# ------------------------------------------------------------------------
