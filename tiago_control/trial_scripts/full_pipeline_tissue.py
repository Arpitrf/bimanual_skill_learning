import sys
import os
parent_dir = os.path.dirname('/home/pal/arpit/bimanual_skill_learning/tiago_control')
sys.path.append(parent_dir)
import time
import rospy
import numpy as np
from tiago_control.tiago_gym import TiagoGym, Listener, ForceTorqueSensor
from control_msgs.msg  import JointTrajectoryControllerState
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm, expm
import pickle
import matplotlib.pyplot as plt
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Vector3
from argparse import ArgumentParser
from utils.camera_utils import Camera, RecordVideo

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

def move_up(env, hand='left_eef'):
    rospy.sleep(2)
    eef_pose = env._observation()[hand]
    eef_pos, eef_quat = eef_pose[:3], eef_pose[3:]
    # move up by 5 cm
    eef_pos[2] += 0.05
    pose_command = env.tiago.create_pose_command(eef_pos, eef_quat)
    # print("---", hand.split('_')[0])
    env.tiago.tiago_pose_writer[hand.split('_')[0]].write(pose_command)
    rospy.sleep(5)

def config_parser(parser=None):
    if parser is None:
        parser = ArgumentParser("Bimanual Skill Learning")
    parser.add_argument('--f_name', type=str)
    parser.add_argument("--record", action="store_true", default=False)
    return parser

def publish_sphere_marker(marker_pub, pos):
    marker = Marker()
    marker.id = 0
    marker.header.frame_id = "base_footprint"  # Set your desired frame_id
    marker.header.stamp = rospy.Time.now()
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.scale.x = 0.05  # Line width
    marker.scale.y = 0.05  # Line width
    marker.scale.z = 0.05  # Line width
    marker.pose.position.x = pos[0] #0.535 
    marker.pose.position.y = pos[1] #-0.165
    marker.pose.position.z = pos[2] #0.736

    marker.color.a = 1.0  # Alpha
    marker.color.r = 0.0  # Red
    marker.color.g = 0.0  # Green
    marker.color.b = 1.0  # Blue

    marker_pub.publish(marker)

def publish_line_marker(marker_pub, axis, q, counter):
    # rospy.init_node('line_marker_publisher', anonymous=True)
    # marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz

    # rospy.sleep(1)

    # delete_all_marker = Marker()
    # delete_all_marker.action = Marker.DELETEALL
    # marker_pub.publish(delete_all_marker)

    rospy.sleep(1)

    # while not rospy.is_shutdown():
    marker = Marker()
    marker.id = counter + 1
    marker.header.frame_id = "base_footprint"  # Set your desired frame_id
    marker.header.stamp = rospy.Time.now()
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.02  # Line width

    # Start and end points of the line
    length = 2
    # axis, q = np.array([-0.101547,  -0.0114104,  0.9947653]), np.array([0.66742979, -0.14872666,  0.10002639])
    # axis = np.array([0,0,1])
    point_1 = q + length * axis
    point_2 = q - length * axis

    start_point = Point()
    start_point.x, start_point.y, start_point.z = point_1[0], point_1[1], point_1[2]
    end_point = Point()
    end_point.x, end_point.y, end_point.z = point_2[0], point_2[1], point_2[2]

    marker.points.append(start_point)
    marker.points.append(end_point)

    marker.color.a = 1.0  # Alpha
    marker.color.r = 0.0  # Red
    marker.color.g = 0.0  # Green
    marker.color.b = 1.0  # Blue

    marker_pub.publish(marker)


    # rospy.sleep(1)
    # Add a marker for q 
    marker = Marker()
    marker.id = counter + 2
    marker.header.frame_id = "base_footprint"  # Set your desired frame_id
    marker.header.stamp = rospy.Time.now()
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.scale.x = 0.05  # Line width
    marker.scale.y = 0.05  # Line width
    marker.scale.z = 0.05  # Line width
    marker.pose.position.x = q[0]
    marker.pose.position.y = q[1]
    marker.pose.position.z = q[2]

    marker.color.a = 1.0  # Alpha
    marker.color.r = 1.0  # Red
    marker.color.g = 1.0  # Green
    marker.color.b = 1.0  # Blue

    marker_pub.publish(marker)


    # rospy.sleep(1)
    # Add a marker for the object location
    marker = Marker()
    marker.id = 0
    marker.header.frame_id = "base_footprint"  # Set your desired frame_id
    marker.header.stamp = rospy.Time.now()
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.scale.x = 0.05  # Line width
    marker.scale.y = 0.05  # Line width
    marker.scale.z = 0.05  # Line width
    marker.pose.position.x = 0.55027766 #0.535 
    marker.pose.position.y = -0.18231454 #-0.165
    marker.pose.position.z = 1.123665 #0.736

    marker.color.a = 1.0  # Alpha
    marker.color.r = 0.0  # Red
    marker.color.g = 0.0  # Green
    marker.color.b = 1.0  # Blue

    marker_pub.publish(marker)

def get_points_prismatic(s_hat, initial_pose, dist_moved=0.2):
    twist = [0, 0, 0] + s_hat.tolist()
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
    T0 = np.array([[1, 0, 0, initial_pose[0,3]],
        [0, 1, 0, initial_pose[1,3]],
        [0, 0, 1, initial_pose[2,3]],
        [0, 0, 0, 1]])
    # T0 = start_T
    # ax.scatter(T0[0][3], T0[1][3], T0[2][3], color='r', marker='o')
    
    final_points = []
    for theta in np.arange(0.015, 0.29, 0.015):
        S_theta = theta * np.array(S)

        T1 = np.dot(T0, expm(S_theta))
        T1 = np.array([
            [initial_pose[0][0], initial_pose[0][1], initial_pose[0][2], T1[0,3]],
            [initial_pose[1][0], initial_pose[1][1], initial_pose[1][2], T1[1,3]],
            [initial_pose[2][0], initial_pose[2][1], initial_pose[2][2], T1[2,3]],
            [0, 0, 0, 1]
        ])
        final_points.append(T1)
        print("T1 POS: ", T1[0:3,3])

        # T1 = np.dot(expm(S_theta), T0)
        # ax.scatter(T1[0][3], T1[1][3], T1[2][3], color='g', marker='o')
        
    return final_points

def wait_right(desired_pos, env, wait_time=30):
    start_time = time.time()
    while True:
        right_eef_pose = env._observation()['right_eef']
        # print("right_eef_pos: ", right_eef_pose[:3])
        error = abs(desired_pos - right_eef_pose[:3])
        current_time = time.time()
        time_diff = current_time - start_time
        if (error < 0.01).all() or time_diff > wait_time:
            break

def wait_left(desired_pos, env, wait_time=30):
    start_time = time.time()
    while True:
        left_eef_pose = env._observation()['left_eef']
        error = abs(desired_pos - left_eef_pose[:3])
        current_time = time.time()
        time_diff = current_time - start_time
        if (error < 0.01).all() or time_diff > wait_time:
            break

def home_right_hand(env):  
#     [ 0.55727367 -0.45815103  1.16181563  0.75473021 -0.11066493 -0.51583154
#  -0.38994027]  
    pos = np.array([0.55727367, -0.45815103,  1.16181563 ])
    quat = np.array([0.75473021, -0.11066493, -0.51583154, -0.38994027])
    pose_command = env.tiago.create_pose_command(pos, quat)
    env.tiago.tiago_pose_writer['right'].write(pose_command)
    wait_right(pos, env, wait_time=15)
    right_eef_pose = env._observation()['right_eef']
    # print("Actual right_eef_pos: ", right_eef_pose[:3])
    # print("error: ", abs(pos - right_eef_pose[:3]))

def home_left_hand(env):
    pos = np.array([0.44245209,  0.59514704,  1.16818457])
    quat = np.array([0.20890626, -0.54033101, -0.29681984, -0.7591433 ])
    # for tissue 2:
    # quat = np.array([-0.76373052, -0.28579711,  0.54594994, -0.1922872 ])
    
    pose_command = env.tiago.create_pose_command(pos, quat)
    env.tiago.tiago_pose_writer['left'].write(pose_command)
    wait_left(pos, env, wait_time=15)
    left_eef_pose = env._observation()['left_eef']
    # print("Actual left_eef_pos: ", left_eef_pose[:3])
    # print("error: ", abs(pos - left_eef_pose[:3]))

def left_gripper_grasp():        
        # Pregrasp pose
        pose_command = env.tiago.create_pose_command(left_eef_pos_pre, left_eef_quat_pre)
        print("Left Hand Going To Pose")
        rospy.sleep(1)
        env.tiago.tiago_pose_writer['left'].write(pose_command)
        wait_left(left_eef_pos_pre, env, wait_time=10)
        rospy.sleep(1)
        left_eef_pose = env._observation()['left_eef']
        print("LEFT HAND: Desired and Actual left_eef_pos: ", left_eef_pos_pre, left_eef_pose[:3])
        print("LEFT HAND: error in position: ", abs(left_eef_pos_pre - left_eef_pose[:3]))

        # Grasp pose
        pose_command = env.tiago.create_pose_command(left_eef_pos, left_eef_quat)
        print("Left Hand Going To Pose")
        rospy.sleep(1)
        env.tiago.tiago_pose_writer['left'].write(pose_command)
        wait_left(left_eef_pos, env)
        rospy.sleep(1)
        left_eef_pose = env._observation()['left_eef']
        print("LEFT HAND: Desired and Actual left_eef_pos: ", left_eef_pos, left_eef_pose[:3])
        print("LEFT HAND: error in position: ", abs(left_eef_pos - left_eef_pose[:3]))

        # Close gripper
        rospy.sleep(1)
        env.tiago.gripper['left'].write(1)


def right_gripper_grasp():
    # Pregrasp pose
    rospy.sleep(1)
    pose_command = env.tiago.create_pose_command(right_eef_pos_pre, right_eef_quat_pre)
    print("Right Hand Going To Pose")
    env.tiago.tiago_pose_writer['right'].write(pose_command)
    wait_right(right_eef_pos_pre, env, wait_time=10)
    rospy.sleep(1)
    right_eef_pose = env._observation()['right_eef']
    print("RIGHT HAND: Desired and Actual right_eef_pos_pre: ", right_eef_pos_pre, right_eef_pose[:3])
    print("RIGHT HAND: error in position: ", abs(right_eef_pos_pre - right_eef_pose[:3]))

    # Grasp pose
    rospy.sleep(1)
    pose_command = env.tiago.create_pose_command(right_eef_pos, right_eef_quat)
    print("Right Hand Going To Pose")
    env.tiago.tiago_pose_writer['right'].write(pose_command)
    wait_right(right_eef_pos, env)
    rospy.sleep(1)
    right_eef_pose = env._observation()['right_eef']
    print("RIGHT HAND: Desired and Actual right_eef_pos: ", right_eef_pos, right_eef_pose[:3])
    print("RIGHT HAND: error in position: ", abs(right_eef_pos - right_eef_pose[:3]))

    # Right hand grasp
    rospy.sleep(1)
    # env.tiago.gripper['right'].write(0.7)
    env.tiago.gripper['right'].write(1.0)

    rospy.sleep(2)
    right_is_grasp = env.tiago.gripper['right'].is_grasped()
    print("right_is_grasp: ", right_is_grasp)

def get_elite_trajs(trajs):
    full_list, traj_len_list, avg_f_list, avg_t_list = [], [], [], []
    MIN_FORCE_THRESHOLD = 4
    for traj in trajs:
        traj_forces = np.array(traj['forces'])
        traj_torques = np.array(traj['torques'])
        traj_forces_sum, traj_torques_sum = [], []
        # Get the FT_sum for all waypints in a trajectory that have a FT sensor reading greater than a minimum threshold
        for i in range(len(traj_forces)):
            forces_sum = (abs(traj_forces[i,0]) + abs(traj_forces[i,1]) + abs(traj_forces[i,2]))
            torques_sum = (abs(traj_torques[i,0]) + abs(traj_torques[i,1]) + abs(traj_torques[i,2]))
            if forces_sum > MIN_FORCE_THRESHOLD:
                traj_forces_sum.append(forces_sum)
                traj_torques_sum.append(torques_sum)

        # Exception handling
        if len(traj_forces_sum) == 0:
            traj_forces_sum = [0.0]
            traj_torques_sum = [0.0]
        
        # Fill up the lists
        avg_f_list.append([traj['traj_number'], np.mean(traj_forces_sum)])
        avg_t_list.append([traj['traj_number'], np.mean(traj_torques_sum)])
        traj_len_list.append([traj['traj_number'], len(traj['forces'])])
        full_list.append([traj['traj_number'], len(traj['forces']), np.mean(traj_forces_sum), np.mean(traj_torques_sum)])
        
    # Sort according to the length of trajectories
    full_list_sorted_by_len = sorted(full_list, key = lambda x: x[1], reverse=True)

    # Keep only the trajs that have max length
    max_traj_len = full_list_sorted_by_len[0][1]
    max_traj_len_list = []
    for elem in full_list_sorted_by_len:
        if elem[1] == max_traj_len:
            max_traj_len_list.append(elem)
    # Now sort according to the FT sensor readings
    full_list_sorted_by_ft = sorted(max_traj_len_list, key = lambda x: x[2])

    # return the trajectory number of the best trajectory
    return full_list_sorted_by_ft
    # return full_list_sorted_by_ft[0][0]

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

# Initializations
seed = 3
args = config_parser().parse_args()
np.random.seed(seed)
rospy.init_node('tiago_test')
env = TiagoGym(frequency=10, right_arm_enabled=True, left_arm_enabled=True, right_gripper_type='robotiq', left_gripper_type='robotiq')
ft = ForceTorqueSensor()

side_cam, top_down_cam, ego_cam = None, None, None
if args.record:
    side_cam = Camera(img_topic="/side_1/color/image_raw", depth_topic="/side_1/aligned_depth_to_color/image_raw")
    # top_down_cam = Camera(img_topic="/top_down/color/image_raw", depth_topic="/top_down/aligned_depth_to_color/image_raw")
    ego_cam = Camera(img_topic="/xtion/rgb/image_raw", depth_topic="/xtion/depth/image_raw")

# Obtain axis from perception
# with open('/home/pal/arpit/tiago_teleop/tiago_control/data/tiago_full_pipeline_3/extrinsic.pickle', 'rb') as handle:
#     extr = pickle.load(handle)
with open('/home/pal/arpit/bimanual_skill_learning/data/tiago_tissue_roll_3/axis.pickle', 'rb') as handle:
    axis = pickle.load(handle)
q_perception = np.array(axis['q'])
s_hat_perception = np.array(axis['s_hat'])

# ------------------------- Hardcoding grasp poses --------------------------
# PUT TISSUE ROLL INSIDE THE BOX
left_eef_pos = np.array([ 0.67880403,  0.1254442,  0.96966923])
left_eef_quat = np.array([ -0.15589715, -0.683488,    0.0989711, -0.70621877])
left_eef_pos_pre = left_eef_pos + np.array([0.0, 0.0, 0.2])
left_eef_quat_pre = left_eef_quat

# Right hand: Set initial pose to a default value
right_eef_pos = np.array([ 0.47584052, -0.38718771,  1.01890969])
right_eef_quat = np.array([0.6701378,  -0.13172846, -0.7114122, -0.1656974])
right_eef_pos_pre = right_eef_pos + np.array([0.0, 0.0, 0.2])
right_eef_quat_pre = right_eef_quat

# Initial pose
right_eef_mat = R.from_quat(right_eef_quat).as_matrix()
T0 = [
        [right_eef_mat[0][0], right_eef_mat[0][1], right_eef_mat[0][2], right_eef_pos[0]],
        [right_eef_mat[1][0], right_eef_mat[1][1], right_eef_mat[1][2], right_eef_pos[1]],
        [right_eef_mat[2][0], right_eef_mat[2][1], right_eef_mat[2][2], right_eef_pos[2]],
        [0, 0, 0, 1]
    ]
T0 = np.array(T0)
# ---------------------------------------------------------------------------

rospy.sleep(2)
marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
delete_all_marker = Marker()
delete_all_marker.action = Marker.DELETEALL
marker_pub.publish(delete_all_marker)

save_folder = f'output/tissue/seed_{seed}/'
os.makedirs(save_folder, exist_ok=True)

# Start recorder
if args.record:
    recorder = RecordVideo(top_down_cam, side_cam, ego_cam)
    recorder.setup_recording()

# Remove later. Adding noise to the initial traj to mimick perception model
# noise = np.random.uniform(low=-0.4, high=0.4, size=(2,))
# noise = np.append(noise, 0.0)
# print("noise: ", noise)
noise = np.array([-0.25, 0.25, 0.0])
org_s_hat_perception  = s_hat_perception.copy()
s_hat_perception = s_hat_perception + noise
s_hat_perception = s_hat_perception / np.linalg.norm(s_hat_perception)
print("original s_hat perception: ", org_s_hat_perception)
print("new s_hat perception: ", s_hat_perception)

# # Open gripper and go home
# rospy.sleep(1)
# env.tiago.gripper['left'].write(0)
# env.tiago.gripper['right'].write(0)
# rospy.sleep(1)
# # go to home
# home_left_hand(env)
# home_right_hand(env)


# -------------------------------------------- CEM START------------------------------------------------------------
counter = 0
interesting_trajectories = 0

mu_x, sigma_x = 0, 0.1
mu_y, sigma_y = 0, 0.1
mu_z, sigma_z = 0, 0.1
min_x, max_x = -0.5, 0.5
min_y, max_y = -0.5, 0.5
min_z, max_z = -0.4, 0.4

# remove later
s_hat_new = s_hat_perception

for epoch in range(2):
    trajs = []
    elite_trajs = []
    print(f"================EPOCH: {epoch}=================")

    for traj_number in range(5):

        print(f"---------------{traj_number}---------------")
        trajectory_dict = {}
        trajectory_dict['epoch'] = epoch        
        trajectory_dict['traj_number'] = traj_number        
        trajectory_dict['left_grasp_lost'] = -1
        trajectory_dict['right_grasp_lost'] = -1

        # ------------------- Obtain the screw axis and the trajectory -------------------------
        # Adding noise to s_hat. This noise can have a larger variance.
        if epoch == 0:
            noise_s_hat_x = np.random.uniform(low=min_x, high=max_x)
            noise_s_hat_y = np.random.uniform(low=min_y, high=max_y)
            # noise_s_hat_z = np.random.uniform(low=min_z, high=max_z)
            noise_s_hat_z = 0.0
            noise_s_hat = np.array([noise_s_hat_x, noise_s_hat_y, noise_s_hat_z])
            print("noise_s_hat: ", noise_s_hat)
        else:
            s_hat_perception = s_hat_new
            noise_s_hat_x = np.random.normal(mu_x, sigma_x)
            noise_s_hat_y = np.random.normal(mu_y, sigma_y)
            noise_s_hat_z = 0.0
            noise_s_hat = np.array([noise_s_hat_x, noise_s_hat_y, noise_s_hat_z])
        
        # # remove later
        # noise_s_hat = np.array([0.0, 0.0, 0.0])
        
        s = s_hat_perception + noise_s_hat
        s_hat = s / np.linalg.norm(s)
        # remove later
        # s_hat = np.array([ 0.40822399, -0.91239034, 0.02995069])
        # s_hat = np.array([0.37233864, 0.92806603, 0.00757534])
        # s_hat = s_hat_new
        # No need to add noise to q here as the axis will definitely pass through the first pose (grasp pose) which we already have.
        q = right_eef_pos

        # -----------------------------------------------
        print("----------- final s_hat: ", s_hat)
        trajectory_dict['s_hat'] = s_hat
        trajectory_dict['s_hat_noise'] = noise_s_hat
        publish_line_marker(marker_pub, s_hat, q, counter)
        counter += 2

        test_traj = get_points_prismatic(s_hat, T0)
        trajectory_dict['final_points'] = test_traj

        # # remove later
        # if epoch == 0:
        #     continue

        # -------------------------robot movement-------------------------
        # # Open gripper and go home
        # move_up(env)
        rospy.sleep(1)
        env.tiago.gripper['left'].write(0)
        env.tiago.gripper['right'].write(0)
        # rospy.sleep(1)
        # # go to home
        # home_left_hand(env)
        home_right_hand(env)

        #  Make left gripper go to pose and grasp
        left_gripper_grasp()

        # Make right gripper go to grasp pose and grasp
        right_gripper_grasp()

    #     rospy.sleep(1)
    #     right_eef_pose = env._observation()['right_eef']
    #     tooltip_ee_offset = np.array([0.26, 0, 0])
    #     print("quat: ", right_eef_pose[3:])
    #     right_eef_pose_mat = R.from_quat(right_eef_pose[3:]).as_matrix()
    #     tooltip_ee_offset_world = np.dot(right_eef_pose_mat, tooltip_ee_offset)
    #     temp_pos = right_eef_pos + tooltip_ee_offset_world[:3]
    #     print("temp_pos: ", temp_pos)
    #     publish_sphere_marker(marker_pub, temp_pos)
        
        # ----------------------- Robot exeuting the task trajectory -------------------- 
        if args.record:
            recorder.start_recording()
            print("Start recording")

        forces, torques = [], []
        forces_sum, torques_sum  = [], []
        interrupted = False

        # input("STARTING ROBOT MOVEMENT")
        print("STARTING ROBOT MOVEMENT")
        rospy.sleep(1)
        for i in range(len(test_traj)):
        # for i in range(8):
            pt = test_traj[i]
            new_pos = np.array(pt[0:3,3])
            new_rot_mat = np.array(pt[:3,:3])
            new_quat = R.from_matrix(new_rot_mat).as_quat()
            pose_command = env.tiago.create_pose_command(new_pos, new_quat)
            env.tiago.tiago_pose_writer['right'].write(pose_command)

            rospy.sleep(2)
            right_eef_pose = env._observation()['right_eef']
            print("desied, actual right_eef_pos: ", new_pos, right_eef_pose[:3])
            print("error: ", abs(new_pos - right_eef_pose[:3]))

            # FT Readings
            rospy.sleep(1)
            val = ft.ft_reader.get_most_recent_msg()
            f = val.wrench.force
            t = val.wrench.torque
            forces.append(np.array([f.x, f.y, f.z]))
            torques.append(np.array([t.x, t.y, t.z]))
            force_sum = abs(f.x) + abs(f.y) + abs(f.z)
            torque_sum = abs(t.x) + abs(t.y) + abs(t.z)
            forces_sum.append(force_sum)
            torques_sum.append(torque_sum)
            print("f, sum(force), sum(torque): ", [f.x, f.y, f.z], force_sum, torque_sum)

            if force_sum > 30:
                print("FORCE GETING TOO HIGH. STOPPING!")
                interrupted = True
                break

            left_is_grasp = env.tiago.gripper['left'].is_grasped()
            right_is_grasp = env.tiago.gripper['right'].is_grasped()
            print("right_is_grasp, left_is_grasp: ", right_is_grasp, left_is_grasp)
            
            if not left_is_grasp:
                interrupted = True
                print("LEFT GRASP LOST. STOPPING!")
                trajectory_dict['left_grasp_lost'] = i
                break
            
            if not right_is_grasp:
                interrupted = True
                print("RIGHT GRASP LOST. STOPPING!")
                trajectory_dict['right_grasp_lost'] = i
                break

        # open gripper again
        print("------------- ROBOT TRAJ COMPLETED -------------")
        rospy.sleep(1)
        env.tiago.gripper['left'].write(0)
        env.tiago.gripper['right'].write(0)
        rospy.sleep(1)

        if args.record:
            recorder.pause_recording()
            recorder.save_video(save_folder, args, epoch, traj_number)
            print("Saved video")
            recorder.reset_frames()

        # ------------------------------------------------------------------------------
        
        trajectory_dict['forces'] = forces
        trajectory_dict['torques'] = torques
        succ = input("Final Task Success. O for Failure and 1 for Success")
        trajectory_dict["task_succ"] = succ
        with open(f'{save_folder}{epoch}_{traj_number}.pickle', 'wb') as handle:
            pickle.dump(trajectory_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        trajs.append(trajectory_dict)

            
    print("len(trajs): ", len(trajs))
    if epoch == 0:
        # # remove later
        # s_hat_new = np.array([0.37233864, 0.92806603, 0.00757534])
        # continue
 
        # currently taking just the top trajectory in the first epoch. Can also try taking some "average" of top x trajectories
        full_list_sorted_by_ft = get_elite_trajs(trajs)
        elite_traj_number = full_list_sorted_by_ft[0][0]
        elite_traj = trajs[elite_traj_number]
        s_hat_new = elite_traj['s_hat']
        print("Elite traj: s_hat, m: ", s_hat_new)    

    if epoch > 0:
        ELITE_TRAJ_THRESHOLD = 3
        noise_s_hat_lis = []
        full_list_sorted_by_ft = get_elite_trajs(trajs)
        upper_bound = min(len(full_list_sorted_by_ft), ELITE_TRAJ_THRESHOLD)
        for traj in full_list_sorted_by_ft[:upper_bound]:
            traj_number_temp = traj[0]
            trajs[traj_number_temp]
            
            noise_s_hat_lis.append(trajs[traj_number_temp]['s_hat_noise'])
            # if successful traj, add the noise 2 more times (increase weight)
            if trajs[traj_number_temp]['task_succ'] == '1':
                for _ in range(2):
                    noise_s_hat_lis.append(trajs[traj_number_temp]['s_hat_noise'])
            print("Elite traj: s_hat, m: ", trajs[traj_number_temp]['s_hat'], trajs[traj_number_temp]['m'])

        for j in noise_s_hat_lis:
            print("noise_s_hat_lis: ", j)
        if len(noise_s_hat_lis) != 0:
            noise_s_hat_lis = np.array(noise_s_hat_lis)
            mu_x_s, sigma_x_s, mu_y_s, sigma_y_s, mu_z_s, sigma_z_s = updateGaussianParams(noise_s_hat_lis)

    if args.record:
        print("Stop recording")
        recorder.stop_recording()
