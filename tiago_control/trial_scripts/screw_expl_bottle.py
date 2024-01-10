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


def get_points_revolute(s_hat, q, initial_pose, angle_moved=4.0):
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
    # print("Actual right_eef_pos: ", right_eef_pose[:3])
    # print("error: ", abs(pos - right_eef_pose[:3]))

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
    # print("Actual left_eef_pos: ", left_eef_pose[:3])
    # print("error: ", abs(pos - left_eef_pose[:3]))

def move_up(env, hand='right_eef'):
    rospy.sleep(2)
    eef_pose = env._observation()[hand]
    eef_pos, eef_quat = eef_pose[:3], eef_pose[3:]
    # move up by 5 cm
    eef_pos[2] += 0.05
    pose_command = env.tiago.create_pose_command(eef_pos, eef_quat)
    # print("---", hand.split('_')[0])
    env.tiago.tiago_pose_writer[hand.split('_')[0]].write(pose_command)
    rospy.sleep(5)

def plot_axis(ax, q, axis, c='b'):
    length = 2
    endpoint_1 = q + length * axis
    endpoint_2 = q - length * axis
    ax.plot([endpoint_2[0], endpoint_1[0]], [endpoint_2[1], endpoint_1[1]], [endpoint_2[2], endpoint_1[2]], c=c)
    plt.show()
    # plt.pause(1)

def check_traj_quality(traj_points):
    x_range = [0.4, 0.8]
    y_range = [-0.28, -0.06]
    z_range = [1.05, 1.18]
    quality = 'good'
    for pose in traj_points:
        point = pose[0:3,3]
        if point[0] > x_range[0] and point[0] < x_range[1] and \
                point[1] > y_range[0] and point[1] < y_range[1] and \
                point[2] > z_range[0] and point[2] < z_range[1]:
            quality = 'good'
        else:
            quality = 'bad'
            break
    if quality == 'bad':
        return False
    else:
        return True


# Initializations
seed = 0
args = config_parser().parse_args()
np.random.seed(seed)
rospy.init_node('tiago_test')
env = TiagoGym(frequency=10, right_arm_enabled=True, left_arm_enabled=True, right_gripper_type='robotiq', left_gripper_type='robotiq')
ft = ForceTorqueSensor()

side_cam, top_down_cam, ego_cam = None, None, None
if args.record:
    side_cam = Camera(img_topic="/side_1/color/image_raw", depth_topic="/side_1/aligned_depth_to_color/image_raw")
    top_down_cam = Camera(img_topic="/top_down/color/image_raw", depth_topic="/top_down/aligned_depth_to_color/image_raw")
    ego_cam = Camera(img_topic="/xtion/rgb/image_raw", depth_topic="/xtion/depth/image_raw")


# Obtain axis from perception
# with open('/home/pal/arpit/tiago_teleop/tiago_control/data/tiago_full_pipeline_3/extrinsic.pickle', 'rb') as handle:
#     extr = pickle.load(handle)
with open('/home/pal/arpit/tiago_teleop/tiago_control/data/tiago_full_pipeline_3/axis.pickle', 'rb') as handle:
    axis = pickle.load(handle)
q_perception = np.array(axis['q'])
s_hat_perception = np.array(axis['s_hat'])
print("s_hat, q: ", s_hat_perception, q_perception)

# # Create a 3D axis
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')  
# ax.set_xlim([0.3, 0.9])
# ax.set_ylim([-0.2, 0.2])
# ax.set_zlim([0.5, 1])
# plot_axis(ax, q_perception, s_hat_perception, c='r')

# ------------------------- Hardcoding grasp poses --------------------------
# Left hand: Set initial pose to a default value
# bottle_right_pose
left_eef_pos_pre = np.array([ 0.54033371,  0.20095334,  0.79674643])
left_eef_quat_pre = np.array([ 0.43567574, -0.54655992,  0.44715073, -0.5581354])
left_eef_pos = np.array([ 0.54033371,  0.00095334,  0.79674643])
left_eef_quat= np.array([ 0.43567574, -0.54655992,  0.44715073, -0.5581354])

# # bottle middle pose
# left_eef_pos_pre = np.array([ 0.61691971,  0.25651524,  0.78789944])
# left_eef_quat_pre = np.array([ 0.5509148,  -0.47211483, 0.39456799, -0.56384091])
# left_eef_pos = np.array([ 0.61691971,  0.17651524,  0.78789944])
# left_eef_quat= np.array([ 0.5509148,  -0.47211483, 0.39456799, -0.56384091])


# Right hand: Set initial pose to a default value
# bottle right 1
right_eef_pos = np.array([ 0.55027766, -0.18231454,  1.1236653])
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
# ---------------------------------------------------------------------------

rospy.sleep(2)
marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
delete_all_marker = Marker()
delete_all_marker.action = Marker.DELETEALL
marker_pub.publish(delete_all_marker)

save_folder = f'output/bottle/seed_{seed}/'
os.makedirs(save_folder, exist_ok=True)

# Start recorder
if args.record:
    recorder = RecordVideo(top_down_cam, side_cam, ego_cam)
    recorder.setup_recording()


counter = 0
interesting_trajectories = 0
for traj_number in range(1):

    print(f"================{traj_number}=================")
    trajectory_dict = {}
    trajectory_dict['left_grasp_lost'] = -1
    trajectory_dict['right_grasp_lost'] = -1

    # ------------------- Obtain the screw axis and the trajectory -------------------------
    # Adding noise to s_hat. This noise can have a larger variance.
    noise = np.random.uniform(low=-0.1, high=0.1, size=(3,))
    # remove later 
    # noise = np.array([0.0, 0.0, 0.0])
    # print("noise for s_hat: ", noise)
    s = s_hat_perception + noise
    s_hat = s / np.linalg.norm(s)
    # Adding noise to q. This noise should have a smaller variance.
    noise = np.random.normal(0, 0.05, 3)
    # print("noise for q: ", noise)
    # noise = np.array([0.0, 0.0, 0.0])
    # print("noise for q: ", noise)
    q = q_perception + noise
    # plot_axis(ax, q, s_hat)
    # -------------- some hardcoded axes --------------
    # s_hat, q = np.array([-0.10081928, -0.04468542,  0.99390074]), np.array([ 0.65978671, -0.12453725,  0.13383721])
    # 24: 
    # s_hat, q = np.array([-0.14850926, -0.05133752,  0.98757757]), np.array([ 0.66089114, -0.20330914,  0.10375334])
    # 11: 
    # s_hat, q = np.array([-0.07162658, -0.11507394,  0.99077123]), np.array([ 0.62366923, -0.11807472,  0.13683665])
    # # 4: 
    # s_hat, q = np.array([-0.23421685, -0.01005412,  0.97213239]), np.array([ 0.74578873, -0.12387543,  0.15782001])
    # # 14: 
    # s_hat, q = np.array([-0.11020542, -0.03296062,  0.99336215]), np.array([ 0.72305601, -0.20245814,  0.1171972 ])
    # 15
    # s_hat, q = np.array([-0.07686291,  0.00660992,  0.99701976]), np.array([ 0.71670167, -0.17766795,  0.04005383])
    # From the offline experiment - worked well
    # s_hat, q = np.array([-0.07537131, -0.10596028,  0.99150975]), np.array([ 0.57661343, -0.08957348,  0.11892584]) # 1.42, 0.19
    # s_hat, q = np.array([-0.11124864,  0.00892458,  0.99375253]), np.array([ 0.67924664, -0.19549993,  0.12938142]) # 0.49, 0.14
    # after tooltip_ee_offset
    # s_hat, q = np.array([-0.08816754, -0.02153198,  0.99587291]), np.array([ 0.59708694, -0.16966684,  0.13509701])
    
    # trial 2 (0, 4) - Good 0.311, 0.069
    s_hat, q = np.array([-0.07296414, -0.02632962,  0.99698695]), np.array([ 0.58814249, -0.14366633,  0.20695057])
    # trial 1 (0, 17) - Goodish 1.412, 0.196 
    # s_hat, q = np.array([-0.0912213,  -0.09355492,  0.99142632]), np.array([ 0.57444014, -0.06673279,  0.11108433])
    # trial 3 (0, 6) - Goodish 1.352, 0.110 
    # s_hat, q = np.array([-0.08974382, -0.03931094,  0.99518877]), np.array([ 0.65867812, -0.1221976,   0.099134 ])
    # trial 4 (0, 5) - Goodish 1.698, 0.186
    # s_hat, q = np.array([-0.1263994,  -0.01573022,  0.9918547 ]), np.array([ 0.5919598,  -0.2103789,   0.18964259]) 
    # trial 5 Bad 2.380 0.412
    # s_hat, q = np.array([-0.07385186, -0.17440865,  0.98189996]), np.array([ 0.68489215, -0.05320495,  0.14008214]) 
    # trial 6 Bad  4.180 0.169
    # s_hat, q = np.array([-0.10365334, -0.06341209,  0.99258999]), np.array([ 0.74633728, -0.20999397,  0.14893376]) 
    # trial 7 Bad 6.191 0.864 
    # s_hat, q = np.array([-0.25974761, -0.08710759,  0.9617398 ]), np.array([ 0.57954888, -0.23462,     0.03248047]) 

    # ideal_1
    # s_hat = np.array([0.0, 0.0, 1.0])
    # q = right_eef_pos

    # s_hat_noise_1
    # s_hat = np.array([-0.10081928, -0.04468542,  0.99390074])
    # s_hat_noise_2
    # s_hat = np.array([-0.07686291,  0.00660992,  0.99701976])
    # s_hat_noise_3
    # s_hat = np.array([-0.08383039, -0.04167278,  0.99560828])
    # s_hat = [-0.06359817  0.00858564  0.99793866]
    # s_hat_noise_4
    # s_hat = np.array([-0.16704007, -0.08923226,  0.98190388])
    # s_hat_noise_5
    # s_hat = np.array([-0.14850926, -0.05133752,  0.98757757])

    # q = right_eef_pos
    # noise = np.random.normal(0, 0.05, 3)
    # q = q + noise
    # q = location of the bottle
    # q = np.array([0.535, -0.165, 0.736])
    # q_noise_1 = 0.044
    # q = np.array([ 0.57601367, -0.14757171,  1.13192928])
    # # q_noise_2 = 0.145
    # q = np.array([ 0.46980572, -0.08515143,  1.05129349])
    # q_noise 3 = 0.01
    # q = np.array([0.54, -0.17, 1.12])
    # -----------------------------------------------
    print("----------- final s_hat, q: ", s_hat, q)
    target_q = np.array([0.55027766, -0.18231454,  1.1236653])
    q_dist = np.linalg.norm(q[:2] - target_q[:2])
    print("q_dist: ", q_dist)
    trajectory_dict['s_hat'] = s_hat
    trajectory_dict['q'] = q
    publish_line_marker(marker_pub, s_hat, q, counter)
    counter += 2

    final_points = get_points_revolute(s_hat, q, T0)
    trajectory_dict['final_points'] = final_points
    
    # keep_traj = check_traj_quality(final_points)
    # print("keep_traj: ", keep_traj)
    # if not keep_traj:
    #     continue
    # interesting_trajectories += 1
    # if traj_number < 5:
    #     continue

    # -------------------------------------------------------------------------------------------

    # # Open gripper and go home
    rospy.sleep(1)
    env.tiago.gripper['left'].write(0)
    env.tiago.gripper['right'].write(0)
    rospy.sleep(1)
    # go to home
    # home_left_hand(env)
    home_right_hand(env)

    # -------- Make left gripper go to pose and grasp ---------------
    # Pregrasp pose
    pose_command = env.tiago.create_pose_command(left_eef_pos_pre, left_eef_quat_pre)
    print("Left Hand Going To Pose")
    rospy.sleep(1)
    env.tiago.tiago_pose_writer['left'].write(pose_command)
    wait_left(left_eef_pos_pre, env)
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
    #-----------------------------------------------------------------

    # --------- Make right gripper go to grasp pose and grasp --------
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
    env.tiago.gripper['right'].write(0.7)
    # -----------------------------------------------------------------

    # rospy.sleep(1)
    # right_eef_pose = env._observation()['right_eef']
    # tooltip_ee_offset = np.array([0.26, 0, 0])
    # print("quat: ", right_eef_pose[3:])
    # right_eef_pose_mat = R.from_quat(right_eef_pose[3:]).as_matrix()
    # tooltip_ee_offset_world = np.dot(right_eef_pose_mat, tooltip_ee_offset)
    # temp_pos = right_eef_pos + tooltip_ee_offset_world[:3]
    # print("temp_pos: ", temp_pos)
    # publish_sphere_marker(marker_pub, temp_pos)
    
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
    for i in range(len(final_points)):
    # for i in range(8):
        pt = final_points[i]
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

        if force_sum > 60:
            print("FORCE GETING TOO HIGH. STOPPING!")
            interrupted = True
            break

        left_is_grasp = env.tiago.gripper['left'].is_grasped()
        right_is_grasp = env.tiago.gripper['right'].is_grasped()
        
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
        # input("press enter")

    # Move the arm up
    if not interrupted:
        move_up(env)

    # open gripper again
    print("------------- ROBOT TRAJ COMPLETED -------------")
    rospy.sleep(1)
    env.tiago.gripper['left'].write(0)
    env.tiago.gripper['right'].write(0)
    rospy.sleep(1)

    if args.record:
        recorder.pause_recording()
        recorder.save_video(save_folder, args)
        print("Saved video")
        recorder.reset_frames()
    # ------------------------------------------------------------------------------
    
    trajectory_dict['forces'] = forces
    trajectory_dict['torques'] = torques
    succ = input("Final Task Success. O for Failure and 1 for Success")
    trajectory_dict["task_succ"] = succ
    # with open(f'{save_folder}{traj_number}.pickle', 'wb') as handle:
    #     pickle.dump(trajectory_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{save_folder}{args.f_name}.pickle', 'wb') as handle:
        pickle.dump(trajectory_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if args.record:
    print("Stop recording")
    recorder.stop_recording()
# print("interesting_trajectories: ", interesting_trajectories)
