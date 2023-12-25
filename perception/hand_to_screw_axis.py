import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math
from scipy.linalg import logm, expm
import pickle
import wandb
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from plotly.subplots import make_subplots
from argparse import ArgumentParser
import random


def plot_plotly_axis(fig, point, dirX, dirY, dirZ, fig_idx, length=20):
    # Calculate the endpoint of the line
    endpoint_1 = point + length * dirX
    endpoint_2 = point + length * dirY
    endpoint_3 = point + length * dirZ
    # Plot the line
    trace_new = go.Scatter3d(
        x=[point[0],endpoint_1[0]], y=[point[1], endpoint_1[1]], z=[point[2], endpoint_1[2]],
        mode='lines',
        line=dict(
            color="#FF0000",
            width=8
        )
    )
    fig.add_trace(trace_new, row=fig_idx[0], col=fig_idx[1])
    trace_new = go.Scatter3d(
        x=[point[0],endpoint_2[0]], y=[point[1], endpoint_2[1]], z=[point[2], endpoint_2[2]],
        mode='lines',
        line=dict(
            color="#00FF00",
            width=8
        )
    )
    fig.add_trace(trace_new, row=fig_idx[0], col=fig_idx[1])
    trace_new = go.Scatter3d(
        x=[point[0],endpoint_3[0]], y=[point[1], endpoint_3[1]], z=[point[2], endpoint_3[2]],
        mode='lines',
        line=dict(
            color="#0000FF",
            width=8
        )
    )
    fig.add_trace(trace_new, row=fig_idx[0], col=fig_idx[1])
    # fig.show()

def get_points_revolute(s_hat, q, initial_pose, angle_moved=6.28):
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
        ax.scatter(T1[0][3], T1[1][3], T1[2][3], color='g', marker='o')
    
    return final_points

# Function to plot the axis lines with arrows
def plot_axis(ax, color, vector, label, pos=[0,0,0]):
    # print("------", pos, vector)
    ax.quiver(pos[0], pos[1], pos[2], vector[0], vector[1], vector[2], color=color, linewidth=2, length=10, normalize=True)

def config_parser(parser=None):
    if parser is None:
        parser = ArgumentParser("Bimanual Skill Learning")
    parser.add_argument('--f_name', type=str)
    # parser.add_argument('--start_frame', type=int, default=1)
    # parser.add_argument('--end_frame', type=int, default=-1)
    parser.add_argument('--show_arc', action='store_true', default=False, help='Run headless or render')
    parser.add_argument('--show_screw', action='store_true', default=False, help='Run headless or render')
    return parser

def compute_screw_axis(start_T, final_T):
    # delta_T = np.dot(final_T, np.linalg.inv(start_T))
    delta_T = np.dot(final_T, np.linalg.inv(start_T))

    # Compute the matrix logarithm
    log_T = logm(delta_T)

    # Extract linear velocities
    linear_velocities = log_T[:3, 3]

    # Extract skew-symmetric matrix (angular velocities)
    S = log_T[:3, :3]
    # print("S: ", S)

    # Calculate angular velocities from the skew-symmetric matrix
    angular_velocities = np.array([S[2, 1] - S[1, 2], S[0, 2] - S[2, 0], S[1, 0] - S[0, 1]]) / 2.0
    # print("angular_velocities: ", angular_velocities)

    # Combine linear and angular velocities to get the twist
    twist = np.concatenate((linear_velocities, angular_velocities))
    # print("Twist vector:", twist)

    screw_axis = angular_velocities / np.linalg.norm(angular_velocities)
    theta = np.linalg.norm(angular_velocities)
    q = np.cross(screw_axis, linear_velocities) / theta
    # print("screw_axis: ", screw_axis)
    # print("q: ", q)
    return screw_axis, q

def circle_equation(x, h, k, l, r):
    # print("x, h, k, l, r: ", x, h, k, l, r)
    return (x[0] - h)**2 + (x[1] - k)**2 + (x[2] - l)**2 - r**2

def compute_arc_axis(Ts, pts_to_compute_normal=[10, 1, 12, 3]):
    # Example 3D points
    data_points =  []
    for T in Ts:
        pos = np.array(T)[:3, 3]
        data_points.append(pos)
    data_points = np.array(data_points)

    initial_guess = [0, 0, 0, 1]
    # Perform least squares fit
    params, covariance = curve_fit(circle_equation, data_points.T, np.zeros(data_points.shape[0]), p0=initial_guess)
    arc_centre_x, arc_centre_y, arc_centre_z, radius = params
    q = np.array([arc_centre_x, arc_centre_y, arc_centre_z])
    # print(f"Center of the circle: ({arc_centre_x}, {arc_centre_y}, {arc_centre_z})")
    # print(f"Radius of the circle: {radius}")

    # TODO: Change this to using two points on the predicted arc
    pt1, pt2, pt3, pt4 = pts_to_compute_normal
    temp_vec_1 = data_points[pt1] - data_points[pt2] 
    temp_vec_2 = data_points[pt3] - data_points[pt4]
    normal_vector = np.cross(temp_vec_1, temp_vec_2)

    # Normalize the normal vector to get the axis of rotation
    axis_of_rotation = normal_vector / np.linalg.norm(normal_vector)
    # print(f"Axis of rotation: {axis_of_rotation}")

    return axis_of_rotation, q

def plot_trajectory_temp(T0, axis, q, mode):
    if mode == 'arc':
        color = [1,0,1]
    else:
        color = [0,0,1]
    for _ in range(1):
        # axis += np.array([0.1, 0, 0.1])
        # axis /= np.linalg.norm(axis)
        T0 = np.array(T0)
        # T0[:3, 3] = np.array([40,  -40, 500])
        # print("to, screw, q: ", T0, axis, q)
        h = 0 # pure rotation
        s_hat = axis
        w = s_hat
        v = -np.cross(s_hat, q)
        twist = np.concatenate((w,v)) 
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
        # S = log_T
        for theta in np.arange(0.1, 6.28, 0.1):
            S_theta = theta * np.array(S)

            # T1 = np.dot(T0, expm(S_theta))
            T1 = np.dot(expm(S_theta), T0)
            ax.scatter(T1[0][3], T1[1][3], T1[2][3], color=color, marker='o')
            plt.pause(1)


        length = 800
        endpoint_1 = q + length * axis
        endpoint_2 = q - length * axis
        ax.plot([endpoint_2[0], endpoint_1[0]], [endpoint_2[1], endpoint_1[1]], [endpoint_2[2], endpoint_1[2]], c=color)
        # input("Press enter")

        # Set labels
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        # ax.set_xlim([-550, 550])
        # ax.set_ylim([-550, 550])
        # ax.set_zlim([-100, 800])
        # if j  == 0:
        #     plt.pause(15)
        # plt.pause(3)
        # plt.cla()


def compute_trajectory_screw(T0, axis, q, ax, fig_screw, fig_arc, len_hand_pts, theta_step=0.05, show_screw=False, show_arc=False):
    computed_Ts = []
    h = 0 # pure rotation
    s_hat = axis
    w = s_hat
    v = -np.cross(s_hat, q)
    twist = np.concatenate((w,v)) 
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
    # S = log_T
    computed_Ts.append(T0)

    # calculate the thetas
    thetas = []
    for i in range(1, len_hand_pts):
        thetas.append(i*theta_step)
    
    # Calculate the transformation of the point when moved by theta along the screw axis
    # for theta in np.arange(0.1, 3.14, 0.1):
    for theta in thetas:
        S_theta = theta * np.array(S)

        # T1 = np.dot(T0, expm(S_theta))
        T1 = np.dot(expm(S_theta), T0)
        computed_Ts.append(T1)
        # print("computed Ts pos: ", T1[0][3], T1[1][3], T1[2][3])
        # Plotting
        # if show_arc:
        #     ax.scatter(T1[0][3], T1[1][3], T1[2][3], color='m', marker='o')
        # trace_new = go.Scatter3d(
        #     x=[T1[0][3]], y=[T1[1][3]], z=[T1[2][3]],
        #     mode='markers',
        #     line=dict(
        #         color="#FF00FF",
        #     ),
        #     marker_size=4
        # )
        # fig_screw.add_trace(trace_new)
        # final_fig.add_trace(trace_new, row=1, col=1)
    
    return computed_Ts

def compute_trajectory_arc(T0, axis, q, ax, fig_screw, fig_arc, len_hand_pts, theta_step=0.05, show_screw=False, show_arc=False):
    computed_Ts = []
    original_orientation = np.array(T0)[:3,:3]
    h = 0 # pure rotation
    s_hat = axis
    w = s_hat
    v = -np.cross(s_hat, q)
    twist = np.concatenate((w,v)) 
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
    # S = log_T
    computed_Ts.append(T0)
    # ax.scatter(T0[0][3], T0[1][3], T0[2][3], color=[0,0,0], marker='o', s=144)
    # if show_screw:
    #     ax.scatter(final_T[0][3], final_T[1][3], final_T[2][3], color=[0,1,1], marker='o',s=144)

    # calculate the thetas
    thetas = []
    for i in range(1, len_hand_pts):
        thetas.append(i*theta_step)
    
    # Calculate the transformation of the point when moved by theta along the screw axis
    # for theta in np.arange(0.1, 3.14, 0.1):
    for theta in thetas:
        S_theta = theta * np.array(S)

        # T1 = np.dot(T0, expm(S_theta))
        T1 = np.dot(expm(S_theta), T0)
        T1 = [
            [original_orientation[0][0], original_orientation[0][1], original_orientation[0][2], T1[0, 3]],
            [original_orientation[1][0], original_orientation[1][1], original_orientation[1][2], T1[1, 3]],
            [original_orientation[2][0], original_orientation[2][1], original_orientation[2][2], T1[2, 3]],
            [0, 0, 0, 1]
        ]
        computed_Ts.append(T1)
        # print("T1: ", T1[0][3], T1[1][3], T1[2][3])
        # # Plotting
        # if show_screw:
        #     ax.scatter(T1[0][3], T1[1][3], T1[2][3], color='b', marker='o')
        # if show_arc:
        #     ax.scatter(T1[0][3], T1[1][3], T1[2][3], color='m', marker='o')
        # trace_new = go.Scatter3d(
        #     x=[T1[0][3]], y=[T1[1][3]], z=[T1[2][3]],
        #     mode='markers',
        #     line=dict(
        #         color="#FF00FF",
        #     ),
        #     marker_size=4
        # )
        # fig_screw.add_trace(trace_new)
        # final_fig.add_trace(trace_new, row=1, col=1)
    
    return computed_Ts

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


args = config_parser().parse_args()
random.seed(1)
np.random.seed(1)
paths = ['/home/arpit/test_projects/frankmocap/mocap_output/screw_axis_test_bottle/hand_poses.pickle',
         '/home/arpit/test_projects/frankmocap/mocap_output/screw_axis_test_laptop/hand_poses.pickle',
         '/home/arpit/test_projects/frankmocap/mocap_output/screw_axis_test_stir/hand_poses.pickle'
]
paths = [f'/home/arpit/test_projects/frankmocap/mocap_output/{args.f_name}/hand_poses.pickle']


for path in paths:
    with open(path, 'rb') as handle:
        hand_dict = pickle.load(handle)
    print("hand_dict: ", len(hand_dict.keys()))
    with open(f'/home/arpit/test_projects/frankmocap/sample_data/{args.f_name}/extrinsic.pickle', 'rb') as handle:
        extr = pickle.load(handle)
    print("extr: ", extr)


    # Create a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plotly 
    fig_screw = go.Figure()
    fig_screw.update_layout(
        scene = dict(
            xaxis = dict(nticks=4, range=[-200,200]),
            yaxis = dict(nticks=4, range=[-200,200]),
            zaxis = dict(nticks=4, range=[400,900]),
        ),
        scene_camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25)
        )
    )

    fig_arc = go.Figure()
    fig_arc.update_layout(
        scene = dict(
            xaxis = dict(nticks=4, range=[-200,200]),
            yaxis = dict(nticks=4, range=[-200,200]),
            zaxis = dict(nticks=4, range=[400,900]),
        ),
        scene_camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25)
        )
    )
    final_fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]])
    final_fig.update_scenes(
        xaxis = dict(nticks=4, range=[-400,400]),
        yaxis = dict(nticks=4, range=[-400,400]),
        zaxis = dict(nticks=4, range=[400,1200]),
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25)
        )
    )

    ax.set_xlim([-150, 150])
    ax.set_ylim([-150, 150])
    ax.set_zlim([400, 900])
    # fig.show()

    # wandb.login()
    # run = wandb.init(
    #     project="bimanual-skill-learning-perception",
    #     notes="trial experiment",
    #     tags=["baseline", "paper1"],
    # )
    # wandb_table = wandb.Table(columns=["Video Name", "Using 6-DoF Pose", "Using 3-DoF Position"])

    # ------------------- Get the 6-DoF hand poses and the delta poses -----------------------
    hand = hand_dict[next(iter(hand_dict))]
    init_pos = hand['left_hand_pos']
    init_pos /= 1000
    init_orn = hand['left_hand_orn']
    init_orn_matrix = R.from_rotvec(np.array(init_orn)).as_matrix()
    T0 = [
            [init_orn_matrix[0][0], init_orn_matrix[0][1], init_orn_matrix[0][2], init_pos[0]],
            [init_orn_matrix[1][0], init_orn_matrix[1][1], init_orn_matrix[1][2], init_pos[1]],
            [init_orn_matrix[2][0], init_orn_matrix[2][1], init_orn_matrix[2][2], init_pos[2]],
            [0, 0, 0, 1]
        ]
    T0_world = np.dot(extr, T0)
    first_hand_pose = T0_world.copy()
    delta_Ts = []
    Ts = []
    Ts.append(T0_world)
    for i, h in enumerate(hand_dict.keys()):
        if i == 0:
            continue
        hand_pos = hand_dict[h]['left_hand_pos']
        # print("---", hand_pos)
        hand_pos /= 1000
        # remove later
        # hand_pos_homo = np.append(hand_pos, 1.0)
        # final_pos = np.dot(extr, hand_pos_homo)[:3]
        # print("final possssssss: ", final_pos)

        hand_orn = hand_dict[h]['left_hand_orn']
        hand_orn_matrix = R.from_rotvec(np.array(hand_orn)).as_matrix()

        transformed_x = np.dot(hand_orn_matrix, [1, 0, 0])
        transformed_y = np.dot(hand_orn_matrix, [0, 1, 0])
        transformed_z = np.dot(hand_orn_matrix, [0, 0, 1])
        # print("transformed_x shape: ", transformed_x.shape)
        
        # if i == 1 or i == 27:
        ax.scatter(hand_pos[0], hand_pos[1], hand_pos[2], c=([[1,0,0]]), marker='o')
        # plot_axis(ax, 'red', transformed_x, 'X', hand_pos)
        # plot_axis(ax, 'green', transformed_y, 'Y', hand_pos)
        # plot_axis(ax, 'blue', transformed_z, 'Z', hand_pos)

        # plotly plots
        # print("hand_pos: ", hand_pos)
        trace_new = go.Scatter3d(
            x=[hand_pos[0]], y=[hand_pos[1]], z=[hand_pos[2]],
            mode='markers',
            line=dict(
                color="#FF0000",
            ),
            marker_size=2
        )
        fig_screw.add_trace(trace_new)
        fig_arc.add_trace(trace_new)
        final_fig.add_trace(trace_new, row=1, col=1)
        final_fig.add_trace(trace_new, row=1, col=2)
        plot_plotly_axis(final_fig, np.array(hand_pos), transformed_x, transformed_y, transformed_z, fig_idx=(1,1))
        plot_plotly_axis(final_fig, np.array(hand_pos), transformed_x, transformed_y, transformed_z, fig_idx=(1,2))

        T1 = [
            [hand_orn_matrix[0][0], hand_orn_matrix[0][1], hand_orn_matrix[0][2], hand_pos[0]],
            [hand_orn_matrix[1][0], hand_orn_matrix[1][1], hand_orn_matrix[1][2], hand_pos[1]],
            [hand_orn_matrix[2][0], hand_orn_matrix[2][1], hand_orn_matrix[2][2], hand_pos[2]],
            [0, 0, 0, 1]
        ]

        # using extrinsic (trial)

        T1_world = np.dot(extr, T1)
        ax.scatter(T1_world[0, 3], T1_world[1, 3], T1_world[2, 3], c=([[0.5,0.5,0.5]]), marker='o')
        print("world---: ", T1_world[:3, 3])

        delta_T_world = np.dot(T1_world, np.linalg.inv(T0_world))
        delta_Ts.append(delta_T_world)
        Ts.append(T1_world)
        T0_world = T1_world
    print("delta_Ts: ", len(delta_Ts))
    print("Ts: ", len(Ts))
    len_hand_pts = len(Ts)
    # ----------------------------------------------------------------------

    min_dist = 1000000
    final_axis_screw = None
    final_q_screw = None
    final_computed_Ts_screw = None
    final_start_T, final_end_T = None, None
    final_idx = None
    for i in range(len(Ts)):
        for j in range(i+1, len(Ts)):
            # Compute the screw axis based on 2 poses 
            start_T = Ts[i]
            final_T = Ts[j]
            axis_screw, q_screw = compute_screw_axis(start_T, final_T)
                       
            T0 = first_hand_pose
            # Compute the trajectory based on the previously obtained screw axis 
            computed_Ts_screw = compute_trajectory_screw(T0, axis_screw, q_screw, ax, fig_screw, fig_arc, len_hand_pts, theta_step=0.06, show_screw=args.show_screw)
            screw_trajectory_dist, screw_orn_dist, screw_orn_dist2 = compute_trajectory_score(Ts, computed_Ts_screw)
            # print("screw_trajectory_dist ", i, j, screw_trajectory_dist)
            if screw_trajectory_dist < min_dist:
                min_dist = screw_trajectory_dist
                final_axis_screw = axis_screw
                final_q_screw = q_screw
                final_computed_Ts_screw = computed_Ts_screw
                final_start_T = start_T
                final_end_T = final_T
                final_idx = (i, j)
    
    # compute_trajectory_score(Ts, final_computed_Ts_screw, log=True)
    print("min_dist: ", min_dist, final_idx)
    print("final_computed_Ts_screw: ", np.array(final_computed_Ts_screw)[:, :3, 3])
    # Plotting
    length = 1
    endpoint_1 = final_q_screw + length * final_axis_screw
    endpoint_2 = final_q_screw - length * final_axis_screw
    if args.show_screw:
        # plot_trajectory_temp(first_hand_pose, final_axis_screw, final_q_screw, mode='screw')
        ax.scatter(final_start_T[0][3], final_start_T[1][3], final_start_T[2][3], color=[0,0,0], marker='o', s=144)
        ax.scatter(final_end_T[0][3], final_end_T[1][3], final_end_T[2][3], color=[0,1,1], marker='o',s=144)
        ax.scatter(final_q_screw[0], final_q_screw[1], final_q_screw[2], color=[0,0.2,0.2], marker='o',s=144)
        ax.plot([endpoint_2[0], endpoint_1[0]], [endpoint_2[1], endpoint_1[1]], [endpoint_2[2], endpoint_1[2]], c=[0,0,1])
        for pose in final_computed_Ts_screw:
            ax.scatter(pose[0][3], pose[1][3], pose[2][3], color='b', marker='o')

    # save to file
    save_dict = {
        's_hat': final_axis_screw,
        'q': final_q_screw
    }
    print("save_dict: ", save_dict)
    with open(f'/home/arpit/test_projects/frankmocap/mocap_output/{args.f_name}/axis.pickle', 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # plot on plotly
    # trace_new = go.Scatter3d(
    #     x=[endpoint_2[0], endpoint_1[0]], y=[endpoint_2[1], endpoint_1[1]], z=[endpoint_2[2], endpoint_1[2]],
    #     mode='lines',
    #     line=dict(
    #         color="#0000FF",
    #     )
    # )
    # fig_screw.add_trace(trace_new)
    # final_fig.add_trace(trace_new, row=1, col=1)

    # # # -------- New method to compute arc ----------
    # # # Plane equation in the form ax + by + cz + d = 0
    # # def plane_equation(points, a, b, c, d):
    # #     # print("points: ", points)
    # #     x, y, z = points
    # #     return (a * x) + (b * y) + (c * z) - d
    
    # # # Example 3D points
    # # data_points =  []
    # # for T in Ts:
    # #     pos = np.array(T)[:3, 3]
    # #     data_points.append(pos)
    # # data_points = np.array(data_points)
    # # print("data_points shape: ", data_points)
    
    # # initial_guess = [0, 0, 0, 1]  # Initial guess for the plane parameters
    # # popt, pcov = curve_fit(plane_equation, data_points.T, np.zeros(data_points.shape[0]), p0=initial_guess)

    # # print("popt: ", popt)
    # # # ---------------------------------------------

    # # --------- compute the arc ----------
    # min_dist = 10000000
    # for i in range(100):
    #     idxs = random.sample(range(0, len(Ts)), 4)
    #     # idxs = [10, 1, 12, 3]
    #     axis_arc, q_arc = compute_arc_axis(Ts, pts_to_compute_normal=idxs)

    #     T0 = first_hand_pose
    #     # Compute the trajectory based on the previously obtained screw axis 
    #     computed_Ts_arc = compute_trajectory_arc(T0, axis_arc, q_arc, ax, fig_screw, fig_arc, len_hand_pts, theta_step=0.3, show_arc=args.show_arc)
    #     arc_trajectory_dist, arc_orn_dist, arc_orn_dist2 = compute_trajectory_score(Ts, computed_Ts_arc)
    #     # print("arc_trajectory_dist ", i, arc_trajectory_dist)
    #     if arc_trajectory_dist < min_dist:
    #         min_dist = arc_trajectory_dist
    #         min_orn_dist = arc_orn_dist
    #         final_axis_arc = axis_arc
    #         final_q_arc = q_arc
    #         final_computed_Ts_arc = computed_Ts_arc
    #         final_idxs = idxs

    # compute_trajectory_score(Ts, final_computed_Ts_arc, log=True)
    # print("min_dist: ", min_dist, min_orn_dist, final_idxs)
    # # Plotting
    # # plot_trajectory_temp(first_hand_pose, final_axis_arc, final_q_arc, mode='arc')
    # length = 800
    # endpoint_1 = np.array([final_q_arc[0], final_q_arc[1], final_q_arc[2]]) + length * final_axis_arc
    # endpoint_2 = np.array([final_q_arc[0], final_q_arc[1], final_q_arc[2]]) - length * final_axis_arc
    # # Plot the line
    # if args.show_arc:
    #     ax.scatter(final_q_arc[0], final_q_arc[1], final_q_arc[2], color='r', marker='o')
    #     ax.plot([endpoint_2[0], endpoint_1[0]], [endpoint_2[1], endpoint_1[1]], [endpoint_2[2], endpoint_1[2]], c=[1,0,1])
    #     for pose in final_computed_Ts_arc:
    #         ax.scatter(pose[0][3], pose[1][3], pose[2][3], color=[1,0,1], marker='o')

    # # plot on plotly
    # trace_new = go.Scatter3d(
    #     x=[q_arc[0]], y=[q_arc[1]], z=[q_arc[2]],
    #     mode='markers',
    #     line=dict(
    #         color="#000000",
    #     )
    # )
    # fig_arc.add_trace(trace_new)
    # final_fig.add_trace(trace_new, row=1, col=2)
    # trace_new = go.Scatter3d(
    #     x=[endpoint_2[0], endpoint_1[0]], y=[endpoint_2[1], endpoint_1[1]], z=[endpoint_2[2], endpoint_1[2]],
    #     mode='lines',
    #     line=dict(
    #         color="#0000FF",
    #     )
    # )
    # fig_arc.add_trace(trace_new)
    # final_fig.add_trace(trace_new, row=1, col=2)
    # # ------------------------------------

    # T0 = first_hand_pose
    # # Compute the trajectory based on the previously obtained screw axis 
    # computed_Ts_screw = compute_trajectory_screw(T0, axis_screw, q_screw, ax, fig_screw, fig_arc, len_hand_pts, theta_step=0.06, show_screw=args.show_screw, )
    # # Compute the trajectory based on the previously obtained arc axis 
    # # TODO: Change this to not change the orientation of the start pose
    # computed_Ts_arc = compute_trajectory_arc(T0, axis_arc, q_arc, ax, fig_screw, fig_arc, len_hand_pts, show_arc=args.show_arc)
    # print("original Ts, computed_screw_Ts, computed_arc_Ts: ", len(Ts), len(computed_Ts_screw), len(computed_Ts_arc))

    # screw_trajectory_dist, screw_orn_dist, screw_orn_dist2 = compute_trajectory_score(Ts, computed_Ts_screw)
    # arc_trajectory_dist, arc_orn_dist, arc_orn_dist2 = compute_trajectory_score(Ts, computed_Ts_arc)
    # print("screw_trajectory_dist, arc_trajectory_dist: ", screw_trajectory_dist, arc_trajectory_dist)
    # print("Orn screw_trajectory_dist, arc_trajectory_dist: ", screw_orn_dist, arc_orn_dist)
    # print("Orn screw_trajectory_dist, arc_trajectory_dist: ", screw_orn_dist2, arc_orn_dist2)

    # Set labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # ax.set_xlim([-150, 150])
    # ax.set_ylim([-150, 150])
    # ax.set_zlim([400, 900])
            
    ax.set_xlim([0.3, 0.9])
    ax.set_ylim([-0.2, 0.2])
    ax.set_zlim([0.5, 1])

    # ax.set_xlim([0, 2])
    # ax.set_ylim([-2, 2])
    # ax.set_zlim([0, 2])

    # wandb_table.add_data('1',
    #                     {"fig1": fig_screw},
    #                     {"fig2": fig_arc}) 

    # print("type: ", type(fig_screw))

    # wandb_table = wandb.Table(
    #             columns=["Video Name"], 
    #             data=[['1']]
    #         )

    # run.log({"perception": wandb_table})   
    # wandb.log({"fig": fig_arc})
    # wandb.log({"fig2": fig_screw})


    # Show the plot
    plt.show() 
    # fig_screw.show()
    # fig_arc.show()


    # final_fig.show()
    # folder_mame = path.split('/')[-2]
    # print("folder_name: ", folder_mame)
    # wandb.log({f"{folder_mame}": final_fig})