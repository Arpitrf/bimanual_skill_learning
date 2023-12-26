import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math
from scipy.linalg import logm, expm
import pickle
from scipy.optimize import curve_fit
from argparse import ArgumentParser
import random

def config_parser(parser=None):
    if parser is None:
        parser = ArgumentParser("Bimanual Skill Learning")
    parser.add_argument('--f_name', type=str)
    # parser.add_argument('--start_frame', type=int, default=1)
    # parser.add_argument('--end_frame', type=int, default=-1)
    parser.add_argument('--show_arc', action='store_true', default=False, help='Run headless or render')
    parser.add_argument('--show_screw', action='store_true', default=False, help='Run headless or render')
    return parser


args = config_parser().parse_args()
paths = [f'data/{args.f_name}/hand_poses.pickle']

for path in paths:
    with open(path, 'rb') as handle:
        hand_dict = pickle.load(handle)
    print("hand_dict: ", len(hand_dict.keys()))
    with open(f'data/{args.f_name}/extrinsic.pickle', 'rb') as handle:
        extr = pickle.load(handle)
    print("extr: ", extr)

    # Create a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

     # ------------------- Get the 6-DoF hand poses and the delta poses -----------------------
    hand = hand_dict[next(iter(hand_dict))]
    init_pos = hand['right_hand_pos']
    init_pos /= 1000
    init_orn = hand['right_hand_orn']
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
        hand_pos = hand_dict[h]['right_hand_pos']
        # print("---", hand_pos)
        hand_pos /= 1000
        # remove later
        # hand_pos_homo = np.append(hand_pos, 1.0)
        # final_pos = np.dot(extr, hand_pos_homo)[:3]
        # print("final possssssss: ", final_pos)

        hand_orn = hand_dict[h]['right_hand_orn']
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

    # temp using 2 points. TODO: fit a line
    start_T, end_T = Ts[0], Ts[-1]
    print("start_T---: ", start_T[:3, 3])
    print("end_T---: ", end_T[:3, 3])
    s = np.array(end_T[:3, 3]) - np.array(start_T[:3, 3])
    print("s: ", s)
    s_hat = s / np.linalg.norm(s)
    print("s_hat: ", s_hat)

    # save to file
    save_dict = {
        's_hat': s_hat,
        'q': np.array([0,0,0])
    }
    print("save_dict: ", save_dict)
    with open(f'data/{args.f_name}/axis.pickle', 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
    T0 = np.array([[1, 0, 0, start_T[0,3]],
        [0, 1, 0, start_T[1,3]],
        [0, 0, 1, start_T[2,3]],
        [0, 0, 0, 1]])
    # T0 = start_T
    ax.scatter(T0[0][3], T0[1][3], T0[2][3], color='r', marker='o')

    for theta in np.arange(0.1, 1.0, 0.01):
        S_theta = theta * np.array(S)

        T1 = np.dot(T0, expm(S_theta))
        # T1 = np.dot(expm(S_theta), T0)
        ax.scatter(T1[0][3], T1[1][3], T1[2][3], color='g', marker='o')


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

    plt.show()
