import numpy as np
import pickle

path = '/home/pal/arpit/bimanual_skill_learning/output/bottle/seed_0/'
folder_names = ['ideal', 'trial_1_good', 'trial_2_good', 'trial_5_bad']
trajs = []

for f in folder_names:
    with open(f'{path}{f}.pickle', 'rb') as handle:
        dict = pickle.load(handle)
        trajs.append(dict)

def cem_update_epoch_zero(trajs):
    full_list, traj_len_list, avg_f_list, avg_t_list = [], [], [], []
    MIN_FORCE_THRESHOLD = 15
    #  Change counter!!
    counter = 0
    for traj in trajs:
        counter += 1
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
        avg_f_list.append([counter, np.mean(traj_forces_sum)])
        avg_t_list.append([counter, np.mean(traj_torques_sum)])
        traj_len_list.append([counter, len(traj['forces'])])
        full_list.append([counter, len(traj['forces']), np.mean(traj_forces_sum), np.mean(traj_torques_sum)])
        
    # Sort according to the length of trajectories
    full_list_sorted_by_len = sorted(full_list, key = lambda x: x[1], reverse=True)

    # Keep only the trajs that have max length
    max_traj_len = full_list_sorted_by_len[0][1]
    # print("max_traj_len: ", max_traj_len)
    max_traj_len_list = []
    for elem in full_list_sorted_by_len:
        if elem[1] == max_traj_len:
            max_traj_len_list.append(elem)
    # Now sort according to the FT sensor readings
    full_list_sorted_by_ft = sorted(max_traj_len_list, key = lambda x: x[2])

    # return the best trajectory
    return full_list_sorted_by_ft[0][0]

mu_x_s, sigma_x_s = 0, 0.05
mu_y_s, sigma_y_s = 0, 0.05
mu_z_s, sigma_z_s = 0, 0.05
mu_x_q, sigma_x_q = 0, 0.03
mu_y_q, sigma_y_q = 0, 0.03
mu_z_q, sigma_z_q = 0, 0.03

s_hat_new, q_new = None, None
for epoch in range(2):
    for traj_number in range(10):
        if epoch > 0:
            s_hat_perception, q_perception = s_hat_new, q_new
            noise_s_hat_x = np.random.normal(mu_x_s, sigma_x_s)
            noise_s_hat_y = np.random.normal(mu_y_s, sigma_y_s)
            noise_s_hat_z = np.random.normal(mu_z_s, sigma_z_s)
            noise_s_hat = np.array([noise_s_hat_x, noise_s_hat_y, noise_s_hat_z])
            noise_q_x = np.random.normal(mu_x_q, sigma_x_q)
            noise_q_y = np.random.normal(mu_y_q, sigma_y_q)
            noise_q_z = np.random.normal(mu_z_q, sigma_z_q)
            noise_q = np.array([noise_q_x, noise_q_y, noise_q_z])

            s = s_hat_perception + noise_s_hat
            s_hat = s / np.linalg.norm(s)
            q = q_perception + noise_q

            print("----------- FINAL s_hat, q: ", s_hat, q)


    if epoch == 0:
        traj_len_list, avg_f_list, avg_t_list = [], [], []
        # currently taking just the top trajectory in the first epoch. Can also try taking some "average" of top x trajectories
        elite_traj_number = cem_update_epoch_zero(trajs)
        # print("elite_traj_number: ", elite_traj_number)
        elite_traj = trajs[elite_traj_number]
        s_hat_new = elite_traj['s_hat']
        q_new = elite_traj['q']