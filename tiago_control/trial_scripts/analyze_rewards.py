import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

folder = "/home/pal/arpit/bimanual_skill_learning/output/tissue_insert_far/seed_0/"
# folder = "/home/pal/arpit/bimanual_skill_learning/output/bottle/seed_0/"
fig, ax = plt.subplots(1,2)
ax[0].set_ylim([0, 70])
ax[1].set_ylim([0, 15])


for file in os.listdir(folder):
    filename = os.fsdecode(file)
    if filename.endswith(".pickle"):
        print(f"---------------------- {filename} --------------------")
        with open(f'{folder}{filename}', 'rb') as handle:
            traj = pickle.load(handle)
                
        forces_sum = []
        torques_sum = []
        forces = np.array(traj['forces'])
        torques = np.array(traj['torques'])
        print("s_hat: ", traj['s_hat'])
        print("type: ", type(forces[0]))
        # print("len(forces), len(torques): ", len(forces), len(torques))
        # print("left_grasp_lost, right_grasp_lost: ", traj['left_grasp_lost'], traj['right_grasp_lost'])


        for i in range(len(forces)):
            # print(forces[i])
            forces_sum.append(abs(forces[i,0]) + abs(forces[i,1]) + abs(forces[i,2]))
            torques_sum.append(abs(torques[i,0]) + abs(torques[i,1]) + abs(torques[i,2]))
        ax[0].plot(np.arange(len(forces_sum)), forces_sum, label=filename.split('.')[0])  # Plot the chart 
        ax[1].plot(np.arange(len(torques_sum)), torques_sum, label=filename.split('.')[0])  # Plot the chart 
plt.legend()
plt.show()