import time
import rospy
import numpy as np
from tiago_gym import TiagoGym, Listener
from control_msgs.msg  import JointTrajectoryControllerState
from scipy.spatial.transform import Rotation as R

rospy.init_node('tiago_test')
env = TiagoGym(frequency=10, right_arm_enabled=True, left_arm_enabled=True, right_gripper_type='robotiq', left_gripper_type='pal')

rospy.sleep(2)

# target
# pos = np.array([ 0.66540788, -0.08981783,  0.93672317])
# quat= np.array([ -0.72167855,  0.01482957,  0.68898529, 0.06526424])
# pose_command = env.tiago.create_pose_command(pos, quat)
# env.tiago.tiago_pose_writer['right'].write(pose_command)

# home
pos = np.array([-0.02927815, -0.44696712,  1.11367162])
quat = np.array([-0.62445749, -0.55287778, -0.39586573,  0.38427767])
pose_command = env.tiago.create_pose_command(pos, quat)
env.tiago.tiago_pose_writer['right'].write(pose_command)
