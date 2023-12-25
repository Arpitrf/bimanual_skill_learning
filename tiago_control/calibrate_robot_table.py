import time
import rospy
import numpy as np
from tiago_gym import TiagoGym, Listener, ForceTorqueSensor


rospy.init_node('tiago_test')
env = TiagoGym(frequency=10, right_arm_enabled=True, left_arm_enabled=True, right_gripper_type='robotiq', left_gripper_type='pal')

# get current joint positions
rospy.sleep(2)
left_joint_pos = env.tiago.left_arm_joint_pos_reader.get_most_recent_msg()
right_joint_pos = env.tiago.right_arm_joint_pos_reader.get_most_recent_msg()
print("left_joint_pos, right_joint_pos: ", left_joint_pos, right_joint_pos)

# # Close the gripper and Move both the hands up (to a safe location that won't hit the table when doing the next joint space motion)
# # left arm joint pos: (-0.7488993541855221, -0.5663182484775039, 1.1562073355934284, 1.4685980855665832, 0.34978101865231986, 1.3578788821947045, 0.8110035348258153)
# # right arm joint pos: (-0.8521065225804967, -0.7271015962739111, 1.4740042927285277, 1.5650019496804675, 0.11740734957885922, 1.338181628120021, -0.001991581630762629)

env.tiago.gripper['right'].write(0.95)
env.tiago.gripper['left'].write(0)

home_left_joint_pos = [-0.7488993541855221, -0.5663182484775039, 1.1562073355934284, 1.4685980855665832, 0.34978101865231986, 1.3578788821947045, 0.8110035348258153]
home_right_joint_pos = [-0.8521065225804967, -0.7271015962739111, 1.4740042927285277, 1.5650019496804675, 0.11740734957885922, 1.338181628120021, -0.001991581630762629]
left_joint_pos_command = env.tiago.create_joint_pos_command(home_left_joint_pos, hand_side='left')
right_joint_pos_command = env.tiago.create_joint_pos_command(home_right_joint_pos, hand_side='right')
rospy.sleep(2)
env.tiago.tiago_joint_pos_writer['left'].write(left_joint_pos_command)
rospy.sleep(10)
env.tiago.tiago_joint_pos_writer['right'].write(right_joint_pos_command)
rospy.sleep(10)

# Check the error in joint pos
rospy.sleep(2)
actual_left_joint_pos = np.array(env.tiago.left_arm_joint_pos_reader.get_most_recent_msg())
actual_right_joint_pos = np.array(env.tiago.right_arm_joint_pos_reader.get_most_recent_msg())
desired_left_joint_pos = np.array(home_left_joint_pos)
desired_right_joint_pos = np.array(home_right_joint_pos)
print("Error in left arm joints: ", abs(desired_left_joint_pos - actual_left_joint_pos))
print("Error in right arm joints: ", abs(desired_right_joint_pos - actual_right_joint_pos))

# TODO: IMPORTANT: Set the torso to the right height for this calibration: [0.15034621940730314]
torso_pos = [0.15034621940730314]
torso_pos_command = env.tiago.create_joint_pos_command(torso_pos, robot_part='torso', traj_time=5)
rospy.sleep(2)
env.tiago.tiago_joint_pos_writer['torso'].write(torso_pos_command)
rospy.sleep(10)
actual_torso_pos = np.array(env.tiago.torso_pos_reader.get_most_recent_msg())
desired_torso_pos = np.array(torso_pos)
# print("left: ", desired_left_joint_pos, actual_left_joint_pos)
# print("right: ", desired_right_joint_pos, actual_right_joint_pos)
print("Error in right arm joints: ", abs(desired_torso_pos - actual_torso_pos))




# Joint-space movement for both the hands
# left: [0.9755480219055362, -1.0662798984491249, 2.5921090981607326, 1.4303861172897268, -1.8388393618046137, 1.2454184359322085, -0.004053563356380074]
# right: [0.7500963030495943, -0.9857035076393257, 2.1205397511647672, 1.378958292008161, -2.054720882341285, 1.3853885969883581, 0.19737592909598867]

calibrate_pos_left = [0.9755480219055362, -1.0662798984491249, 2.5921090981607326, 1.4303861172897268, -1.8388393618046137, 1.2454184359322085, -0.004053563356380074]
calibrate_pos_right = [0.7500963030495943, -0.9857035076393257, 2.1205397511647672, 1.378958292008161, -2.054720882341285, 1.3853885969883581, 0.19737592909598867]
left_joint_pos_command = env.tiago.create_joint_pos_command(calibrate_pos_left, hand_side='left')
right_joint_pos_command = env.tiago.create_joint_pos_command(calibrate_pos_right, hand_side='right')
rospy.sleep(2)
env.tiago.tiago_joint_pos_writer['left'].write(left_joint_pos_command)
rospy.sleep(10)
env.tiago.tiago_joint_pos_writer['right'].write(right_joint_pos_command)
rospy.sleep(10)

# Check the error in joint pos
rospy.sleep(2)
actual_left_joint_pos = np.array(env.tiago.left_arm_joint_pos_reader.get_most_recent_msg())
actual_right_joint_pos = np.array(env.tiago.right_arm_joint_pos_reader.get_most_recent_msg())
desired_left_joint_pos = np.array(calibrate_pos_left)
desired_right_joint_pos = np.array(calibrate_pos_right)
# print("left: ", desired_left_joint_pos, actual_left_joint_pos)
# print("right: ", desired_right_joint_pos, actual_right_joint_pos)
print("Error in left arm joints: ", abs(desired_left_joint_pos - actual_left_joint_pos))
print("Error in right arm joints: ", abs(desired_right_joint_pos - actual_right_joint_pos))
