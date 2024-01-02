import gym
import time
import numpy as np
import rospy
from threading import Lock
from collections import OrderedDict
from tiago_control.transformations import euler_to_quat, quat_to_euler, add_angles

import tf
from std_msgs.msg import Header
from control_msgs.msg  import JointTrajectoryControllerState
from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped, WrenchStamped 
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String, Bool


class TiagoGym(gym.Env):

    def __init__(self, frequency=10, right_arm_enabled=True, left_arm_enabled=True, right_gripper_type=None, left_gripper_type=None):
        super(TiagoGym).__init__()

        self.frequency = frequency
        self.right_arm_enabled = right_arm_enabled
        self.left_arm_enabled = left_arm_enabled
        self.right_gripper_enabled = right_gripper_type is not None
        self.left_gripper_enabled = left_gripper_type is not None

        self.tiago = Tiago(right_arm_enabled=right_arm_enabled, left_arm_enabled=left_arm_enabled, right_gripper_type=right_gripper_type, left_gripper_type=left_gripper_type)

    @property
    def observation_space(self):
        ob_space = OrderedDict()

        ob_space['right_eef'] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3+4,),
        )

        ob_space['left_eef'] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3+4,),
        )

        ob_space['right_gripper'] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
        )

        ob_space['left_gripper'] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
        )

        return gym.spaces.Dict(ob_space)

    @property
    def action_space(self):
        act_space = OrderedDict()
        
        if self.right_arm_enabled:
            act_space['right'] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3+4+int(self.right_gripper_enabled)),
            )

        if self.left_arm_enabled:
            act_space['left'] = gym.spaces.Box(
                low=-np.insef,
                high=np.inf,
                shape=(3+4+int(self.left_gripper_enabled)),
            )

        return gym.spaces.Dict(act_space)


    def _observation(self):
        proprio = {
            'right_eef': self.tiago.right_arm_pose,
            'left_eef': self.tiago.left_arm_pose,
            'right_gripper': self.tiago.right_gripper_pos,
            'left_gripper': self.tiago.left_gripper_pos
        }

        return proprio

    def reset(self):
        self.start_time = None
        self.end_time = None
        
        return self._observation()
    
    def step(self, action):
        
        if action is not None:
            self.tiago.step(action)
        
        self.end_time = time.time()
        if self.start_time is not None:
            # print('Idle time:', 1/self.frequency - (self.end_time-self.start_time))
            rospy.sleep(max(0., 1/self.frequency - (self.end_time-self.start_time)))
        self.start_time = time.time()

        obs = self._observation()
        rew = 0
        done = False
        info = {}

        return obs, rew, done, info

class Tiago:

    def __init__(self, right_arm_enabled=True, left_arm_enabled=True, right_gripper_type=None, left_gripper_type=None):
        self.right_arm_enabled = right_arm_enabled
        self.left_arm_enabled = left_arm_enabled
        self.right_gripper_type = right_gripper_type
        self.left_gripper_type= left_gripper_type

        self.gripper = {'right': None, 'left': None}
        if self.right_gripper_type == 'pal':
            self.gripper['right'] = PALGripper('right')
        elif self.right_gripper_type == 'robotiq':
            self.gripper['right'] = RobotiqGripper('right')
        elif self.right_gripper_type is not None:
            raise NotImplementedError

        if self.left_gripper_type == 'pal':
            self.gripper['left'] = PALGripper('left')
        elif self.left_gripper_type == 'robotiq':
            self.gripper['left'] = RobotiqGripper('left')            
        elif self.left_gripper_type is not None:
            raise NotImplementedError

        self.setup_listeners()
        self.setup_actors()
    
    def setup_listeners(self):
        self.tiago_pose_reader = TiagoPoseListener('/base_footprint')

        def process_joint_pos(message):
            return list(message.actual.positions)
        # listeners for joint positions for each arm
        self.left_arm_joint_pos_reader = Listener(input_topic_name=f'/arm_left_controller/state', input_message_type=JointTrajectoryControllerState, post_process_func=process_joint_pos)
        self.right_arm_joint_pos_reader = Listener(input_topic_name=f'/arm_right_controller/state', input_message_type=JointTrajectoryControllerState, post_process_func=process_joint_pos)
        self.torso_pos_reader = Listener(input_topic_name=f'/torso_controller/state', input_message_type=JointTrajectoryControllerState, post_process_func=process_joint_pos)
    
    def setup_actors(self):
        self.tiago_pose_writer = {'right': None, 'left': None}
        self.tiago_joint_pos_writer = {'right': None, 'left': None, 'arm': None}
        self.tiago_joint_pos_writer['torso'] = Publisher('/torso_controller/command', JointTrajectory)
        if self.right_arm_enabled:
            self.tiago_pose_writer['right'] = Publisher('/whole_body_kinematic_controller/arm_right_tool_link_goal', PoseStamped)
            self.tiago_joint_pos_writer['right'] = Publisher('/arm_right_controller/command', JointTrajectory)
        if self.left_arm_enabled:
            self.tiago_pose_writer['left'] = Publisher('/whole_body_kinematic_controller/arm_left_tool_link_goal', PoseStamped)
            self.tiago_joint_pos_writer['left'] = Publisher('/arm_left_controller/command', JointTrajectory)

        self.head_writer = Publisher('/whole_body_kinematic_controller/gaze_objective_xtion_optical_frame_goal', PoseStamped)

        
    @property
    def get_camera_extrinsic(self):
        pos, quat = self.tiago_pose_reader.get_pose('/xtion_rgb_optical_frame')
        print("RGB EXTR: pos, quat: ", pos, quat)
        return pos, quat

    @property
    def get_depth_camera_extrinsic(self):
        pos, quat = self.tiago_pose_reader.get_pose('/xtion_depth_optical_frame')
        print(" DEPTH EXTR: pos, quat: ", pos, quat)
        return pos, quat
    
    @property
    def right_arm_pose(self):
        pos, quat = self.tiago_pose_reader.get_pose('/arm_right_tool_link')

        if pos is None:
            return None
        return np.concatenate((pos, quat))

    @property
    def left_arm_pose(self):
        pos, quat = self.tiago_pose_reader.get_pose('/arm_left_tool_link')

        if pos is None:
            return None
        return np.concatenate((pos, quat))

    @property
    def right_gripper_pos(self):
        if self.gripper['right'] is None:
            return None
        return self.gripper['right'].get_state()

    @property
    def left_gripper_pos(self):
        if self.gripper['left'] is None:
            return None
        return self.gripper['left'].get_state()

    def create_pose_command(self, trans, quat):
        header = Header(stamp=rospy.Time.now(), frame_id='/base_footprint')
        pose = Pose(position=Point(trans[0], trans[1], trans[2]), orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]))
        message = PoseStamped(header=header, pose=pose)

        return message

    def create_joint_pos_command(self, joint_pos, robot_part='arm', hand_side='left', traj_time=12):
        header = Header()        
        if robot_part == 'arm':
            joint_names  = [f'arm_{hand_side}_1_joint', f'arm_{hand_side}_2_joint', f'arm_{hand_side}_3_joint', f'arm_{hand_side}_4_joint', f'arm_{hand_side}_5_joint', f'arm_{hand_side}_6_joint', f'arm_{hand_side}_7_joint']
        elif robot_part == 'torso':
            joint_names = ['torso_lift_joint']
        joint_points = [JointTrajectoryPoint(positions=joint_pos, time_from_start=rospy.Duration(traj_time))]
        message = JointTrajectory(header=header, joint_names=joint_names, points=joint_points)

        return message


    def process_action(self, action, arm):

        # convert deltas to absolute positions
        pos_delta, euler_delta, gripper = action[:3], action[3:6], action[6]

        cur_pose = self.right_arm_pose if arm=='right' else self.left_arm_pose
        cur_pos, cur_euler = cur_pose[:3], quat_to_euler(cur_pose[3:])
        
        target_pos = cur_pos + pos_delta
        target_euler = add_angles(euler_delta, cur_euler)
        target_quat = euler_to_quat(target_euler)    

        return target_pos, target_quat, gripper # pos, quat, gripper

    def step(self, action):

        for arm in ['right', 'left']:
            if action[arm] is None:
                continue

            pos, quat, gripper_act = self.process_action(action[arm], arm)

            # print(arm, pos, quat, gripper)
            pose_command = self.create_pose_command(pos, quat)
            if self.tiago_pose_writer[arm] is not None:
                self.tiago_pose_writer[arm].write(pose_command)
            if self.gripper[arm] is not None:
                self.gripper[arm].write(gripper_act)

class TiagoPoseListener:
    
    def __init__(self, base_link):
        self.listener = tf.TransformListener()    
        self.base_link = base_link


    def get_transform(self, rel_link, base_link):
        try:
            (trans, rot) = self.listener.lookupTransform(base_link, rel_link, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            # print('TF Connecting...')
            trans, rot = None, None

        return trans, rot

    def get_pose(self, link):
        trans, rot = self.get_transform(link, self.base_link)
        # print("trans, rot: ", trans, rot)
        return trans, rot

class RobotiqGripper:

    def __init__(self, side):
        self.side = side
        self.gripper_min = 0.0
        self.gripper_max = 0.7

        def process_gripper(message):
            return message.actual.positions[0]
        
        self.gripper_reader = Listener(input_topic_name=f'/gripper_{self.side}_controller/state', input_message_type=JointTrajectoryControllerState, post_process_func=process_gripper)
        self.gripper_writer = Publisher(f'/gripper_{self.side}_controller/command', JointTrajectory)
        self.grasp_status = Listener(input_topic_name=f'/gripper_{self.side}/is_grasped', input_message_type=Bool)

    def get_state(self):
        dist =  self.gripper_reader.get_most_recent_msg()
        # print(f"in get gripper state for {self.side}: ", dist)

        # normalize gripper state and return
        return (dist - self.gripper_min)/(self.gripper_max - self.gripper_min)

    def create_gripper_command(self, dist):
        message = JointTrajectory()
        message.header = Header()
        message.joint_names = ['gripper_right_finger_joint']
        point = JointTrajectoryPoint(positions=[dist], time_from_start = rospy.Duration(1))
        message.points.append(point)
        
        return message

    def write(self, gripper_act):
        # unnormalize gripper action
        gripper_act = gripper_act*(self.gripper_max - self.gripper_min) + self.gripper_min
        # print("gripper_act: ", gripper_act)
        gripper_cmd = self.create_gripper_command(gripper_act)

        self.gripper_writer.write(gripper_cmd)

    def is_grasped(self):
        is_grasped = self.grasp_status.get_most_recent_msg()
        return is_grasped.data


class PALGripper:

    def __init__(self, side):
        self.side = side
        print("side: ", side)

        self.gripper_min = 0.0
        self.gripper_max = 0.09
        
        def process_gripper(message):
            return message.actual.positions[0]

        self.gripper_reader = Listener(input_topic_name=f'/parallel_gripper_{self.side}_controller/state', input_message_type=JointTrajectoryControllerState, post_process_func=process_gripper)
        self.gripper_writer = Publisher(f'/parallel_gripper_{self.side}_controller/command', JointTrajectory)

    def get_state(self):
        dist =  self.gripper_reader.get_most_recent_msg()
        return dist
        # print(f"in get gripper state for {self.side}: ", dist)

        # normalize gripper state and return
        # return (dist - self.gripper_min)/(self.gripper_max - self.gripper_min)

    def create_gripper_command(self, dist):
        message = JointTrajectory()
        message.header = Header()
        message.joint_names = ['parallel_gripper_joint']
        point = JointTrajectoryPoint(positions=[dist], time_from_start = rospy.Duration(1))
        message.points.append(point)
        
        return message

    def write(self, gripper_act):
        # unnormalize gripper action
        gripper_act = gripper_act*(self.gripper_max - self.gripper_min) + self.gripper_min
        gripper_cmd = self.create_gripper_command(gripper_act)

        self.gripper_writer.write(gripper_cmd)

    def is_grasped(self):
        gripper_pos = self.get_state()
        print("gripper_pos: ", gripper_pos)
        if gripper_pos < 0.015:
            return False
        else:
            return True


class Publisher:

    def __init__(self, pub_name, pub_message_type, queue_size=5):
        self.publisher = rospy.Publisher(pub_name, pub_message_type, queue_size=queue_size)
    
    def write(self, message):
        self.publisher.publish(message)
        
class Listener:
    
    def __init__(self, input_topic_name, input_message_type, post_process_func=None):
        self.inputlock = Lock()
        self.input_topic_name = input_topic_name
        self.input_message_type = input_message_type
        self.post_process_func = post_process_func

        self.most_recent_message = None
        self.init_listener()

    def callback(self, data):
        with self.inputlock:
            self.most_recent_message = data
    
    def init_listener(self):
        rospy.Subscriber(self.input_topic_name, self.input_message_type, self.callback)
    
    def get_most_recent_msg(self):
        while self.most_recent_message is None:
            print(f'Waiting for topic {self.input_topic_name} to publish.')
            rospy.sleep(0.02)
        
        data = self.most_recent_message if self.post_process_func is None else self.post_process_func(self.most_recent_message)
        return data
    
class ForceTorqueSensor(object):
    def __init__(self):
        self.last_msg = None
        self.ft_reader = Listener(input_topic_name=f'/wrist_right_ft/corrected', input_message_type=WrenchStamped)
        
        # rospy.loginfo(
        #     "Subscribed to: '" + str(self.force_torque_sub.resolved_name) + "' topic.")

    # def force_torque_cb(self, msg):
    #     """
    #     :type msg: WrenchStamped
    #     """
    #     self.last_msg = msg
    #     # print("msg: ", msg)

    def run(self):
        """Show information on what was found in the joint states current"""
        rospy.loginfo("Waiting for first WrenchStamped message...")
        while not rospy.is_shutdown() and self.last_msg is None:
            # print(self.last_msg)
            rospy.sleep(0.2)

        print("---------------- out of first loop -------------")

        # Check at a 5Hz rate to not spam
        r = rospy.Rate(2)
        while not rospy.is_shutdown():
            self.do_stuff_with_last_msg()
            r.sleep()

    def do_stuff_with_last_msg(self):
        """Print funny sentences about what we can guess of our status thanks to the force torque sensor"""
        # self.last_msg = WrenchStamped()  # line for autocompletion pourposes
        f = self.last_msg.wrench.force
        t = self.last_msg.wrench.torque
        print("force, torque: ", [f.x, f.y, f.z], [t.x, t.y, t.z])
        total_torque = abs(t.x) + abs(t.y) + abs(t.z)
        if f.z > 10.0:
            rospy.loginfo("Looks someone is pulling from my hand!")
        elif f.z < -10.0:
            rospy.loginfo("Looks someone is pushing my hand!")
        elif total_torque > 2.0:
            rospy.loginfo("Hey, why are you twisting my hand? :(")
        else:
            rospy.loginfo(
                "Do something with my hand and I'll tell you what I feel")


def create_gripper_command(dist):
    message = JointTrajectory()
    message.header = Header()
    message.joint_names = ['parallel_gripper_joint']
    point = JointTrajectoryPoint(positions=[dist], time_from_start = rospy.Duration(1))
    message.points.append(point)
    
    return message

if __name__=='__main__':
    rospy.init_node('tiago_test')

    pub = Publisher('/parallel_gripper_right_controller/command', JointTrajectory)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        command = create_gripper_command(0.09)
        pub.write(command)
    rate.sleep()