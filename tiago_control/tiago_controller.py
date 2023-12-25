import numpy as np
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped
from control_msgs.msg import PointHeadActionGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def create_command(joint):
        message = JointTrajectory()
        message.header = Header()
        message.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint(positions=joint, time_from_start = rospy.Duration(0.2))
        message.points.append(point)
        
        return message

joints = [[0.4, 0.0], [-0.4, 0.0], [0.0, 0.3], [0.0, -0.3]]

rospy.init_node('tiago_oculus_control')

# h = Header(stamp=rospy.Time.now(), frame_id='/base_footprint')
# m = PointHeadActionGoal()
# m.header.frame_id = '/base_footprint'
# m.goal.target.header.frame_id = '/base_footprint' 
# m.goal.target.point = Point(1.0, 0.0, 0.5)

# publisher = rospy.Publisher('/head_controller/point_head_action/goal', PointHeadActionGoal, queue_size=5)
publisher = rospy.Publisher('/head_controller/command', JointTrajectory, queue_size=5)
r = rospy.Rate(10)
while not rospy.is_shutdown():

    l = input()
    m = create_command(joints[int(l)])
    print(m)
    publisher.publish(m)
    r.sleep()







