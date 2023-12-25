import numpy as np
import rospy
import tf
from threading import Lock
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped

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

class TiagoPoseListener:
    
    def __init__(self, base_link):
        import tf
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
        return trans, rot


def create_tiago_command(trans, quat):
    header = Header(stamp=rospy.Time.now(), frame_id='/base_footprint')
    pose = Pose(position=Point(trans[0], trans[1], trans[2]), orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]))
    pose_stamped = PoseStamped(header=header, pose=pose)

    return pose_stamped
    
if __name__=='__main__':
    rospy.init_node('tiago_test')

    tiago_pose = TiagoPoseListener('/base_footprint') 
    while not rospy.is_shutdown():
        link = input('Enter link name:')
        if link == 'q':
            break
        trans, rot = tiago_pose.get_pose(link)
        if trans is not None:
            print((np.array(trans)*1000).astype(int)/1000, (np.array(rot)*1000).astype(int)/1000)
    






