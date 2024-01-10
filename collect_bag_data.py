import time
import cv2
import rospy
import numpy as np
from real_tiago.utils.ros_utils import Listener
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import matplotlib.pyplot as plt
import pickle
from tiago_control.tiago_gym import TiagoGym, Listener
from scipy.spatial.transform import Rotation as R


def get_intr():
    pass

def img_processing(data):
    br = CvBridge()
    img = cv2.cvtColor(br.imgmsg_to_cv2(data), cv2.COLOR_BGR2RGB)
    return img

def depth_processing(data):
    br = CvBridge()
    img = br.imgmsg_to_cv2(data)
    return img

if __name__=='__main__':
    rospy.init_node('tiago_test')
    env = TiagoGym(frequency=10, right_arm_enabled=True, left_arm_enabled=True, right_gripper_type='robotiq', left_gripper_type='pal')


    img_topic = "/xtion/rgb/image_raw" #"/camera/color/image_raw"
    img_topic2 =  "/top_down/color/image_raw"

    img_listener = Listener(
                            input_topic_name=img_topic,
                            input_message_type=Image,
                            post_process_func=img_processing
                        )
    img_listener2 = Listener(
                            input_topic_name=img_topic2,
                            input_message_type=Image,
                            post_process_func=img_processing
                        )

    r = rospy.Rate(10)
    counter = 1
    save_path_1 = '/home/pal/arpit/kalibr/data4/cam0/'
    save_path_2 = '/home/pal/arpit/kalibr/data4/cam1/'
    os.makedirs(save_path_1, exist_ok=True)
    os.makedirs(save_path_2, exist_ok=True)

    rospy.sleep(1)
    time.sleep(4)
    print("================STARTING CAPTURE===================")

    while not rospy.is_shutdown():
        color_img = img_listener.get_most_recent_msg()
        color_img2 = img_listener2.get_most_recent_msg()
        # print('img', img.shape)
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(color_img)
        # ax[1].imshow(color_img2)
        # plt.show()        
        # save to disk

        # print(time.time_ns())
        f_name = time.time_ns()
        cv2.imwrite(f'{save_path_1}{f_name}.png', color_img) 
        cv2.imwrite(f'{save_path_2}{f_name}.png', color_img2) 

        counter += 1
        time.sleep(0.1)
        # cv2.waitKey(10)
        # r.sleep()