import time
import cv2
import rospy
import numpy as np
from real_tiago.user_interfaces.oculus_control import VRPolicy
from real_tiago.utils.ros_utils import Listener
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import matplotlib.pyplot as plt
import pickle
from tiago_gym import TiagoGym, Listener
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
    depth_topic = "/xtion/depth/image_raw" #/camera/aligned_depth_to_color/image_raw"

    # img_topic =  "/camera/color/image_raw"
    # depth_topic = "/camera/aligned_depth_to_color/image_raw"


    img_listener = Listener(
                            input_topic_name=img_topic,
                            input_message_type=Image,
                            post_process_func=img_processing
                        )
    depth_listener = Listener(
                            input_topic_name=depth_topic,
                            input_message_type=Image,
                            post_process_func=depth_processing
                        )

    r = rospy.Rate(10)
    counter = 1
    save_path = '/home/pal/arpit/tiago_teleop/tiago_control/data/tiago_full_pipeline_3/'
    os.makedirs(save_path+'color_img', exist_ok=True)
    os.makedirs(save_path+'depth_img', exist_ok=True)
    os.makedirs(save_path+'depth', exist_ok=True)

    rospy.sleep(2)
    # write the extrinsic to file
    extr_position, extr_quat = env.tiago.get_camera_extrinsic
    extr_rotation = R.from_quat(extr_quat).as_matrix()
    R_world_cam = extr_rotation
    T_world_cam = np.array([
        [R_world_cam[0][0], R_world_cam[0][1], R_world_cam[0][2], extr_position[0]],
        [R_world_cam[1][0], R_world_cam[1][1], R_world_cam[1][2], extr_position[1]],
        [R_world_cam[2][0], R_world_cam[2][1], R_world_cam[2][2], extr_position[2]],
        [0, 0, 0, 1]
    ])
    with open(f'{save_path}extrinsic.pickle', 'wb') as handle:
        pickle.dump(T_world_cam, handle, protocol=pickle.HIGHEST_PROTOCOL)

    time.sleep(10)
    print("================STARTING CAPTURE===================")

    while not rospy.is_shutdown():
        color_img = img_listener.get_most_recent_msg()
        # print('img', img.shape)
        depth_img = depth_listener.get_most_recent_msg()
        cv2.imshow('img', color_img)
        cv2.imshow('depth', depth_img)
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(color_img)
        # ax[1].imshow(depth_img)
        # plt.show()
        time.sleep(0.5)
        
        # save to disk
        with open(f'{save_path}depth/{counter:04d}.pickle', 'wb') as handle:
            pickle.dump(depth_img, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # final_frame = cv2.cvtColor(color_img.copy(), cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{save_path}color_img/{counter:04d}.jpg', color_img) 
        cv2.imwrite(f'{save_path}depth_img/{counter:04d}.jpg', depth_img)

        counter += 1
        cv2.waitKey(10)
        # r.sleep()

    
    
    
    
    
    
    
    
    
    
    # intr = [
    #     [910.143310546875, 0.0, 622.4730224609375], 
    #     [0.0, 910.373779296875, 383.2250061035156], 
    #     [0.0, 0.0, 1.0]
    # ]
    # intr = [
    #     [523.9963414139355, 0.0, 328.83202929614686], 
    #     [0.0, 524.4907272320442, 237.83703502879925], 
    #     [0.0, 0.0, 1.0]
    # ]

    # pix_x, pix_y = 0, 0
    # click_z = depth_img[pix_x, pix_y]
    # # click_z *= depth_scale
    # click_x = (pix_x-intr[0, 2]) * \
    #     click_z/intr[0, 0]
    # click_y = (pix_y-intr[1, 2]) * \
    #     click_z/intr[1, 1]
    # if click_z == 0:
    #     raise Exception('Invalid pick point')
    # # 3d point in camera coordinates
    # point_3d = np.asarray([click_x, click_y, click_z])

    # get 

    # from tiago_gym import TiagoGym
    
    # env = TiagoGym(frequency=10, head_enabled=True, right_arm_enabled=True, left_arm_enabled=True, right_gripper_type='robotiq', left_gripper_type='pal')
    # obs = env.reset()

    # vr = VRPolicy()
    # def shutdown_helper():
    #     vr.stop()
    
    # r = rospy.Rate(10)
    # while not rospy.is_shutdown():
    #     # print(obs)
    #     action = {'right': None, 'left': None}
    #     for arm in ['right', 'left']:
            
    #         eef = obs[f'{arm}_eef']
    #         if eef is None:
    #             continue
            
    #         cartesian = np.concatenate((eef[:3], eef[3:7]))
    #         gripper = obs[f'{arm}_gripper']
    #         vr_obs = {'cartesian_position': cartesian, 'gripper_position': gripper}
    #         print(arm, cartesian)
    #         action[arm] = vr.forward(vr_obs, arm=arm)

    #     # print(action)

    #     obs, _,  _, _ = env.step(action)

    # rospy.on_shutdown(shutdown_helper)