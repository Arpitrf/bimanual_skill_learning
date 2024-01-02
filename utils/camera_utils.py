import cv2
import numpy as np
from cv_bridge import CvBridge
from real_tiago.utils.ros_utils import Listener
from sensor_msgs.msg import Image
from threading import Thread
import imageio
import time
import matplotlib.pyplot as plt

def img_processing(data):
    br = CvBridge()
    img = cv2.cvtColor(br.imgmsg_to_cv2(data), cv2.COLOR_BGR2RGB)
    return np.asarray(img).astype(int)

def depth_processing(data):
    br = CvBridge()
    img = br.imgmsg_to_cv2(data)
    return np.expand_dims(np.asarray(img), -1).astype(int)

# Handle when no depth is available
class Camera:

    def __init__(self, img_topic, depth_topic, img_post_proc_func=None, depth_post_proc_func=None, *args, **kwargs) -> None:
        self.img_topic = img_topic
        self.depth_topic = depth_topic

        self.img_listener = Listener(
                            input_topic_name=self.img_topic,
                            input_message_type=Image,
                            post_process_func=img_processing if img_post_proc_func is None else img_post_proc_func
                        )
        self.depth_listener = Listener(
                                input_topic_name=self.depth_topic,
                                input_message_type=Image,
                                post_process_func=depth_processing if depth_post_proc_func is None else depth_post_proc_func
                            )
        
        self._img_shape  = self.img_listener.get_most_recent_msg().shape
        self._depth_shape = self.depth_listener.get_most_recent_msg().shape

    def get_img(self):
        return self.img_listener.get_most_recent_msg()
    
    def get_depth(self):
        return self.depth_listener.get_most_recent_msg()

    def get_camera_obs(self):
        return {
            'color': self.get_img().astype(np.uint8),
            'depth': self.get_depth(),
        }
    
    @property
    def img_shape(self):
        return self._img_shape

    @property
    def depth_shape(self):
        return self._depth_shape
    

class RecordVideo:

    def __init__(self,
                 camera_interface_top,
                 camera_interface_side,
                 camera_interface_ego) -> None:
        self.recording = False
        self.env_video_frames = {}
        self.env_video_frames['top'] = []
        self.env_video_frames['side'] = []
        self.env_video_frames['ego'] = []
        self.camera_interface_top = camera_interface_top
        self.camera_interface_side = camera_interface_side
        self.camera_interface_ego = camera_interface_ego

    def reset_frames(self):
        self.env_video_frames['top'] = []
        self.env_video_frames['side'] = []
        self.env_video_frames['ego'] = []
    
    def setup_thread(self, target):
        print('SETUP THREAD', target)
        # print(list(map(lambda t:t.name,threading.enumerate())))
        thread = Thread(target=target)
        thread.daemon = True
        thread.start()
        print('started', thread.name)
        return thread

    def record_video_daemon_fn(self):
        counter = 0
        print("IN Daemon self.recording ", self.recording)
        while self.recorder_on:
            while self.recording:
                # if counter % 1000 == 0:
                time.sleep(0.1)
                top_view = self.camera_interface_top.get_camera_obs()
                side_view = self.camera_interface_side.get_camera_obs()
                ego_view = self.camera_interface_ego.get_camera_obs()
                capture_top = top_view["color"]
                capture_side = side_view["color"]
                capture_ego = ego_view["color"]
                self.env_video_frames['top'].append(cv2.cvtColor(capture_top.copy(), cv2.COLOR_BGR2RGB))
                self.env_video_frames['side'].append(cv2.cvtColor(capture_side.copy(), cv2.COLOR_BGR2RGB))
                self.env_video_frames['ego'].append(cv2.cvtColor(capture_ego.copy(), cv2.COLOR_BGR2RGB))
                # cv2.imshow("", cv2.cvtColor(capture.copy(), cv2.COLOR_BGR2RGB))
                # cv2.waitKey(10)
                # if counter % 100000 == 0:
                #     cv2.imwrite(f'temp/{counter}.jpg', top_view)
                # counter += 1
                # print("counter: ", counter)

    def setup_recording(self):
        print("--------------self.recording: ", self.recording)
        if self.recording:
            return
        self.recorder_on = True
        self.recording_daemon = self.setup_thread(
            target=self.record_video_daemon_fn)

    def start_recording(self):
        self.recording = True
    
    def pause_recording(self):
        self.recording = False

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.recorder_on = False
        self.recording_daemon.join()
        self.recording_daemon = None

    def save_video(self, save_folder, args):
        # print("self.env_video_frames.items(): ", self.env_video_frames.items())
        for key, frames in self.env_video_frames.items():
            if len(frames) == 0:
                continue
            print("len of frames: ", len(frames))

            path = f'{save_folder}/{args.f_name}_{key}.mp4'
            with imageio.get_writer(path, mode='I', fps=10) as writer: # originally 24
                for frame in frames:
                    writer.append_data(frame)