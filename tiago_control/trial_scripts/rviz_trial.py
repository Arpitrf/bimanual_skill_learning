#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Vector3
import numpy as np 

def publish_line_marker(axis, q):
    rospy.init_node('line_marker_publisher', anonymous=True)
    marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz

    rospy.sleep(1)

    delete_all_marker = Marker()
    delete_all_marker.action = Marker.DELETEALL
    marker_pub.publish(delete_all_marker)

    rospy.sleep(1)

    # while not rospy.is_shutdown():
    marker = Marker()
    marker.id = 1
    marker.header.frame_id = "base_footprint"  # Set your desired frame_id
    marker.header.stamp = rospy.Time.now()
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.02  # Line width

    # Start and end points of the line
    length = 2
    # axis, q = np.array([-0.101547,  -0.0114104,  0.9947653]), np.array([0.66742979, -0.14872666,  0.10002639])
    # axis = np.array([0,0,1])
    point_1 = q + length * axis
    point_2 = q - length * axis

    start_point = Point()
    start_point.x, start_point.y, start_point.z = point_1[0], point_1[1], point_1[2]
    end_point = Point()
    end_point.x, end_point.y, end_point.z = point_2[0], point_2[1], point_2[2]

    marker.points.append(start_point)
    marker.points.append(end_point)

    marker.color.a = 1.0  # Alpha
    marker.color.r = 0.0  # Red
    marker.color.g = 0.0  # Green
    marker.color.b = 1.0  # Blue

    marker_pub.publish(marker)

    rospy.sleep(1)
    # Add a marker for the object location
    marker = Marker()
    marker.id = 2
    marker.header.frame_id = "base_footprint"  # Set your desired frame_id
    marker.header.stamp = rospy.Time.now()
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.scale.x = 0.05  # Line width
    marker.scale.y = 0.05  # Line width
    marker.scale.z = 0.05  # Line width
    marker.pose.position.x = 0.535
    marker.pose.position.y = -0.165
    marker.pose.position.z = 0.736

    marker.color.a = 1.0  # Alpha
    marker.color.r = 0.0  # Red
    marker.color.g = 0.0  # Green
    marker.color.b = 1.0  # Blue

    marker_pub.publish(marker)

    # rate.sleep()

if __name__ == '__main__':
    try:
        # Set the point and direction of the line
        line_point = Point(x=1.0, y=1.0, z=0.0)
        line_direction = Vector3(x=1.0, y=0.0, z=0.0)

        publish_line_marker(line_point, line_direction)
    except rospy.ROSInterruptException:
        pass