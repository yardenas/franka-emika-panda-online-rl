#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped

def publish_dummy_pose():
    pub = rospy.Publisher('pose', PoseStamped, queue_size=10)
    rospy.init_node('dummy_pose_publisher', anonymous=True)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "panda_link0"
        # Fill in dummy pose values
        pose_msg.pose.position.x = 0.66105
        pose_msg.pose.position.y = 0.
        pose_msg.pose.position.z = 0.05

        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0

        pub.publish(pose_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_dummy_pose()
    except rospy.ROSInterruptException:
        pass