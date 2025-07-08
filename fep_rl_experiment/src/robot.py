#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from std_srvs.srv import Empty, SetBool
import tf2_ros
import tf2_geometry_msgs


class Robot:
    def __init__(self):
        self._desired_ee_pose_pub = rospy.Publisher(
            "/cmd_pose", PoseStamped, queue_size=1
        )
        self._gripper_pub = rospy.Publisher("/cmd_gripper", JointState, queue_size=1)
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_sub = rospy.Subscriber(
            "/camera/image_raw", Image, self.image_callback
        )
        self.ee_pose_sub = rospy.Subscriber(
            "/ee_pose", PoseStamped, self.ee_pose_callback
        )
        # Setup reset service
        self.reset_service = rospy.Service(
            "/reset_controller",
            Empty,
            self.reset_service_cb,
        )
        self.start_service = rospy.Service(
            "/start_controller",
            SetBool,
            self.start_service_cb,
        )
        self._action_scale = 0.005
        self.current_tip_pos = None
        # FIXME (yarden): get this value from the mujoco environment
        self.goal_tip_quat = None

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")

    def ee_pose_callback(self, msg: PoseStamped):
        pos = msg.pose.position
        self.current_tip_pos = np.array([pos.x, pos.y, pos.z])
        # Update internal orientation too if needed
        ori = msg.pose.orientation
        self.current_tip_quat = np.array([ori.x, ori.y, ori.z, ori.w])

    def get_camera_image(self) -> np.ndarray:
        return self.latest_image
    
    def start_service_cb(self, req):
        self._running = req.data
        return True, "Started controller."
        
    def reset_service_cb(self, req):
        """Resets the controller."""
        # FIXME (yarden): get this from the mujoco environment
        target_pose = PoseStamped()
        target_pose.header.frame_id = "panda_link0"
        target_pose.header.stamp = rospy.Time.now()
        target_pose.pose.position.x = 0.476
        target_pose.pose.position.y = 0.040
        target_pose.pose.position.z = 0.15
        target_pose.pose.orientation.x = -0.675
        target_pose.pose.orientation.y = 0.737
        target_pose.pose.orientation.z = 0.00
        target_pose.pose.orientation.w = 0.000
        self._desired_ee_pose_pub.publish(target_pose)
        self._running = False
        return []

    def act(self, action: np.ndarray) -> np.ndarray:
        """
        action: np.array of shape (4,) -> [dx, dy, dz, gripper]
        dx, dy, dz in range [-1, 1], scaled by action_scale
        gripper: <0 means close, >=0 means open
        """

        # Scale and apply limits
        delta_pos = action[:3] * self._action_scale
        new_tip_pos = self.current_tip_pos + delta_pos

        # Clip new tip position within safe workspace bounds
        new_tip_pos[0] = np.clip(new_tip_pos[0], 0.25, 0.77)
        new_tip_pos[1] = np.clip(new_tip_pos[1], -0.32, 0.32)
        new_tip_pos[2] = np.clip(new_tip_pos[2], 0.02, 0.5)

        # Keep orientation fixed (e.g., downward)
        quat = self.goal_tip_quat

        # Publish EE pose
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "panda_link0"
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.pose.position.x = new_tip_pos[0]
        pose_msg.pose.position.y = new_tip_pos[1]
        pose_msg.pose.position.z = new_tip_pos[2]
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        self._desired_ee_pose_pub.publish(pose_msg)

        # Gripper action
        gripper_close = action[3] < 0  # if < 0 → close
        gripper_pos = 0.0 if gripper_close else 1.0

        gripper_msg = JointState()
        gripper_msg.header.stamp = rospy.Time.now()
        gripper_msg.position = [gripper_pos]
        self._gripper_pub.publish(gripper_msg)
        # Update internal state (assume target reached)
        return new_tip_pos

    def get_end_effector_pos(self) -> np.ndarray:
        return self.current_tip_pos


class BoxInteractor:
    def __init__(self):
        self.robot = Robot()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Assume pose of box published in camera frame
        self.box_pose_sub = rospy.Subscriber(
            "/box_pose_cam", PoseStamped, self.box_pose_callback
        )
        self.latest_box_pose_global = None

    def box_pose_callback(self, msg: PoseStamped):
        try:
            # Transform pose from camera frame to base frame (e.g., panda_link0)
            transform = self.tf_buffer.lookup_transform(
                "panda_link0",  # target frame
                msg.header.frame_id,  # source frame
                rospy.Time(0),
                rospy.Duration(1.0),
            )

            pose_transformed = tf2_geometry_msgs.do_transform_pose(msg, transform)
            self.latest_box_pose_global = pose_transformed
            rospy.loginfo(f"Box pose in base frame: {pose_transformed.pose.position}")

        except Exception as e:
            rospy.logwarn(f"Transform failed: {e}")
