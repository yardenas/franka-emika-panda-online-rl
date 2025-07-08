#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from std_srvs.srv import Empty, SetBool
import tf2_ros
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation as R


class Robot:
    def __init__(self, init_node=False):
        if init_node:
            rospy.init_node("franka_emika_robot_interface")
        self._desired_ee_pose_pub = rospy.Publisher(
            "/cartesian_impedance_example_controller/equilibrium_pose",
            PoseStamped,
            queue_size=1,
        )
        self._gripper_pub = rospy.Publisher("cmd_gripper", JointState, queue_size=1)
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_sub = rospy.Subscriber(
            "/camera/image_raw", Image, self.image_callback, queue_size=1
        )
        self.ee_pose_sub = rospy.Subscriber(
            "/ee_pose", PoseStamped, self.ee_pose_callback, queue_size=1
        )
        # Setup reset service
        self.reset_service = rospy.Service(
            "reset_controller",
            Empty,
            self.reset_service_cb,
        )
        self.start_service = rospy.Service(
            "start_controller",
            SetBool,
            self.start_service_cb,
        )
        self._action_scale = 0.005
        self.current_tip_pos = None
        self.goal_tip_quat = np.array(
            [
                [9.9849617e-01, 9.4118714e-04, 5.4812428e-02, 6.6105318e-01],
                [1.0211766e-03, -9.9999845e-01, -1.4304515e-03, -5.1778345e-04],
                [5.4810949e-02, 1.4842749e-03, -9.9849564e-01, 1.7906836e-01],
                [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
            ]
        )
        self.start_pos = np.array([6.6105318e-01, -5.1778345e-04, 1.7906836e-01])

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
        rospy.loginfo("Resetting robot...")
        target_pose = PoseStamped()
        target_pose.header.frame_id = "panda_link0"
        target_pose.header.stamp = rospy.Time.now()
        target_pose.pose.position.x = float(self.start_pos[0])
        target_pose.pose.position.y = float(self.start_pos[1])
        target_pose.pose.position.z = float(self.start_pos[2])

        # Extract rotation matrix (top-left 3x3 of the matrix)
        rotation_matrix = self.goal_tip_quat[:3, :3]

        # Convert rotation matrix to quaternion
        quat = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]

        # Set orientation from quaternion
        target_pose.pose.orientation.x = quat[0]
        target_pose.pose.orientation.y = quat[1]
        target_pose.pose.orientation.z = quat[2]
        target_pose.pose.orientation.w = quat[3]
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
