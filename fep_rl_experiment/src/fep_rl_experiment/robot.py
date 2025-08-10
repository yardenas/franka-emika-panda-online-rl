import rospy
import numpy as np
from collections import deque
import cv2
import actionlib
from franka_gripper.msg import HomingAction, HomingGoal
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from std_srvs.srv import Empty, SetBool
import tf2_ros


class Robot:
    def __init__(self, init_node=False):
        if init_node:
            rospy.init_node("franka_emika_robot_interface")
        self._desired_ee_pose_pub = rospy.Publisher(
            "/cartesian_impedance_example_controller/equilibrium_pose",
            PoseStamped,
            queue_size=1,
        )

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.image_callback, queue_size=1
        )
        self.gripper_command_client = actionlib.SimpleActionClient(
            "/franka_gripper/gripper_action", GripperCommandAction
        )
        self.gripper_homing_action = actionlib.SimpleActionClient(
            "/franka_gripper/homing", HomingAction
        )
        self.image_pub = rospy.Publisher("processed_image", Image, queue_size=1)
        self.ee_pose_sub = rospy.Subscriber(
            "/cartesian_impedance_example_controller/measured_pose",
            PoseStamped,
            self.ee_pose_callback,
            queue_size=1,
        )
        self.qpos_sub = rospy.Subscriber(
            "/joint_states", JointState, self.joint_state_callback, queue_size=1
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
        self.joint_state = None
        self.latest_image = None
        self.last_image_time = None
        self.last_tip_pos_time = None
        self.last_joint_state_time = None
        self.last_cube_time = None
        self.goal_tip_quat = np.array([-1.0, 0.0, 0.0, 0.0])
        self.start_pos = np.array([6.6105e-1, -0.05, 1.7906836e-01])
        self.ee_velocity_estimator = LinearVelocityEstimator()
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

    def image_callback(self, msg: Image):
        try:
            bgr_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            grayscale = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            image = _preprocess_image(grayscale)
            self.latest_image = image
            self.last_image_time = msg.header.stamp
            output_msg = self.bridge.cv2_to_imgmsg(
                (image * 255).astype(np.uint8), encoding="mono8"
            )
            output_msg.header.stamp = rospy.Time.now()  # Optional: add timestamp
            self.image_pub.publish(output_msg)
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")

    def ee_pose_callback(self, msg: PoseStamped):
        pos = msg.pose.position
        self.current_tip_pos = np.array([pos.x, pos.y, pos.z])
        # Update internal orientation too if needed
        ori = msg.pose.orientation
        self.current_tip_quat = np.array([ori.x, ori.y, ori.z, ori.w])
        self.ee_velocity_estimator.add_measurement(
            self.current_tip_pos, msg.header.stamp
        )
        self.last_tip_pos_time = msg.header.stamp

    def joint_state_callback(self, msg: JointState):
        self.joint_state = np.array(msg.position)
        self.last_joint_state_time = msg.header.stamp

    def get_camera_image(self) -> np.ndarray:
        return self.latest_image

    def get_joint_state(self):
        return self.joint_state

    def get_cube_pos(self, frame="panda_link0") -> np.ndarray:
        try:
            transformed_pose = self.tf_buffer.lookup_transform(
                frame, "aruco_cube_frame", rospy.Time(0)
            )
            pos = transformed_pose.transform.translation
            self.last_cube_time = transformed_pose.header.stamp
            return np.array([pos.x, pos.y, pos.z])
        except (
            tf2_ros.LookupException,
            tf2_ros.ExtrapolationException,
            tf2_ros.TransformException,
        ) as e:
            rospy.logerr(f"Transform error in get_cube_pos: {e}")
            return None

    def get_cube_quat(self, frame="panda_link0") -> np.ndarray:
        try:
            transformed_pose = self.tf_buffer.lookup_transform(
                frame, "aruco_cube_frame", rospy.Time(0)
            )
            quat = transformed_pose.transform.rotation
            return np.array([quat.x, quat.y, quat.z, quat.w])
        except (
            tf2_ros.LookupException,
            tf2_ros.ExtrapolationException,
            tf2_ros.TransformException,
        ) as e:
            rospy.logerr(f"Transform error in get_cube_quat: {e}")
            return None

    def start_service_cb(self, req):
        self._running = req.data
        return True, "Started controller."

    def reset_service_cb(self, req):
        """Resets the controller."""
        rospy.loginfo("Resetting robot...")
        goal = HomingGoal()
        self.gripper_homing_action.send_goal(goal)
        target_pose = PoseStamped()
        target_pose.header.frame_id = "panda_link0"
        target_pose.header.stamp = rospy.Time.now()
        target_pose.pose.position.x = float(self.start_pos[0])
        target_pose.pose.position.y = float(self.start_pos[1])
        target_pose.pose.position.z = float(self.start_pos[2])
        # Set orientation from quaternion
        target_pose.pose.orientation.x = float(self.goal_tip_quat[0])
        target_pose.pose.orientation.y = float(self.goal_tip_quat[1])
        target_pose.pose.orientation.z = float(self.goal_tip_quat[2])
        target_pose.pose.orientation.w = float(self.goal_tip_quat[3])
        self._desired_ee_pose_pub.publish(target_pose)
        self.gripper_homing_action.wait_for_result()
        self._running = False
        return []

    def act(self, action: np.ndarray) -> np.ndarray:
        """
        action: np.array of shape (4,) -> [dx, dy, dz, gripper]
        dx, dy, dz in range [-1, 1], scaled by action_scale
        gripper: <0 means close, >=0 means open
        """
        if not self.ok:
            rospy.logwarn("Not ready yet. Cannot execute action.")
            return
        # Scale and apply limits
        delta_pos = action[:3] * self._action_scale
        new_tip_pos = self.current_tip_pos + delta_pos

        # Clip new tip position within safe workspace bounds
        new_tip_pos[0] = np.clip(new_tip_pos[0], 0.25, 0.77)
        new_tip_pos[1] = np.clip(new_tip_pos[1], -0.32, 0.32)
        new_tip_pos[2] = np.clip(new_tip_pos[2], 0.02, 0.5)
        # Publish EE pose
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "panda_link0"
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.pose.position.x = float(self.start_pos[0])
        pose_msg.pose.position.y = float(new_tip_pos[1])
        pose_msg.pose.position.z = float(new_tip_pos[2])
        pose_msg.pose.orientation.x = float(self.goal_tip_quat[0])
        pose_msg.pose.orientation.y = float(self.goal_tip_quat[1])
        pose_msg.pose.orientation.z = float(self.goal_tip_quat[2])
        pose_msg.pose.orientation.w = float(self.goal_tip_quat[3])
        self._desired_ee_pose_pub.publish(pose_msg)

        if action[3] >= 0.0:
            goal = GripperCommandGoal()
            goal.command.position = 0.03
            goal.command.max_effort = 50.0
            self.gripper_command_client.send_goal(goal)
        elif action[3] < 0.0:
            goal = GripperCommandGoal()
            goal.command.position = 0.015
            goal.command.max_effort = 50.0
            self.gripper_command_client.send_goal(goal)
        return new_tip_pos

    def get_end_effector_pos(self) -> np.ndarray:
        return self.current_tip_pos

    @property
    def ok(self):
        ready = True
        if self.current_tip_pos is None:
            rospy.logwarn("current_tip_pos is None")
            ready = False
        if self.latest_image is None:
            rospy.logwarn("latest_image is None")
            ready = False
        if self.get_cube_pos() is None:
            rospy.logwarn("get_cube_pos() returned None")
            ready = False
        if self.joint_state is None:
            rospy.logwarn("joint_state is None")
            ready = False
        return ready

    @property
    def safe(self):
        pos = self.get_end_effector_pos()
        out_of_bounds = np.any(np.abs(pos - self.start_pos) > 0.3)
        if out_of_bounds:
            rospy.logwarn(
                f"Robot out of bounds. Position is: {self.get_end_effector_pos()}"
            )
        velocity = self.ee_velocity_estimator.estimate_velocity()
        high_velocity = np.any(np.abs(velocity) > 0.5)
        if high_velocity:
            rospy.logwarn(f"EE high velocity. Velocity is: {velocity}")
        return not out_of_bounds and not high_velocity


class LinearVelocityEstimator:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.positions = deque(maxlen=window_size)  # List of 3D position vectors
        self.timestamps = deque(maxlen=window_size)  # Corresponding timestamps

    def add_measurement(self, position, timestamp):
        self.positions.append(np.array(position))
        self.timestamps.append(float(timestamp.to_sec()))

    def estimate_velocity(self):
        if len(self.positions) < 2 or np.array(self.timestamps).std() < 1e-6:
            return None  # Not enough data yet
        t = np.array(self.timestamps)
        p = np.vstack(self.positions)  # Shape: (N, 3)
        # Normalize time to improve numerical stability
        t_centered = t - t.mean()
        # Solve for slope a in p = a*t + b, using least squares
        A = t_centered[:, np.newaxis]  # Shape: (N, 1)
        v_est, _, _, _ = np.linalg.lstsq(A, p - p.mean(axis=0), rcond=None)
        return v_est.flatten()  # Shape: (3,)


def _crop_and_resize(grayscale):
    crop_top = 50  # pixels to crop from the top
    crop_bottom = 0  # pixels to crop from the bottom
    crop_left = 100  # optional: pixels from the left
    crop_right = 125  # optional: pixels from the right
    height, width = grayscale.shape
    # Ensure you don't go out of bounds
    grayscale = grayscale[
        crop_top : height - crop_bottom, crop_left : width - crop_right
    ]
    grayscale = cv2.resize(grayscale, (84, 84), interpolation=cv2.INTER_LINEAR)
    return grayscale


def _preprocess_image(grayscale):
    grayscale = _crop_and_resize(grayscale)
    out = grayscale
    # Normalize to [0, 1] and convert to float32
    normalized = out.astype(np.float32) / 255.0
    return normalized
