import rospy
import numpy as np
from collections import deque
import cv2
from franka_gripper.msg import GraspActionGoal, MoveActionGoal
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from std_srvs.srv import Empty, SetBool
import tf2_ros
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
        self.grasp_publisher = rospy.Publisher(
            "/franka_gripper/grasp/goal", GraspActionGoal, queue_size=1
        )
        self.move_publisher = rospy.Publisher(
            "/franka_gripper/move/goal", MoveActionGoal, queue_size=1
        )
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.image_callback, queue_size=1
        )
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
        self.goal_tip_transform = np.array(
            [
                [9.9849617e-01, 9.4118714e-04, 5.4812428e-02, 6.6105318e-01],
                [1.0211766e-03, -9.9999845e-01, -1.4304515e-03, -5.1778345e-04],
                [5.4810949e-02, 1.4842749e-03, -9.9849564e-01, 1.7906836e-01],
                [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
            ]
        )
        self.goal_tip_quat = R.from_matrix(self.goal_tip_transform[:3, :3]).as_quat()
        self.start_pos = np.array([6.6105318e-01, -5.1778345e-04, 1.7906836e-01])
        self.ee_velocity_estimator = LinearVelocityEstimator()
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

    def image_callback(self, msg: Image):
        try:
            bgr_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            rgb_image = cv2.resize(rgb_image, (64, 64), interpolation=cv2.INTER_LINEAR)
            # Normalize to [0, 1] and convert to float32
            rgb_image_normalized = rgb_image.astype(np.float32) / 255.0
            # Now rgb_image_normalized is an RGB image with float values in [0, 1]
            self.latest_image = rgb_image_normalized
            self.last_image_time = msg.header.stamp
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
                "aruco_cube_frame", frame, rospy.Time(0)
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
                "aruco_cube_frame", frame, rospy.Time(0)
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
        goal = MoveActionGoal()
        goal.goal.width = 0.06
        goal.goal.speed = 10.
        self.move_publisher.publish(goal)
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
        pose_msg.pose.position.x = float(new_tip_pos[0])
        pose_msg.pose.position.y = float(new_tip_pos[1])
        pose_msg.pose.position.z = float(new_tip_pos[2])
        pose_msg.pose.orientation.x = float(self.current_tip_quat[0])
        pose_msg.pose.orientation.y = float(self.current_tip_quat[1])
        pose_msg.pose.orientation.z = float(self.current_tip_quat[2])
        pose_msg.pose.orientation.w = float(self.current_tip_quat[3])
        self._desired_ee_pose_pub.publish(pose_msg)
        if action[3] >= 0.0:
            goal = GraspActionGoal()
            goal.goal.width = 0.04
            goal.goal.speed = 0.1
            goal.goal.force = 20.0
            goal.goal.epsilon.inner = 0.04
            goal.goal.epsilon.outer = 0.04
            self.grasp_publisher.publish(goal)
        else:
            goal = MoveActionGoal()
            goal.goal.width = 0.06
            goal.goal.speed = 10.
            self.move_publisher.publish(goal)
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
    def in_sync(self):
        timestamp_dict = {
            "last_image_time": self.last_image_time,
            "last_tip_pos_time": self.last_tip_pos_time,
            "last_joint_state_time": self.last_joint_state_time,
            "last_cube_time": self.last_cube_time,
        }
        # Check for None values
        for name, ts in timestamp_dict.items():
            if ts is None:
                rospy.logwarn(f"Timestamp '{name}' is None.")
                return False
        keys = list(timestamp_dict.keys())
        max_diff = rospy.Duration.from_sec(0.15)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                t1_name = keys[i]
                t2_name = keys[j]
                t1 = timestamp_dict[t1_name]
                t2 = timestamp_dict[t2_name]
                diff = abs((t1 - t2).to_sec())
                if diff > max_diff.to_sec():
                    rospy.logwarn(
                        f"Timestamps '{t1_name}' and '{t2_name}' differ by {diff:.6f} seconds, "
                        f"which exceeds threshold of {max_diff.to_sec():.6f} seconds."
                    )
                    return False
        return True

    @property
    def safe(self):
        pos = self.get_end_effector_pos()
        out_of_bounds = np.any(np.abs(pos) > 1.0)
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
