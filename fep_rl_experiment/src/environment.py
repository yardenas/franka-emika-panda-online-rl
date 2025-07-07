
from typing import Any, NamedTuple
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

import rclpy.logging

class Transition(NamedTuple):
  """Container for a transition."""

  observation: Any
  action: Any
  reward: Any
  discount: Any
  next_observation: Any
  extras: Any = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray

import numpy as np
from typing import Dict, Any

class PandaPickCubeROS:
    def __init__(self, robot, config: Dict[str, Any]):
        self.robot = robot
        self.config = config
        self.prev_action = np.zeros(3)
        self.prev_reward = 0.0
        self.reached_box = 0.0
        self._steps = 0
        self.current_pos = self.robot.get_end_effector_pos()

    def reset(self) -> Dict[str, Any]:
        self.robot.reset_pose()
        self.step_count = 0
        self.gripper_closed = False
        img = self.robot.get_camera_image()
        obs = {"pixels/view_0": img.astype(np.float32) / 255.0}
        return obs

    def step(self, action: np.ndarray):
        # 2. Reset if first step
        newly_reset = self._steps == 0
        if newly_reset:
            self.prev_reward = 0.0
            self.reached_box = 0.0
            self.prev_action = np.zeros(3)
            self.current_pos = self.robot.get_end_effector_pos()

        # 3. Occasionally aid exploration (optional in real world)
        # This is risky or unnecessary in real-world environments, so skip or log-only.

        # 4. Cartesian control
        delta_pos = action[:2]  # y, z movement
        gripper_cmd = action[2] > 0  # binary open/close
        delta_xyz = np.array([0.0, *delta_pos])  # No x control

        new_pos = self.robot.move_tip(delta_xyz)
        self.robot.set_gripper(gripper_cmd)
        self.current_pos = new_pos

        # 5. Rewards
        reward = 0.0

        # Reward for small action changes (action smoothness)
        da = np.linalg.norm(action - self.prev_action)
        reward += self.config.reward_config.action_rate * da
        self.prev_action = action

        # Sparse reward: check box lifted
        lifted = self.robot.check_box_lifted()
        if lifted:
            reward += self.config.reward_config.lifted_reward
            reward += self.config.reward_config.success_reward

        # Progress reward
        reward = max(reward - self.prev_reward, 0.0)
        self.prev_reward = max(reward + self.prev_reward, self.prev_reward)
        # Observations
        obs = {}
        obs['pixels/view_0'] = self.robot.get_camera_image()
        # Done condition
        done = lifted or self._steps >= self.config.episode_length
        # Update step count
        self._steps = 0 if done else self._steps + 1
        return obs, reward, done, {
            'lifted': lifted,
            'steps': self._steps,
        }

    def _get_reward(self, info):
        # info should include:
        # - target_pos: np.array (3,)
        # - box_pos: np.array (3,)
        # - box_rot_quat: np.array (4,)  # (w, x, y, z)
        # - gripper_pos: np.array (3,)
        # - robot_qpos: np.array (n_joints,)
        # - init_qpos: np.array (n_joints,)
        # - target_rot_quat: np.array (4,)

        target_pos = info["target_pos"]
        box_pos = info["box_pos"]
        gripper_pos = info["gripper_pos"]
        robot_qpos = info["robot_qpos"]
        init_qpos = self._init_q  # initial robot joint pos stored in env
        target_rot_quat = info["target_rot_quat"]
        # Position error between box and target
        pos_err = np.linalg.norm(target_pos - box_pos)
        # Orientation error between box and target: use rotation matrix difference norm
        box_rot_mat = R.from_quat(box_rot_quat[[1,2,3,0]]).as_matrix()  # scipy uses (x,y,z,w)
        target_rot_mat = R.from_quat(target_rot_quat[[1,2,3,0]]).as_matrix()
        # Compare first 6 elements as in sim (2 columns of 3x3)
        rot_err = np.linalg.norm(target_rot_mat.ravel()[:6] - box_rot_mat.ravel()[:6])
        box_target = 1 - np.tanh(5 * (0.9 * pos_err + 0.1 * rot_err))
        # Distance between box and gripper
        gripper_box_dist = np.linalg.norm(box_pos - gripper_pos)
        gripper_box = 1 - np.tanh(5 * gripper_box_dist)
        # Robot joint deviation from initial pose
        robot_deviation = np.linalg.norm(robot_qpos - init_qpos)
        robot_target_qpos = 1 - np.tanh(robot_deviation)
        # Floor collision check — assume you have a method or flag
        floor_collision = self.check_floor_collision()  # bool
        no_floor_collision = 1.0 if not floor_collision else 0.0
        # Update reached_box in info (persistent flag)
        if "reached_box" not in info:
            info["reached_box"] = 0.0
        info["reached_box"] = max(
            info["reached_box"], float(gripper_box_dist < 0.012)
        )
        rewards = {
            "gripper_box": gripper_box,
            "box_target": box_target * info["reached_box"],
            "no_floor_collision": no_floor_collision,
            "robot_target_qpos": robot_target_qpos,
        }
        return rewards
            