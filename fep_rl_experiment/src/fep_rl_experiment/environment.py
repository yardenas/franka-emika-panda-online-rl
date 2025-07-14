from typing import Any, NamedTuple
import numpy as np
import time
from fep_rl_experiment.robot import Robot

from typing import Dict


class Transition(NamedTuple):
    """Container for a transition."""

    observation: Any
    action: Any
    reward: Any
    discount: Any
    next_observation: Any
    extras: Any = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray


_REWARD_CONFIG = {
    "reward_scales": {
        "gripper_box": 4.0,
        "box_target": 8.0,
        "no_floor_collision": 0.25,
        "no_box_collision": 0.05,
        "robot_target_qpos": 0.0,
    },
    "action_rate": -0.0005,
    "no_soln_reward": -0.01,
    "lifted_reward": 0.5,
    "success_reward": 2.0,
}

_SUCCESS_THRESHOLD = 0.05


class PandaPickCube:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.prev_reward = 0.0
        self.reached_box = 0.0
        self.current_pos = self.robot.get_end_effector_pos()
        x_plane = self.robot.goal_tip_transform[0, 3] - 0.03
        self.target_pos = np.array([x_plane, 0.0, 0.2])
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.init_joint_state = np.array(
            [
                -2.00000e-05,
                4.78040e-01,
                -5.50000e-04,
                -1.81309e00,
                -1.61000e-03,
                2.34597e00,
                7.85010e-01,
            ]
        )

    def reset(self) -> Dict[str, Any]:
        self.robot.reset_service_cb(None)
        self.prev_reward = 0.0
        self.reached_box = 0.0
        time.sleep(5.0)
        img = self.robot.get_camera_image()
        obs = {"pixels/view_0": img.astype(np.float32) / 255.0}
        return obs

    def step(self, action: np.ndarray):
        # TODO (yarden): listen to e-stop and terminate
        only_yz = np.array([0.0, *action[1:]])  # No x control
        new_pos = self.robot.act(only_yz)
        self.current_pos = new_pos
        raw_rewards = self._get_reward()
        rewards = {
            k: v * self._config.reward_config.reward_scales[k]
            for k, v in raw_rewards.items()
        }
        # FIXME (yarden): should be measured somehow
        hand_box = False
        raw_rewards["no_box_collision"] = np.where(hand_box, 0.0, 1.0)
        total_reward = np.clip(sum(rewards.values()), -1e4, 1e4)
        box_pos = self.robot.get_cube_pos()
        total_reward += (box_pos[2] > 0.05) * _REWARD_CONFIG["lifted_reward"]
        success = np.linalg.norm(box_pos[2], self.target_pos) < _SUCCESS_THRESHOLD
        total_reward += success * _REWARD_CONFIG["success_reward"]
        # Progress reward
        reward = max(total_reward - self.prev_reward, 0.0)
        self.prev_reward = max(reward + self.prev_reward, self.prev_reward)
        # Observations
        img = self.robot.get_camera_image()
        obs = {"pixels/view_0": img.astype(np.float32) / 255.0}
        out_of_bounds = np.any(np.abs(box_pos) > 1.0)
        out_of_bounds |= box_pos[2] < 0.0
        # TODO (yarden): measure this with estop
        done = out_of_bounds or not self.robot.safe
        info = {**rewards, "reached_box": success}
        return obs, reward, done, info

    def _get_reward(self):
        box_pos = self.robot.get_cube_pos()
        # FIXME (yarden): double check that end effector pos == gripper pos
        gripper_pos = self.robot.get_end_effector_pos()
        pos_err = np.linalg.norm(box_pos - self.target_pos)
        box_mat = _quat_to_mat(self.robot.get_cube_quat())
        target_mat = _quat_to_mat(self.target_quat)
        rot_err = np.linalg.norm(target_mat.ravel()[:6] - box_mat.ravel()[:6])
        box_target = 1.0 - np.tanh(5 * (0.9 * pos_err + 0.1 * rot_err))
        gripper_box = 1 - np.tanh(5 * np.linalg.norm(box_pos - gripper_pos))
        qpos = self.robot.get_joint_state()
        robot_target_qpos = 1 - np.tanh(np.linalg.norm(qpos - self.init_joint_state))
        # FIXME (yarden): collisions
        hand_floor_collision = 0.0
        no_floor_collision = 1 - hand_floor_collision
        self.reached_box = np.maximum(
            self.reached_box, np.linalg.norm(box_pos - gripper_pos) < 0.012
        )
        rewards = {
            "gripper_box": gripper_box,
            "box_target": box_target * self.reached_box,
            "no_floor_collision": no_floor_collision,
            "robot_target_qpos": robot_target_qpos,
        }
        return rewards


def _quat_to_mat(q):
    q = np.outer(q, q)
    return np.array(
        [
            [
                q[0, 0] + q[1, 1] - q[2, 2] - q[3, 3],
                2 * (q[1, 2] - q[0, 3]),
                2 * (q[1, 3] + q[0, 2]),
            ],
            [
                2 * (q[1, 2] + q[0, 3]),
                q[0, 0] - q[1, 1] + q[2, 2] - q[3, 3],
                2 * (q[2, 3] - q[0, 1]),
            ],
            [
                2 * (q[1, 3] - q[0, 2]),
                2 * (q[2, 3] + q[0, 1]),
                q[0, 0] - q[1, 1] - q[2, 2] + q[3, 3],
            ],
        ]
    )
