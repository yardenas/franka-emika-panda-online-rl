from collections import deque
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
        x_plane = self.robot.start_pos[0]
        self.target_pos = np.array([x_plane, -0.05, 0.2])
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.init_joint_state = np.array(
            [
                [
                    -0.00002,
                    0.47804,
                    -0.00055,
                    -1.81309,
                    -0.00161,
                    2.34597,
                    0.78501,
                    0.04000,
                    0.04000,
                ]
            ]
        )
        self.gripper_act = deque(maxlen=5)

    def reset(self) -> Dict[str, Any]:
        self.robot.reset_service_cb(None)
        self.prev_reward = 0.0
        self.reached_box = 0.0
        self.gripper_act.extend([0.0] * 5)
        time.sleep(2.0)
        img = self.robot.get_camera_image()
        ee = self.robot.get_end_effector_pos()
        action_history = np.array(self.gripper_act)
        fingers = self.robot.get_joint_state()[-2:] / 0.04
        propreiceptive = np.concatenate([ee, fingers, action_history])
        obs = {"pixels/view_0": img, "state": propreiceptive}
        return obs

    def step(self, action: np.ndarray):
        only_yz = np.concatenate(
            ([self.robot.start_pos[0] - self.robot.get_end_effector_pos()[0]], action)
        )
        self.robot.act(only_yz)
        raw_rewards = self._get_reward()
        rewards = {
            k: v * _REWARD_CONFIG["reward_scales"][k] for k, v in raw_rewards.items()
        }
        hand_box = False
        raw_rewards["no_box_collision"] = np.where(hand_box, 0.0, 1.0)
        total_reward = np.clip(sum(rewards.values()), -1e4, 1e4)
        box_pos = self.robot.get_cube_pos()
        total_reward += (box_pos[2] > 0.05) * _REWARD_CONFIG["lifted_reward"]
        success = np.linalg.norm(box_pos[2] - self.target_pos[2]) < _SUCCESS_THRESHOLD
        total_reward += success * _REWARD_CONFIG["success_reward"]
        # Progress reward
        reward = max(total_reward - self.prev_reward, 0.0)
        self.prev_reward = max(reward + self.prev_reward, self.prev_reward)
        # Observations
        img = self.robot.get_camera_image()
        ee = self.robot.get_end_effector_pos()
        self.gripper_act.append(action[-1])
        action_history = np.array(self.gripper_act)
        fingers = self.robot.get_joint_state()[-2:] / 0.04
        propreiceptive = np.concatenate([ee, fingers, action_history])
        obs = {"pixels/view_0": img, "state": propreiceptive}
        out_of_bounds = np.any(np.abs(box_pos) > 1.0)
        out_of_bounds |= box_pos[2] < 0.0
        done = not self.robot.safe or success
        info = {
            **raw_rewards,
            "reached_box": self.reached_box,
            "success": success,
            "lifted": box_pos[2] > 0.05,
        }
        return obs, reward, done, info

    def _get_reward(self):
        box_pos = self.robot.get_cube_pos()
        # FIXME (yarden): double check that end effector pos == gripper pos
        gripper_pos = self.robot.get_end_effector_pos()
        pos_err = np.linalg.norm(box_pos - self.target_pos)
        box_target = 1.0 - np.tanh(5 * (pos_err))
        gripper_box = 1 - np.tanh(5 * np.linalg.norm(box_pos - gripper_pos))
        qpos = self.robot.get_joint_state()
        robot_target_qpos = 1 - np.tanh(np.linalg.norm(qpos - self.init_joint_state))
        hand_floor_collision = gripper_pos[-1] < -0.001
        no_floor_collision = 1 - hand_floor_collision
        self.reached_box = np.maximum(
            self.reached_box, np.linalg.norm(box_pos - gripper_pos) < 0.025
        )
        rewards = {
            "gripper_box": gripper_box,
            "box_target": box_target * self.reached_box,
            "no_floor_collision": no_floor_collision,
            "robot_target_qpos": robot_target_qpos,
        }
        return rewards
