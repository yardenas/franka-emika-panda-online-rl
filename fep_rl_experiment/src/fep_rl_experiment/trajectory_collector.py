from typing import Any, NamedTuple
import rospy
from fep_rl_experiment.environment import PandaPickCube


class Transition(NamedTuple):
    observation: Any
    action: Any
    reward: Any
    discount: Any
    next_observation: Any
    extras: Any = ()


class TrajectoryCollector:
    def __init__(self, env: PandaPickCube, trajectory_length: int):
        self.transitions = []
        self.running = False
        self.trajectory_length = trajectory_length
        self.reward = 0
        self.policy = None
        self.env = env
        self.terminated = False
        self.prev_obs = None

    def step(self):
        if not self.running:
            return
        action = self.policy(self.prev_obs)
        obs, reward, done, info = self.env.step(action)
        truncation = self.current_step >= self.trajectory_length and not done
        transition = _make_transition(
            self.prev_obs, action, reward, done, obs, info, truncation
        )
        self.transitions.append(transition)
        self.reward += reward
        self.prev_obs = obs
        self.terminated = done

    def start(self, policy):
        self.transitions = []
        self.terminated = False
        self.running = True
        self.reward = 0
        self.policy = policy
        rospy.loginfo("Resetting robot...")
        self.prev_obs = self.env.reset()

    def end(self):
        self.running = False
        self.policy = None

    @property
    def trajectory_done(self):
        return self.current_step >= self.trajectory_length or self.terminated

    @property
    def current_step(self):
        return len(self.transitions)


def _make_transition(obs, action, reward, done, next_obs, info, truncation):
    return Transition(
        observation=obs,
        action=action,
        reward=reward,
        next_observation=next_obs,
        discount=1 - done,
        extras={"trancation": truncation, **info},
    )
