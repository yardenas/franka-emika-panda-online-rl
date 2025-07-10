from typing import Any, NamedTuple
from fep_rl_experiment.environment import PandaPickCube

class Transition(NamedTuple):
  observation: Any
  action: Any
  reward: Any
  discount: Any
  next_observation: Any
  extras: Any = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray

class TrajectoryCollector:
    def __init__(self, env: PandaPickCube):
        self.current_step = 0
        self.transitions = []
        self.running = False
        self.terminated = False
        self.trajectory_length = 0
        self.reward = 0
        self.policy = None
        self.env = env
        self.prev_obs = None

    def step(self):
        if not self.running:
            return
        action = self.policy(self.prev_obs)
        obs, reward, done, info = self.env.step(action)
        self.current_step += 1
        self.terminated = done
        self.reward += reward
        truncation = self.current_step >= self.trajectory_length and not self.terminated
        transition = _make_transition(msg, self.terminated, truncation)
        self.transitions.append(transition)

    def start(self, requested_length, policy):
        self.current_step = 0
        self.joint_limit_counter = 0
        self.transitions = []
        self.terminated = False
        self.running = True
        self.trajectory_length = requested_length
        self.reward = 0
        self.policy = policy
        self.prev_obs = self.env.reset()

    def end(self):
        self.running = False
        self.trajectory_length = 0
        self.policy = None

    @property
    def trajectory_done(self):
        return self.current_step >= self.trajectory_length or self.terminated

def _make_transition(obs, action, reward, done, next_obs, info, truncation):
    return Transition(
        observation=obs,
        action=action,
        reward=reward,
        next_observation=next_obs,
        discount=1-done,
        extras={
            "policy_extras": {},
            "state_extras": {
               "trancation":  truncation,
               **info
            }
        },
    )