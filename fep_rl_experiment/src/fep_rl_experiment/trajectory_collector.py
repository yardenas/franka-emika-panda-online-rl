from typing import Any, NamedTuple
from fep_rl_experiment.environment import PandaPickCube

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

def _make_transition(obs, ):
    observation = {
        "state": _make_state(msg.observation),
        "privileged_state": _make_privileged_state(msg.observation),
    }
    next_observation = {
        "state": _make_state(msg.next_observation),
        "privileged_state": _make_privileged_state(msg.next_observation),
    }
    info = {kv.key: kv.value for kv in msg.info}
    reward = msg.reward
    # Correct reward for estops
    if terminated and not msg.done:
        reward -= 1
    info["truncation"] = truncated
    info["termination"] = -terminated
    return Transition(
        observation=observation,
        action=msg.action,
        reward=reward,
        next_observation=next_observation,
        done=msg.done,
        info=info,
    )