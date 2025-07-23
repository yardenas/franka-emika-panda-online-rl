from typing import Any, NamedTuple
from collections import defaultdict
import rospy
import threading
import time
from fep_rl_experiment.transitions_server import TransitionsServer
from fep_rl_experiment.session import Session
from fep_rl_experiment.environment import PandaPickCube
from fep_rl_experiment.robot import Robot


class Transition(NamedTuple):
    observation: Any
    action: Any
    reward: Any
    discount: Any
    next_observation: Any
    extras: Any = ()


class ExperimentDriver:
    def __init__(self):
        rospy.init_node("franka_emika_robot_interface")
        self.dt = rospy.get_param("~dt")
        self.trajectory_length = rospy.get_param("~trajectory_length")
        session_id = rospy.get_param("~session_id")
        self.session = Session(filename=session_id, directory="experiment_sessions")
        num_steps = len(self.session.steps)
        self.robot = Robot()
        self.env = PandaPickCube(self.robot)
        self.running = False
        self.run_id = num_steps
        self.transitions_server = TransitionsServer(self, safe_mode=True)
        self.server_thread = threading.Thread(
            target=self.transitions_server.loop, daemon=True
        )
        self.server_thread.start()
        rospy.loginfo("Experiment driver initialized.")

    def sample_trajectory(self, policy):
        rospy.loginfo(f"Starting trajectory sampling... Run id: {self.run_id}")
        trajectory = _collect_trajectory(self.env, policy, self.trajectory_length, self.dt)
        self.summarize_trial(trajectory)
        return trajectory

    def summarize_trial(self, transitions):
        infos = [transition.extras["state_extras"] for transition in transitions]
        table_data = defaultdict(float)
        for info in infos:
            for key, value in info.items():
                table_data[key] += value
        table_data["steps"] = len(infos)
        table_data["reward"] = float(
            sum(transition.reward for transition in transitions)
        )
        table_data["terminated"] = (
            1 - transitions[-1].discount and not infos[-1]["truncation"]
        )
        rospy.loginfo(
            f"Total reward: {table_data['reward']}\nTotal cost: {table_data['cost']}\n{_format_reward_summary(table_data)}"
        )
        self.session.update(table_data)
        self.run_id += 1

    @property
    def robot_ok(self):
        return self.robot.ok


def _collect_trajectory(env, policy, episode_length, dt):
    transitions: list[Transition] = []
    done = False
    steps = 0
    obs = env.reset()
    while not done and steps < episode_length:
        start = time.time()
        action = policy(obs)
        next_obs, reward, done, info = env.step(action)
        end = time.time()
        elapsed = end - start
        steps += 1
        truncated = not done and steps == episode_length - 1
        transition = Transition(
            obs,
            action,
            reward,
            1 - done,
            next_obs,
            {"state_extras": {"truncation": truncated, **info}},
        )
        transition.extras["state_extras"]["time"] = elapsed
        transitions.append(transition)
        # Wait or warn if behind schedule
        remaining = dt - (time.time() - start)
        if remaining > 0:
            time.sleep(remaining)
        else:
            rospy.logwarn(f"Iteration took too long: {elapsed:.4f}s > dt={dt:.4f}s")
        obs = next_obs  # Advance observation for next step
    if not done:
        assert len(transitions) == episode_length
    return transitions


def _format_reward_summary(table_data):
    lines = []
    header = f"{'Reward Component':<20} {'Total Value':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for key, value in table_data.items():
        lines.append(f"{key:<20} {value:>12.2f}")
    return "\n".join(lines)
