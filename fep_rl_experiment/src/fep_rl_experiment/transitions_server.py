import pickle
from typing import Mapping
import zmq
import rospy
import time
import numpy as np
import onnxruntime as ort


class TransitionsServer:
    def __init__(self, experiment_driver, safe_mode=False, address="tcp://*:5559"):
        self.experiment_driver = experiment_driver
        self.address = address
        self.safe_mode = safe_mode

    def loop(self):
        with zmq.Context() as ctx:
            with ctx.socket(zmq.REP) as socket:
                socket.bind(self.address)
                while True:
                    message = socket.recv()
                    policy, num_steps = pickle.loads(message)
                    if num_steps < self.experiment_driver.trajectory_length:
                        rospy.logerr("Invalid num_steps: {}".format(num_steps))
                    trials = self.run(policy, num_steps)
                    if trials is None:
                        continue
                    socket.send(pickle.dumps(trials))

    def run(self, policy, num_steps):
        trials = []
        num_transitions = 0
        while num_transitions < num_steps:
            trial = self.do_trial(policy)
            new_num_transitions = len(trial)
            if num_transitions + new_num_transitions > num_steps:
                trial = trial[: num_steps - num_transitions]
                trial[-1].extras["state_extras"]["truncation"] = True
                rospy.loginfo("Truncating trajectory")
            num_transitions += len(trial)
            trials.append(trial)
            rospy.loginfo("Completed trial")
        transitions = flatten_trajectories(trials)
        assert len(transitions[2]) == num_steps, (
            f"Expected {num_steps} transitions, got {len(transitions)}"
        )
        return transitions

    def do_trial(self, policy_bytes):
        rospy.loginfo("Starting sampling")
        while not self.experiment_driver.robot_ok:
            rospy.loginfo("Waiting the robot to be ready...")
            time.sleep(2.5)
        if self.safe_mode:
            while True:
                answer = input("Press Y/y when ready to collect trajectory\n")
                if not (answer == "Y" or answer == "y"):
                    rospy.loginfo("Skipping trajectory")
                    continue
                else:
                    break
        policy_fn = self.parse_policy(policy_bytes)
        while True:
            try:
                trajectory = self.experiment_driver.sample_trajectory(policy_fn)
                break
            except RuntimeError:
                rospy.logwarn("Could not sample trajectory")
        rospy.loginfo("Sampling finished")
        return trajectory

    def parse_policy(self, policy_bytes):
        session = ort.InferenceSession(
            policy_bytes,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        # Get input and output names (assuming 1 input and 1 output)
        output_name = session.get_outputs()[0].name

        def infer(inputs: Mapping[str, np.ndarray]) -> np.ndarray:
            inputs = {k: v.astype(np.float32)[None] for k, v in inputs.items()}
            result = session.run([output_name], inputs)
            return result[0][0]

        return infer


def flatten_trajectories(trajectories):
    observations = {
        key: np.array(
            [t.observation[key] for traj in trajectories for t in traj],
            dtype=np.float32,
        )
        for key in trajectories[0][0].observation
    }
    actions = np.array(
        [t.action for traj in trajectories for t in traj], dtype=np.float32
    )
    rewards = np.array(
        [t.reward for traj in trajectories for t in traj], dtype=np.float32
    )
    next_observations = {
        key: np.array(
            [t.next_observation[key] for traj in trajectories for t in traj],
            dtype=np.float32,
        )
        for key in trajectories[0][0].next_observation
    }
    discount = np.array(
        [t.discount for traj in trajectories for t in traj], dtype=np.float32
    )
    state_extras = {
        key: np.array(
            [t.extras["state_extras"][key] for traj in trajectories for t in traj],
            dtype=np.float32,
        )
        for key in trajectories[0][0].extras["state_extras"]
    }
    extras = {"state_extras": state_extras, "policy_extras": {}}
    return (
        observations,
        actions,
        rewards,
        next_observations,
        discount,
        extras,
    )
