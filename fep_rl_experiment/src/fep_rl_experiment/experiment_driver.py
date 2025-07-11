from collections import defaultdict
import rospy
import threading
from fep_rl_experiment.transitions_server import TransitionsServer
from fep_rl_experiment.trajectory_collector import TrajectoryCollector
from fep_rl_experiment.session import Session
from fep_rl_experiment.environment import PandaPickCube
from fep_rl_experiment.robot import Robot


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
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.timer_callback)
        self.transitions_server = TransitionsServer(self, safe_mode=True)
        self.trajectory_collector = TrajectoryCollector(
            self.env, self.trajectory_length
        )
        self.server_thread = threading.Thread(
            target=self.transitions_server.loop, daemon=True
        )
        self.server_thread.start()
        
        rospy.loginfo("Experiment driver initialized.")
    def start_sampling(self, policy):
        if self.running:
            rospy.logerr("Already running, please finish your previous run.")
            return
        rospy.loginfo(f"Starting command sampling... Run id: {self.run_id}")
        self.running = True
        self.trajectory_collector.start(policy)

    def timer_callback(self, event):
        if not self.running:
            return
        if event.current_real - event.current_expected > 1.5 * self.dt:
            rospy.logwarn("Missed previous step call time.")
        self.trajectory_collector.step()
        if self.trajectory_collector.trajectory_done:
            rospy.loginfo("Command sampling completed. Returning to standing mode.")
            self.running = False
            self.run_id += 1
            self.summarize_trial()
            self.trajectory_collector.end()

    def summarize_trial(self):
        infos = [
            transition.info for transition in self.trajectory_collector.transitions
        ]
        table_data = defaultdict(float)
        for info in infos:
            for key, value in info.items():
                table_data[key] += value
        table_data["steps"] = len(infos)
        table_data["reward"] = self.trajectory_collector.reward
        rospy.loginfo(
            f"Total reward: {self.trajectory_collector.reward}\n{_format_reward_summary(table_data)}"
        )
        self.session.update(table_data)

    def get_trajectory(self):
        return self.trajectory_collector.transitions

    @property
    def robot_ok(self):
        return self.robot.ok


def _format_reward_summary(table_data):
    lines = []
    header = f"{'Reward Component':<20} {'Total Value':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for key, value in table_data.items():
        lines.append(f"{key:<20} {value:>12.2f}")
    return "\n".join(lines)
