#!/usr/bin/env python
import rospy
import numpy as np
from fep_rl_experiment.robot import Robot

def main():
    robot = Robot(init_node=True)
    rospy.loginfo("Robot node is running.")
    # Wait for Gazebo and publishers to be ready
    rospy.sleep(1.0)
    # Define a series of actions: [dx, dy, dz, gripper]
    test_actions = [
        np.array([1., 0.0, 0.0, 1.0]),   # Move forward, open gripper
        np.array([0.0, 1., 0.0, -1.0]),  # Move right, close gripper
        np.array([0.0, 0.0, 1., 1.0]),   # Move up, open gripper
        np.array([-1., 0.0, 0.0, -1.0]), # Move back, close gripper
    ]
    rate = rospy.Rate(20)
    for i, action in enumerate(test_actions):
        rospy.loginfo(f"Sending action {i+1}: {action}")
        for _ in range(100): 
            new_pos = robot.act(action)
            rate.sleep()
        rospy.loginfo(f"New position: {new_pos}")
    rospy.sleep(1.0)
    robot.reset_service_cb(None)
    rospy.loginfo("Resetting to starting position")
    rospy.sleep(1.0)

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass