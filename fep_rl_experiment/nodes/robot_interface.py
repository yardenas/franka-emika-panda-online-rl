#!/usr/bin/env python
import rospy
from fep_rl_experiment.robot import Robot

def main():
    Robot(init_node=True)
    rospy.loginfo("Robot node is running.")
    # Keep the node alive
    rospy.spin()
    rospy.Rate.sleep

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass