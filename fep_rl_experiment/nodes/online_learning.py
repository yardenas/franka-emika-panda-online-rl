#!/usr/bin/env python
import rospy
from fep_rl_experiment.experiment_driver import ExperimentDriver

def main():
    ExperimentDriver()
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass