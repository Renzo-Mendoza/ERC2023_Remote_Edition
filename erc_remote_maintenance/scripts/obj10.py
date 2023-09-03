#!/usr/bin/env python3

import sys
import copy
import rospy
from rospkg import RosPack

RosPkg = 'erc_remote_maintenance'
sys.path.insert(0, RosPack().get_path(RosPkg))

from orionlib.RobotController import *

def main():
    global controller, mrkr_detect
    # Start class
    controller = RobotController(markers_detect = True, load_markers_base=True)
    rospy.loginfo("# ========== STARTING OBJECTIVE 10 ========== #")
    rospy.loginfo("Loading controller class.")
    # Start test
    rospy.loginfo("Loading scene objects.")
    controller.add_all_scene()
    controller.go_home()
    controller.control_gripper(gripper_state = GRIPPER_OPEN)
    
    try:
        rospy.loginfo(controller.get_current_joints_state())
        rospy.loginfo("Going to home position.")
        controller.go_home_safe()
        rospy.loginfo(controller.get_current_joints_state())
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        controller.go_home()
        return
    except Exception:
        controller.go_home()
        rospy.loginfo(Exception)
        return
    rospy.loginfo("Removing scene objects.")
    controller.remove_all_scene()
    rospy.loginfo("# ========== FINISHING OBJECTIVE 10 ========== #")

if __name__ == "__main__":
    main()