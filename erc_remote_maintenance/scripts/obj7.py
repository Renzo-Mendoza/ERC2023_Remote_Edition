#!/usr/bin/env python3

import sys
import copy
import rospy
from rospkg import RosPack
from aruco_msgs.msg import MarkerArray
import numpy as np

RosPkg = 'erc_remote_maintenance'
sys.path.insert(0, RosPack().get_path(RosPkg))

from orionlib.RobotController import *

VIEW_HIDDEN_JOINTS = np.array([-50.0, -100.0, 90.0, -125.0, -91.0, 74.0])*np.pi/180

global controller 

def main():
    global controller
    # Start class
    controller = RobotController(markers_detect = True, load_markers_base = True)
    rospy.loginfo("# ========== STARTING OBJECTIVE 2 ========== #")
    rospy.loginfo("Loading controller class.")
    # Start test
    rospy.loginfo("Loading scene objects.")
    controller.add_all_scene()
    controller.go_home()
    controller.control_gripper(gripper_state = GRIPPER_OPEN)
    try:
        controller.control_joint_state(VIEW_HIDDEN_JOINTS)
        mrkr_id, _ = controller.detect_aruco_hiden(OTHER_MRKR_SIZE)
        rospy.loginfo("Hiden aruco is "+str(mrkr_id))
        controller.go_home()

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
    rospy.loginfo("# ========== FINISHING OBJECTIVE 2 ========== #")



if __name__ == "__main__":
    main()