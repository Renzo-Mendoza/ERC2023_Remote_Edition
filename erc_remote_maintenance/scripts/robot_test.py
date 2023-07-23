#!/usr/bin/env python3

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import std_msgs.msg
import numpy as np
import time
from math import pi, tau, dist, fabs, cos
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from rospkg import RosPack

RosPkg = 'erc_remote_maintenance'
sys.path.insert(0, RosPack().get_path(RosPkg))

import lib.RobotController as rbt_ctr

def main():
    # Start class
    try:
        controller = rbt_ctr.RobotController(markers_detect = True)
    except:
        controller = rbt_ctr.RobotController()
    # Start test
    try:

        current_pose = controller.get_current_pose()
        pos, x_axis, y_axis, z_axis = controller.pose2vectors(current_pose)
        new_pose = controller.rot_pose_axis(current_pose, 15, pos-np.array([[0.0], [0.0], [0.0]]), x_axis, degrees=True)
        controller.control_pose_state(new_pose)
        controller.go_home()
        
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return
    except:
        controller.go_home()



if __name__ == "__main__":
    main()