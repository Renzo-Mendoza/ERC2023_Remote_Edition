#!/usr/bin/env python3

import sys
import rospy
import numpy as np
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from rospkg import RosPack

RosPkg = 'erc_remote_maintenance'
sys.path.insert(0, RosPack().get_path(RosPkg))

from orionlib.RobotController import *

def main():
    # Start class
    try:
        controller = RobotController(markers_detect = True)
    except:
        controller = RobotController()
    # Start test
    try:

        current_pose = controller.get_current_pose()
        pos, x_axis, y_axis, z_axis = pose2vectors(current_pose)
        new_pose = rot_pose_axis(current_pose, 15, pos-np.array([[0.0], [0.0], [0.0]]), x_axis, degrees=True)
        controller.control_pose_state(new_pose)
        # Add scene objects
        controller.add_imu()
        controller.add_inspection_box()
        controller.add_scene_planes()
        delay(5000)
        # Remove scene objects
        controller.remove_scene_objects()
        controller.remove_scene_planes()
        # Go to home position
        controller.go_home()
        return
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        controller.go_home()
        return
    except:
        controller.go_home()



if __name__ == "__main__":
    main()