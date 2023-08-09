#!/usr/bin/env python3

import sys
import copy
import rospy
from rospkg import RosPack
from aruco_msgs.msg import MarkerArray

RosPkg = 'erc_remote_maintenance'
sys.path.insert(0, RosPack().get_path(RosPkg))

from orionlib.RobotController import *

global controller 

def callback_id(data):
    global controller
    for position in range(len(data.markers)):
        controller.update_mrkr_pose('tag'+str(data.markers[position].id),data.markers[position].pose.pose)

def main():
    global controller
    # Start class
    controller = RobotController(markers_detect = False)
    # Start test
    controller.go_home()
    controller.control_gripper(gripper_state = GRIPPER_OPEN)
    try:
        # Co to view pose
        controller.control_joint_state(VIEW_JOINT_STATE)
        # Getting markers poses
        _mrkr_sub = rospy.Subscriber(MRKRS_ARUCO_TOPIC_NAME, MarkerArray, callback_id)
        delay(1000)
        _mrkr_sub.unregister()
        # Show dictionary
        rospy.loginfo(controller.get_mrkr_dictionary())
        # Check aruco markers
        controller.check_aruco()
        controller.go_home()
        # Add scene objects and plane
        controller.add_buttons()
        controller.add_buttons_plane()
        controller.control_gripper(gripper_state = GRIPPER_CLOSE)
        controller.clear_waypoints()
        for raw in range(BUTTONS_GRID_BASE.shape[0]):
            for colum in range(BUTTONS_GRID_BASE.shape[1]):
                if BUTTONS_GRID_BASE[raw, colum] != None:
                    button_pose = controller.get_button_pose(BUTTONS_GRID_BASE[raw,colum]) 
                    if button_pose != None:
                        button_pos, button_or = pose2array(button_pose, or_as = ROT_MATRIX_REP)
                        # Align orientation
                        arm_or = controller.alig_orientation_to_mrkr(button_or)
                        # Out of button
                        arm_pos = copy.deepcopy(button_pos)
                        arm_pos[X_AXIS_INDEX,0] = arm_pos[X_AXIS_INDEX,0]-TIP_TOOL_CLOSED_DFL-OFFSET_FROM_BUTTON
                        arm_pose = array2pose(arm_pos, arm_or)
                        controller.load_waypoint(arm_pose)
                        controller.plan_cartesian_path()
                        controller.display_trajectory()
                        controller.execute_plan()
                        controller.clear_waypoints()
                        # Press button
                        arm_pos[X_AXIS_INDEX,0] = arm_pos[X_AXIS_INDEX,0]+DISP_PRESS_BUTTON+OFFSET_FROM_BUTTON-OFFSET_SF
                        arm_pose = array2pose(arm_pos, arm_or)
                        controller.control_pose_state(arm_pose)
                        # Out of button
                        arm_pos[X_AXIS_INDEX,0] = arm_pos[X_AXIS_INDEX,0]-DISP_PRESS_BUTTON-OFFSET_FROM_BUTTON+OFFSET_SF
                        arm_pose = array2pose(arm_pos, arm_or)
                        controller.load_waypoint(arm_pose)
                        controller.plan_cartesian_path()
                        controller.display_trajectory()
                        controller.execute_plan()
                        controller.clear_waypoints()
        # Remove scene objects and plane
        controller.remove_scene_planes()
        controller.remove_buttons()
        controller.go_home()
        controller.control_gripper(gripper_state = GRIPPER_OPEN)
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        controller.go_home()
        return
    except Exception:
        controller.go_home()
        rospy.loginfo(Exception)
        return



if __name__ == "__main__":
    main()