#!/usr/bin/env python3

import sys
import rospy
from rospkg import RosPack

RosPkg = 'erc_remote_maintenance'
sys.path.insert(0, RosPack().get_path(RosPkg))

from orionlib.RobotController import *

global controller 

def main():
    global controller
    # Start class
    rospy.loginfo("# ========== STARTING OBJECTIVE 9 ========== #")
    rospy.loginfo("Loading controller class.")
    controller = RobotController(markers_detect = True, load_markers_base = True)
    try: 
        button = rospy.get_param('~tag')
        rospy.loginfo("Hiden button = "+str(button))
    except:
        button = 0
    # Start test
    rospy.loginfo("Loading scene objects.")
    controller.add_all_scene()
    controller.go_home()
    controller.control_gripper(gripper_state = GRIPPER_OPEN)
    try:
        # Check aruco markers
        controller.go_home()
        # Add scene objects and plane
        rospy.loginfo("Closing the gripper.")
        button_pose = controller.get_button_pose(button) 
        if button_pose != None:
            rospy.loginfo("|------GO TO BUTTON "+str(button)+"------|")
            button_pos, button_or = pose2array(button_pose, or_as = ROT_MATRIX_REP)

            # Align orientation
            arm_or = controller.alig_orientation_to_mrkr(button_or)

            # Out of button
            rospy.loginfo("Go out of button "+str(button))
            controller.control_cartesian(move_pose_axis(array2pose(button_pos, arm_or), 
                                                        -TIP_TOOL_OPEN_DFL-OFFSET_FROM_BUTTON, 
                                                        Z_AXIS_INDEX, 
                                                        absolute = False))
            
            controller.control_cartesian(move_pose_axis(controller.get_current_pose(), 
                                                        TIP_TOOL_OPEN_DFL-TIP_TOOL_CLOSED_DFL, 
                                                        Z_AXIS_INDEX, 
                                                        absolute = False))
            controller.control_gripper(gripper_state = GRIPPER_CLOSE)

            # Press button
            rospy.loginfo("Pressing button "+str(button))
            controller.control_cartesian(move_pose_axis(controller.get_current_pose(), 
                                                        DISP_PRESS_BUTTON+OFFSET_FROM_BUTTON-OFFSET_SF, 
                                                        Z_AXIS_INDEX, 
                                                        absolute = False))
            delay(1000)

            # Out of button
            rospy.loginfo("Realising button "+str(button))
            controller.control_cartesian(move_pose_axis(controller.get_current_pose(), 
                                                        -DISP_PRESS_BUTTON-OFFSET_FROM_BUTTON+OFFSET_SF, 
                                                        Z_AXIS_INDEX, 
                                                        absolute = False))
            controller.control_gripper(gripper_state = GRIPPER_OPEN)
            
            controller.control_cartesian(move_pose_axis(controller.get_current_pose(), 
                                                        -TIP_TOOL_OPEN_DFL+TIP_TOOL_CLOSED_DFL, 
                                                        Z_AXIS_INDEX, 
                                                        absolute = False))
        # Remove scene objects and plane
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
    rospy.loginfo("Removing scene objects.")
    controller.remove_all_scene()
    rospy.loginfo("# ========== FINISHING OBJECTIVE 9 ========== #")
    return

if __name__ == "__main__":
    main()