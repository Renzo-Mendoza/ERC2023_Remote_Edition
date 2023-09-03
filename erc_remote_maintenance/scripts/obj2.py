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
    controller = RobotController(markers_detect = True, load_markers_base = True)
    rospy.loginfo("# ========== STARTING OBJECTIVE 2 ========== #")
    rospy.loginfo("Loading controller class.")
    try:
        buttons = [int(tag) for tag in rospy.get_param('~tags').split(',')]
        if all(button == 0 for button in buttons):
            rospy.loginfo("No order recived...")
            return
        else:
            rospy.loginfo("Order recived: "+ rospy.get_param('~tags'))
    except:
        buttons = [0,0,0,0]
    # Start test
    rospy.loginfo("Loading scene objects.")
    controller.add_all_scene()
    controller.go_home()
    controller.control_gripper(gripper_state = GRIPPER_OPEN)
    try:
        # Check aruco markers
        controller.go_home()
        arm_init_pose = controller.get_current_pose()
        controller.control_gripper(gripper_state = GRIPPER_OPEN)
        # Add scene objects and plane
        #rospy.loginfo("Closing the gripper.")
        #controller.control_gripper(gripper_state = GRIPPER_CLOSE)
        for button in buttons:
            button_pose = controller.get_button_pose(button) 
            if button_pose != None:
                rospy.loginfo("|------GO TO BUTTON "+str(button)+"------|")
                button_pos, button_or = pose2array(button_pose, or_as = ROT_MATRIX_REP)
                # Align orientation
                arm_or = controller.alig_orientation_to_mrkr(button_or)
                # Out of button 1
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
        arm_pose = controller.get_current_pose()
        controller.control_cartesian_disp(arm_init_pose.position.y - arm_pose.position.y,Y_AXIS_INDEX,absolute=True)
        controller.control_cartesian_disp(arm_init_pose.position.z - arm_pose.position.z,Z_AXIS_INDEX,absolute=True)
        controller.control_cartesian_disp(arm_init_pose.position.x - arm_pose.position.x,X_AXIS_INDEX,absolute=True)
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
    rospy.loginfo("# ========== FINISHING OBJECTIVE 2 ========== #")
    return

if __name__ == "__main__":
    main()