#!/usr/bin/env python3

import sys
import copy
import rospy
from rospkg import RosPack

RosPkg = 'erc_remote_maintenance'
sys.path.insert(0, RosPack().get_path(RosPkg))

from orionlib.RobotController import *

IMU_PANEL_POSITION = np.array([[ 0.18],[0.254],[0.05]])
IMU_PANEL_ORIENTATION = np.array([-90, 180,  90 ])

LEFT_PANEL_POSITION = np.array([[0.141],[0.205],[0.406]])
LEFT_PANEL_ORIENTATION = np.array([-90,  90, 118])

GROUND_PANEL_POSITION = np.array([[ 0.195],[-0.230],[ 0.076]]) 
GROUND_PANEL_ORIENTATION = np.array([-90,  180,  90])

UPPER_RIGHT_PANEL_POSITION = np.array([[0.254],[-0.159],[0.31]])
UPPER_RIGHT_PANEL_ORIENTATION = np.array([-90, 135,  60])

FRONT_RIGHT_PANEL_POSITION = np.array([[ 0.258],[-0.240],[0.194]])
FRONT_RIGHT_PANEL_ORIENTATION = np.array([-90, 120,  72])

global controller

################## BUTTONS MARKERS #####################
def detect_buttons_mrkrs():
    global controller, marker_sub
    rospy.loginfo("|------DETECTING BUTTONS------|")
    controller.active_aruco_subscriber(OTHER_MRKR_SIZE)
    steps = int(BUTTONS_PANEL_SIZE[Z_AXIS_INDEX]//(-DISP_REL_TOOL_TO_CAM[X_AXIS_INDEX,0]*2))
    controller.control_cartesian_disp(0.07, X_AXIS_INDEX, absolute=True)
    _, robot_or = pose2array(controller.get_current_pose())
    for step in range(steps):
        robot_pos = np.array([[controller.get_current_pose().position.x],
                              [BUTTONS_PANEL_SIZE[Y_AXIS_INDEX]*((step % 2 == 0)-0.5)],
                              [BUTTONS_PANEL_SIZE[Z_AXIS_INDEX]*(1-0.5/steps-step/steps)]])
        controller.control_cartesian(array2pose(robot_pos,robot_or))
        controller.control_cartesian_disp(-np.sign(controller.get_current_pose().position.y)*BUTTONS_PANEL_SIZE[Y_AXIS_INDEX], Y_AXIS_INDEX, absolute=True)
    controller.disable_aruco_subscriber(OTHER_MRKR_SIZE)
    for button_mrkr in range(1,TOTAL_BUTTONS+1):
        button_mrkr_pose = copy.deepcopy(controller.get_mrkr_pose(BUTTON_MRKR,button_position = button_mrkr))
        if button_mrkr_pose != None:
            rospy.loginfo("Going to tag"+str(button_mrkr))
            mrkr_pos, _ = pose2array(button_mrkr_pose)
            _, arm_or = pose2array(controller.get_current_pose())
            controller.control_cartesian(move_pose_xyz(array2pose(mrkr_pos, arm_or), 
                                                       np.array([-DISP_REL_TOOL_TO_CAM[X_AXIS_INDEX,0]/2, 
                                                                 0.0, 
                                                                 -TIP_TOOL_OPEN_DFL-OFFSET_FROM_MRKR]),
                                                       absolute=False))
            controller.update_aruco_detect(BUTTON_MRKR, button_mrkr)
    controller.go_home()

#################### IMU MARKERS #######################
def detect_imu_mrkrs():
    global controller
    # LEFT PANEL
    rospy.loginfo("|------DETECTING LEFT PANEL------|")
    controller.control_pose_state(array2pose(LEFT_PANEL_POSITION, LEFT_PANEL_ORIENTATION))
    if(controller.update_aruco_detect(IMU_DEST_MRKR)):
        rospy.loginfo("Going to "+IMU_DEST_MRKR)
        left_panel_pos, _ = pose2array(controller.get_mrkr_pose(IMU_DEST_MRKR))
        _, arm_or = pose2array(controller.get_current_pose())
        controller.control_cartesian(move_pose_xyz(array2pose(left_panel_pos, arm_or), 
                                                   np.array([-DISP_REL_TOOL_TO_CAM[X_AXIS_INDEX,0]/2, 
                                                              0.0, 
                                                              -TIP_TOOL_OPEN_DFL-OFFSET_FROM_MRKR-0.05]),
                                                   absolute=False))
        controller.update_aruco_detect(IMU_DEST_MRKR)
    controller.go_home()
    # IMU
    robot_pos, _ = pose2array(controller.get_current_pose())
    rospy.loginfo("|------DETECTING IMU------|")
    controller.control_pose_state(rot_pose_axis(controller.get_current_pose(),90,robot_pos-np.array([[0.1],[0.0],[0.0]]),Y_AXIS,degrees=True))
    controller.control_pose_state(rot_pose_axis(controller.get_current_pose(),45,ORIGIN,Z_AXIS,degrees=True))
    controller.control_pose_state(array2pose(IMU_PANEL_POSITION, IMU_PANEL_ORIENTATION))
    if(controller.update_aruco_detect(IMU_MRKR)):
        rospy.loginfo("Going to "+IMU_MRKR)
        imu_pos, _ = pose2array(controller.get_mrkr_pose(IMU_MRKR))
        _, robot_or = pose2array(controller.get_current_pose())
        controller.control_cartesian(move_pose_xyz(array2pose(imu_pos, robot_or), 
                                                   np.array([-DISP_REL_TOOL_TO_CAM[X_AXIS_INDEX,0]/2, 
                                                              0.0, 
                                                              -TIP_TOOL_OPEN_DFL-OFFSET_FROM_MRKR]),
                                                   absolute=False))
        controller.update_aruco_detect(IMU_MRKR)
    controller.go_home()

################### COVER MARKERS ######################
def detect_cover_mrkrs():
    global controller
    # GROUND PANEL
    rospy.loginfo("|------DETECTING GROUND PANEL------|")
    robot_pos, _ = pose2array(controller.get_current_pose())
    controller.control_pose_state(rot_pose_axis(controller.get_current_pose(),90,robot_pos-np.array([[0.1],[0.0],[0.0]]),Y_AXIS,degrees=True))
    controller.control_pose_state(rot_pose_axis(controller.get_current_pose(),-45,ORIGIN,Z_AXIS,degrees=True))
    controller.control_pose_state(array2pose(GROUND_PANEL_POSITION, GROUND_PANEL_ORIENTATION))
    if(controller.update_aruco_detect(INSP_PNL_CVR_STG_MRKR)):
        dest_plane_pos, _ = pose2array(controller.get_mrkr_pose(INSP_PNL_CVR_STG_MRKR))
        _, robot_or = pose2array(controller.get_current_pose())
        controller.control_cartesian(move_pose_xyz(array2pose(dest_plane_pos, robot_or), 
                                                   np.array([-DISP_REL_TOOL_TO_CAM[X_AXIS_INDEX,0]/2, 
                                                              0.0, 
                                                              -TIP_TOOL_OPEN_DFL-OFFSET_FROM_MRKR]),
                                                   absolute=False))
        rospy.loginfo("Going to "+INSP_PNL_CVR_STG_MRKR)
        controller.update_aruco_detect(INSP_PNL_CVR_STG_MRKR)
    controller.go_home()
    # FRONT PANEL
    rospy.loginfo("|------DETECTING BOX------|")
    controller.control_pose_state(rot_pose_axis(controller.get_current_pose(),-45,ORIGIN,Z_AXIS,degrees=True))
    controller.control_pose_state(array2pose(FRONT_RIGHT_PANEL_POSITION, FRONT_RIGHT_PANEL_ORIENTATION))
    controller.update_aruco_detect(INSP_PNL_MRKR)
    # UPPER PANEL
    controller.go_home()
    rospy.loginfo("|------DETECTING COVER------|")
    controller.control_pose_state(rot_pose_axis(controller.get_current_pose(),-45,ORIGIN,Z_AXIS,degrees=True))
    controller.control_pose_state(array2pose(UPPER_RIGHT_PANEL_POSITION, UPPER_RIGHT_PANEL_ORIENTATION))
    if(controller.update_aruco_detect(INSP_PNL_CVR_MRKR)):
        rospy.loginfo("Going to "+INSP_PNL_CVR_MRKR)
        upper_panel_pos, _ = pose2array(controller.get_mrkr_pose(INSP_PNL_CVR_MRKR))
        _, robot_or = pose2array(controller.get_current_pose())
        controller.control_cartesian(move_pose_xyz(array2pose(upper_panel_pos, robot_or), 
                                                   np.array([-DISP_REL_TOOL_TO_CAM[X_AXIS_INDEX,0]/2, 
                                                              0.0, 
                                                              -TIP_TOOL_OPEN_DFL-OFFSET_FROM_MRKR-0.03]),
                                                   absolute=False))
        controller.update_aruco_detect(INSP_PNL_CVR_MRKR)
    controller.go_home()

def main():
    global controller
    # Start class
    controller = RobotController(markers_detect = False, load_markers_base = True)
    rospy.loginfo("# ========== STARTING OBJECTIVE 1 ========== #")
    rospy.loginfo("Loading controller class.")
    rospy.loginfo("Loading scene objects.")
    controller.add_all_scene()
    # Start test
    controller.go_home()
    controller.control_gripper(gripper_state = GRIPPER_OPEN)
    
    try:
        detect_buttons_mrkrs()
        controller.add_buttons_plane()
        controller.add_buttons()
        detect_imu_mrkrs()
        controller.add_imu()
        controller.add_left_panel()
        detect_cover_mrkrs()
        controller.add_inspection_box()
        controller.add_inspection_box_cover()
        controller.write_mrkrs()
        controller.check_aruco()
        rospy.loginfo(controller.get_mrkr_dictionary())
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
    rospy.loginfo("# ========== FINISHING OBJECTIVE 1 ========== #")
    return 


if __name__ == "__main__":
    main()