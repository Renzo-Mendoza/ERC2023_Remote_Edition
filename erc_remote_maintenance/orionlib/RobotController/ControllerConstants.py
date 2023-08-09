#!/usr/bin/env python3

# ===================================================================================== #
# =================== ERC2023 REMOTE EDITION - LIBRARY - CONSTANTS ==================== #
# ===================================================================================== #

import numpy as np
from scipy.spatial.transform import Rotation
from .FunctionsConstants import *
from .ControllerFunctions import *

# Manipulator constants
UR_GROUP_NAME                 = "manipulator"
GRIPPER_GROUP_NAME            = "gripper"
UR3_ERC_NAME                  = "ur3_erc"
UR3_ERC_GRIPPER_CONTROL_TOPIC = "/gripper_command"
DISPLAY_PATH_TOPIC            = "/move_group/display_planned_path"
CONTROLLER_NAME               = "robot_controller"
SRV_NAME                      = 'erc_aruco_score'
MRKRS_ARUCO_TOPIC_NAME        = "/aruco_marker_publisher/markers"

# UR3 ERC Remote constants
HOME_POS_JOINTS     = np.array([0.0, -120.0, 100.0, 20.0, 90.0, -90.0])*np.pi/180
TOLERANCE           = 0.01
OFFSET_TOOL_JOINT   = 0.12084
TCP_OPEN            = 0.1515
TCP_SEMI_OPEN       = 0.1860
TCP_SEMI_CLOSED     = 0.1925
TCP_CLOSED          = 0.2000
PAT_SIZE            = 0.0372
TIP_TOOL_CLOSED_DFJ = TCP_CLOSED+PAT_SIZE/2 # DFJ: Distance from joint 
TIP_TOOL_CLOSED_DFL = TIP_TOOL_CLOSED_DFJ-OFFSET_TOOL_JOINT # DFJ: Distance from link (eef_link)
OFFSET_FROM_BUTTON  = 0.000
OFFSET_SF           = -0.003

# Scene constants
PRESSED_BUTTON_DIST = 0.017
WIDTH_BUTTON        = 0.023
DISP_PRESS_BUTTON   = WIDTH_BUTTON-PRESSED_BUTTON_DIST
CUBOID              = "cuboid"
SPHERE              = "sphere"
TOTAL_BUTTONS       = 9
MRKR_WIDTH          = 0.0006
BUTTONS_PLANE_NAME  = "buttons_plane"
IMU_DEST_PANEL_NAME = "imu_destination_panel"
COV_BOX_NAME        = "cover_box"
BOX_NAME            = "box"
IMU_NAME            = "imu"
BUTTON_NAME         = "button"


# Scene description
BUTTONS_PANEL_SIZE    = np.array([0.300, 0.500, 0.005])
BUTT_BOX_SIZE         = np.array([0.060, 0.060, PRESSED_BUTTON_DIST])
LEFT_PANEL_SIZE       = np.array([0.250, 0.350, 0.005])
RIGHT_PANEL_SIZE      = np.array([0.005, 0.250, 0.180])
FUZZY_BOX_SIZE        = np.array([0.035, 0.035, 0.035])
FLAT_COVER_SIZE       = np.array([0.150, 0.100, 0.005])
CUT_DIST_BOX          = FLAT_COVER_SIZE[Z_AXIS_INDEX]*2.0
INSPECT_BOX_SIZE      = np.array([0.150, 0.0734-CUT_DIST_BOX, 0.100])
IMU_SIZE              = np.array([0.100, 0.050, 0.050])
MRKRS_PKG             = 'erc_remote_maintenance'
MRKRS_FOLDER          = "/config/"
MRKRS_FILE_NAME       = "markers.yaml"
BUTTON_MRKR           = 'tag'
IMU_MRKR              = 'tag10'
IMU_DEST_MRKR         = 'tag11'
INSP_PNL_MRKR         = 'tag12'
INSP_PNL_CVR_MRKR     = 'tag13'
INSP_PNL_CVR_STG_MRKR = 'tag14'
PANEL_INCL_ANGLE_DEG  = 18.00
IMU_COVER_MRKR_SIZE   = 0.040
OTHER_MRKR_SIZE       = 0.050
XY_DIST_MRKR_COVER    = 0.005
DIST_MRKR_IP_TO_IP    = 0.005 #Inspection panel
X_DIST_MRKR_IP        = 0.035
Y_DIST_MRKR_IP        = 0.005
Y_DIST_MRKR_BUTTON    = 0.030
VIEW_MIDDLE_PANAL_POS = np.array([[0.06],[-0.0],[0.38361]]) 
VIEW_MIDDLE_PANAL_OR  = Rotation.from_euler('ZYX', [0, 90, 0], degrees = True).as_quat()
VIEW_JOINT_STATE      = np.array([55.0, -146.0, 88.0, 58.0, 145.0, -90.0])*np.pi/180
BUTTONS_GRID_BASE     = np.array([[1, 2, 3],
                                  [4, 5, 6],
                                  [7, 8, 9]])

# Displacements
DISP_MRKR_BOX_TO_BOX    = np.array([[0.0], 
                                    [-CUT_DIST_BOX/2], 
                                    [-MRKR_WIDTH-INSPECT_BOX_SIZE[Z_AXIS_INDEX]/2]])
DISP_MRKR_COV_TO_FZZ    = np.array([[-XY_DIST_MRKR_COVER-IMU_COVER_MRKR_SIZE/2+FLAT_COVER_SIZE[X_AXIS_INDEX]/2], 
                                    [-XY_DIST_MRKR_COVER-IMU_COVER_MRKR_SIZE/2+FLAT_COVER_SIZE[Y_AXIS_INDEX]/2], 
                                    [-MRKR_WIDTH+FUZZY_BOX_SIZE[Z_AXIS_INDEX]/2]])
DISP_FUZZY_TO_FLAT      = np.array([[0.0], 
                                    [0.0], 
                                    [-FUZZY_BOX_SIZE[Z_AXIS_INDEX]/2-FLAT_COVER_SIZE[Z_AXIS_INDEX]/2]])
DISP_MRKR_IMU_TO_IMU    = np.array([[0.0], 
                                    [0.0], 
                                    [-MRKR_WIDTH-IMU_SIZE[Z_AXIS_INDEX]/2]])
DISP_MRKR_LP_TO_LP      = np.array([[-OTHER_MRKR_SIZE/2-X_DIST_MRKR_IP+LEFT_PANEL_SIZE[X_AXIS_INDEX]/2], 
                                    [OTHER_MRKR_SIZE/2+Y_DIST_MRKR_IP-LEFT_PANEL_SIZE[Y_AXIS_INDEX]/2], 
                                    [-DIST_MRKR_IP_TO_IP-MRKR_WIDTH-LEFT_PANEL_SIZE[Z_AXIS_INDEX]/2]])
DISP_MRKR_BUT_TO_BUT    = np.array([[0.0], 
                                    [-OTHER_MRKR_SIZE/2-Y_DIST_MRKR_BUTTON], 
                                    [-MRKR_WIDTH+WIDTH_BUTTON]])
DISP_BUT_TO_BUT_BOX     = np.array([[0.0], 
                                    [0.0], 
                                    [-WIDTH_BUTTON-BUTT_BOX_SIZE[Z_AXIS_INDEX]/2]])
IMU_MRKR_ALIGN_MATRIX   = Rotation.from_euler('ZYX', [0, 180, 0], degrees = True).as_dcm()
OTHER_MRKR_ALIGN_MATRIX = Rotation.from_euler('ZYX', [90, 180, 0], degrees = True).as_dcm()

# Gripper states
GRIPPER_CLOSE      = 'close'
GRIPPER_OPEN       = 'open'
GRIPPER_SEMI_CLOSE = 'semi_close'
GRIPPER_SEMI_OPEN  = 'semi_open'

# Constants
TIMEOUT            = 4.0 #seconds
WAIT_TIME_GRIPPER  = 5000.0 #miliseconds