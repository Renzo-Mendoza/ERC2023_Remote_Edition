#!/usr/bin/env python3

# ===================================================================================== #
# =================== ERC2023 REMOTE EDITION - LIBRARY - CONSTANTS ==================== #
# ===================================================================================== #

import numpy as np
from .FunctionsConstants import *
from .ControllerFunctions import *

# Manipulator constants
UR_GROUP_NAME                 = "manipulator"
GRIPPER_GROUP_NAME            = "gripper"
UR3_ERC_NAME                  = "ur3_erc"
UR3_ERC_GRIPPER_CONTROL_TOPIC = "/gripper_command"
DISPLAY_PATH_TOPIC            = "/move_group/display_planned_path"

# UR3 ERC Remote constants
HOME_POS_JOINTS     = np.array([0.0, -120.0, 100.0, 20.0, 90.0, -90.0])*np.pi/180
TOLERANCE           = 0.01
OFFSET_TOOL_JOINT   = 0.12084
TCP_OPEN            = 0.1515
TCP_SEMI_OPEN       = 0.1860
TCP_SEMI_CLOSED     = 0.1925
TCP_CLOSED          = 0.2000
TIP_TOOL_CLOSED_DFJ = TCP_CLOSED+0.0372/2 # DFJ: Distance from joint 
TIP_TOOL_CLOSED_DFL = TIP_TOOL_CLOSED_DFJ-OFFSET_TOOL_JOINT # DFJ: Distance from link (eef_link)

# Scene constants
PRESSED_BUTTON_DIST = 0.017
WIDTH_BUTTON        = 0.023
CUBOID              = "cuboid"
SPHERE              = "sphere"
TOTAL_BUTTONS       = 9
MRKR_WIDTH          = 0.0006
IMU_DEST_PLANE_NAME = "imu_destination_plane"
BUTTONS_PLANE_NAME  = "buttons_plane"
FUZZY_BOX_NAME      = "fuzzy_box"
FLAT_COV_BOX_NAME   = "flat_cover_box"
BOX_NAME            = "box"
IMU_NAME            = "imu"

# Scene description
BUTTONS_PANEL_SIZE    = np.array([0.005, 0.300, 0.500])
LEFT_PANEL_SIZE       = np.array([0.005, 0.250, 0.350])
RIGHT_PANEL_SIZE      = np.array([0.005, 0.250, 0.180])
FUZZY_BOX_SIZE        = np.array([0.035, 0.035, 0.035])
FLAT_COVER_SIZE       = np.array([0.100, 0.150, 0.005])
CUT_DIST_BOX          = FLAT_COVER_SIZE[Z_AXIS_INDEX]*2.0
INSPECT_BOX_SIZE      = np.array([0.100, 0.150, 0.0734-CUT_DIST_BOX])
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
PANEL_INCL_ANGLE_DEG  = 18.0
IMU_COVER_MRKR_SIZE   = 0.040
OTHER_MRKR_SIZE       = 0.050
XY_DIST_MRKR_COVER    = 0.005
Y_DIST_MRKR_BOX       = INSPECT_BOX_SIZE[Y_AXIS_INDEX]/2
XYZ_DIST_MRKR_BOX     = INSPECT_BOX_SIZE[Z_AXIS_INDEX]/2
DISP_MRKR_BOX_TO_BOX  = rot_vector_axis(np.array([[MRKR_WIDTH+INSPECT_BOX_SIZE[X_AXIS_INDEX]/2], 
                                                  [0.0], 
                                                  [0.0-CUT_DIST_BOX/2]]), -PANEL_INCL_ANGLE_DEG, Z_AXIS, degrees = True)
DISP_MRKR_BOX_TO_FZZ  = rot_vector_axis(np.array([[MRKR_WIDTH+INSPECT_BOX_SIZE[X_AXIS_INDEX]/2], 
                                                  [0.0], 
                                                  [INSPECT_BOX_SIZE[Z_AXIS_INDEX]/2+FUZZY_BOX_SIZE[Z_AXIS_INDEX]/2]]), -PANEL_INCL_ANGLE_DEG, Z_AXIS, degrees = True)
DISP_MRKR_COV_TO_BOX  = rot_vector_axis(np.array([[-XY_DIST_MRKR_COVER-IMU_COVER_MRKR_SIZE/2+INSPECT_BOX_SIZE[X_AXIS_INDEX]/2], 
                                                  [-XY_DIST_MRKR_COVER-IMU_COVER_MRKR_SIZE/2+INSPECT_BOX_SIZE[Y_AXIS_INDEX]/2], 
                                                  [-MRKR_WIDTH-INSPECT_BOX_SIZE[Z_AXIS_INDEX]/2-CUT_DIST_BOX/2]]), -PANEL_INCL_ANGLE_DEG, Z_AXIS, degrees = True)
DISP_MRKR_COV_TO_FZZ  = rot_vector_axis(np.array([[-XY_DIST_MRKR_COVER-IMU_COVER_MRKR_SIZE/2+FLAT_COVER_SIZE[X_AXIS_INDEX]/2], 
                                                  [XY_DIST_MRKR_COVER+IMU_COVER_MRKR_SIZE/2-FLAT_COVER_SIZE[Y_AXIS_INDEX]/2], 
                                                  [-MRKR_WIDTH+FUZZY_BOX_SIZE[Z_AXIS_INDEX]/2]]), -PANEL_INCL_ANGLE_DEG, Z_AXIS, degrees = True)
DISP_FUZZY_TO_FLAT    = rot_vector_axis(np.array([[0.0], 
                                                  [0.0], 
                                                  [-FUZZY_BOX_SIZE[Z_AXIS_INDEX]/2-FLAT_COVER_SIZE[Z_AXIS_INDEX]/2]]), -PANEL_INCL_ANGLE_DEG, Z_AXIS, degrees = True)
DISP_MRKR_IMU_TO_IMU  = np.array([[0.0], 
                                  [0.0], 
                                  [-MRKR_WIDTH-IMU_SIZE[Z_AXIS_INDEX]/2]])
# Gripper states
GRIPPER_CLOSE      = 'close'
GRIPPER_OPEN       = 'open'
GRIPPER_SEMI_CLOSE = 'semi_close'
GRIPPER_SEMI_OPEN  = 'semi_open'

# Constants
TIMEOUT            = 4.0 #seconds