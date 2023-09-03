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
ERC_SRV_NAME                  = 'erc_aruco_score'
ARUCO_SRV_NAME                = 'aruco_erc_detect'
MRKRS50_ARUCO_TOPIC_NAME      = "/aruco_marker_publisher50/markers"
MRKRS40_ARUCO_TOPIC_NAME      = "/aruco_marker_publisher40/markers"

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
TIP_TOOL_OPEN_DFJ   = TCP_OPEN+PAT_SIZE/2 
TIP_TOOL_CLOSED_DFL = TIP_TOOL_CLOSED_DFJ-OFFSET_TOOL_JOINT # DFJ: Distance from link (eef_link)
TIP_TOOL_OPEN_DFL   = TIP_TOOL_OPEN_DFJ-OFFSET_TOOL_JOINT 
OFFSET_FROM_BUTTON  = 0.002
OFFSET_SF           = -0.006
OFFSET_FROM_MRKR    = 0.050
PANELS_ANGLE        = 18

# Scene constants
PRESSED_BUTTON_DIST = 0.017
WIDTH_BUTTON        = 0.023
DISP_PRESS_BUTTON   = WIDTH_BUTTON-PRESSED_BUTTON_DIST
CUBOID              = "cuboid"
SPHERE              = "sphere"
TOTAL_BUTTONS       = 9
MRKR_WIDTH          = 0.0006
BASICS_SCENE        = "basics_scene"
BUTTONS_PLANE_NAME  = "buttons_plane"
IMU_DEST_PANEL_NAME = "imu_destination_panel"
COV_BOX_NAME        = "cover_box"
BOX_NAME            = "box"
IMU_NAME            = "imu"
BUTTON_NAME         = "button"


# Scene description
BUTTONS_PANEL_SIZE    = np.array([0.005, 0.300, 0.500])
BUTT_BOX_SIZE         = np.array([0.060, 0.060, PRESSED_BUTTON_DIST])
LEFT_PANEL_SIZE       = np.array([0.250, 0.350, 0.005])
RIGHT_PANEL_SIZE      = np.array([0.005, 0.250, 0.180])
FUZZY_BOX_SIZE        = np.array([0.035, 0.035, 0.035])
FLAT_COVER_SIZE       = np.array([0.150, 0.100, 0.005])
CUT_DIST_BOX          = FLAT_COVER_SIZE[Z_AXIS_INDEX]*2.0
OFFSET_PLANE_TO_GND   = -0.0025
BASE_OFF_ARM_TO_GND   = -0.140
INSPECT_BOX_SIZE      = np.array([0.150, 0.0734-CUT_DIST_BOX, 0.100])
IMU_SIZE              = np.array([0.100, 0.050, 0.050])
SUP_FRAME_UR5         = np.array([1.000, 0.150, 0.140])
MRKRS_PKG             = 'erc_remote_maintenance'
MRKRS_FOLDER          = "/config/"
MRKRS_FILE_NAME       = "markers.yaml"
MRKRS_BASE_FILE_NAME  = "markers_base.yaml"
NO_MRKR               = ''
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
ORIGIN                = np.array([[0.0],[0.0],[0.0]]) 
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
DISP_REL_TOOL_TO_CAM    = np.array([[-0.06514], 
                                    [0.0], 
                                    [-0.03924]])
DISP_ARM_TO_FRAME       = np.array([[0.0], 
                                    [0.0], 
                                    [-SUP_FRAME_UR5[Z_AXIS_INDEX]/2]])
SUP_FRAME_ARM_QUAT      = Rotation.from_euler('ZYX', [0, 0, 0], degrees = True).as_quat()
IMU_MRKR_ALIGN_MATRIX   = Rotation.from_euler('ZYX', [0, 180, 0], degrees = True).as_dcm()
OTHER_MRKR_ALIGN_MATRIX = Rotation.from_euler('ZYX', [90, 180, 0], degrees = True).as_dcm()

#safe positions
INTERMEDIATE_LL    = np.array([1.0282466288518766, -0.9994517881435176, 1.2463500497358053, -0.24663217815094285, 2.599192944052879, -1.5704337971786222])
INTERMEDIATE_RL    = np.array([-1.828271436788567, -1.8480477091687741, 2.3721020073776184, -0.524057613985387, -0.2576311009120209, -1.5703879869240858])
INTERMEDIATE_ML    = np.array([8.046308494336074e-05, -1.710698363271757, 2.476217455297027, -0.7654400620065607, 1.5708933184580163, -1.5707881413739786])
INTERMEDIATE_LH    = np.array([0.8683877274117888, -0.3353650746870369, -1.1214792535463092, -1.6849483097419338, -2.439195806956418, 1.5706696227110477])
INTERMEDIATE_RH    = np.array([-1.7697271156666297, -0.4129268541657303, -1.4615763066115814, -1.267258884552624, 0.19910651741313412, 1.5706990460060792])
INTERMEDIATE_MH    = np.array([0.0001721952024205109, -1.5935340292398235, 0.4283228210460148, 1.1653052935619819, 1.5708222304639508, -1.5708001985733535])

# Gripper states
GRIPPER_CLOSE      = 'close'
GRIPPER_OPEN       = 'open'
GRIPPER_SEMI_CLOSE = 'semi_close'
GRIPPER_SEMI_OPEN  = 'semi_open'

# Constants
TIMEOUT            = 4.0 #seconds
WAIT_TIME_GRIPPER  = 5000.0 #miliseconds