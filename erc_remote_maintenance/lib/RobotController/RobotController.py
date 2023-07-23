#!/usr/bin/env python3

# ===================================================================================== #
# ========================= ERC2023 REMOTE EDITION - LIBRARY ========================== #
# ===================================================================================== #
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import std_msgs.msg
import numpy as np
import time
from math import pi, dist, fabs, cos
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list, list_to_pose
from scipy.spatial.transform import Rotation
from rospkg import RosPack
import yaml

# Manipulator constants
UR_GROUP_NAME                 = "manipulator"
UR3_ERC_NAME                  = "ur3_erc"
UR3_ERC_GRIPPER_CONTROL_TOPIC = "/gripper_command"
DISPLAY_PATH_TOPIC            = "/move_group/display_planned_path"

# UR3 ERC Remote constants
HOME_POS_JOINTS     = np.array([0.0, -120.0, 100.0, 20.0, 90.0, -90.0])*pi/180
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

# Scene description
MARKER_STICK          = 0.0006
BUTTONS_PANEL_SIZE    = (0.005, 0.300, 0.500)
LEFT_PANEL_SIZE       = (0.005, 0.250, 0.350)
RIGHT_PANEL_SIZE      = (0.005, 0.250, 0.180)
INSPECT_BOX_SIZE      = (0.070, 0.150, 0.100)
BUTTON_MRKR           = 'tag'
IMU_MRKR              = 'tag10'
IMU_DEST_MRKR         = 'tag11'
INSP_PNL_MRKR         = 'tag12'
INSP_PNL_CVR_MRKR     = 'tag13'
INSP_PNL_CVR_STG_MRKR = 'tag14'

# Gripper states
GRIPPER_CLOSE      = 'close'
GRIPPER_OPEN       = 'open'
GRIPPER_SEMI_CLOSE = 'semi_close'
GRIPPER_SEMI_OPEN  = 'semi_open'

# Constants
ROT_MATRIX_REP     = 'rot'
ROT_QUAT_REP       = 'quat'
ROT_EULER_REP      = 'euler'
X_AXIS             = np.array([[1], 
                               [0], 
                               [0]])
Y_AXIS             = np.array([[0], 
                               [1], 
                               [0]])
Z_AXIS             = np.array([[0], 
                               [0], 
                               [1]])
X_AXIS_INDEX       = 0
Y_AXIS_INDEX       = 1
Z_AXIS_INDEX       = 2
TIMEOUT            = 2.0 #seconds
MRKRS_PKG          = 'erc_remote_maintenance'
MRKRS_FOLDER       = "/config/"
MRKRS_FILE_NAME    = "markers.yaml"


# ===================================================================================== #
# ==================================== ROBOT CLASS ==================================== #
# ===================================================================================== #
# RobotController: Class for controlling a manipulator using the moveit framework.
# @param: group_name -> String, main group name.
# @param: robot_name -> String, robot name.
# @returns: void
class RobotController(object):

    # ---------------------------------------------------------------------------------- #
    # -------------------------------- CONSTRUCTOR ------------------------------------- #
    # ---------------------------------------------------------------------------------- #

    def __init__(self, group_name = UR_GROUP_NAME, 
                       robot_name = UR3_ERC_NAME, 
                       display_path_topic = DISPLAY_PATH_TOPIC,
                       markers_detect = False,
                       markers_pkg = MRKRS_PKG,
                       markers_folder = MRKRS_FOLDER,
                       markers_file_name = MRKRS_FILE_NAME):
        # Allow using properties and methods of parent class
        super(RobotController, self).__init__()
        # Initialize MoveIt!
        moveit_commander.roscpp_initialize(sys.argv)
        # Initialize node
        rospy.init_node("robot_controller", anonymous=True)
        _rate = rospy.Rate(10)
        # Object to get robot parameters and properties
        _robot = moveit_commander.RobotCommander()
        # Object to get robot scene parameters and description
        _scene = moveit_commander.PlanningSceneInterface()
        # Object to controll move groups
        _group_name = group_name
        _robot_name = robot_name
        _display_path_topic = display_path_topic
        _move_group = moveit_commander.MoveGroupCommander(_group_name)
        # Trajectory publisher
        _trajectory_pub = rospy.Publisher(
            _display_path_topic,
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=10,
        )
        # Grippers
        if _robot_name == UR3_ERC_NAME:
            _gripper_pub = rospy.Publisher(
                UR3_ERC_GRIPPER_CONTROL_TOPIC,
                std_msgs.msg.String,
                queue_size=10,
            )
            self._gripper_pub = _gripper_pub
        # Planning frame
        _planning_frame = _move_group.get_planning_frame()
        # End-efector link
        _eef_link = _move_group.get_end_effector_link()
        # Groups names
        _group_names = _robot.get_group_names()
        # Robot MoveIt! states
        _current_joints = _move_group.get_current_joint_values()
        _previous_joints = _current_joints
        _current_pose = _move_group.get_current_pose().pose
        _previous_pose = _current_pose
        _waypoints = []
        _pose_path = geometry_msgs.msg.Pose()
        _gripper_state = std_msgs.msg.String(GRIPPER_OPEN)
        # Scene description
        if markers_detect:
            _mrkrs = yaml.safe_load(open(RosPack().get_path(markers_pkg)+markers_folder+markers_file_name,'r'))
            self._mrkrs = _mrkrs
        # Class objects
        self._robot = _robot
        self._scene = _scene 
        self._rate = _rate
        self._group_name = _group_name
        self._robot_name = _robot_name
        self._move_group = _move_group 
        self._trajectory_pub = _trajectory_pub
        self._planning_frame = _planning_frame
        self._eef_link = _eef_link
        self._group_names = _group_names
        self._current_joints = _current_joints
        self._previous_joints = _previous_joints
        self._current_pose = _current_pose
        self._previous_pose = _previous_pose
        self._waypoints = _waypoints
        self._pose_path = _pose_path
        self._trajectory_displayer = moveit_msgs.msg.DisplayTrajectory()
        self._gripper_state = _gripper_state
        self._display_path_topic = _display_path_topic
        self._rotation = Rotation.from_euler('zyz', [0, 0, 0], degrees=True)

    # ---------------------------------------------------------------------------------- #
    # -------------------------------- MAIN METHODS ------------------------------------ #
    # ---------------------------------------------------------------------------------- #

    # all_close: Convenience method for testing if the values in two lists are within a 
    # tolerance of each other. For Pose and PoseStamped inputs, the angle between the two 
    # quaternions is compared (the angle between the identical orientations q and -q is 
    # calculated correctly).
    # @param: goal      -> A list of floats, a Pose or a PoseStamped.
    # @param: actual    -> A list of floats, a Pose or a PoseStamped.
    # @param: tolerance -> A float.
    # @returns: bool
    def all_close(self, goal, actual, tolerance):
        if type(goal) is list:
            for index in range(len(goal)):
                if abs(actual[index] - goal[index]) > tolerance:
                    return False

        elif type(goal) is geometry_msgs.msg.PoseStamped:
            return self.all_close(goal.pose, actual.pose, tolerance)

        elif type(goal) is geometry_msgs.msg.Pose:
            x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
            x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
            # Euclidean distance
            d = dist((x1, y1, z1), (x0, y0, z0))
            # phi = angle between orientations
            cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
            return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

        return True

    # control_gripper: Method for controlling the gripper state: 'open', 'semi_open', 'close', 
    # 'semi_close'. The desired gripper opening state is sent to the gripper_controller node.
    # @param: gripper_state -> String, desired gripper opening.
    # @returns: void    
    def control_gripper(self, gripper_state=GRIPPER_OPEN):
        self._gripper_state.data = gripper_state
        for i in range(3):
            self._gripper_pub.publish(self._gripper_state)
            self.sleep()

    # control_joint_state: Method that positions the group of links at each the desired 
    # joint position angles.
    # @param: joints_goal -> Array, desired joints state.
    # @returns: bool
    def control_joint_state(self, joints_goal):
        self._move_group.go(joints_goal, wait=True)
        self._move_group.stop()
        return self.all_close(joints_goal, self.get_current_joints_state(), TOLERANCE)

    # go_home: Method that positions the group of links at each of the home joints configuration.
    # @returns: bool    
    def go_home(self):
        self.control_joint_state(HOME_POS_JOINTS)
    
    # control_pose_state: Method that positions the end-efector link at the desired pose
    # @param: pose_goal -> Pose, desired end-efector pose.
    # @returns: bool
    def control_pose_state(self, pose_goal):
        self._move_group.set_pose_target(pose_goal)
        self._move_group.go(wait=True)
        self._move_group.stop()
        self._move_group.clear_pose_targets()
        _current_pose = self.get_current_pose()
        return self.all_close(pose_goal, _current_pose, TOLERANCE)
    
    # plan_cartesian_path: Method that plans the trajectory for a waypoints secuence.
    # @returns: void
    def plan_cartesian_path(self):
        (self._plan, self._fraction) = self._move_group.compute_cartesian_path(self._waypoints, 0.01, 0.0)
    
    # display_trajectory: Method that displays teh desired trajectory.
    # @returns: void
    def display_trajectory(self):
        self._trajectory_displayer.trajectory_start = self._robot.get_current_state()
        self._trajectory_displayer.trajectory.append(self._plan)
        self._trajectory_pub.publish(self._trajectory_displayer)
    
    # execute_plan: Method that executes movement to follow the desired trajectory.
    # @returns: void
    def execute_plan(self):
        self._move_group.execute(self._plan, wait=True)

    # ---------------------------------------------------------------------------------- #
    # ----------------------------- GEOMETRIC METHODS ---------------------------------- #
    # ---------------------------------------------------------------------------------- #
    
    # rot2rot: Method that converts a rototion to a specified rotation representation.
    # @param: orientation  -> Array, orientation array
    # @param: rot_rep      -> String, desired rotation representation: ROT_MATRIX_REP, 
    #                         ROT_QUAT_REP and ROT_EULER_REP, 
    # @param: euler_rep    -> String, current euler representation, in case.
    # @param: to_euler_rep -> String, desired euler representation, in case.
    # @returns: Array
    def rot2rot(self, orientation, rot_rep = ROT_MATRIX_REP, euler_rep = 'zxz', to_euler_rep = 'zxz'):
        if len(orientation.shape) == 1 and orientation.shape[0] == 4:
            self._rotation = Rotation.from_quat(orientation)
        elif len(orientation.shape) == 2:
            self._rotation = Rotation.from_dcm(orientation)
        elif len(orientation.shape) == 1 and orientation.shape[0] == 3:
            self._rotation = Rotation.from_euler(euler_rep, orientation, degrees=True)
        else:
            return False, orientation
        
        if rot_rep == ROT_MATRIX_REP:
            return True, self._rotation.as_dcm()
        elif rot_rep == ROT_QUAT_REP:
            return True, self._rotation.as_quat()
        elif rot_rep == ROT_EULER_REP:
            return True, self._rotation.as_euler(to_euler_rep, degrees=True)

    # move_point: Method that determines the position of a point displaced either along a 
    #             specific axis relative to an orientation or along an absolute axis.
    # @param: point       -> Array, initial point.
    # @param: orientation -> Array, orientation for relative displacement.
    # @param: axis        -> Int, desired axis of movement.
    # @param: absolute    -> Bool, absolute or relative.
    # @returns: Array
    def move_point(self, point, orientation, dist, axis = Z_AXIS_INDEX, absolute = False):
        if absolute:
            matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        else:
            _, matrix = self.rot2rot(orientation, ROT_MATRIX_REP)
        return  point + dist*matrix[:,[axis]]
    
    # move_pose_axis: Method that determines the final pose displaced either along a 
    #                 specific axis relative to an orientation or along an absolute axis,
    #                 without changing its orientation.
    # @param: pose        -> Pose, initial pose.
    # @param: orientation -> Array, orientation for relative displacement.
    # @param: axis        -> Int, desired axis of movement.
    # @param: absolute    -> Bool, absolute or relative.
    # @returns: Pose
    def move_pose_axis(self, pose, dist, axis = Z_AXIS_INDEX, absolute = False):
        pos_array, or_array = self.pose2array(pose)
        return self.array2pose(self.move_point(pos_array, or_array, dist, axis = axis, absolute = absolute), or_array)
    
    # cross_vector: Cross vector method.
    # @param: e -> Array, vector e.
    # @param: u -> Array, vector u.
    # @returns: Array
    def cross_vector(self, e, u):
        return np.reshape(np.cross(np.reshape(e,3),np.reshape(u,3)),(3,1))
    
    # dot_vector: Dot vector method.
    # @param: e -> Array, vector e.
    # @param: u -> Array, vector u.
    # @returns: Array
    def dot_vector(self, e, u):
        return np.dot(np.reshape(e,3),np.reshape(u,3))

    # rot_vector_axis: Method that rotates a vector given a axis and rotation angle,
    #                  this method uses Rodrigues' rotation formula.
    # @param: vector  -> Array, initial vector.
    # @param: angle   -> Float, rotation angle.
    # @param: axis    -> Float, rotation axis.
    # @param: degrees -> Bool, degrees or radians.
    # @returns: Array
    def rot_vector_axis(self, vector, angle, axis, degrees = False):
        if degrees:
            theta = angle*pi/180
        else:
            theta = angle
        e = axis/np.linalg.norm(axis)
        return np.cos(theta)*vector + np.sin(theta)*(self.cross_vector(e, vector)) + (1-np.cos(theta))*(self.dot_vector(e, vector))*e
    
    # rot_matrix_orientation_axis: Method that rotates an orientation matrix given a axis and 
    #                              rotation angle.
    # @param: matrix  -> Array, initial matrix orientation.
    # @param: angle   -> Float, rotation angle.
    # @param: axis    -> Float, rotation axis.
    # @param: degrees -> Bool, degrees or radians.
    # @returns: Array
    def rot_matrix_orientation_axis(self, matrix, angle, axis, degrees = False):
        return np.concatenate((self.rot_vector_axis(matrix[:,[X_AXIS_INDEX]], angle, axis, degrees = degrees), 
                               self.rot_vector_axis(matrix[:,[Y_AXIS_INDEX]], angle, axis, degrees = degrees), 
                               self.rot_vector_axis(matrix[:,[Z_AXIS_INDEX]], angle, axis, degrees = degrees)), axis=1)
    
    # rot_pose_axis: Method that rotates a pose given a axis and rotation angle.
    # @param: pose        -> Pose, initial pose.
    # @param: angle       -> Float, rotation angle.
    # @param: point_axis  -> Array, some point on the axis.
    # @param: vector_axis -> Array, Axis vector.
    # @param: degrees     -> Bool, degrees or radians.
    # @returns: Pose
    def rot_pose_axis(self, pose, angle, point_axis, vector_axis, degrees = False):
        pos_array, matrix = self.pose2array(pose, or_as=ROT_MATRIX_REP)
        pos_rot = self.rot_vector_axis((pos_array-point_axis), angle, vector_axis, degrees = degrees) + point_axis
        matrix_rot = self.rot_matrix_orientation_axis(matrix, angle, vector_axis, degrees = degrees)
        return self.array2pose(pos_rot, matrix_rot)

    # ---------------------------------------------------------------------------------- #
    # -------------------------------- scene METHODS ----------------------------------- #
    # ---------------------------------------------------------------------------------- #
    
    # get_mrkr_pose: Method that returns an specific marker pose.
    # @param: mrkr            -> String, marker name: BUTTON_MRKR, IMU_MRKR, IMU_DEST_MRKR,
    #                            INSP_PNL_MRKR, INSP_PNL_CVR_MRKR, INSP_PNL_CVR_STG_MRKR.
    # @param: button_position -> String, button position, in case, 
    # @returns: Pose
    def get_mrkr_pose(self, mrkr, button_position = 1):
        if mrkr == BUTTON_MRKR:
            return list_to_pose(self._mrkrs.get('tag'+str(button_position)))
        else:
            return list_to_pose(self._mrkrs.get(mrkr))

    # ---------------------------------------------------------------------------------- #
    # ----------------------------- SECONDARY METHODS ---------------------------------- #
    # ---------------------------------------------------------------------------------- #
    
    # get_current_joints_state: Method that returns the current joints state.
    # @returns: array, current joints state.
    def get_current_joints_state(self):
        self._current_joints = self._move_group.get_current_joint_values()
        return self._current_joints
    
    # update_previous_joint_state: Method that updates the previous joints state.
    # @returns: void.
    def update_previous_joints_state(self):
        self._previous_joints = self._current_joints
    
    # get_previous_joints_state: Method that returns the previous joints state.
    # @returns: array, current joints state.
    def get_previous_joints_state(self):
        return self._previous_joints
    
    # get_current_pose: Method that returns the current end-efector pose.
    # @returns: pose, current end-efector pose.
    def get_current_pose(self):
        self._current_pose = self._move_group.get_current_pose().pose
        return self._current_pose
    
    # update_previous_joint_state: Method that updates the previous registered end-efector
    # pose.
    # @returns: void.
    def update_previous_pose_state(self):
        self._previous_pose = self._current_pose
    
    # get_previous_pose: Method that returns the previous end-efector pose.
    # @returns: pose, previous end-efector pose.
    def get_previous_pose(self):
        return self._previous_pose, self._previous_position_list, self._previous_orientation_list
    
    # load_waypoint: Method that loads a waypoint for the desired trajectory.
    # @param: point -> Pose, next waypoint.
    # @returns: void.
    def load_waypoint(self, point):
        self._waypoints.append(copy.deepcopy(point))
    
    # clear_waypoints: Method that clear all load waypoints.
    # @returns: void.
    def clear_waypoints(self):
        self._waypoints = []

    # get_waypoints: Method that returns the registered waypoints.
    # @returns: array, waypoints.
    def get_waypoints(self):
        return self._waypoints
    
    # get_waypoints: Method that generates a delay acording to the specified rate.
    # @returns: void.
    def sleep(self):
        self._rate.sleep()
    
    # delay: Method that generates a delay of miliseconds.
    # @param: miliseconds -> int, time delay in miliseconds.
    # @returns: array, waypoints.
    def delay(self, miliseconds):
        time.sleep(miliseconds/1000)

    # pose2array: Method that converts pose class to position and orientation arrays.
    # @param: pose         -> Pose, pose class
    # @param: or_as        -> String, desired rotation representation: ROT_MATRIX_REP, 
    #                         ROT_QUAT_REP and ROT_EULER_REP, 
    # @param: euler_rep    -> String, current euler representation, in case.
    # @param: to_euler_rep -> String, desired euler representation, in case.
    # @returns: Array
    def pose2array(self, pose, or_as = ROT_QUAT_REP, euler_rep = 'zxz', to_euler_rep = 'zxz'):
        position_array = np.transpose(np.array([pose_to_list(pose)[0:3]]))
        _, orientation_array = self.rot2rot(np.array(pose_to_list(pose)[3:7]), rot_rep = or_as, euler_rep = euler_rep, to_euler_rep = to_euler_rep)
        return position_array, orientation_array
    
    # pose2array: Method that converts position and orientation arrays to pose class.
    # @param: position_array    -> Array, position array
    # @param: orientation_array -> Array, orientation array
    # @returns: Pose
    def array2pose(self, position_array, orientation_array):
        _, orientation_quat = self.rot2rot(orientation_array, ROT_QUAT_REP)
        return list_to_pose([position_array[0,0], position_array[1,0], position_array[2,0], 
                             orientation_quat[0], orientation_quat[1], orientation_quat[2], orientation_quat[3]])

    # pose2vectors: Method that converts pose (position and orientation) into vector.
    # @param: pose -> Pose, actual pose.
    # @returns: Arrays
    def pose2vectors(self, pose):
        pos_vector = np.transpose(np.array([pose_to_list(pose)[0:3]]))
        _, or_matrix = self.rot2rot(np.array(pose_to_list(pose)[3:7]), rot_rep = ROT_MATRIX_REP)
        return pos_vector, or_matrix[:,[X_AXIS_INDEX]], or_matrix[:,[Y_AXIS_INDEX]], or_matrix[:,[Z_AXIS_INDEX]]

