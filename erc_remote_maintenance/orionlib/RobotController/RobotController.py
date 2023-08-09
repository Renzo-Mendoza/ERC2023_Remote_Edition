#!/usr/bin/env python3

# ===================================================================================== #
# ========================= ERC2023 REMOTE EDITION - LIBRARY ========================== #
# ===================================================================================== #

import sys
import copy
import rospy
import pandas as pd
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import std_msgs.msg
import numpy as np
import rosservice
from erc_aruco_msg.srv import ErcArucoRequest, ErcArucoResponse, ErcAruco
from math import dist, fabs, cos
from .ControllerConstants import *
from .ControllerFunctions import *

# ===================================================================================== #
# ==================================== ROBOT CLASS ==================================== #
# ===================================================================================== #
# RobotController: Class for controlling a manipulator using the moveit framework.
# @param: robot_group_name   -> String, main group name.
# @param: gripper_group_name -> String, gripper group name.
# @param: robot_name         -> String, robot name.
# @param: display_path_topic -> String, Topic for path display.
# @param: markers_detect     -> Bool, Indicates if the markers where detected.
# @param: markers_pkg        -> String, Package where markers information is located.
# @param: markers_folder     -> String, Folder inside the package.
# @param: markers_file_name  -> String, Markers information file name.
# @param: node_name          -> String, Node name.
# @returns: void
class RobotController(object):

    # ---------------------------------------------------------------------------------- #
    # ---------------------------------- BUILDER --------------------------------------- #
    # ---------------------------------------------------------------------------------- #

    def __init__(self, robot_group_name = UR_GROUP_NAME, 
                       gripper_group_name = GRIPPER_GROUP_NAME,
                       robot_name = UR3_ERC_NAME, 
                       display_path_topic = DISPLAY_PATH_TOPIC,
                       markers_detect = False,
                       markers_pkg = MRKRS_PKG,
                       markers_folder = MRKRS_FOLDER,
                       markers_file_name = MRKRS_FILE_NAME,
                       node_name = CONTROLLER_NAME):
        # Allow using properties and methods of parent class
        super(RobotController, self).__init__()
        # Initialize MoveIt!
        moveit_commander.roscpp_initialize(sys.argv)
        # Initialize node
        rospy.init_node(node_name, anonymous=True)
        _rate = rospy.Rate(10)
        # Object to get robot parameters and properties
        _robot = moveit_commander.RobotCommander()
        # Object to get robot scene parameters and description
        _scene = moveit_commander.PlanningSceneInterface()
        # Object to controll move groups
        _robot_group_name = robot_group_name
        _gripper_group_name = gripper_group_name
        _robot_name = robot_name
        _display_path_topic = display_path_topic
        _move_group = moveit_commander.MoveGroupCommander(_robot_group_name)
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
        # Planning frame
        _planning_frame = _robot.get_planning_frame()
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
        _mrkrs = {}
        _buttons = {}
        _erc_srv = None
        _srv_request = ErcArucoRequest()
        _srv_response = ErcArucoResponse()
        # Scene description
        if markers_detect:
            _mrkrs = load_markers_poses(markers_pkg , markers_folder, markers_file_name)
            _buttons = self.get_buttons_dictionary
        if "/"+SRV_NAME in rosservice.get_service_list():
            rospy.wait_for_service(SRV_NAME)
            _erc_srv = rospy.ServiceProxy(SRV_NAME,ErcAruco)
            rospy.loginfo("Service erc aruco checker available")
        else:
            rospy.loginfo("No service erc aruco checker available")
        # Class objects
        self._robot = _robot
        self._scene = _scene 
        self._rate = _rate
        self._robot_group_name = _robot_group_name
        self._gripper_group_name = _gripper_group_name
        self._robot_name = _robot_name
        self._move_group = _move_group 
        self._trajectory_pub = _trajectory_pub
        self._planning_frame = _planning_frame
        self._eef_link = _eef_link
        self._planning_frame = _planning_frame
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
        self._fuzzy_box_pose = geometry_msgs.msg.Pose()
        self._imu_pose = geometry_msgs.msg.Pose()
        self._insp_box_pose = geometry_msgs.msg.Pose()
        self._mrkrs = _mrkrs
        self._buttons = _buttons
        self._erc_srv = _erc_srv
        self._srv_request = _srv_request
        self._srv_response = _srv_response
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
            for _index in range(len(goal)):
                if abs(actual[_index] - goal[_index]) > tolerance:
                    return False

        elif type(goal) is geometry_msgs.msg.PoseStamped:
            return self.all_close(goal.pose, actual.pose, tolerance)

        elif type(goal) is geometry_msgs.msg.Pose:
            _x0, _y0, _z0, _qx0, _qy0, _qz0, _qw0 = pose_to_list(actual)
            _x1, _y1, _z1, _qx1, _qy1, _qz1, _qw1 = pose_to_list(goal)
            # Euclidean distance
            _d = dist((_x1, _y1, _z1), (_x0, _y0, _z0))
            # phi = angle between orientations
            _cos_phi_half = fabs(_qx0 * _qx1 + _qy0 * _qy1 + _qz0 * _qz1 + _qw0 * _qw1)
            return _d <= tolerance and _cos_phi_half >= cos(tolerance / 2.0)

        return True

    # control_gripper: Method for controlling the gripper state: 'open', 'semi_open', 'close', 
    # 'semi_close'. The desired gripper opening state is sent to the gripper_controller node.
    # @param: gripper_state -> String, desired gripper opening.
    # @returns: void    
    def control_gripper(self, gripper_state=GRIPPER_OPEN):
        self._gripper_state.data = gripper_state
        self._gripper_pub.publish(self._gripper_state)
        self.sleep()
        delay(WAIT_TIME_GRIPPER)

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
        return self.all_close(pose_goal, self.get_current_pose(), TOLERANCE)
    
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
    # ------------------------------ GEOMETRIC METHODS --------------------------------- #
    # ---------------------------------------------------------------------------------- #
    
    # rot_orientation_relative: Method that aligns the gripper orientation opposite to an
    #                           marker.
    # @param: mrkr_orientation -> Array, marker orientation.
    # @param: or_align         -> Array, IMU_MRKR_ALIGN_MATRIX for imu, 
    #                             OTHER_MRKR_ALIGN_MATRIX for others.
    # @param: or_as            -> String, desired rotation representation: ROT_MATRIX_REP, 
    #                             ROT_QUAT_REP and ROT_EULER_REP, 
    # @param: euler_rep        -> String, current euler representation, in case.
    # @param: to_euler_rep     -> String, desired euler representation, in case.
    # @returns: Array
    def alig_orientation_to_mrkr(self, mrkr_orientation, or_align = OTHER_MRKR_ALIGN_MATRIX, or_as = ROT_MATRIX_REP, euler_rep = 'zxz', to_euler_rep = 'zxz'):
        return rot2rot(rot2rot(mrkr_orientation, rot_rep = ROT_MATRIX_REP)@or_align, rot_rep = or_as, in_euler = euler_rep, out_euler = to_euler_rep)

    # ---------------------------------------------------------------------------------- #
    # -------------------------------- SCENE METHODS ----------------------------------- #
    # ---------------------------------------------------------------------------------- #
    
    # reload_mrkrs: Method that reload markers poses.
    # @param: markers_pkg        -> String, Package where markers information is located.
    # @param: markers_folder     -> String, Folder inside the package.
    # @param: markers_file_name  -> String, Markers information file name.
    # @returns: void
    def reload_mrkrs(self, markers_pkg = MRKRS_PKG, markers_folder = MRKRS_FOLDER, markers_file_name = MRKRS_FILE_NAME):
        self._mrkrs = load_markers_poses(markers_pkg , markers_folder, markers_file_name)

    # update_mrkr_pose: Method that updates any marker pose in the dictionary.
    # @param: mrkr -> String, marker name: BUTTON_MRKR, IMU_MRKR, IMU_DEST_MRKR,
    #                 INSP_PNL_MRKR, INSP_PNL_CVR_MRKR, INSP_PNL_CVR_STG_MRKR.
    # @param: pose -> Pose, new pose.
    # @returns: void
    def update_mrkr_pose(self, mrkr, pose):
        try:
            self._mrkrs[mrkr] = pose_to_list(pose)
        except:
            self._mrkrs = {mrkr: pose_to_list(pose)}

    # get_mrkr_pose: Method that returns an specific marker pose.
    # @param: mrkr            -> String, marker name: BUTTON_MRKR, IMU_MRKR, IMU_DEST_MRKR,
    #                            INSP_PNL_MRKR, INSP_PNL_CVR_MRKR, INSP_PNL_CVR_STG_MRKR.
    # @param: button_position -> String, button position, in case. 
    # @returns: Pose
    def get_mrkr_pose(self, mrkr, button_position = 1):
        try:
            if mrkr == BUTTON_MRKR and (button_position in range(1,TOTAL_BUTTONS+1)):
                return list_to_pose(self._mrkrs.get('tag'+str(button_position)))
            else:
                return list_to_pose(self._mrkrs.get(mrkr))
        except:
            return None
    
    # get_button_pose: Method that return specified button pose (open state).
    # @param: button_position -> String, button position.
    # @returns: Pose
    def get_button_pose(self, button_position):
        _button_mrkr_pose = self.get_mrkr_pose(BUTTON_MRKR, button_position = button_position)
        if _button_mrkr_pose == None:
            return None
        else:
            _button_mrkr_pos, _button_mrkr_or = pose2array(_button_mrkr_pose, or_as = ROT_QUAT_REP)
            _button_pos = _button_mrkr_pos+rot_displacement(DISP_MRKR_BUT_TO_BUT, _button_mrkr_or)
            return array2pose(_button_pos, _button_mrkr_or)
    
    # get_imu_pose: Method that returns imu pose.
    # @returns: Pose
    def get_imu_pose(self):
        _imu_mrkr_pose = self.get_mrkr_pose(IMU_MRKR)
        if _imu_mrkr_pose == None:
            return None
        else:
            _imu_mrkr_pos, _imu_mrkr_or = pose2array(_imu_mrkr_pose, or_as = ROT_QUAT_REP)
            _imu_pos = _imu_mrkr_pos+rot_displacement(DISP_MRKR_IMU_TO_IMU,_imu_mrkr_or)
            return array2pose(_imu_pos, _imu_mrkr_or)
    
    # get_fuzzy_box_pose: Method that returns fuzzy box pose.
    # @returns: Pose
    def get_fuzzy_box_pose(self):
        _cover_mrkr_pose = self.get_mrkr_pose(INSP_PNL_CVR_MRKR)
        if _cover_mrkr_pose == None:
            return False
        else:
            _cover_mrkr_pos, _cover_mrkr_or = pose2array(_cover_mrkr_pose, or_as = ROT_QUAT_REP)
            _fuzzy_box_pos = _cover_mrkr_pos + rot_displacement(DISP_MRKR_COV_TO_FZZ, _cover_mrkr_or)
            return array2pose(_fuzzy_box_pos, _cover_mrkr_or)
    
    # add_object: Method that adds an cuboid or sphere in the scene.
    # @param: object_name -> String, cuboid name.
    # @param: geometry    -> String, geometry type.
    # @param: pose        -> Pose, object pose.
    # @param: size        -> List, object size.
    # @param: timeout     -> Float, wait time.
    # @returns: bool
    def add_object(self, object_name, shape, pose, size, timeout = TIMEOUT):
        _object_pose = geometry_msgs.msg.PoseStamped()
        _object_pose.header.frame_id = self._planning_frame
        _object_pose.pose = pose
        if shape == CUBOID:
            self._scene.add_box(object_name, _object_pose, size = size)
        elif shape == SPHERE:
            self._scene.add_sphere(object_name, _object_pose, radius = size)

        return self.wait_for_state_update(object_name, object_is_known = True, timeout = timeout)
    
    # add_plane: Method that adds a plane in the scene.
    # @param: plane_name -> String, plane name.
    # @param: pose       -> Pose, plane pose.
    # @param: normal     -> List, normal vector.
    # @param: timeout    -> Float, wait time.
    # @returns: bool
    def add_plane(self, plane_name, normal, offset, timeout = TIMEOUT):
        _plane_pose = geometry_msgs.msg.PoseStamped()
        _plane_pose.header.frame_id = self._planning_frame
        _plane_pose.pose.orientation.w = 1.0
        self._scene.add_plane(plane_name, _plane_pose, normal = normal, offset = offset)
        return self.wait_for_state_update(plane_name, object_is_known = True, timeout = timeout)
    
    # remove_box: Method that removes an cuboid or sphere in the scene.
    # @param: object_name -> String, cuboid name.
    # @param: timeout     -> Float, wait time.
    # @returns: bool
    def remove_object(self, object_name, timeout = TIMEOUT):
        self._scene.remove_world_object(object_name)
        return self.wait_for_state_update(object_name, object_is_known = False, object_is_attached = False, timeout = timeout)

    # wait_for_state_update: Method that confirms scene state update.
    # @param: object_name     -> String, cuboid name.
    # @param: object_is_known -> bool, for added object update.
    # @param: object_is_known -> bool, for attached object update.
    # @param: timeout         -> Float, wait time.
    # @returns: bool
    def wait_for_state_update(self, object_name, object_is_known = False, object_is_attached = False, timeout = TIMEOUT):
        _start = rospy.get_time()
        _seconds = rospy.get_time()
        while (_seconds - _start < timeout) and not rospy.is_shutdown():
            # Test if the box is in attached objects
            _attached_objects = self._scene.get_attached_objects([object_name])
            _is_attached = len(_attached_objects.keys()) > 0
            # Test if the box is in the scene.
            # Note that attaching the box will remove it from known_objects
            _is_known = object_name in self._scene.get_known_object_names()
            # Test if we are in the expected state
            if (object_is_attached == _is_attached) and (object_is_known == _is_known):
                return True
            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            _seconds = rospy.get_time()
        # If we exited the while loop without returning then we timed out
        return False
    
    # add_buttons_plane: Method that adds buttons plane to the scene.
    # @param: timeout -> Float, wait time.
    # @param: z_disp  -> Float, displacement in z axis.
    # @returns: bool
    def add_buttons_plane(self, z_disp = 0.0, timeout = TIMEOUT):
        for _button_index in range(1,TOTAL_BUTTONS+1):
            _button_pose = self.get_mrkr_pose(BUTTON_MRKR,_button_index)
            if _button_pose != None:
                _p, _, _, _z = pose2vectors(move_pose_axis(_button_pose, -MRKR_WIDTH, axis = Z_AXIS_INDEX, absolute = False))
                return self.add_plane(BUTTONS_PLANE_NAME, normal=_z.flatten(), offset=dot_vector(_z,_p+_z*z_disp), timeout = timeout)
        return False
    
    # add_buttons: Method that adds buttons to the scene.
    # @param: timeout -> Float, wait time.
    # @param: z_disp  -> Float, displacement in z axis.
    # @returns: bool
    def add_buttons(self, z_disp = 0.0, timeout = TIMEOUT):
        # Searching for button tag
        for _button_index in range(1,TOTAL_BUTTONS+1):
            _button_pose = self.get_button_pose(_button_index)
            if _button_pose != None:
                _button_pos, _button_or = pose2array(_button_pose, or_as = ROT_QUAT_REP)
                _button_box_pos = _button_pos+rot_displacement(DISP_BUT_TO_BUT_BOX+np.array([[0.0],[0.0],[z_disp]]), _button_or)
                _box_pose = array2pose(_button_box_pos, _button_or)
                self.add_object(BUTTON_NAME+str(_button_index), CUBOID, _box_pose, BUTT_BOX_SIZE, timeout = timeout)


    # add_inspection_box_cover: Method that adds inspection box cover to scene.
    # @param: timeout -> Float, wait time.
    # @returns: bool
    def add_inspection_box_cover(self, timeout = TIMEOUT):
        _cover_mrkr_pose = self.get_mrkr_pose(INSP_PNL_CVR_MRKR)
        if _cover_mrkr_pose == None:
            return False
        else:
            _cover_mrkr_pos, _cover_mrkr_or = pose2array(_cover_mrkr_pose, or_as = ROT_QUAT_REP)
            _fuzzy_box_pos = _cover_mrkr_pos+rot_displacement(DISP_MRKR_COV_TO_FZZ, _cover_mrkr_or)
            _cover_pos = _fuzzy_box_pos+rot_displacement(DISP_FUZZY_TO_FLAT, _cover_mrkr_or)

            _fuzzy_box_pose = array2pose(_fuzzy_box_pos, _cover_mrkr_or)
            _cover_pose = array2pose(_cover_pos, _cover_mrkr_or)
            self.add_object(COV_BOX_NAME+"1", CUBOID, _fuzzy_box_pose, FUZZY_BOX_SIZE, timeout = timeout)
            self.add_object(COV_BOX_NAME+"2", CUBOID, _cover_pose, FLAT_COVER_SIZE, timeout = timeout)
            return True

    # add_inspection_box: Method that adds inspection box to scene.
    # @param: timeout -> Float, wait time.
    # @returns: bool
    def add_inspection_box(self, timeout = TIMEOUT):
        _box_mrkr_pose = self.get_mrkr_pose(INSP_PNL_MRKR)
        if _box_mrkr_pose == None:
            return False
        else:
            _box_mrkr_pos, _box_mrkr_or = pose2array(_box_mrkr_pose, or_as = ROT_QUAT_REP)
            _box_pos = _box_mrkr_pos+rot_displacement(DISP_MRKR_BOX_TO_BOX, _box_mrkr_or)
            _box_pose = array2pose(_box_pos, _box_mrkr_or)
            self.add_object(BOX_NAME, CUBOID, _box_pose, INSPECT_BOX_SIZE, timeout = timeout)
            return True
    
    # add_left_panel: Method that adds left panel to scene.
    # @param: timeout -> Float, wait time.
    # @param: z_disp  -> Float, displacement in z axis.
    # @returns: bool
    def add_left_panel(self, z_disp = 0.0, timeout = TIMEOUT):
        _lp_mrkr_pose = self.get_mrkr_pose(IMU_DEST_MRKR)
        if _lp_mrkr_pose == None:
            return False
        else:
            _lp_mrkr_pos, _lp_mrkr_or = pose2array(_lp_mrkr_pose, or_as = ROT_QUAT_REP)
            _lp_pos = _lp_mrkr_pos+rot_displacement(DISP_MRKR_LP_TO_LP+np.array([[0.0],[0.0],[z_disp]]), _lp_mrkr_or)
            _lp_pose = array2pose(_lp_pos, _lp_mrkr_or)
            self.add_object(IMU_DEST_PANEL_NAME, CUBOID, _lp_pose, LEFT_PANEL_SIZE, timeout = timeout)
            return True

    # add_imu: Method that adds imu to scene.
    # @param: timeout -> Float, wait time.
    # @returns: bool
    def add_imu(self, timeout = TIMEOUT):
        _imu_mrkr_pose = self.get_mrkr_pose(IMU_MRKR)
        if _imu_mrkr_pose == None:
            return False
        else:
            _imu_mrkr_pos, _imu_mrkr_or = pose2array(_imu_mrkr_pose, or_as = ROT_QUAT_REP)
            _imu_pos = _imu_mrkr_pos+DISP_MRKR_IMU_TO_IMU
            _imu_pose = array2pose(_imu_pos, _imu_mrkr_or)
            self.add_object(IMU_NAME, CUBOID, _imu_pose, IMU_SIZE, timeout = timeout)

    # remove_scene_planes: Method that removes available scene planes.
    # @param: timeout -> Float, wait time.
    # @returns: void
    def remove_scene_planes(self, timeout = TIMEOUT):
        self.remove_object(BUTTONS_PLANE_NAME, timeout = timeout)
    
    # remove_buttons: Method that removes buttons in scene.
    # @param: timeout -> Float, wait time.
    # @returns: void
    def remove_buttons(self, timeout = TIMEOUT):
        for _button_index in range(1,TOTAL_BUTTONS+1):
            self.remove_object(BUTTON_NAME+str(_button_index), timeout = timeout)

    # remove_scene_objects: Method that removes available scene objects.
    # @param: object_name -> String, object name.
    # @param: timeout     -> Float, wait time.
    # @returns: bool
    def remove_scene_object(self, object_name, timeout = TIMEOUT):
        if object_name == COV_BOX_NAME:
            self.remove_object(object_name+"1", timeout = timeout)
            self.remove_object(object_name+"2", timeout = timeout)
        else:
            self.remove_object(object_name, timeout = timeout)
    
    # attach_object: Method that attaches an scene object to the gripper.
    # @param: object_name -> String, object name.
    # @param: timeout     -> Float, wait time.
    # @returns: void
    def attach_object(self, object_name, timeout = TIMEOUT):
        if object_name == COV_BOX_NAME:
            self._scene.attach_box(self._eef_link, object_name+"1", touch_links = self._robot.get_link_names(group=self._gripper_group_name))
            self._scene.attach_box(self._eef_link, object_name+"2", touch_links = self._robot.get_link_names(group=self._gripper_group_name))
            return self.wait_for_state_update(object_name+"1", object_is_attached=True, object_is_known=False, timeout=timeout) and self.wait_for_state_update(object_name+"2", object_is_attached=True, object_is_known=False, timeout=timeout)
        else:
            self._scene.attach_box(self._eef_link, object_name, touch_links = self._robot.get_link_names(group=self._gripper_group_name))
            return self.wait_for_state_update(object_name, object_is_attached=True, object_is_known=False, timeout=timeout)

    # detach_object: Method that detaches an scene object from the gripper.
    # @param: object_name -> String, object name.
    # @param: timeout     -> Float, wait time.
    # @returns: void
    def detach_object(self, object_name, timeout = TIMEOUT):
        if object_name == COV_BOX_NAME:
            self._scene.remove_attached_object(self._eef_link, object_name+"1")
            self._scene.remove_attached_object(self._eef_link, object_name+"2")
            return self.wait_for_state_update(object_name+"1", object_is_attached=True, object_is_known=False, timeout=timeout) and self.wait_for_state_update(object_name+"2", object_is_attached=True, object_is_known=False, timeout=timeout)
        else:
            self._scene.remove_attached_object(self._eef_link, object_name)
            return self.wait_for_state_update(object_name, object_is_attached=True, object_is_known=False, timeout=timeout)
    
    # get_buttons_dictionary: Method that the buttons dictionary.
    # @returns: array.
    def get_buttons_dictionary(self):
        _buttons_dict = {}
        for _button_position in range(1,TOTAL_BUTTONS+1):
            _button_pose = self.get_button_pose(_button_position)
            if _button_pose != None:
                try:
                    _buttons_dict["tag"+str(_button_position)]=pose_to_list(_button_pose)
                except:
                    _buttons_dict = {"tag"+str(_button_position):pose_to_list(_button_pose)}
        return _buttons_dict
    # ---------------------------------------------------------------------------------- #
    # ----------------------------- SECONDARY METHODS ---------------------------------- #
    # ---------------------------------------------------------------------------------- #

    # get_current_joints_state: Method that chacks aruco results.
    # @returns: array, current joints state.
    def check_aruco(self):
        try:
            _aruco_pos = np.empty((14,3))
            for raw in range(0,14):
                _aruco_pose = self.get_mrkr_pose("tag"+str(raw+1))
                if _aruco_pose != None:
                    _aruco_pos[raw,:] = np.array([_aruco_pose.position.x, _aruco_pose.position.y, _aruco_pose.position.z])
                else:
                    _aruco_pos[raw,:] = np.array([0.0, 0.0, 0.0])

            self._srv_request.tag1 = _aruco_pos[0,:].tolist()
            self._srv_request.tag2 = _aruco_pos[1,:].tolist()
            self._srv_request.tag3 = _aruco_pos[2,:].tolist()
            self._srv_request.tag4 = _aruco_pos[3,:].tolist()
            self._srv_request.tag5 = _aruco_pos[4,:].tolist()
            self._srv_request.tag6 = _aruco_pos[5,:].tolist()
            self._srv_request.tag7 = _aruco_pos[6,:].tolist()
            self._srv_request.tag8 = _aruco_pos[7,:].tolist()
            self._srv_request.tag9 = _aruco_pos[8,:].tolist()
            self._srv_request.tag10 = _aruco_pos[9,:].tolist()
            self._srv_request.tag11 = _aruco_pos[10,:].tolist()
            self._srv_request.tag12 = _aruco_pos[11,:].tolist()
            self._srv_request.tag13 = _aruco_pos[12,:].tolist()
            self._srv_request.tag14 = _aruco_pos[13,:].tolist()
            self._srv_response = self._erc_srv(self._srv_request)
            rospy.loginfo(f"ORION TEAM aruco markers score: {self._srv_response.score}")
        except:
            rospy.loginfo("Can't use erc_aruco_checker service")

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
    
    # get_mrkr_dictionary: Method that returns marker dictionary.
    # @returns: pose, current end-efector pose.
    def get_mrkr_dictionary(self):
        return self._mrkrs
    
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

