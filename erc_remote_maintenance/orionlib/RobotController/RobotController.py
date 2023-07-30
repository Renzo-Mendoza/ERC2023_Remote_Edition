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
from math import dist, fabs, cos
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list, list_to_pose
from scipy.spatial.transform import Rotation
from .ControllerConstants import *
from .ControllerFunctions import *

# ===================================================================================== #
# ==================================== ROBOT CLASS ==================================== #
# ===================================================================================== #
# RobotController: Class for controlling a manipulator using the moveit framework.
# @param: group_name         -> String, main group name.
# @param: robot_name         -> String, robot name.
# @param: display_path_topic -> String, Topic for path display.
# @param: markers_detect     -> Bool, Indicates if the markers where detected.
# @param: markers_pkg        -> String, Package where markers information is located.
# @param: markers_folder     -> String, Folder inside the package.
# @param: markers_file_name  -> String, Markers information file name.
# @returns: void
class RobotController(object):

    # ---------------------------------------------------------------------------------- #
    # ---------------------------------- BUILDER --------------------------------------- #
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
        # Scene description
        if markers_detect:
            _mrkrs = load_markers_poses(markers_pkg , markers_folder, markers_file_name)
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
    # -------------------------------- SCENE METHODS ----------------------------------- #
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
            try:
                return list_to_pose(self._mrkrs.get(mrkr))
            except:
                return None
    
    # add_object: Method that adds an cuboid or sphere in the scene.
    # @param: object_name -> String, cuboid name.
    # @param: geometry    -> String, geometry type.
    # @param: pose        -> Pose, object pose.
    # @param: size        -> List, object size.
    # @param: timeout     -> Float, wait time.
    # @returns: bool
    def add_object(self, object_name, shape, pose, size, timeout = TIMEOUT):
        object_pose = geometry_msgs.msg.PoseStamped()
        object_pose.header.frame_id = self._planning_frame
        object_pose.pose = pose
        if shape == CUBOID:
            self._scene.add_box(object_name, object_pose, size = size)
        elif shape == SPHERE:
            self._scene.add_sphere(object_name, object_pose, radius = size)

        return self.wait_for_state_update(object_name, object_is_known = True, timeout = timeout)
    
    # add_plane: Method that adds a plane in the scene.
    # @param: plane_name -> String, plane name.
    # @param: pose       -> Pose, plane pose.
    # @param: normal     -> List, normal vector.
    # @param: timeout    -> Float, wait time.
    # @returns: bool
    def add_plane(self, plane_name, normal, offset, timeout = TIMEOUT):
        plane_pose = geometry_msgs.msg.PoseStamped()
        plane_pose.header.frame_id = self._planning_frame
        plane_pose.pose.orientation.w = 1.0
        self._scene.add_plane(plane_name, plane_pose, normal = normal, offset = offset)
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
        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Test if the box is in attached objects
            attached_objects = self._scene.get_attached_objects([object_name])
            is_attached = len(attached_objects.keys()) > 0
            # Test if the box is in the scene.
            # Note that attaching the box will remove it from known_objects
            is_known = object_name in self._scene.get_known_object_names()
            # Test if we are in the expected state
            if (object_is_attached == is_attached) and (object_is_known == is_known):
                return True
            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()
        # If we exited the while loop without returning then we timed out
        return False
    
    # add_scene_planes: Method that adds available scene planes.
    # @param: timeout -> Float, wait time.
    # @returns: void
    def add_scene_planes(self, timeout = TIMEOUT):
        # Searching for button tag
        for button_index in range(TOTAL_BUTTONS):
            button_pose = self.get_mrkr_pose(BUTTON_MRKR,button_index+1)
            if button_pose != None:
                p, _, _, z = pose2vectors(move_pose_axis(button_pose, -MRKR_WIDTH, axis = Z_AXIS_INDEX, absolute = False))
                self.add_plane(BUTTONS_PLANE_NAME, normal=z.flatten(), offset=dot_vector(z,p), timeout = timeout)
                break
        # Searching for imu destination plane
        imu_dest_pose = self.get_mrkr_pose(IMU_DEST_MRKR)
        if imu_dest_pose != None:
            p, _, _, z = pose2vectors(move_pose_axis(imu_dest_pose, -MRKR_WIDTH, axis = Z_AXIS_INDEX, absolute = False))
            self.add_plane(IMU_DEST_PLANE_NAME, normal=z.flatten(), offset=dot_vector(z,p), timeout = timeout)
        # Searching for cover_plane
        '''insp_pnl_pose = self.get_mrkr_pose(INSP_PNL_MRKR)
        if insp_pnl_pose != None:
            p, _, _, z = pose2vectors(move_pose_axis(insp_pnl_pose, -MRKR_WIDTH, axis = Z_AXIS_INDEX, absolute = False))
            self.add_plane(PANEL_PLANE_NAME, normal=z.flatten(), offset=dot_vector(z,p), timeout = timeout)'''
    
    # add_inspection_box: Method that adds inspection box to scene.
    # @param: timeout -> Float, wait time.
    # @returns: void
    def add_inspection_box(self, timeout = TIMEOUT):
        box_mrkr_pose = self.get_mrkr_pose(INSP_PNL_MRKR)
        cover_mrkr_pose = self.get_mrkr_pose(INSP_PNL_CVR_MRKR)
        if (box_mrkr_pose == None) and (cover_mrkr_pose == None):
            return False
        else:
            box_mrkr_pos, box_mrkr_or = pose2array(box_mrkr_pose, or_as = ROT_QUAT_REP)
            cover_mrkr_pos, cover_mrkr_or = pose2array(cover_mrkr_pose, or_as = ROT_QUAT_REP)
            '''if (box_mrkr_pose!=None) and (cover_mrkr_pose!=None):
                box_pos = ((box_mrkr_pos+DISP_MRKR_BOX_TO_BOX)+(cover_mrkr_pos+DISP_MRKR_COV_TO_BOX))/2
                fuzzy_box_pos = ((box_mrkr_pos+DISP_MRKR_BOX_TO_FZZ)+(cover_mrkr_pos+DISP_MRKR_COV_TO_FZZ))/2
            else:
                box_pos = ((box_mrkr_pos+DISP_MRKR_BOX_TO_BOX)+(cover_mrkr_pos+DISP_MRKR_COV_TO_BOX))
                fuzzy_box_pos = ((box_mrkr_pos+DISP_MRKR_BOX_TO_FZZ)+(cover_mrkr_pos+DISP_MRKR_COV_TO_FZZ))
            '''
            box_pos = box_mrkr_pos+DISP_MRKR_BOX_TO_BOX
            fuzzy_box_pos = cover_mrkr_pos+DISP_MRKR_COV_TO_FZZ
            cover_pos = fuzzy_box_pos+DISP_FUZZY_TO_FLAT

            _ , _, _, z = pose2vectors(box_mrkr_pose)
            insp_box_or = rot_align_axis(z, axis = X_AXIS_INDEX, degrees = True)
            box_pose = array2pose(box_pos, insp_box_or)
            fuzzy_box_pose = array2pose(fuzzy_box_pos, insp_box_or)
            cover_pose = array2pose(cover_pos, insp_box_or)
            self.add_object(BOX_NAME, CUBOID, box_pose, INSPECT_BOX_SIZE, timeout = timeout)
            self.add_object(FUZZY_BOX_NAME, CUBOID, fuzzy_box_pose, FUZZY_BOX_SIZE, timeout = timeout)
            self.add_object(FLAT_COV_BOX_NAME, CUBOID, cover_pose, FLAT_COVER_SIZE, timeout = timeout)

    # add_imu: Method that adds imu to scene.
    # @param: timeout -> Float, wait time.
    # @returns: void
    def add_imu(self, timeout = TIMEOUT):
        imu_mrkr_pose = self.get_mrkr_pose(IMU_MRKR)
        if imu_mrkr_pose == None:
            return False
        else:
            imu_mrkr_pos, imu_mrkr_or = pose2array(imu_mrkr_pose, or_as = ROT_QUAT_REP)
            imu_pos = imu_mrkr_pos+DISP_MRKR_IMU_TO_IMU
            imu_pose = array2pose(imu_pos, imu_mrkr_or)
            self.add_object(IMU_NAME, CUBOID, imu_pose, IMU_SIZE, timeout = timeout)

    # remove_scene_planes: Method that removes available scene planes.
    # @param: timeout -> Float, wait time.
    # @returns: void
    def remove_scene_planes(self, timeout = TIMEOUT):
        self.remove_object(BUTTONS_PLANE_NAME, timeout = timeout)
        self.remove_object(IMU_DEST_PLANE_NAME, timeout = timeout)

    # remove_scene_objects: Method that removes available scene objects.
    # @param: timeout -> Float, wait time.
    # @returns: void
    def remove_scene_objects(self, timeout = TIMEOUT):
        self.remove_object(BOX_NAME, timeout = timeout)
        self.remove_object(FUZZY_BOX_NAME, timeout = timeout)
        self.remove_object(FLAT_COV_BOX_NAME, timeout = timeout)
        self.remove_object(IMU_NAME, timeout = timeout)
                
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

