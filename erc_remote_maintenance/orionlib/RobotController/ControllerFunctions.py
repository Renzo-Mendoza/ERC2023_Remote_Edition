#!/usr/bin/env python3

# ===================================================================================== #
# ==================== ERC2023 REMOTE EDITION - FUNCTION LIBRARY ====================== #
# ===================================================================================== #

import numpy as np
import time
from moveit_commander.conversions import pose_to_list, list_to_pose
from scipy.spatial.transform import Rotation
from rospkg import RosPack
import yaml
from .FunctionsConstants import *

# ---------------------------------------------------------------------------------- #
# ---------------------------- GEOMETRIC FUNCTIONS --------------------------------- #
# ---------------------------------------------------------------------------------- #

# rot2rot: Method that converts a rotation to a specified rotation representation.
# @param: orientation  -> Array, orientation array
# @param: rot_rep      -> String, desired rotation representation: ROT_MATRIX_REP, 
#                         ROT_QUAT_REP and ROT_EULER_REP, 
# @param: euler_rep    -> String, current euler representation, in case.
# @param: to_euler_rep -> String, desired euler representation, in case.
# @returns: Array
def rot2rot(orientation, rot_rep = ROT_MATRIX_REP, in_rotvect = False, in_euler = 'zxz', out_euler = 'zxz', in_degrees = True, out_degrees = True):
    if in_rotvect:
        _rotation = Rotation.from_rotvec(orientation)
    elif len(orientation.shape) == 1 and orientation.shape[0] == 4:
        _rotation = Rotation.from_quat(orientation)
    elif len(orientation.shape) == 2:
        _rotation = Rotation.from_dcm(orientation)
    elif len(orientation.shape) == 1 and orientation.shape[0] == 3:
        _rotation = Rotation.from_euler(in_euler, orientation, degrees = in_degrees)
    else:
        return False, orientation
    
    if rot_rep == ROT_MATRIX_REP:
        return True, _rotation.as_dcm()
    elif rot_rep == ROT_QUAT_REP:
        return True, _rotation.as_quat()
    elif rot_rep == ROT_EULER_REP:
        return True, _rotation.as_euler(out_euler, degrees = out_degrees)
    elif rot_rep == ROT_VECTOR_REP:
        return True, _rotation.as_rotvec()

# move_point: Method that determines the position of a point displaced either along a 
#             specific axis relative to an orientation or along an absolute axis.
# @param: point       -> Array, initial point.
# @param: orientation -> Array, orientation for relative displacement.
# @param: axis        -> Int, desired axis of movement.
# @param: absolute    -> Bool, absolute or relative.
# @returns: Array
def move_point(point, orientation, dist, axis = Z_AXIS_INDEX, absolute = False):
    if absolute:
        matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    else:
        _, matrix = rot2rot(orientation, ROT_MATRIX_REP)
    return  point + dist*matrix[:,[axis]]

# move_pose_axis: Method that determines the final pose displaced either along a 
#                 specific axis relative to an orientation or along an absolute axis,
#                 without changing its orientation.
# @param: pose        -> Pose, initial pose.
# @param: orientation -> Array, orientation for relative displacement.
# @param: axis        -> Int, desired axis of movement.
# @param: absolute    -> Bool, absolute or relative.
# @returns: Pose
def move_pose_axis(pose, dist, axis = Z_AXIS_INDEX, absolute = False):
    pos_array, or_array = pose2array(pose)
    return array2pose(move_point(pos_array, or_array, dist, axis = axis, absolute = absolute), or_array)

# rot_vector_axis: Method that rotates a vector given a axis and rotation angle,
#                  this method uses Rodrigues' rotation formula.
# @param: vector  -> Array, initial vector.
# @param: angle   -> Float, rotation angle.
# @param: axis    -> Float, rotation axis.
# @param: degrees -> Bool, degrees or radians.
# @returns: Array
def rot_vector_axis(vector, angle, axis, degrees = False):
    if degrees:
        theta = angle*np.pi/180
    else:
        theta = angle
    e = axis/np.linalg.norm(axis)
    return np.cos(theta)*vector + np.sin(theta)*(cross_vector(e, vector)) + (1-np.cos(theta))*(dot_vector(e, vector))*e

# rot_matrix_orientation_axis: Method that rotates an orientation matrix given a axis and 
#                              rotation angle.
# @param: matrix  -> Array, initial matrix orientation.
# @param: angle   -> Float, rotation angle.
# @param: axis    -> Float, rotation axis.
# @param: degrees -> Bool, degrees or radians.
# @returns: Array
def rot_matrix_orientation_axis(matrix, angle, axis, degrees = False):
    return np.concatenate((rot_vector_axis(matrix[:,[X_AXIS_INDEX]], angle, axis, degrees = degrees), 
                            rot_vector_axis(matrix[:,[Y_AXIS_INDEX]], angle, axis, degrees = degrees), 
                            rot_vector_axis(matrix[:,[Z_AXIS_INDEX]], angle, axis, degrees = degrees)), axis=1)

# rot_pose_axis: Method that rotates a pose given a axis and rotation angle.
# @param: pose        -> Pose, initial pose.
# @param: angle       -> Float, rotation angle.
# @param: point_axis  -> Array, some point on the axis.
# @param: vector_axis -> Array, Axis vector.
# @param: degrees     -> Bool, degrees or radians.
# @returns: Pose
def rot_pose_axis(pose, angle, point_axis, vector_axis, degrees = False):
    pos_array, matrix = pose2array(pose, or_as=ROT_MATRIX_REP)
    pos_rot = rot_vector_axis((pos_array-point_axis), angle, vector_axis, degrees = degrees) + point_axis
    matrix_rot = rot_matrix_orientation_axis(matrix, angle, vector_axis, degrees = degrees)
    return array2pose(pos_rot, matrix_rot)

# cross_vector: Cross vector method.
# @param: e -> Array, vector e.
# @param: u -> Array, vector u.
# @returns: Array
def cross_vector(e, u):
    return np.reshape(np.cross(np.reshape(e,3),np.reshape(u,3)),(3,1))

# dot_vector: Dot vector method.
# @param: e -> Array, vector e.
# @param: u -> Array, vector u.
# @returns: Array
def dot_vector(e, u):
    return np.dot(np.reshape(e,3),np.reshape(u,3))

# dist2plane: Distance between a point and a plane.
# @param: p      -> Array, vector from origin to point.
# @param: normal -> Array, plane normal vector.
# @param: offset -> Float, plane offset.
# @returns: Float
def dist2plane(p, normal, offset):
    if normal[X_AXIS_INDEX,0]!=0:
        return abs(dot_vector(np.array([[-offset/normal[X_AXIS_INDEX,0]], [0.], [0.]])-p, normal/np.linalg.norm(normal)))
    elif normal[Y_AXIS_INDEX,0]!=0:
        return abs(dot_vector(np.array([[0.], [-offset/normal[Y_AXIS_INDEX,0]], [0.]])-p, normal/np.linalg.norm(normal)))
    elif normal[Z_AXIS_INDEX,0]!=0:
        return abs(dot_vector(np.array([[0.], [0.], [-offset/normal[Z_AXIS_INDEX,0]]])-p, normal/np.linalg.norm(normal)))
    else:
        return None

# vector2plane: Vector from a point to a plane.
# @param: p      -> Array, vector from origin to point.
# @param: normal -> Array, plane normal vector.
# @param: offset -> Float, plane offset.
# @returns: Array
def vector2plane(p, normal, offset):
    if normal[X_AXIS_INDEX,0]!=0:
        return dot_vector(np.array([[-offset/normal[X_AXIS_INDEX,0]], [0.], [0.]])-p, normal/np.linalg.norm(normal))*normal/np.linalg.norm(normal)
    elif normal[Y_AXIS_INDEX,0]!=0:
        return dot_vector(np.array([[0.], [-offset/normal[Y_AXIS_INDEX,0]], [0.]])-p, normal/np.linalg.norm(normal))*normal/np.linalg.norm(normal)
    elif normal[Z_AXIS_INDEX,0]!=0:
        return dot_vector(np.array([[0.], [0.], [-offset/normal[Z_AXIS_INDEX,0]]])-p, normal/np.linalg.norm(normal))*normal/np.linalg.norm(normal)
    else:
        return None

# angle_vectors: Angle between vectors.
# @param: e       -> Array, vector e.
# @param: u       -> Array, vector u.
# @param: degrees -> Bool, degrees or radians.
# @returns: Float
def angle_vectors(e, u, degrees = False):
    return np.arccos(dot_vector(e,u)/np.linalg.norm(e)/np.linalg.norm(u))*(1.0*(not degrees) + 180/np.pi*(degrees))

# rot_align_axis: Method that aligns and axis from a coordinate frame to a vector.
# @param: vector  -> Array, vector to align.
# @param: matrix  -> Array, orientation matrix.
# @param: axis    -> Float, axis to align.
# @param: degrees -> Bool, degrees or radians.
# @returns: Array
def rot_align_axis(vector, matrix = BASE_MATRIX, axis = X_AXIS_INDEX, degrees = False):
    return rot_matrix_orientation_axis(matrix, 
                                       angle_vectors(matrix[:,[axis]], vector, degrees = degrees), 
                                       cross_vector(matrix[:,[axis]],vector)/np.linalg.norm(cross_vector(vector,matrix[:,[axis]])), 
                                       degrees = degrees)

# ---------------------------------------------------------------------------------- #
# ------------------------------- SCENE FUNCTIONS ---------------------------------- #
# ---------------------------------------------------------------------------------- #

# get_mrkr_pose: Method that returns an specific marker pose.
# @param: mrkr            -> String, marker name: BUTTON_MRKR, IMU_MRKR, IMU_DEST_MRKR,
#                            INSP_PNL_MRKR, INSP_PNL_CVR_MRKR, INSP_PNL_CVR_STG_MRKR.
# @param: button_position -> String, button position, in case, 
# @returns: Pose
def load_markers_poses(markers_pkg, markers_folder, markers_file_name):
    return yaml.safe_load(open(RosPack().get_path(markers_pkg)+markers_folder+markers_file_name,'r'))
            
# ---------------------------------------------------------------------------------- #
# ---------------------------- SECONDARY FUNCTIONS --------------------------------- #
# ---------------------------------------------------------------------------------- #

# delay: Method that generates a delay of miliseconds.
# @param: miliseconds -> int, time delay in miliseconds.
# @returns: array, waypoints.
def delay(miliseconds):
    time.sleep(miliseconds/1000)

# pose2array: Method that converts pose class to position and orientation arrays.
# @param: pose         -> Pose, pose class
# @param: or_as        -> String, desired rotation representation: ROT_MATRIX_REP, 
#                         ROT_QUAT_REP and ROT_EULER_REP, 
# @param: euler_rep    -> String, current euler representation, in case.
# @param: to_euler_rep -> String, desired euler representation, in case.
# @returns: Array
def pose2array(pose, or_as = ROT_QUAT_REP, euler_rep = 'zxz', to_euler_rep = 'zxz'):
    position_array = np.transpose(np.array([pose_to_list(pose)[0:3]]))
    _, orientation_array = rot2rot(np.array(pose_to_list(pose)[3:7]), rot_rep = or_as, in_euler = euler_rep, out_euler = to_euler_rep)
    return position_array, orientation_array

# pose2array: Method that converts position and orientation arrays to pose class.
# @param: position_array    -> Array, position array
# @param: orientation_array -> Array, orientation array
# @returns: Pose
def array2pose(position_array, orientation_array):
    _, orientation_quat = rot2rot(orientation_array, ROT_QUAT_REP)
    return list_to_pose([position_array[0,0], position_array[1,0], position_array[2,0], 
                            orientation_quat[0], orientation_quat[1], orientation_quat[2], orientation_quat[3]])

# pose2vectors: Method that converts pose (position and orientation) into vector.
# @param: pose -> Pose, actual pose.
# @returns: Arrays
def pose2vectors(pose):
    pos_vector = np.transpose(np.array([pose_to_list(pose)[0:3]]))
    _, or_matrix = rot2rot(np.array(pose_to_list(pose)[3:7]), rot_rep = ROT_MATRIX_REP)
    return pos_vector, or_matrix[:,[X_AXIS_INDEX]], or_matrix[:,[Y_AXIS_INDEX]], or_matrix[:,[Z_AXIS_INDEX]]