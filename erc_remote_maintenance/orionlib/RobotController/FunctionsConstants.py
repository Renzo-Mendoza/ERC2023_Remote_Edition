#!/usr/bin/env python3

# ===================================================================================== #
# =================== ERC2023 REMOTE EDITION - FUNCTIONS CONSTANTS ==================== #
# ===================================================================================== #

import numpy as np

# GeomEtric Constants
ROT_MATRIX_REP     = 'rot'
ROT_QUAT_REP       = 'quat'
ROT_EULER_REP      = 'euler'
ROT_VECTOR_REP     = "rotvector"
X_AXIS             = np.array([[1], 
                               [0], 
                               [0]])
Y_AXIS             = np.array([[0], 
                               [1], 
                               [0]])
Z_AXIS             = np.array([[0], 
                               [0], 
                               [1]])
BASE_MATRIX        = np.array([[1, 0, 0], 
                               [0, 1, 0], 
                               [0, 0, 1]])
X_AXIS_INDEX       = 0
Y_AXIS_INDEX       = 1
Z_AXIS_INDEX       = 2