import numpy as np

import sys
sys.path.append('/workspaces/mavlab/')
from ros2_ws.src.student.tut_03.module_kinematics import Smat

def coriolis_matrix(M, vel):
    """
    Calculate the Coriolis matrix parametrized in terms of the mass 
    matrix M and the state vector.
    
    Args:
        M (array): Mass matrix (6x6)
        vel (array): Velocity vector (6x1)
        
    Returns:
        array: Coriolis matrix (6x6)
    """
    C = np.zeros((6, 6))

    #===========================================================================
    # TODO: Implement the code to calculate the Coriolis matrix
    #===========================================================================

    # Write your code here
    M11 = M[:3, :3]
    M12 = M[:3, 3:]
    M21 = M[3:, :3]
    M22 = M[3:, 3:]

    v1 = vel[:3]
    v2 = vel[3:]

    C[:3, 3:] = -Smat(M11 @ v1 + M12 @ v2)
    C[3:, :3] = -Smat(M11 @ v1 + M12 @ v2)
    C[3:, 3:] = -Smat(M21 @ v1 + M22 @ v2)
   
    #===========================================================================

    return C