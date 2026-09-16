import numpy as np
import warnings

# Coordinate frames:
#
# 1. Earth Centered Inertial (ECI) frame - {i}
# 2. Earth Centered Earth Fixed (ECEF) frame - {e}
# 3. North-East-Down (NED) frame - {n}
# 4. BODY frame - {b}
#
# You may further assume the following:
# 
# eul = [phi, theta, psi] 
# 
# with the order of rotation being ZYX and
#   phi being angle about x-axis
#   theta being angle about y-axis
#   psi being angle about z-axis
#
# We assume ZYX rotation order in this file. For example,
# this means that the NED frame is rotated about 
# its z-axis by angle psi followed by a rotation about 
# the resultant y-axis by angle theta followed by a 
# rotation about the resultant x-axis by angle phi
# to reach the BODY frame.
#
# quat = [qw, qx, qy, qz] is considered to be an unit quaternion
#
# rotm = 3 x 3 matrix
# 
# The rotation matrix rotm when pre-multiplied by a vector
# in BODY frame will yield a vector in NED frame

def Smat(vec):
    """
    Calculate the skew-symmetric matrix for a given vector
    
    Args:
        vec (array): Vector to be converted to skew-symmetric matrix
        
    Returns:
        array: Skew-symmetric matrix
    """
    S = np.zeros((3,3))
    #===========================================================================
    # TODO: Implement the skew-symmetric matrix
    #===========================================================================
    # Write your code here
    S = np.array([[0, -vec[2], vec[1]],
                  [vec[2], 0, -vec[0]],
                  [-vec[1], vec[0], 0]])
    
     
    #===========================================================================               
    return S
Smat(np.array([1,2,3]))

def eul_to_rotm(angles):
    """
    Convert Euler angles to rotation matrix
    
    Args:
        angles (array): Euler angles [phi, theta, psi]
        
    Returns:
        array: Rotation matrix
    """
    R = np.zeros((3,3))

    phi = angles[0]
    theta = angles[1]
    psi = angles[2]

    #===========================================================================
    # TODO: Implement the rotation matrix
    #===========================================================================
    # Write your code here
    R = np.array([[np.cos(psi)*np.cos(theta), -np.sin(psi)*np.cos(phi) + np.cos(psi)*np.sin(theta)*np.sin(phi), np.sin(psi)*np.sin(phi) + np.cos(psi)*np.cos(phi)*np.sin(theta)],
                  [np.sin(psi)*np.cos(theta), np.cos(psi)*np.cos(phi) + np.sin(phi)*np.sin(theta)*np.sin(psi), -np.cos(psi)*np.sin(phi) + np.sin(theta)*np.sin(psi)*np.cos(phi)],
                  [-np.sin(theta), np.cos(theta)*np.sin(phi), np.cos(theta)*np.cos(phi)]])

      # return R
    #===========================================================================
    
    return R

def rotm_to_eul(rotm, order='ZYX', deg=False):
    """
    Convert rotation matrix to Euler angles
    
    Args:
        rotm (array): Rotation matrix
        order (str): Order of Euler angles (default: 'ZYX')
        deg (bool): Return angles in degrees (default: False)
        
    Returns:
        array: Euler angles
    """

    # Initialize the Euler angles
    eul = np.zeros(3, dtype=float)

    # Check if the order is ZYX
    if order != 'ZYX':
        raise ValueError('Any order other than ZYX is not currently available!')

    if order == 'ZYX':

        #===========================================================================
        # TODO: Implement the code to convert rotation matrix to Euler angles
        #===========================================================================

        # Write your code here
        theta = -np.arcsin(rotm[2, 0])
        phi = np.arctan2(rotm[2, 1], rotm[2, 2])
        psi = np.arctan2(rotm[1, 0], rotm[0, 0])

        eul = np.array([phi, theta, psi])



         
        #===========================================================================

        if deg:
            eul = eul * 180 / np.pi       

    return eul

def eul_rate_matrix(angles, order='ZYX'):
    """
    Calculate the Euler rate matrix J2
    
    Args:
        angles (array): Euler angles [phi, theta, psi]
        
    Returns:
        array: Euler rate matrix
    """
    J2 = np.zeros((3,3))
    
    phi = angles[0]
    theta = angles[1]
    psi = angles[2]

    #===========================================================================
    # TODO: Implement the Euler rate matrix J2
    #===========================================================================
    # Write your code here
    J2 = np.array([[1, np.sin(phi)*np.sin(theta)/np.cos(theta), np.cos(phi)*np.sin(theta)/np.cos(theta)],
                   [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]])

     
    #===========================================================================
    
    return J2

def eul_rate(eul, w_nb_b, order='ZYX'):
    """
    Calculate the Euler rate
    
    Args:
        eul (array): Euler angles [phi, theta, psi]
        w_nb_b (array): Angular velocity in body frame [wx, wy, wz]
        order (str): Order of Euler angles (default: 'ZYX')
        
    Returns:
        array: Euler rate
    """
    deul = np.zeros(3, dtype=float)

    if order != 'ZYX':
        raise ValueError('Any order other than ZYX is not currently available!')

    if order == 'ZYX':

        #===========================================================================
        # TODO: Implement the code to calculate the Euler rate
        #===========================================================================

        # Write your code here
        J2 = eul_rate_matrix(eul)
        deul = J2 @ w_nb_b

         
        #===========================================================================

    return deul

def eul_to_quat(eul, order='ZYX', deg=False):
    """
    Convert Euler angles to quaternion
    
    Args:
        eul (array): Euler angles [phi, theta, psi]
        
    Returns:
        array: Quaternion
    """
    quat = np.zeros(4, dtype=float)
    quat[0] = 1.0

    if order != 'ZYX':
        raise ValueError('Any order other than ZYX is not currently available!')

    # Write your code here

    if order == 'ZYX':
        
        if deg:
            phi = eul[0] * np.pi / 180
            theta = eul[1] * np.pi / 180
            psi = eul[2] * np.pi / 180
        else:
            phi = eul[0]
            theta = eul[1]
            psi = eul[2]

        #===========================================================================
        # TODO: Implement the code to convert Euler angles to quaternion
        #===========================================================================

        # Write your code here
        qw = np.cos(phi/2) * np.cos(theta/2) * np.cos(psi/2) + np.sin(phi/2) * np.sin(theta/2) * np.sin(psi/2)
        qx = np.sin(phi/2) * np.cos(theta/2) * np.cos(psi/2) - np.cos(phi/2) * np.sin(theta/2) * np.sin(psi/2)
        qy = np.cos(phi/2) * np.sin(theta/2) * np.cos(psi/2) + np.sin(phi/2) * np.cos(theta/2) * np.sin(psi/2)
        qz = np.cos(phi/2) * np.cos(theta/2) * np.sin(psi/2) - np.sin(phi/2) * np.sin(theta/2) * np.cos(psi/2)
        quat = np.array([qw,qx,qy,qz])

         
        #===========================================================================

        quat = quat / np.linalg.norm(quat)

    return quat

def quat_to_eul(quat, order='ZYX', deg=False):
    """
    Convert quaternion to Euler angles
    
    Args:
        quat (array): Quaternion
        order (str): Order of Euler angles (default: 'ZYX')
        deg (bool): Return angles in degrees (default: False)
        
    Returns:
        array: Euler angles
    """
    eul = np.zeros(3, dtype=float)
    
    if order != 'ZYX':
        raise ValueError('Any order other than ZYX is not currently available!')

    # Write your code here

    if order == 'ZYX':
        qw = quat[0]
        qx = quat[1]
        qy = quat[2]
        qz = quat[3]

        #===========================================================================
        # TODO: Implement the code to convert quaternion to Euler angles
        #===========================================================================

        # Write your code here
        phi = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
        theta = -np.arcsin(2 * (qz * qx - qw * qy))
        psi = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    
        eul = np.array([phi, theta, psi])


         
        #===========================================================================

    return eul

def quat_to_rotm(quat):    
    """
    Convert quaternion to rotation matrix
    
    Args:
        quat (array): Quaternion [qw, qx, qy, qz]
        
    Returns:
        array: Rotation matrix
    """
    rotm = np.eye(3, dtype=float)

    #===========================================================================
    # TODO: Implement the code to convert quaternion to rotation matrix
    #===========================================================================

    # Write your code here
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]

    rotm = np.array([[1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                     [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
                     [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)]])



     
    #===========================================================================

    return rotm

def quat_multiply(q1, q2):
    """
    Multiply two quaternions
    
    Args:
        q1 (array): Quaternion 1
        q2 (array): Quaternion 2
        
    Returns:
        array: Resultant quaternion
    """
    w1 = q1[0]; x1 = q1[1]; y1 = q1[2]; z1 = q1[3]
    w2 = q2[0]; x2 = q2[1]; y2 = q2[2]; z2 = q2[3]

    q_prod = np.zeros(4, dtype=float)
    q_prod[0] = 1.0

    #===========================================================================
    # TODO: Implement the code to multiply two quaternions
    #===========================================================================

    # Write your code here
    w_prod = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x_prod = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y_prod = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z_prod = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    q_prod = np.array([w_prod,x_prod,y_prod,z_prod])

     
    #===========================================================================

    return q_prod

def quat_conjugate(quat):
    """
    Calculate the conjugate of a quaternion
    
    Args:
        quat (array): Quaternion
        
    Returns:
        array: Conjugate of the quaternion
    """
    q_conj = np.zeros(4, dtype=float)
    q_conj[0] = 1.0
    
    #===========================================================================
    # TODO: Implement the code to calculate the conjugate of a quaternion
    #===========================================================================

    # Write your code here
    q_conj = np.array([quat[0], -quat[1], -quat[2], -quat[3]])

     
    #===========================================================================

    return q_conj

def rotate_vec_by_quat(vec_a, q_a_b):
    """
    Rotate a vector by a quaternion
    
    Args:
        vec_a (array): Vector to be rotated
        q_a_b (array): Quaternion
        
    Returns:
        array: Rotated vector
    """
    vec_b = np.zeros(3, dtype=float)
    #===========================================================================
    # TODO: Implement the code to rotate a vector by a quaternion
    #================== =========================================================

    # Write your code here
    q_vec = np.array([0, *vec_a])
    rot_vec_b = quat_multiply(quat_multiply(q_a_b,q_vec),quat_conjugate(q_a_b))
    vec_b = rot_vec_b[1:]

     
    #===========================================================================

    return vec_b

def quat_rate_matrix(quat):
    """
    Calculate the quaternion rate matrix
    
    Args:
        quat (array): Quaternion
        
    Returns:
        array: Quaternion rate matrix
    """
    Tmat = np.zeros((4, 3))

    #===========================================================================
    # TODO: Implement the code to calculate the quaternion rate matrix
    #===========================================================================

    # Write your code here
    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]
    Tmat = 0.5 * np.array([[-q1, -q2, -q3],
                           [ q0, -q3,  q2],
                           [ q3,  q0, -q1],
                           [-q2,  q1,  q0]])

     
    #===========================================================================

    return Tmat

def quat_rate(quat, w_nb_b):
    """
    Calculate the quaternion rate
    
    Args:
        quat (array): Quaternion
        w_nb_b (array): Angular velocity in body frame [wx, wy, wz]
        
    Returns:
        array: Quaternion rate
    """
    dquat = np.zeros(4, dtype=float)

    #===========================================================================
    # TODO: Implement the code to calculate the quaternion rate
    #===========================================================================

    # Write your code here
    Tmat = quat_rate_matrix(quat)
    dquat = Tmat @ w_nb_b

     
    #===========================================================================

    return dquat

def ssa(ang, deg=False):
    """
    Smallest Signed Angle (SSA) function to wrap angle to [-180, 180] degrees or 
    [-pi, pi] radians
    
    Args:
        ang (float): Angle to be wrapped
        deg (bool): Return angle in degrees (default: False)
        
    Returns:
        float: Wrapped angle
    """

    #===========================================================================
    # TODO: Implement the code to wrap angle to [-180, 180] degrees or 
    # [-pi, pi] radians
    #===========================================================================

    # Write your code here
    if deg:
         ang = ((ang + 180) % 360) - 180
    else:
         ang = ((ang + np.pi) % (2 * np.pi)) - np.pi

     
    #===========================================================================

    return ang

def ned_to_llh(ned, llh0):
    """
    Convert NED coordinates to LLH coordinates
    
    Args:
        ned (array): NED coordinates [xn, yn, zn]
        llh0 (array): Initial LLH coordinates [mu0, l0, h0]
        
    Returns:
        array: LLH coordinates [mu, l, h]
    """
    xn = ned[0]
    yn = ned[1]
    zn = ned[2]

    mu0 = llh0[0] * np.pi / 180
    l0 = llh0[1] * np.pi / 180
    h0 = llh0[2]

    llh = np.zeros(3, dtype=float)

    #===========================================================================
    # TODO: Implement the code to convert NED coordinates to LLH coordinates
    #===========================================================================

    # Write your code here
    re = 6378137.0 # Equatorial Radius
    e2 = 0.0818**2 # Eccentricity squared
    Rn = re/np.sqrt(1 - e2*(np.sin(mu0)**2))
    Rm = Rn * (1 - e2)/np.sqrt(1 - e2*(np.sin(mu0)**2))
    del_l = yn * np.arctan2(1, Rm*np.cos(mu0))
    del_mu = xn * np.atan2(1,Rn)

    l = np.rad2deg(ssa(l0 + del_l))
    mu = np.rad2deg(ssa(mu0 + del_mu))
    h = h0 - zn

    llh = np.array([mu, l, h])
     
    #===========================================================================

    return llh

def llh_to_ned(llh, llh0):
    """
    Convert LLH coordinates to NED coordinates
    
    Args:
        llh (array): LLH coordinates [mu, l, h]
        llh0 (array): Initial LLH coordinates [mu0, l0, h0]
        
    Returns:
        array: NED coordinates [xn, yn, zn]
    """
    
    ned = np.zeros(3, dtype=float)
    #===========================================================================
    # TODO: Implement the code to convert LLH coordinates to NED coordinates
    #===========================================================================

    # Write your code here
    mu0 = np.radians(llh0[0])
    l0 = np.radians(llh0[1])
    h0 = llh0[2]
    
    mu = np.radians(llh[0])
    l = np.radians(llh[1])
    h = llh[2]
    
    re = 6378137.0 # Equatorial Radius
    e2 = 0.0066943800230119255 # Eccentricity squared
    Rn = re/np.sqrt(1 - e2*(np.sin(mu0)**2))
    Rm = Rn * (1 - e2)/np.sqrt(1 - e2*(np.sin(mu0)**2))

    xn = (mu - mu0) / np.arctan2(1,Rn)
    yn = (l - l0) / np.arctan2(1, Rm*np.cos(mu0))
    zn = h0 - h

    ned = np.array([xn, yn, zn], dtype=np.float64)

     
    #===========================================================================

    return ned

def rotm_ned_to_ecef(llh):
    """
    Calculate the rotation matrix from NED to ECEF frame
    
    Args:
        llh (array): LLH coordinates [mu, l, h]
        
    Returns:
        array: Rotation matrix
    """
    mu = llh[0] * np.pi / 180
    l = llh[1] * np.pi / 180
    h = llh[2]

    rotm = np.zeros((3,3), dtype=float)

    #===========================================================================
    # TODO: Implement the code to calculate the rotation matrix from NED to ECEF frame
    #===========================================================================

    # Write your code here

    rotm = np.array([[-np.cos(l)*np.sin(mu), -np.sin(l) , -np.cos(l)*np.cos(mu)],
                     [-np.sin(l)*np.sin(mu), np.cos(l), -np.sin(l)*np.cos(mu)],
                     [np.cos(mu), 0, -np.sin(mu)]])

   
    #===========================================================================

    return rotm

def ecef_to_llh(ecef):
    """
    Convert ECEF coordinates to LLH coordinates
    
    Args:
        ecef (array): ECEF coordinates [xe, ye, ze]
        llh0 (array): Initial LLH coordinates [mu0, l0, h0]
        
    Returns:
        array: LLH coordinates [mu, l, h]
    """
    llh = np.zeros(3, dtype=float)

    #===========================================================================
    # TODO: Implement the code to convert ECEF coordinates to LLH coordinates
    #===========================================================================

    # Write your code here
    re = 6378137.0 # Equatorial Radius
    rp = 6356752.314245 # Polar Radius
    e2 = 0.0066943800230119255 # Eccentricity squared

    xe = ecef[0]
    ye = ecef[1]
    ze = ecef[2]

    l = np.arctan2(ye, xe)
    p = np.sqrt(xe**2 + ye**2)
    mu = np.arctan2(ze, p*(1 - e2))
    mu0 = 0

    while abs(mu - mu0) > 1e-9:
        mu0 = mu
        N = re**2 / np.sqrt((re*np.cos(mu0))**2 + (rp*np.sin(mu0))**2)
        h = (p/np.cos(mu0)) - N
        mu = np.arctan2(ze, p*(1 - e2*(N/(N+h))))

    llh = np.array([np.rad2deg(mu), np.rad2deg(l), h], dtype=np.float64)





    
    #===========================================================================

    return llh

def llh_to_ecef(llh):
    """
    Convert LLH coordinates to ECEF coordinates
    
    Args:
        llh (array): LLH coordinates [mu, l, h]
        
    Returns:
        array: ECEF coordinates [xe, ye, ze]
    """
    ecef = np.zeros(3, dtype=float)

    #===========================================================================
    # TODO: Implement the code to convert LLH coordinates to ECEF coordinates
    #===========================================================================

    # Write your code here
    mu = np.radians(llh[0])
    l = np.radians(llh[1])
    h = llh[2]

    re = 6378137.0 # Equatorial Radius
    rp = 6356752.314245 # Polar Radius
    e2 = 0.0818**2 # Eccentricity squared

    
    N = re**2 / np.sqrt((re*np.cos(mu))**2 + (rp*np.sin(mu))**2)
    
    x = (N + h) * np.cos(mu) * np.cos(l)
    y = (N + h) * np.cos(mu) * np.sin(l)
    z = ((rp**2/re**2) * N + h) * np.sin(mu)

    ecef = np.array([x, y, z])

    
    #===========================================================================

    return ecef
def clip(value, threshold):
    """Clip a value to +/- threshold.
    
    Args:
        value (float): Input value
        threshold (float or None): Maximum absolute value. If None, return value unchanged
    
    Returns:
        float: Clipped value
    """
    if threshold is None:
        return value
    if np.abs(value) > threshold:
        return np.sign(value) * threshold
    return value