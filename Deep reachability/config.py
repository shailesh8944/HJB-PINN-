"""Inputs: physics constants, state layout, and domain bounds for the
Sookshma two-player HJI reachability trainer. Single source of truth --
change values here, nothing else needs to change."""
import math

G          = 9.80665
MASS       = 20.0

# COUPLED effective mass matrix (body origin at centre of mass), identical to
# config/vessel_config.yaml and src/dynamics.py:
#   M = [[M11, 0,   0   ],
#        [ 0,  M22, M23 ],
#        [ 0,  M32, M33 ]]
# M11 = m - X_ud = 21.72 ; M22 = m - Y_vd = 29.214 ;
# M33 = Izz - N_rd = 2.44(CAD) + 0.4232 = 2.8632 ;
M11, M22, M33 = 21.7200, 29.2140, 2.8632
M23, M32   = 0.7626, 0.7614
DELTA      = M22 * M33 - M23 * M32               # determinant of the sway-yaw 2x2 block
ARM        = 0.21                                # thruster lateral arm [m] (CAD: y = +/-0.21)
FMAX       = 1.82 * G                            # 17.848 N per thruster (static)
ALPHA_U    = FMAX / math.sqrt(2.0)               # inscribed-ellipse semi-axes
ALPHA_R    = ARM * FMAX / math.sqrt(2.0)
SIG0, SIG1 = ALPHA_U ** 2, ALPHA_R ** 2
EPS_H      = 1.0e-6                              # rounds the Hamiltonian norm corner
R_COLLIDE  = 1.0                                 # collision radius [m]
T_HORIZON  = 20.0                                # reachability horizon [s]
DAMP_SIGN  = -1.0                                # D(v)v subtracted

# state indices: zeta = [X, Y, psi_rel, u_e, v_e, u_i, v_i, r_e, r_i]
IX, IY, IPSI, IUE, IVE, IUI, IVI, IRE, IRI = range(9)

# training-domain bounds for the 9 states (time bound t in [-T_HORIZON, 0] is
# appended separately wherever a 10-vector [state, t] is needed)
STATE_LO = [-15.0, -15.0, -math.pi, -0.2, -0.6, -0.2, -0.6, -1.5, -1.5]
STATE_HI = [ 15.0,  15.0,  math.pi,  1.2,  0.6,  1.2,  0.6,  1.5,  1.5]
