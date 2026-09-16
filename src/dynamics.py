"""
dynamics.py — Standard Fossen 3-DOF manoeuvring model for the Sookshma ASV.

State  s = [x, y, psi, u, v, r]
    x, y  : position in NED (inertial) frame  [m]
    psi   : heading                            [rad]
    u,v,r : body-frame surge, sway, yaw-rate   [m/s, m/s, rad/s]

Kinematics (body -> NED):
    x_dot   =  u cos(psi) - v sin(psi)
    y_dot   =  u sin(psi) + v cos(psi)
    psi_dot =  r

Kinetics (paper eqs 1-6):
    M v_dot + C(v) v + D(v) v = tau
    v_dot = M^{-1} [ tau - C(v) v - D(v) v ]

Mass matrix  M = M_rigid + M_added, with
    M_rigid = diag(m, m, Izz)
    M_added = [[ -X_ud,   0,     0    ],
               [   0,    -Y_vd, -Y_rd ],
               [   0,    -N_vd, -N_rd ]]
Added mass may be supplied as NON-DIMENSIONAL hydrodynamic derivatives and is
dimensionalized here with the prime (bis) system (0.5*rho*L^n).  The sway and
yaw axes are therefore coupled and M is a full (non-diagonal) 3x3 matrix.

Coriolis (skew-symmetric, general form built from M):
    C(v) = [[0, 0, -(m22 v + m23 r)],
            [0, 0,   m11 u         ],
            [ (m22 v + m23 r), -m11 u, 0]]
When the cross term m23 = 0 this reduces to the paper's diagonal-M Coriolis.

Damping  D(v) v = identified polynomial [X_D, Y_D, N_D] (from the config).
"""

import numpy as np


class VesselDynamics:
    def __init__(self, cfg):
        rb = cfg["rigid_body"]
        m = float(rb["mass_kg"])
        Izz = float(rb["Izz_kg_m2"])

        # ---- rigid-body inertia (CoG at origin, port-stbd symmetric) --------
        M_rb = np.diag([m, m, Izz]).astype(float)

        # ---- added mass -----------------------------------------------------
        am = cfg["added_mass"]
        X_ud, Y_vd, Y_rd, N_vd, N_rd = self._added_mass_terms(am)
        M_a = np.array([[-X_ud, 0.0,   0.0 ],
                        [ 0.0, -Y_vd, -Y_rd],
                        [ 0.0, -N_vd, -N_rd]], float)

        self.M = M_rb + M_a
        self.Minv = np.linalg.inv(self.M)
        # effective diagonal masses (kept for callers such as relative_dynamics.py)
        self.M11 = self.M[0, 0]
        self.M22 = self.M[1, 1]
        self.M33 = self.M[2, 2]
        self.m23 = self.M[1, 2]       # sway-yaw coupling (for Coriolis)

        # ---- model switches -------------------------------------------------
        mdl = cfg["model"]
        self.include_coriolis = bool(mdl["include_coriolis"])
        self.damping_sign = float(mdl["damping_sign"])
        self.active_set = mdl["active_coeff_set"]
        self.c = cfg["coefficients"][self.active_set]

    # -----------------------------------------------------------------------
    @staticmethod
    def _added_mass_terms(am):
        """Return dimensional (X_ud, Y_vd, Y_rd, N_vd, N_rd)."""
        source = am.get("source", "dimensional")
        if source == "nondimensional":
            rho = float(am["rho_kg_m3"])
            L = float(am["L_m"])
            s3 = 0.5 * rho * L ** 3     # surge / sway added mass
            s4 = 0.5 * rho * L ** 4     # sway-yaw cross terms
            s5 = 0.5 * rho * L ** 5     # yaw added inertia
            return (float(am["X_ud_nd"]) * s3,
                    float(am["Y_vd_nd"]) * s3,
                    float(am["Y_rd_nd"]) * s4,
                    float(am["N_vd_nd"]) * s4,
                    float(am["N_rd_nd"]) * s5)
        # dimensional fallback (accept either *_dot or 0 defaults)
        return (float(am.get("X_udot", 0.0)),
                float(am.get("Y_vdot", 0.0)),
                float(am.get("Y_rdot", 0.0)),
                float(am.get("N_vdot", 0.0)),
                float(am.get("N_rdot", 0.0)))

    # -----------------------------------------------------------------------
    def damping(self, u, v, r):
        """Identified damping vector D(v) v = [X_D, Y_D, N_D]."""
        c = self.c
        au, av, ar = abs(u), abs(v), abs(r)

        X_D = (c["Dx_u"] * u
               + c["Dx_u2"] * u * u
               + c["Dx_v2"] * v * v
               + c["Dx_r2"] * r * r
               + c["Dx_vr"] * (v * r)
               + c["Dx_u3"] * u ** 3
               + c["Dx_uv2"] * (u * v * v)
               + c["Dx_ur2"] * (u * r * r)
               + c["Dx_uvr"] * (u * v * r)
               + c["Dx_absu_u"] * (au * u))

        Y_D = (c["Dy_v"] * v
               + c["Dy_r"] * r
               + c["Dy_uv"] * (u * v)
               + c["Dy_ur"] * (u * r)
               + c["Dy_v3"] * v ** 3
               + c["Dy_r3"] * r ** 3
               + c["Dy_u2v"] * (u * u * v)
               + c["Dy_u2r"] * (u * u * r)
               + c["Dy_v2r"] * (v * v * r)
               + c["Dy_vr2"] * (v * r * r)
               + c["Dy_absv_v"] * (av * v)
               + c["Dy_u_absv"] * (u * av))

        N_D = (c["Dn_v"] * v
               + c["Dn_r"] * r
               + c["Dn_uv"] * (u * v)
               + c["Dn_ur"] * (u * r)
               + c["Dn_v3"] * v ** 3
               + c["Dn_r3"] * r ** 3
               + c["Dn_u2v"] * (u * u * v)
               + c["Dn_u2r"] * (u * u * r)
               + c["Dn_v2r"] * (v * v * r)
               + c["Dn_vr2"] * (v * r * r)
               + c["Dn_absr_r"] * (ar * r)
               + c["Dn_u_absr"] * (u * ar))

        return np.array([X_D, Y_D, N_D])

    def coriolis(self, u, v, r):
        """Skew-symmetric Coriolis force C(v) v built from the full M."""
        a = self.M22 * v + self.m23 * r          # = m22 v + m23 r
        C = np.array([[0.0,        0.0,      -a       ],
                      [0.0,        0.0,       self.M11 * u],
                      [a,   -self.M11 * u,    0.0      ]])
        return C @ np.array([u, v, r])

    # -----------------------------------------------------------------------
    def nu_dot(self, nu, tau):
        """Body-frame acceleration v_dot given velocity nu and generalised force tau."""
        u, v, r = nu
        rhs = tau + self.damping_sign * self.damping(u, v, r)   # tau - D(v)v
        if self.include_coriolis:
            rhs = rhs - self.coriolis(u, v, r)                  # - C(v) v
        return self.Minv @ rhs

    def state_dot(self, s, tau):
        """Full 6-state derivative s_dot = f(s, tau)."""
        x, y, psi, u, v, r = s
        cpsi, spsi = np.cos(psi), np.sin(psi)
        x_dot = u * cpsi - v * spsi
        y_dot = u * spsi + v * cpsi
        psi_dot = r
        nu_dot = self.nu_dot(np.array([u, v, r]), tau)
        return np.array([x_dot, y_dot, psi_dot, nu_dot[0], nu_dot[1], nu_dot[2]])
