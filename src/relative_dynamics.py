"""
relative_dynamics.py — 9-state relative pursuit dynamics for one pursuer
(subscript i) chasing one APF-driven evader (subscript e), both Sookshma ASVs.

State (pursuer body frame):
    zeta = [X, Y, psi_rel, u_e, v_e, u_i, v_i, r_e, r_i]  in R^9
             0  1    2       3    4    5    6    7    8
    (X, Y)   : evader position relative to the pursuer, in the pursuer body frame
    psi_rel  : relative heading  psi_e - psi_i
    u_*,v_*,r_*: surge, sway, yaw-rate of each vessel (own body frame)

The system is control-affine in the PURSUER generalised force u_i = (tau_u^i, tau_r^i):
    zeta_dot = abar(zeta) + G_i @ u_i

The drift abar collects (a) the relative kinematics, (b) the evader velocity
dynamics closed by the APF law, and (c) the CONTROL-INDEPENDENT part of the
pursuer velocity dynamics. With coupled sway-yaw added mass, yaw moment enters
both the v_i and r_i rows through the same M^{-1} used by VesselDynamics.

Relative kinematics (pursuer body frame, evader at (X, Y)):
    X_dot     = u_e cos(psi_rel) - v_e sin(psi_rel) - u_i + r_i * Y
    Y_dot     = u_e sin(psi_rel) + v_e cos(psi_rel) - v_i - r_i * X
    psi_rel_dot = r_e - r_i

Velocity dynamics use the vessel's own Fossen model (dynamics.VesselDynamics).
Because nu_dot is affine in tau  (nu_dot(nu, tau) = nu_dot(nu, 0) + M^{-1} tau),
the pursuer rows split cleanly into a drift part nu_dot(nu_i, 0) plus G_i @ u_i.
"""

import numpy as np

# state indices
IX, IY, IPSI = 0, 1, 2
IUE, IVE = 3, 4
IUI, IVI = 5, 6
IRE, IRI = 7, 8


class RelativeDynamics:
    def __init__(self, pursuer_dyn, evader_dyn, apf_evader):
        """
        pursuer_dyn, evader_dyn : VesselDynamics instances (may share a config or
                                  differ only in the active damping-coefficient set)
        apf_evader              : APFEvader instance providing the evader control
        """
        self.pursuer = pursuer_dyn
        self.evader = evader_dyn
        self.apf = apf_evader

        # Exact control columns from nu_dot = nu_dot0 + M^{-1}[tau_u,0,tau_r].
        self.G = np.zeros((9, 2))
        self.G[IUI, 0] = self.pursuer.Minv[0, 0]
        self.G[IVI, 1] = self.pursuer.Minv[1, 2]
        self.G[IRI, 1] = self.pursuer.Minv[2, 2]

    # -----------------------------------------------------------------------
    def G_matrix(self):
        """Constant 9x2 pursuer input matrix G_i."""
        return self.G

    # -----------------------------------------------------------------------
    def drift(self, zeta):
        """Drift abar(zeta) in R^9 (evader APF folded in, pursuer control excluded)."""
        X, Y = zeta[IX], zeta[IY]
        psi = zeta[IPSI]
        u_e, v_e, r_e = zeta[IUE], zeta[IVE], zeta[IRE]
        u_i, v_i, r_i = zeta[IUI], zeta[IVI], zeta[IRI]
        cp, sp = np.cos(psi), np.sin(psi)

        a = np.zeros(9)

        # (a) relative kinematics
        a[IX] = u_e * cp - v_e * sp - u_i + r_i * Y
        a[IY] = u_e * sp + v_e * cp - v_i - r_i * X
        a[IPSI] = r_e - r_i

        # (b) evader velocity dynamics, closed by the APF law
        tau_e = self.apf.tau(zeta)                       # realised evader force [tau_u,0,tau_r]
        due, dve, dre = self.evader.nu_dot(np.array([u_e, v_e, r_e]), tau_e)
        a[IUE], a[IVE], a[IRE] = due, dve, dre

        # (c) pursuer velocity dynamics, CONTROL-INDEPENDENT part (tau = 0)
        dui0, dvi0, dri0 = self.pursuer.nu_dot(np.array([u_i, v_i, r_i]), np.zeros(3))
        a[IUI], a[IVI], a[IRI] = dui0, dvi0, dri0

        return a

    # -----------------------------------------------------------------------
    def zeta_dot(self, zeta, tau_i):
        """Full relative-state derivative: abar(zeta) + G_i @ (tau_u^i, tau_r^i)."""
        tau_i = np.asarray(tau_i, float).reshape(2)
        return self.drift(zeta) + self.G @ tau_i

    # -----------------------------------------------------------------------
    def rk4_step(self, zeta, tau_i, dt):
        """One fixed-step RK4 update of the relative dynamics (constant tau_i over dt)."""
        k1 = self.zeta_dot(zeta, tau_i)
        k2 = self.zeta_dot(zeta + 0.5 * dt * k1, tau_i)
        k3 = self.zeta_dot(zeta + 0.5 * dt * k2, tau_i)
        k4 = self.zeta_dot(zeta + dt * k3, tau_i)
        return zeta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
