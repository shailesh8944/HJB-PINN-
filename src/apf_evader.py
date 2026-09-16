"""
apf_evader.py — Artificial-potential-field (APF) feedback law for the evader.

The evader does not optimise: it runs a fixed repulsive APF policy that drives it
directly away from the pursuer at a commanded flee speed. Because this law is a
known function of the *relative* state, the two-player game collapses to a
single-player optimal control problem for the pursuer (hence HJB, not HJI).

Relative state (pursuer body frame):
    zeta = [X, Y, psi_rel, u_e, v_e, u_i, v_i, r_e, r_i]
             0  1    2       3    4    5    6    7    8
The pursuer is at the origin; the evader is at (X, Y). "Away from the pursuer"
is therefore the bearing beta = atan2(Y, X) measured in the pursuer frame, and
the evader's heading in that same frame is psi_rel, so the flee heading error is
    e_psi = wrap(beta - psi_rel),
which is expressible purely in zeta (no absolute headings needed).

The APF law produces a desired generalised force (surge force, yaw moment); this
is then allocated to the evader's two forward-only thrusters and clamped to the
configured band, so the evader obeys the same actuator limits as the pursuer.
"""

import numpy as np


def wrap_pi(a):
    """Wrap an angle to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class APFEvader:
    def __init__(self, game_cfg, thruster):
        apf = game_cfg["evader_apf"]
        self.V_e_max = float(apf["V_e_max_mps"])
        self.K_u = float(apf["K_u"])
        self.K_psi = float(apf["K_psi"])
        self.K_r = float(apf["K_r"])
        self.thr = thruster                       # for allocation + clamping
        self.d = self.thr.arm
        self.g = self.thr.g
        self.Fmax_N = self.thr.kgf_to_N(self.thr.f_max_kgf)
        self.Fmin_N = self.thr.kgf_to_N(self.thr.f_min_kgf)

    # -----------------------------------------------------------------------
    def command(self, zeta):
        """
        Desired evader generalised force BEFORE actuator clamping.
        Returns tau_cmd = [tau_u, 0, tau_r]  (N, N, N.m).
        """
        X, Y = zeta[0], zeta[1]
        psi_rel = zeta[2]
        u_e, r_e = zeta[3], zeta[7]

        beta = np.arctan2(Y, X)                   # escape bearing, pursuer frame
        e_psi = wrap_pi(beta - psi_rel)           # flee heading error

        tau_u = self.K_u * (self.V_e_max - u_e)
        tau_u = max(0.0, tau_u)                    # forward-only surge command
        tau_r = self.K_psi * e_psi - self.K_r * r_e
        return np.array([tau_u, 0.0, tau_r])

    # -----------------------------------------------------------------------
    def tau(self, zeta):
        """
        Realised evader generalised force AFTER allocation + forward-only clamp,
        i.e. what the two evader thrusters can actually deliver.
        Returns tau_app = [tau_u, 0, tau_r]  (N, N, N.m).
        """
        tau_cmd = self.command(zeta)
        return self._allocate_clamp(tau_cmd[0], tau_cmd[2])

    # -----------------------------------------------------------------------
    def _allocate_clamp(self, tau_u, tau_r):
        """Invert (tau_u, tau_r) -> thruster forces, clamp to [Fmin, Fmax] (N),
        recompute the generalised force actually applied."""
        d = self.d
        Fs_N = tau_u / 2.0 + tau_r / (2.0 * d)
        Fp_N = tau_u / 2.0 - tau_r / (2.0 * d)
        Fs_N = np.clip(Fs_N, self.Fmin_N, self.Fmax_N)
        Fp_N = np.clip(Fp_N, self.Fmin_N, self.Fmax_N)
        tau_u_a = Fp_N + Fs_N
        tau_r_a = d * (Fs_N - Fp_N)
        return np.array([tau_u_a, 0.0, tau_r_a])
