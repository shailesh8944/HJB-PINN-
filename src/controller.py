"""
controller.py — Guidance and control for waypoint tracking.

Guidance : line-of-sight (LOS) toward the active waypoint, with an acceptance
           radius that advances to the next waypoint.
Control  : heading PD (proportional-derivative) -> yaw moment tau_r
           speed   PI (proportional-integral)    -> surge force tau_u
Allocation: inverse of the thrust map, with per-thruster clamping to the
            forward-only 0-1.82 kgf band. Yaw authority saturates realistically.
"""

import numpy as np


def wrap_pi(a):
    """Wrap an angle to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class WaypointController:
    def __init__(self, cfg, thruster):
        m = cfg["mission"]
        g = m["gains"]
        self.wps = [np.array(w, float) for w in m["waypoints"]]
        self.accept = float(m["acceptance_radius_m"])
        self.slow_r = float(m["slowdown_radius_m"])
        self.u_cruise = float(m["cruise_speed_mps"])

        self.Kp_u = float(g["speed_Kp"])
        self.Ki_u = float(g["speed_Ki"])
        self.Imax = float(g["speed_Imax"])
        self.Kp_psi = float(g["heading_Kp"])
        self.Kd_psi = float(g["heading_Kd"])

        self.thr = thruster
        self.idx = 0                 # active waypoint index
        self.u_int = 0.0             # speed integral state
        self.done = False

    def active_waypoint(self):
        return self.wps[min(self.idx, len(self.wps) - 1)]

    def is_final(self):
        return self.idx >= len(self.wps) - 1

    def update(self, state, dt):
        """
        Given full state [x,y,psi,u,v,r], return:
          tau_cmd (desired [tau_u,0,tau_r]), tau_app (after saturation),
          (F_port,F_stbd) applied kgf, and telemetry dict.
        """
        x, y, psi, u, v, r = state
        wp = self.active_waypoint()
        dx, dy = wp[0] - x, wp[1] - y
        dist = np.hypot(dx, dy)

        # ---- waypoint switching -------------------------------------------
        if dist < self.accept:
            if self.is_final():
                self.done = True
            else:
                self.idx += 1
                wp = self.active_waypoint()
                dx, dy = wp[0] - x, wp[1] - y
                dist = np.hypot(dx, dy)

        # ---- LOS guidance: desired heading toward the waypoint ------------
        psi_d = np.arctan2(dy, dx)
        e_psi = wrap_pi(psi_d - psi)

        # ---- speed command: cruise, easing down near the final waypoint ---
        u_d = self.u_cruise
        if self.is_final():
            u_d = self.u_cruise * min(1.0, dist / max(self.slow_r, 1e-6))
        # Reduce speed while heading error is large (turn first, then run)
        u_d *= max(0.15, np.cos(e_psi))

        # ---- heading PD -> yaw moment -------------------------------------
        tau_r = self.Kp_psi * e_psi - self.Kd_psi * r

        # ---- speed PI (with anti-windup) -> surge force -------------------
        e_u = u_d - u
        self.u_int = np.clip(self.u_int + e_u * dt, -self.Imax / max(self.Ki_u, 1e-9),
                             self.Imax / max(self.Ki_u, 1e-9))
        tau_u = self.Kp_u * e_u + self.Ki_u * self.u_int
        tau_u = max(0.0, tau_u)          # forward-only

        tau_cmd = np.array([tau_u, 0.0, tau_r])

        # ---- allocation with saturation -----------------------------------
        tau_app, (Fp, Fs) = self._allocate(tau_u, tau_r)

        tele = dict(dist=dist, e_psi=e_psi, psi_d=psi_d, u_d=u_d,
                    wp_idx=self.idx, tau_u_cmd=tau_u, tau_r_cmd=tau_r)
        return tau_cmd, tau_app, (Fp, Fs), tele

    def _allocate(self, tau_u, tau_r):
        """Invert tau=(tau_u,tau_r) to thruster forces, clamp, recompute actual tau."""
        g = self.thr.g
        d = self.thr.arm
        Fmax = self.thr.kgf_to_N(self.thr.f_max_kgf)
        Fmin = self.thr.kgf_to_N(self.thr.f_min_kgf)
        # desired forces in Newtons
        Fs_N = tau_u / 2.0 + tau_r / (2.0 * d)
        Fp_N = tau_u / 2.0 - tau_r / (2.0 * d)
        # clamp to forward-only band
        Fs_N = np.clip(Fs_N, Fmin, Fmax)
        Fp_N = np.clip(Fp_N, Fmin, Fmax)
        # actual generalised force after clamping
        tau_u_a = Fp_N + Fs_N
        tau_r_a = d * (Fs_N - Fp_N)
        tau_app = np.array([tau_u_a, 0.0, tau_r_a])
        return tau_app, (Fp_N / g, Fs_N / g)   # forces back in kgf
