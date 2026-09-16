"""
thruster.py — Twin-thruster model for the Sookshma ASV.

Provides:
  * PWM <-> force lookup on the Blue Robotics T200 curve (16 V by default)
  * force clamping to the configured operating band [F_min, F_max] (kgf)
  * thrust allocation  (F_port, F_stbd) -> (tau_u, tau_r)

Allocation follows the paper's B = [[1, 1], [-d, d]]:
      tau_u = F_port + F_stbd                 (surge force, N)
      tau_r = arm * (F_stbd - F_port)         (yaw moment, N.m)
Body frame: x forward, y starboard, z down; +r = bow to starboard.
"""

import os
import csv
import numpy as np


class Thruster:
    def __init__(self, cfg, config_dir):
        t = cfg["thrusters"]
        self.count = t["count"]
        self.arm = float(t["arm_m"])
        self.g = float(t["gravity_m_s2"])
        self.f_min_kgf = float(t["force_min_kgf"])
        self.f_max_kgf = float(t["force_max_kgf"])

        # Load the performance curve (PWM, RPM, force) for lookups.
        curve_path = os.path.join(config_dir, t["curve_csv"])
        pwm, rpm, f_kgf, f_N = [], [], [], []
        with open(curve_path, newline="") as fh:
            for row in csv.DictReader(fh):
                pwm.append(float(row["pwm_us"]))
                rpm.append(float(row["rpm"]))
                f_kgf.append(float(row["force_kgf"]))
                f_N.append(float(row["force_N"]))
        # sort by force for monotonic force->pwm lookup on the forward branch
        idx = np.argsort(pwm)
        self.pwm = np.array(pwm)[idx]
        self.rpm = np.array(rpm)[idx]
        self.f_kgf = np.array(f_kgf)[idx]
        self.f_N = np.array(f_N)[idx]

    # ---- unit helpers ------------------------------------------------------
    def kgf_to_N(self, f_kgf):
        return f_kgf * self.g

    def clamp_kgf(self, f_kgf):
        return float(np.clip(f_kgf, self.f_min_kgf, self.f_max_kgf))

    # ---- curve lookups (forward branch, PWM >= neutral) --------------------
    def force_N_from_pwm(self, pwm_us):
        """Interpolate force [N] for a PWM [us] from the T200 curve."""
        return float(np.interp(pwm_us, self.pwm, self.f_N))

    def pwm_from_force_kgf(self, f_kgf):
        """Inverse lookup: PWM [us] that produces a given forward force [kgf]."""
        f_kgf = self.clamp_kgf(f_kgf)
        fwd = self.f_kgf >= -1e-9              # forward + neutral branch
        order = np.argsort(self.f_kgf[fwd])
        fk = self.f_kgf[fwd][order]
        pw = self.pwm[fwd][order]
        return float(np.interp(f_kgf, fk, pw))

    # ---- allocation --------------------------------------------------------
    def allocate(self, F_port_kgf, F_stbd_kgf):
        """
        Map two thruster forces (kgf) to generalised force tau = (tau_u, 0, tau_r).
        Returns tau [N, N, N.m] and the clamped forces actually applied (kgf).
        """
        Fp = self.clamp_kgf(F_port_kgf)
        Fs = self.clamp_kgf(F_stbd_kgf)
        Fp_N = self.kgf_to_N(Fp)
        Fs_N = self.kgf_to_N(Fs)
        tau_u = Fp_N + Fs_N
        tau_r = self.arm * (Fs_N - Fp_N)
        tau = np.array([tau_u, 0.0, tau_r])
        return tau, (Fp, Fs)

    # ---- reporting ---------------------------------------------------------
    def limits_N(self):
        return self.kgf_to_N(self.f_min_kgf), self.kgf_to_N(self.f_max_kgf)
