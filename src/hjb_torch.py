"""
hjb_torch.py — Differentiable, batched PyTorch port of the relative drift,
the HJB Hamiltonian, and the terminal function, for use inside the
physics-informed network. Every quantity here is a faithful re-implementation of
the NumPy modules (dynamics.py, apf_evader.py, relative_dynamics.py, hjb.py);
train_pinn.py validates the two against each other before training.

State (batched):  zeta shape (B, 9) = [X, Y, psi_rel, u_e, v_e, u_i, v_i, r_e, r_i]
Costate (batched): p shape (B, 9) = grad_zeta V
"""

import numpy as np
import torch

# state indices
IX, IY, IPSI = 0, 1, 2
IUE, IVE = 3, 4
IUI, IVI = 5, 6
IRE, IRI = 7, 8

_DAMP_KEYS_X = ["Dx_u", "Dx_u2", "Dx_v2", "Dx_r2", "Dx_vr", "Dx_u3",
                "Dx_uv2", "Dx_ur2", "Dx_uvr", "Dx_absu_u"]
_DAMP_KEYS_Y = ["Dy_v", "Dy_r", "Dy_uv", "Dy_ur", "Dy_v3", "Dy_r3",
                "Dy_u2v", "Dy_u2r", "Dy_v2r", "Dy_vr2", "Dy_absv_v", "Dy_u_absv"]
_DAMP_KEYS_N = ["Dn_v", "Dn_r", "Dn_uv", "Dn_ur", "Dn_v3", "Dn_r3",
                "Dn_u2v", "Dn_u2r", "Dn_v2r", "Dn_vr2", "Dn_absr_r", "Dn_u_absr"]


def wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class HJBTorch:
    """Batched torch physics for the pursuer(HJB)/evader(APF) game."""

    def __init__(self, vessel_cfg, game_cfg, dtype=torch.float64):
        self.dtype = dtype
        rb = vessel_cfg["rigid_body"]; am = vessel_cfg["added_mass"]
        m = float(rb["mass_kg"]); Izz = float(rb["Izz_kg_m2"])
        self.M11 = m - float(am["X_udot"])
        self.M22 = m - float(am["Y_vdot"])
        self.M33 = Izz - float(am["N_rdot"])

        mdl = vessel_cfg["model"]
        self.include_coriolis = bool(mdl["include_coriolis"])
        self.damping_sign = float(mdl["damping_sign"])

        cp = game_cfg["vessels"]["pursuer_coeff_set"]
        ce = game_cfg["vessels"]["evader_coeff_set"]
        self.cP = vessel_cfg["coefficients"][cp]
        self.cE = vessel_cfg["coefficients"][ce]

        # thruster / actuator
        th = vessel_cfg["thrusters"]
        self.arm = float(th["arm_m"]); self.g = float(th["gravity_m_s2"])
        self.Fmax = float(th["force_max_kgf"]) * self.g
        self.Fmin = float(th["force_min_kgf"]) * self.g

        # APF gains
        apf = game_cfg["evader_apf"]
        self.V_e_max = float(apf["V_e_max_mps"])
        self.K_u = float(apf["K_u"]); self.K_psi = float(apf["K_psi"]); self.K_r = float(apf["K_r"])

        # HJB / ellipse
        gm = game_cfg["game"]
        self.rho = float(gm["capture_radius_m"])
        self.T = float(gm["horizon_s"])
        self.eps_H = float(gm["hamiltonian_eps"])
        self.alpha_u = self.Fmax / np.sqrt(2.0)
        self.alpha_r = self.arm * self.Fmax / np.sqrt(2.0)
        self.Sig0 = self.alpha_u ** 2
        self.Sig1 = self.alpha_r ** 2

    # -----------------------------------------------------------------------
    def _damping(self, u, v, r, c):
        au, av, ar = u.abs(), v.abs(), r.abs()
        X_D = (c["Dx_u"] * u + c["Dx_u2"] * u * u + c["Dx_v2"] * v * v
               + c["Dx_r2"] * r * r + c["Dx_vr"] * v * r + c["Dx_u3"] * u ** 3
               + c["Dx_uv2"] * u * v * v + c["Dx_ur2"] * u * r * r
               + c["Dx_uvr"] * u * v * r + c["Dx_absu_u"] * au * u)
        Y_D = (c["Dy_v"] * v + c["Dy_r"] * r + c["Dy_uv"] * u * v + c["Dy_ur"] * u * r
               + c["Dy_v3"] * v ** 3 + c["Dy_r3"] * r ** 3 + c["Dy_u2v"] * u * u * v
               + c["Dy_u2r"] * u * u * r + c["Dy_v2r"] * v * v * r + c["Dy_vr2"] * v * r * r
               + c["Dy_absv_v"] * av * v + c["Dy_u_absv"] * u * av)
        N_D = (c["Dn_v"] * v + c["Dn_r"] * r + c["Dn_uv"] * u * v + c["Dn_ur"] * u * r
               + c["Dn_v3"] * v ** 3 + c["Dn_r3"] * r ** 3 + c["Dn_u2v"] * u * u * v
               + c["Dn_u2r"] * u * u * r + c["Dn_v2r"] * v * v * r + c["Dn_vr2"] * v * r * r
               + c["Dn_absr_r"] * ar * r + c["Dn_u_absr"] * u * ar)
        return X_D, Y_D, N_D

    def _nu_dot(self, u, v, r, tau_u, tau_r, c):
        """Body-frame acceleration (du,dv,dr) for one vessel, tau=(tau_u,0,tau_r)."""
        X_D, Y_D, N_D = self._damping(u, v, r, c)
        rhs_u = tau_u + self.damping_sign * X_D
        rhs_v = 0.0 + self.damping_sign * Y_D
        rhs_r = tau_r + self.damping_sign * N_D
        if self.include_coriolis:
            rhs_u = rhs_u - (-self.M22 * v * r)
            rhs_v = rhs_v - (self.M11 * u * r)
            rhs_r = rhs_r - ((self.M22 - self.M11) * u * v)
        return rhs_u / self.M11, rhs_v / self.M22, rhs_r / self.M33

    # -----------------------------------------------------------------------
    def _apf_tau(self, zeta):
        """Realised evader (tau_u, tau_r) after allocation + forward-only clamp."""
        X, Y, psi = zeta[:, IX], zeta[:, IY], zeta[:, IPSI]
        u_e, r_e = zeta[:, IUE], zeta[:, IRE]
        beta = torch.atan2(Y, X)
        e_psi = wrap_pi(beta - psi)
        tau_u = torch.clamp(self.K_u * (self.V_e_max - u_e), min=0.0)
        tau_r = self.K_psi * e_psi - self.K_r * r_e
        d = self.arm
        Fs = torch.clamp(tau_u / 2.0 + tau_r / (2.0 * d), self.Fmin, self.Fmax)
        Fp = torch.clamp(tau_u / 2.0 - tau_r / (2.0 * d), self.Fmin, self.Fmax)
        return Fp + Fs, d * (Fs - Fp)

    # -----------------------------------------------------------------------
    def drift(self, zeta):
        """Relative drift abar(zeta), shape (B, 9)."""
        X, Y, psi = zeta[:, IX], zeta[:, IY], zeta[:, IPSI]
        u_e, v_e, r_e = zeta[:, IUE], zeta[:, IVE], zeta[:, IRE]
        u_i, v_i, r_i = zeta[:, IUI], zeta[:, IVI], zeta[:, IRI]
        cpsi, spsi = torch.cos(psi), torch.sin(psi)

        # evader velocity dynamics via APF
        tau_ue, tau_re = self._apf_tau(zeta)
        due, dve, dre = self._nu_dot(u_e, v_e, r_e, tau_ue, tau_re, self.cE)
        # pursuer velocity dynamics, control-independent (tau = 0)
        dui0, dvi0, dri0 = self._nu_dot(u_i, v_i, r_i,
                                        torch.zeros_like(u_i), torch.zeros_like(u_i), self.cP)

        a = torch.empty_like(zeta)
        a[:, IX] = u_e * cpsi - v_e * spsi - u_i + r_i * Y
        a[:, IY] = u_e * spsi + v_e * cpsi - v_i - r_i * X
        a[:, IPSI] = r_e - r_i
        a[:, IUE] = due; a[:, IVE] = dve; a[:, IRE] = dre
        a[:, IUI] = dui0; a[:, IVI] = dvi0; a[:, IRI] = dri0
        return a

    # -----------------------------------------------------------------------
    def terminal(self, zeta):
        """ell(zeta) = sqrt(X^2 + Y^2) - rho, shape (B,)."""
        return torch.sqrt(zeta[:, IX] ** 2 + zeta[:, IY] ** 2 + 1e-12) - self.rho

    def hamiltonian(self, zeta, p):
        """H(zeta, p) = <p, abar> + Fmax*p_ui/M11 - sqrt(Sig0 c0^2 + Sig1 c1^2 + eps), shape (B,)."""
        a = self.drift(zeta).detach()             # drift does not depend on network params
        drift_term = (p * a).sum(dim=1)
        c0 = p[:, IUI] / self.M11
        c1 = p[:, IRI] / self.M33
        bias = self.Fmax * c0
        R = torch.sqrt(self.Sig0 * c0 ** 2 + self.Sig1 * c1 ** 2 + self.eps_H)
        return drift_term + bias - R

    def vi_residual(self, zeta, V, dVdt, gradV):
        """min{ dVdt + H(zeta, gradV),  ell(zeta) - V }, shape (B,)."""
        hj = dVdt + self.hamiltonian(zeta, gradV)
        obstacle = self.terminal(zeta) - V
        return torch.minimum(hj, obstacle)
