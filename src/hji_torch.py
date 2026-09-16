"""
hji_torch.py — Two-player Hamilton-Jacobi-Isaacs (HJI) game physics: one pursuer
(minimiser) versus one optimising evader (maximiser), both Sookshma vessels.

This replaces the single-player HJB setup (where the evader ran a fixed
artificial-potential-field law folded into the drift) with a genuine zero-sum
differential game:

    zeta_dot = a0(zeta) + G_i u_i + G_e u_e

    a0(zeta) : pure drift = relative kinematics + control-INDEPENDENT part of both
               vessels' velocity dynamics (no APF, no thrust)
    G_i      : pursuer input matrix (rows u_i, r_i)
    G_e      : evader  input matrix (rows u_e, r_e)

Isaacs Hamiltonian (pursuer minimises, evader maximises); the controls enter
additively and separately, so the min and max decouple and min-max = max-min:

    H(zeta,p) = <p, a0> + [ p^T G_i u0 - R_i(p) ]        (pursuer, min)
                        + [ p^T G_e u0 + R_e(p) ]        (evader, max)

    R_i = sqrt(Sig0 (p_ui/M11)^2 + Sig1 (p_ri/M33)^2 + eps)
    R_e = sqrt(Sig0 (p_ue/M11)^2 + Sig1 (p_re/M33)^2 + eps)

Both vessels use the same inscribed-ellipse actuator set (Sig0, Sig1, offset u0)
built in the base class HJBTorch. The value function's zero-sublevel set {V<=0}
is the guaranteed-capture region; {V>0} is guaranteed escape; {V=0} is the barrier.
"""

import torch
from hjb_torch import HJBTorch, IX, IY, IPSI, IUE, IVE, IUI, IVI, IRE, IRI


class GameHJITorch(HJBTorch):
    """Adds the two-player Isaacs drift, Hamiltonian, and optimal controls."""

    # ------------------------------------------------------------------
    def drift0(self, zeta):
        """Pure drift a0(zeta): both vessels control-independent (no APF/thrust)."""
        X, Y, psi = zeta[:, IX], zeta[:, IY], zeta[:, IPSI]
        u_e, v_e, r_e = zeta[:, IUE], zeta[:, IVE], zeta[:, IRE]
        u_i, v_i, r_i = zeta[:, IUI], zeta[:, IVI], zeta[:, IRI]
        cpsi, spsi = torch.cos(psi), torch.sin(psi)
        z0 = torch.zeros_like(u_e)

        due, dve, dre = self._nu_dot(u_e, v_e, r_e, z0, z0, self.cE)   # evader, tau=0
        dui, dvi, dri = self._nu_dot(u_i, v_i, r_i, z0, z0, self.cP)   # pursuer, tau=0

        a = torch.empty_like(zeta)
        a[:, IX] = u_e * cpsi - v_e * spsi - u_i + r_i * Y
        a[:, IY] = u_e * spsi + v_e * cpsi - v_i - r_i * X
        a[:, IPSI] = r_e - r_i
        a[:, IUE] = due; a[:, IVE] = dve; a[:, IRE] = dre
        a[:, IUI] = dui; a[:, IVI] = dvi; a[:, IRI] = dri
        return a

    # ------------------------------------------------------------------
    def _R(self, c0, c1):
        return torch.sqrt(self.Sig0 * c0 ** 2 + self.Sig1 * c1 ** 2 + self.eps_H)

    def game_hamiltonian(self, zeta, p):
        """Isaacs Hamiltonian H = <p,a0> + (b_i - R_i) + (b_e + R_e)."""
        a = self.drift0(zeta).detach()
        drift_term = (p * a).sum(dim=1)
        ci0 = p[:, IUI] / self.M11; ci1 = p[:, IRI] / self.M33      # pursuer costate dir
        ce0 = p[:, IUE] / self.M11; ce1 = p[:, IRE] / self.M33      # evader  costate dir
        Ri = self._R(ci0, ci1)
        Re = self._R(ce0, ce1)
        bias_i = self.Fmax * ci0
        bias_e = self.Fmax * ce0
        return drift_term + (bias_i - Ri) + (bias_e + Re)

    def game_vi_residual(self, zeta, V, dVdt, gradV):
        """min{ dVdt + H_game(zeta, gradV),  ell(zeta) - V }."""
        hj = dVdt + self.game_hamiltonian(zeta, gradV)
        return torch.minimum(hj, self.terminal(zeta) - V)

    # ------------------------------------------------------------------
    def optimal_controls(self, p):
        """Saddle-point controls: pursuer tau_i* = u0 - Sig c_i / R_i (min),
        evader tau_e* = u0 + Sig c_e / R_e (max). Returns (tau_i, tau_e), each
        (tau_u, tau_r)."""
        ci0 = p[:, IUI] / self.M11; ci1 = p[:, IRI] / self.M33
        ce0 = p[:, IUE] / self.M11; ce1 = p[:, IRE] / self.M33
        Ri = self._R(ci0, ci1); Re = self._R(ce0, ce1)
        tau_i = torch.stack([self.Fmax - self.Sig0 * ci0 / Ri, -self.Sig1 * ci1 / Ri], dim=1)
        tau_e = torch.stack([self.Fmax + self.Sig0 * ce0 / Re,  self.Sig1 * ce1 / Re], dim=1)
        return tau_i, tau_e

    # ------------------------------------------------------------------
    def zeta_dot(self, zeta, tau_i, tau_e):
        """Full game dynamics zeta_dot = a0(zeta) + G_i tau_i + G_e tau_e, for
        closed-loop rollout of the saddle-point controls (used by the verifier)."""
        a = self.drift0(zeta).clone()
        a[:, IUI] = a[:, IUI] + tau_i[:, 0] / self.M11
        a[:, IRI] = a[:, IRI] + tau_i[:, 1] / self.M33
        a[:, IUE] = a[:, IUE] + tau_e[:, 0] / self.M11
        a[:, IRE] = a[:, IRE] + tau_e[:, 1] / self.M33
        return a
