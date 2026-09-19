"""
hjb.py — Hamilton-Jacobi-Bellman (HJB) reachability formulation for the pursuer.

This module assembles the objects that define the value function's governing
equation, up to (but not including) the neural solver:

  * terminal / capture function      ell(zeta) = sqrt(X^2 + Y^2) - rho
  * pursuer actuator set             inscribed ellipse in the true thrust diamond
  * HJB Hamiltonian                  H(zeta, p) = min_{u_i in Ein} <p, abar + G_i u_i>
  * optimal pursuer control          u_i*(zeta, p)

The value function V(zeta, t) is the unique viscosity solution of the variational
inequality
    min{ dV/dt + H(zeta, grad V),  ell(zeta) - V } = 0,   V(zeta, 0) = ell(zeta),
whose zero-sublevel set is the backward reachable tube of guaranteed capture. The
variational inequality itself is solved later by the physics-informed network;
this module provides every term that appears inside it.

Actuator model (see the methodology note). Two forward-only thrusters,
F_p, F_s in [0, Fmax], allocation tau_u = F_p + F_s, tau_r = d (F_s - F_p), give a
feasible diamond in (tau_u, tau_r) centred at u0 = (Fmax, 0). We use the largest
ellipse inscribed in that diamond (a conservative inner approximation, so every
commanded thrust is realisable and the tube is never optimistic):
    Ein = { u0 + w : w^T Sigma^{-1} w <= 1 },
    Sigma = diag( (Fmax/sqrt2)^2, (d Fmax/sqrt2)^2 ).
The support function of this ellipse gives H in closed form with a smooth
(differentiable) dependence on the costate p.
"""

import numpy as np

from relative_dynamics import IX, IY


class HJBFormulation:
    def __init__(self, rel_dyn, game_cfg, thruster):
        self.rel = rel_dyn
        self.rho = float(game_cfg["game"]["capture_radius_m"])
        self.T = float(game_cfg["game"]["horizon_s"])
        self.eps_H = float(game_cfg["game"]["hamiltonian_eps"])

        # pursuer actuator ellipse, in (tau_u, tau_r) space
        d = thruster.arm
        Fmax = thruster.kgf_to_N(thruster.f_max_kgf)      # single-thruster max force [N]
        self.Fmax_N = Fmax
        self.d = d
        self.u0 = np.array([Fmax, 0.0])                   # diamond centre (both thrusters half)
        # inscribed-ellipse semi-axes: half-diagonals / sqrt(2)
        self.alpha_u = Fmax / np.sqrt(2.0)                # surge semi-axis
        self.alpha_r = d * Fmax / np.sqrt(2.0)            # yaw   semi-axis
        self.Sigma = np.array([self.alpha_u ** 2, self.alpha_r ** 2])   # diag entries

    # -----------------------------------------------------------------------
    def terminal(self, zeta):
        """Signed distance to the capture disc: ell(zeta) = ||(X,Y)|| - rho."""
        return np.hypot(zeta[IX], zeta[IY]) - self.rho

    def in_capture_set(self, zeta):
        return self.terminal(zeta) <= 0.0

    # -----------------------------------------------------------------------
    def _costate_control_dir(self, p):
        """Exact control direction c = G_i^T p, including sway-yaw coupling."""
        return self.rel.G_matrix().T @ np.asarray(p)

    def hamiltonian(self, zeta, p):
        """
        HJB Hamiltonian  H(zeta, p) = <p, abar(zeta)> + p^T G_i u0
                                       - sqrt( c^T Sigma c + eps_H ),
        the single-minimisation (pursuer-only) value; the negative square-root
        term is the support function of the inscribed ellipse.
        """
        a = self.rel.drift(zeta)
        drift_term = float(p @ a)
        c = self._costate_control_dir(p)
        bias = self.Fmax_N * c[0]                          # = p^T G_i u0 = Fmax * p_{u_i}/M11_i
        R = np.sqrt(self.Sigma[0] * c[0] ** 2 + self.Sigma[1] * c[1] ** 2 + self.eps_H)
        return drift_term + bias - R

    # -----------------------------------------------------------------------
    def optimal_control(self, p):
        """
        Optimal pursuer generalised force u_i* = u0 - Sigma c / R  (in the
        inscribed ellipse), returned as (tau_u^i*, tau_r^i*)  [N, N.m].
        Always forward: tau_u* in [Fmax(1 - 1/sqrt2), Fmax(1 + 1/sqrt2)].
        """
        c = self._costate_control_dir(p)
        R = np.sqrt(self.Sigma[0] * c[0] ** 2 + self.Sigma[1] * c[1] ** 2 + self.eps_H)
        tau_u = self.u0[0] - self.Sigma[0] * c[0] / R
        tau_r = self.u0[1] - self.Sigma[1] * c[1] / R
        return np.array([tau_u, tau_r])

    # -----------------------------------------------------------------------
    def vi_residual(self, zeta, t, V, dVdt, gradV):
        """
        Residual of the HJB variational inequality at (zeta, t):
            min{ dVdt + H(zeta, gradV),  ell(zeta) - V }.
        Provided for later use inside the physics-informed-network loss; a correct
        value function drives this to zero.
        """
        hj = dVdt + self.hamiltonian(zeta, gradV)
        obstacle = self.terminal(zeta) - V
        return min(hj, obstacle)
