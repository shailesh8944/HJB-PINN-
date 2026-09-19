"""Two-player relative-kinematics vessel dynamics and Isaacs game math:
drift, terminal cost, Hamiltonian, saddle-point optimal controls, and the
self-consistency checks that validate the closed-form solution against a
brute-force min-max."""
import numpy as np
import torch

from config import (IX, IY, IPSI, IUE, IVE, IUI, IVI, IRE, IRI,
                     M11, M22, M33, MC, DELTA, ARM, FMAX, ALPHA_U, ALPHA_R,
                     SIG0, SIG1, EPS_H, R_COLLIDE)
from hydrodynamics import nu_dot0


def drift0(z):
    """Pure drift a0(zeta): relative kinematics + both vessels' control-independent
    velocity dynamics (no thrust). Shape (B,9)."""
    X, Y, psi = z[:, IX], z[:, IY], z[:, IPSI]
    ue, ve, re = z[:, IUE], z[:, IVE], z[:, IRE]
    ui, vi, ri = z[:, IUI], z[:, IVI], z[:, IRI]
    cp, sp = torch.cos(psi), torch.sin(psi)
    due, dve, dre = nu_dot0(ue, ve, re)
    dui, dvi, dri = nu_dot0(ui, vi, ri)
    a = torch.empty_like(z)
    a[:, IX]  = ue*cp - ve*sp - ui + ri*Y
    a[:, IY]  = ue*sp + ve*cp - vi - ri*X
    a[:, IPSI] = re - ri
    a[:, IUE] = due; a[:, IVE] = dve; a[:, IRE] = dre
    a[:, IUI] = dui; a[:, IVI] = dvi; a[:, IRI] = dri
    return a


def terminal(z):
    """ell(zeta) = sqrt(X^2+Y^2) - R  (signed distance to collision disc)."""
    return torch.sqrt(z[:, IX]**2 + z[:, IY]**2 + 1e-12) - R_COLLIDE


def _R(c0, c1):
    return torch.sqrt(SIG0 * c0**2 + SIG1 * c1**2 + EPS_H)


def _cdirs(p):
    """Control costate directions with the COUPLED input matrix G:
    tau_u acts on surge (1/M11); tau_r acts on yaw (M22/DELTA) AND sway (-MC/DELTA)."""
    ci0 = p[:, IUI] / M11
    ci1 = (M22 * p[:, IRI] - MC * p[:, IVI]) / DELTA
    ce0 = p[:, IUE] / M11
    ce1 = (M22 * p[:, IRE] - MC * p[:, IVE]) / DELTA
    return ci0, ci1, ce0, ce1


def game_hamiltonian(z, p):
    """Isaacs H = <p,a0> + (Fmax*ci0 - Ri) + (Fmax*ce0 + Re)."""
    a = drift0(z).detach()
    drift_term = (p * a).sum(dim=1)
    ci0, ci1, ce0, ce1 = _cdirs(p)
    return drift_term + (FMAX*ci0 - _R(ci0, ci1)) + (FMAX*ce0 + _R(ce0, ce1))


def game_vi_residual(z, V, dVdt, gradV):
    return torch.minimum(dVdt + game_hamiltonian(z, gradV), terminal(z) - V)


def optimal_controls(p):
    """Saddle-point thruster commands (tau_i, tau_e) read off the costate p=gradV.
    Returns ti=(tau_u_i, tau_r_i) for the pursuer, te=(tau_u_e, tau_r_e) for the evader."""
    ci0, ci1, ce0, ce1 = _cdirs(p)
    Ri, Re = _R(ci0, ci1), _R(ce0, ce1)
    ti = torch.stack([FMAX - SIG0*ci0/Ri, -SIG1*ci1/Ri], 1)
    te = torch.stack([FMAX + SIG0*ce0/Re,  SIG1*ce1/Re], 1)
    return ti, te


def isaacs_checks(device, n=200, seed=0):
    """Brute-force cross-check of the closed-form Hamiltonian/controls:
    (1) closed-form H matches brute-force min-max over the thruster ellipse,
    (2) min-max == max-min (a genuine game value exists, no duality gap),
    (3) the saddle-point controls stay inside the thruster ellipse."""
    rng = np.random.default_rng(seed)
    Z = rng.uniform([-14,-14,-np.pi,-0.2,-0.6,-0.2,-0.6,-1.5,-1.5],
                    [14,14,np.pi,1.2,0.6,1.2,0.6,1.5,1.5], size=(n,9))
    Z[np.hypot(Z[:,0], Z[:,1]) < 1.0, 0] += 3.0
    P = rng.normal(size=(n,9))
    Zt = torch.tensor(Z, dtype=torch.float64, device=device)
    Pt = torch.tensor(P, dtype=torch.float64, device=device)
    Hcf = game_hamiltonian(Zt, Pt).cpu().numpy()
    a0 = drift0(Zt).cpu().numpy()
    th = np.linspace(0, 2*np.pi, 3000)
    cu = FMAX + ALPHA_U*np.cos(th); cr = ALPHA_R*np.sin(th)
    e_ham = e_is = 0.0
    for k in range(n):
        base = float(P[k] @ a0[k])
        ci0 = P[k,IUI]/M11; ci1 = (M22*P[k,IRI] - MC*P[k,IVI])/DELTA
        ce0 = P[k,IUE]/M11; ce1 = (M22*P[k,IRE] - MC*P[k,IVE])/DELTA
        mn = (ci0*cu + ci1*cr).min(); mx = (ce0*cu + ce1*cr).max()
        e_ham = max(e_ham, abs(Hcf[k] - (base+mn+mx)))
        e_is = max(e_is, abs((base+mn+mx) - (base+mx+mn)))
    ti, te = optimal_controls(Pt)
    def outside(tau):
        Fs = tau[:,0]/2 + tau[:,1]/(2*ARM); Fp = tau[:,0]/2 - tau[:,1]/(2*ARM)
        return float(torch.maximum(torch.clamp(-torch.minimum(Fs,Fp),min=0),
                                   torch.clamp(torch.maximum(Fs,Fp)-FMAX,min=0)).max())
    worst = max(outside(ti), outside(te))
    return e_ham, e_is, worst
