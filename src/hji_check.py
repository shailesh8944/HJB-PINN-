"""
hji_check.py — Math checks for the two-player Isaacs game physics (hji_torch.py),
before any training.

Checks:
  (1) Isaacs Hamiltonian closed form == brute-force min-max over both ellipses.
  (2) Isaacs condition: min-max == max-min (the game value exists).
  (3) Saddle-point controls are realisable by the forward-only thrusters.
  (4) Reduction: with the evader costate zeroed, H reduces to the pursuer-only
      (single-minimisation) Hamiltonian.
"""

import os, sys, numpy as np, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hji_torch import GameHJITorch
from hjb_torch import IUE, IUI, IRE, IRI

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)


def main():
    with open(os.path.join(ROOT, "config", "vessel_config.yaml")) as f: vcfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, "config", "game_config.yaml")) as f: gcfg = yaml.safe_load(f)
    phys = GameHJITorch(vcfg, gcfg, dtype=torch.float64)
    M11, M33 = phys.M11, phys.M33
    au, ar, Fmax = phys.alpha_u, phys.alpha_r, phys.Fmax

    rng = np.random.default_rng(0)
    n = 200
    Z = rng.uniform([-14, -14, -np.pi, -0.2, -0.6, -0.2, -0.6, -1.5, -1.5],
                    [14, 14, np.pi, 1.2, 0.6, 1.2, 0.6, 1.5, 1.5], size=(n, 9))
    Z[np.hypot(Z[:, 0], Z[:, 1]) < 1.0, 0] += 3.0
    P = rng.normal(size=(n, 9))
    Zt, Pt = torch.tensor(Z), torch.tensor(P)

    Hcf = phys.game_hamiltonian(Zt, Pt).numpy()
    a0 = phys.drift0(Zt).numpy()

    th = np.linspace(0, 2 * np.pi, 2000)
    ci_u = Fmax + au * np.cos(th); ci_r = ar * np.sin(th)     # pursuer control on ellipse
    ce_u = Fmax + au * np.cos(th); ce_r = ar * np.sin(th)     # evader  control on ellipse

    e_ham, e_isaacs = 0.0, 0.0
    for k in range(n):
        base = float(P[k] @ a0[k])
        ci0, ci1 = P[k, IUI] / M11, P[k, IRI] / M33
        ce0, ce1 = P[k, IUE] / M11, P[k, IRE] / M33
        inner_i = ci0 * ci_u + ci1 * ci_r        # <G_i^T p, u_i> over the ellipse
        inner_e = ce0 * ce_u + ce1 * ce_r        # <G_e^T p, u_e> over the ellipse
        min_i, max_e = float(np.min(inner_i)), float(np.max(inner_e))
        minmax = base + min_i + max_e            # min_i then max_e
        maxmin = base + max_e + min_i            # max_e then min_i
        e_ham = max(e_ham, abs(Hcf[k] - minmax))
        e_isaacs = max(e_isaacs, abs(minmax - maxmin))

    # (3) saddle controls realisable
    ti, te = phys.optimal_controls(Pt)
    d = phys.arm
    def outside(tau):
        Fs = tau[:, 0] / 2 + tau[:, 1] / (2 * d)
        Fp = tau[:, 0] / 2 - tau[:, 1] / (2 * d)
        return float(torch.maximum(torch.clamp(-torch.minimum(Fs, Fp), min=0),
                                   torch.clamp(torch.maximum(Fs, Fp) - Fmax, min=0)).max())
    worst = max(outside(ti), outside(te))

    # (4) reduction: zero the evader costate -> pursuer-only Hamiltonian
    Pz = Pt.clone(); Pz[:, IUE] = 0; Pz[:, IRE] = 0
    Hgame_z = phys.game_hamiltonian(Zt, Pz).numpy()
    ci0 = (Pz[:, IUI] / M11).numpy(); ci1 = (Pz[:, IRI] / M33).numpy()
    Ri = np.sqrt(phys.Sig0 * ci0 ** 2 + phys.Sig1 * ci1 ** 2 + phys.eps_H)
    Hpursuer_only = (Pz.numpy() * a0).sum(1) + Fmax * ci0 - Ri + np.sqrt(phys.eps_H)
    e_reduce = float(np.max(np.abs(Hgame_z - Hpursuer_only)))

    print("=" * 66)
    print(" HJI game physics — math checks")
    print("=" * 66)
    print(f" (1) H closed-form vs brute-force min-max : {e_ham:.2e}  "
          f"[{'PASS' if e_ham < 1e-3 else 'FAIL'}]")
    print(f" (2) Isaacs gap |min-max - max-min|       : {e_isaacs:.2e}  "
          f"[{'PASS' if e_isaacs < 1e-9 else 'FAIL'}]  (game value exists)")
    print(f" (3) saddle controls outside thruster band: {worst:.2e}  "
          f"[{'PASS' if worst < 1e-9 else 'FAIL'}]")
    print(f" (4) reduction to pursuer-only H          : {e_reduce:.2e}  "
          f"[{'PASS' if e_reduce < 1e-9 else 'FAIL'}]")
    print("=" * 66)


if __name__ == "__main__":
    main()
