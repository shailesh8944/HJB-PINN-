"""
hjb_demo.py — Initial-setup demonstration and self-checks for the pursuer(HJB) /
evader(APF) formulation, up to the HJB Hamiltonian (no neural solver yet).

What it does:
  1. Builds both vessels' Fossen dynamics, the APF evader, the 9-state relative
     dynamics, and the HJB formulation from the two config files.
  2. Prints the formulation at a sample relative state and costate: drift abar,
     input matrix G_i, terminal ell, Hamiltonian H, and optimal pursuer control.
  3. Runs formal self-checks:
       (a) control-affine split of the pursuer velocity dynamics,
       (b) Hamiltonian closed form == brute-force minimum over the ellipse,
       (c) optimal control is realisable by the two forward-only thrusters,
       (d) G_i structure.
  4. Runs a short closed-loop rollout (placeholder pure-pursuit pursuer vs APF
     evader) to exercise the relative dynamics and report the closest approach
     min_s ell(zeta(s)) -- the quantity the value function V will formalise --
     and saves a chase plot.

Usage:
    python src/hjb_demo.py
    python src/hjb_demo.py --plot
"""

import os
import sys
import argparse
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dynamics import VesselDynamics                 # noqa: E402
from thruster import Thruster                       # noqa: E402
from apf_evader import APFEvader, wrap_pi           # noqa: E402
from relative_dynamics import RelativeDynamics, IX, IY, IPSI, IUE, IUI, IVI, IRI  # noqa: E402
from hjb import HJBFormulation                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build(vessel_cfg_path, game_cfg_path):
    vcfg = load_yaml(vessel_cfg_path)
    gcfg = load_yaml(game_cfg_path)
    config_dir = os.path.dirname(os.path.abspath(vessel_cfg_path))

    # two vessels; allow different damping-coefficient sets per config
    vcfg_p = dict(vcfg); vcfg_p["model"] = dict(vcfg["model"])
    vcfg_e = dict(vcfg); vcfg_e["model"] = dict(vcfg["model"])
    vcfg_p["model"]["active_coeff_set"] = gcfg["vessels"]["pursuer_coeff_set"]
    vcfg_e["model"]["active_coeff_set"] = gcfg["vessels"]["evader_coeff_set"]

    pursuer_dyn = VesselDynamics(vcfg_p)
    evader_dyn = VesselDynamics(vcfg_e)
    thr = Thruster(vcfg, config_dir)
    apf = APFEvader(gcfg, thr)
    rel = RelativeDynamics(pursuer_dyn, evader_dyn, apf)
    hjb = HJBFormulation(rel, gcfg, thr)
    return dict(vcfg=vcfg, gcfg=gcfg, pursuer=pursuer_dyn, evader=evader_dyn,
                thr=thr, apf=apf, rel=rel, hjb=hjb)


# ---------------------------------------------------------------------------
def self_checks(B):
    rel, hjb, thr = B["rel"], B["hjb"], B["thr"]
    d, Fmax = thr.arm, thr.kgf_to_N(thr.f_max_kgf)
    rng = np.random.default_rng(0)
    results = []

    # (a) control-affine split: nu_dot(nu, tau) == nu_dot(nu, 0) + Minv @ tau
    nu = rng.normal(size=3)
    tau = np.array([rng.normal(), 0.0, rng.normal()])
    lhs = rel.pursuer.nu_dot(nu, tau)
    rhs = rel.pursuer.nu_dot(nu, np.zeros(3)) + rel.pursuer.Minv @ tau
    err_affine = float(np.max(np.abs(lhs - rhs)))
    results.append(("control-affine split of pursuer dynamics", err_affine, err_affine < 1e-12))

    # (b) Hamiltonian closed form == brute-force min over the ellipse boundary
    zeta = rng.normal(size=9); zeta[0] += 8.0     # place evader ~8 m ahead
    p = rng.normal(size=9)
    H_closed = hjb.hamiltonian(zeta, p)
    a = rel.drift(zeta); G = rel.G_matrix()
    th = np.linspace(0, 2 * np.pi, 4000)
    w = np.stack([hjb.alpha_u * np.cos(th), hjb.alpha_r * np.sin(th)], axis=1)  # ellipse boundary
    u_samples = hjb.u0[None, :] + w
    vals = (p @ a) + u_samples @ (G.T @ p)
    H_brute = float(np.min(vals))
    err_ham = abs(H_closed - H_brute)
    results.append(("Hamiltonian closed form vs brute-force min", err_ham, err_ham < 1e-3))

    # (c) optimal control realisable by forward-only thrusters
    worst = 0.0
    for _ in range(500):
        pp = rng.normal(size=9)
        tau_u, tau_r = hjb.optimal_control(pp)
        Fs = tau_u / 2.0 + tau_r / (2.0 * d)
        Fp = tau_u / 2.0 - tau_r / (2.0 * d)
        # distance outside the forward band [0, Fmax], should be ~0 (ellipse subset diamond)
        worst = max(worst, -min(Fs, Fp, 0.0), max(Fs - Fmax, Fp - Fmax, 0.0))
    results.append(("optimal control realisable (ellipse subset diamond)", worst, worst < 1e-9))

    # (d) G is exactly the relevant columns of the full coupled M^{-1}
    Gok = (abs(G[IUI, 0] - rel.pursuer.Minv[0, 0]) < 1e-15 and
           abs(G[IVI, 1] - rel.pursuer.Minv[1, 2]) < 1e-15 and
           abs(G[IRI, 1] - rel.pursuer.Minv[2, 2]) < 1e-15 and
           np.count_nonzero(G) == 3)
    results.append(("G_i structure matches coupled M^{-1}", 0.0, Gok))
    return results


# ---------------------------------------------------------------------------
def pursuit_controller(zeta, hjb, u_des=1.0, Kpsi=6.0, Kd=4.5, Ku=45.0):
    """Placeholder pure-pursuit pursuer (NOT the HJB controller): point at the
    evader and thrust forward. Used only to exercise the dynamics until the
    physics-informed network supplies grad V for the true optimal control."""
    X, Y = zeta[IX], zeta[IY]
    u_i, r_i = zeta[IUI], zeta[IRI]
    e_psi = np.arctan2(Y, X)                 # bearing to evader in pursuer frame (want 0)
    tau_u = max(0.0, Ku * (u_des - u_i))
    tau_r = Kpsi * e_psi - Kd * r_i
    # project onto the inscribed ellipse so the pursuer respects the HJB control set
    w = np.array([tau_u, tau_r]) - hjb.u0
    scale = np.sqrt((w[0] / hjb.alpha_u) ** 2 + (w[1] / hjb.alpha_r) ** 2)
    if scale > 1.0:
        w = w / scale
    return hjb.u0 + w


def rollout(B, zeta0, dt=0.05, T=25.0):
    rel, hjb = B["rel"], B["hjb"]
    n = int(round(T / dt)) + 1
    zeta = np.array(zeta0, float)
    log = []
    ell_min = np.inf
    for k in range(n):
        t = k * dt
        ell = hjb.terminal(zeta)
        ell_min = min(ell_min, ell)
        log.append([t, zeta[IX], zeta[IY], ell])
        if ell <= 0.0:                       # captured
            break
        tau_i = pursuit_controller(zeta, hjb)
        zeta = rel.rk4_step(zeta, tau_i, dt)
    return np.array(log), ell_min


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vessel", default=os.path.join(ROOT, "config", "vessel_config.yaml"))
    ap.add_argument("--game", default=os.path.join(ROOT, "config", "game_config.yaml"))
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    B = build(args.vessel, args.game)
    rel, hjb = B["rel"], B["hjb"]

    # ---- sample state + costate -------------------------------------------
    #  evader 8 m ahead and 3 m to starboard, both drifting; arbitrary costate
    zeta = np.array([8.0, 3.0, 0.3, 0.6, 0.0, 0.5, 0.0, 0.0, 0.0])
    p = np.array([0.4, -0.2, 0.1, 0.0, 0.0, 0.7, 0.0, 0.0, -0.5])

    print("=" * 70)
    print(" HJB / APF pursuit formulation — initial setup")
    print("=" * 70)
    print(" pursuer M =", np.array2string(rel.pursuer.M, precision=4))
    print(f" ellipse semi-axes: surge {hjb.alpha_u:.3f} N, yaw {hjb.alpha_r:.3f} N.m; "
          f"centre u0 = ({hjb.u0[0]:.3f}, 0) N")
    print(f" capture radius rho = {hjb.rho} m, horizon T = {hjb.T} s")
    print("-" * 70)
    print(" sample zeta =", np.array2string(zeta, precision=3))
    print(" drift abar  =", np.array2string(rel.drift(zeta), precision=3))
    print(f" terminal ell(zeta) = {hjb.terminal(zeta):.4f} m  "
          f"(in capture set: {hjb.in_capture_set(zeta)})")
    print(f" Hamiltonian H(zeta,p) = {hjb.hamiltonian(zeta, p):.5f}")
    tau_star = hjb.optimal_control(p)
    print(f" optimal pursuer control tau_i* = (tau_u {tau_star[0]:.3f} N, tau_r {tau_star[1]:.3f} N.m)")

    # ---- self-checks ------------------------------------------------------
    print("-" * 70)
    print(" self-checks:")
    allok = True
    for name, err, ok in self_checks(B):
        allok = allok and ok
        print(f"   [{'PASS' if ok else 'FAIL'}] {name:<52} (residual {err:.2e})")
    print(f"   overall: {'ALL PASS' if allok else 'SOME FAILED'}")

    # ---- rollout ----------------------------------------------------------
    print("-" * 70)
    zeta0 = np.array([10.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    log, ell_min = rollout(B, zeta0)
    captured = ell_min <= 0.0
    print(f" placeholder rollout (pure-pursuit vs APF): closest approach "
          f"min_s ell = {ell_min:.3f} m  ->  {'CAPTURED' if captured else 'escaped'}")
    print(f"   (this closest-approach number is what the value function V(zeta,t) formalises)")
    print("=" * 70)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = os.path.join(ROOT, "outputs"); os.makedirs(out, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(log[:, 2], log[:, 1], "b-", label="evader rel. path (pursuer frame)")
        ax[0].plot(0, 0, "ks", label="pursuer (origin)")
        thc = np.linspace(0, 2 * np.pi, 100)
        ax[0].plot(hjb.rho * np.sin(thc), hjb.rho * np.cos(thc), "r--", label="capture disc")
        ax[0].set_xlabel("Y rel [m]"); ax[0].set_ylabel("X rel [m]")
        ax[0].axis("equal"); ax[0].grid(True); ax[0].legend()
        ax[0].set_title("Relative chase")
        ax[1].plot(log[:, 0], log[:, 3], "b-")
        ax[1].axhline(0.0, color="r", ls="--")
        ax[1].set_xlabel("t [s]"); ax[1].set_ylabel(r"$\ell(\zeta)$ = gap - $\rho$ [m]")
        ax[1].grid(True); ax[1].set_title("Closest-approach score over time")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "hjb_demo.png"), dpi=130)
        print(" saved plot -> outputs/hjb_demo.png")


if __name__ == "__main__":
    main()
