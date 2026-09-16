"""
verify.py — Automated correctness checks for the Sookshma 3-DOF dynamics.

These checks answer "is the implementation faithful to the intended physics?"
They do NOT prove the identified coefficients match the real vessel — that
requires validation against withheld experimental data (see README / notes).

Run:
    python src/verify.py
    python src/verify.py --set theta_boot

Each check prints PASS / FAIL / WARN with the numbers behind it.
"""

import os
import sys
import argparse
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dynamics import VesselDynamics          # noqa: E402
from thruster import Thruster                # noqa: E402
import simulate as sim                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

RESULTS = []


def record(name, status, detail):
    RESULTS.append((name, status, detail))
    tag = {"PASS": "[ PASS ]", "FAIL": "[ FAIL ]", "WARN": "[ WARN ]"}[status]
    print(f"{tag}  {name}\n          {detail}")


# ---------------------------------------------------------------------------
# 1. Equilibrium: zero velocity + zero force -> zero acceleration
# ---------------------------------------------------------------------------
def check_equilibrium(dyn):
    nu = np.zeros(3)
    tau = np.zeros(3)
    a = dyn.nu_dot(nu, tau)
    ok = np.linalg.norm(a) < 1e-12
    record("Equilibrium (nu=0, tau=0 -> nu_dot=0)",
           "PASS" if ok else "FAIL",
           f"|nu_dot| = {np.linalg.norm(a):.2e}")


# ---------------------------------------------------------------------------
# 2. Coriolis is workless:  nu^T C(nu) nu = 0  (skew-symmetry requirement)
# ---------------------------------------------------------------------------
def check_coriolis_workless(dyn):
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(10000):
        nu = rng.uniform(-2, 2, 3)
        C_nu = dyn.coriolis(*nu)
        worst = max(worst, abs(nu @ C_nu))
    ok = worst < 1e-9
    record("Coriolis workless (nu^T C nu = 0)",
           "PASS" if ok else "FAIL",
           f"max |nu^T C nu| over 1e4 random states = {worst:.2e}")


# ---------------------------------------------------------------------------
# 3. Port-starboard symmetry: symmetric thrust from rest -> sway=yaw=0 exactly
# ---------------------------------------------------------------------------
def check_symmetry(dyn, thr):
    tau, _ = thr.allocate(1.82, 1.82)              # symmetric
    s = np.zeros(6)
    dt = 0.01
    max_vr = 0.0
    for _ in range(3000):
        s = sim.rk4_step(dyn, s, tau, dt)
        max_vr = max(max_vr, abs(s[4]), abs(s[5]))  # |v|, |r|
    ok = max_vr < 1e-10
    record("Port/starboard symmetry (symmetric thrust -> v=r=0)",
           "PASS" if ok else "FAIL",
           f"max(|v|,|r|) over 30 s = {max_vr:.2e}")


# ---------------------------------------------------------------------------
# 4. Damping dissipativity: power P = nu^T D(nu)nu >= 0 over the envelope
#    (D must remove energy, not inject it). Reports anti-damping regions.
# ---------------------------------------------------------------------------
def check_dissipativity(dyn):
    us = np.linspace(0.0, 1.4, 15)          # forward-only surge envelope
    vs = np.linspace(-0.5, 0.5, 15)
    rs = np.linspace(-1.5, 1.5, 15)
    neg = 0
    total = 0
    worst = (0.0, None)
    for u in us:
        for v in vs:
            for r in rs:
                if u == 0 and v == 0 and r == 0:
                    continue
                D = dyn.damping(u, v, r)
                P = u * D[0] + v * D[1] + r * D[2]
                total += 1
                if P < 0:
                    neg += 1
                    if P < worst[0]:
                        worst = (P, (u, v, r))
    frac = 100.0 * neg / total
    status = "PASS" if neg == 0 else "WARN"
    detail = (f"{neg}/{total} ({frac:.1f}%) grid points have negative "
              f"dissipated power (anti-damping).")
    if worst[1] is not None:
        detail += f" Worst P={worst[0]:.2f} at (u,v,r)={tuple(round(x,2) for x in worst[1])}."
    else:
        detail += " Damping is dissipative everywhere sampled."
    record("Damping dissipativity (power >= 0)", status, detail)


# ---------------------------------------------------------------------------
# 5. Principal damping signs: linear surge/sway/yaw damping should be > 0
# ---------------------------------------------------------------------------
def check_principal_signs(dyn):
    c = dyn.c
    items = {"Dx_u (linear surge)": c["Dx_u"],
             "Dy_v (linear sway)": c["Dy_v"],
             "Dn_r (linear yaw)": c["Dn_r"]}
    bad = {k: v for k, v in items.items() if v <= 0}
    small = {k: v for k, v in items.items() if 0 < v < 1.0}
    if bad:
        record("Principal damping signs (> 0)", "FAIL",
               f"Non-dissipative linear terms: {bad}")
    elif small:
        record("Principal damping signs (> 0)", "WARN",
               f"All positive, but very weak: {small} "
               f"(weak linear yaw damping -> oscillatory turns).")
    else:
        record("Principal damping signs (> 0)", "PASS",
               f"All positive: { {k: round(v,3) for k,v in items.items()} }")


# ---------------------------------------------------------------------------
# 6. Integrator convergence: RK4 vs a high-accuracy reference (scipy if present)
#    and self-convergence under step halving (expect ~4th order).
# ---------------------------------------------------------------------------
def check_integrator(dyn, thr):
    tau, _ = thr.allocate(1.82, 0.6)          # a turning maneuver (exercises all DOF)
    s0 = np.zeros(6)
    T = 10.0

    def rollout(dt):
        s = s0.copy()
        n = int(round(T / dt))
        for _ in range(n):
            s = sim.rk4_step(dyn, s, tau, dt)
        return s

    s_c = rollout(0.02)
    s_f = rollout(0.01)
    s_ff = rollout(0.005)
    e1 = np.linalg.norm(s_c - s_ff)
    e2 = np.linalg.norm(s_f - s_ff)
    ratio = e1 / e2 if e2 > 0 else float("inf")
    # RK4 error ~ dt^4, so halving dt should cut error ~16x
    self_ok = e2 < 1e-3

    detail = (f"self-convergence: |dt.02 - dt.005|={e1:.2e}, "
              f"|dt.01 - dt.005|={e2:.2e}, ratio={ratio:.1f} (RK4 ideal ~16)")

    ref_ok = True
    try:
        from scipy.integrate import solve_ivp
        sol = solve_ivp(lambda t, s: dyn.state_dot(s, tau), [0, T], s0,
                        method="RK45", rtol=1e-10, atol=1e-12, dense_output=True)
        s_ref = sol.y[:, -1]
        e_ref = np.linalg.norm(s_f - s_ref)
        ref_ok = e_ref < 1e-3
        detail += f" | vs scipy RK45 (tol 1e-10): |diff|={e_ref:.2e}"
    except Exception as ex:
        detail += f" | scipy reference skipped ({ex})"

    record("Integrator convergence (RK4)",
           "PASS" if (self_ok and ref_ok) else "WARN", detail)


# ---------------------------------------------------------------------------
# 7. Free decay: no thrust + initial motion -> kinetic energy decreases
# ---------------------------------------------------------------------------
def check_free_decay(dyn):
    s = np.array([0, 0, 0, 1.2, 0.2, 0.5], float)   # some initial motion
    tau = np.zeros(3)
    dt = 0.01

    def ke(state):
        nu = state[3:6]
        return 0.5 * nu @ dyn.M @ nu               # full (coupled) mass matrix

    prev = ke(s)
    increases = 0
    for _ in range(4000):
        s = sim.rk4_step(dyn, s, tau, dt)
        cur = ke(s)
        if cur > prev + 1e-9:
            increases += 1
        prev = cur
    status = "PASS" if increases == 0 else "WARN"
    record("Free decay (no thrust -> kinetic energy non-increasing)",
           status,
           f"kinetic-energy-increasing steps: {increases}/4000; "
           f"final KE={prev:.4f} J")


# ---------------------------------------------------------------------------
# 8. Coefficient-set robustness: theta_boot vs theta_anneal on same maneuver
# ---------------------------------------------------------------------------
def check_set_divergence(cfg, config_dir):
    thr = Thruster(cfg, config_dir)
    tau, _ = thr.allocate(1.82, 0.6)
    s0 = np.zeros(6)
    dt = 0.01

    def rollout(setname):
        c2 = {**cfg, "model": {**cfg["model"], "active_coeff_set": setname}}
        d = VesselDynamics(c2)
        s = s0.copy()
        for _ in range(1000):
            s = sim.rk4_step(d, s, tau, dt)
        return s

    sa = rollout("theta_anneal")
    sb = rollout("theta_boot")
    dpos = np.linalg.norm(sa[:2] - sb[:2])
    dpsi = np.rad2deg(abs(sa[2] - sb[2]))
    status = "WARN" if (dpos > 1.0 or dpsi > 10.0) else "PASS"
    record("Coefficient-set robustness (boot vs anneal, 10 s turn)",
           status,
           f"position gap={dpos:.2f} m, heading gap={dpsi:.1f} deg. "
           f"Large gap => identification not well constrained; needs more/better data.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "vessel_config.yaml"))
    ap.add_argument("--set", dest="cset", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    config_dir = os.path.dirname(os.path.abspath(args.config))
    if args.cset:
        cfg["model"]["active_coeff_set"] = args.cset

    dyn = VesselDynamics(cfg)
    thr = Thruster(cfg, config_dir)

    print("=" * 70)
    print(f" Verifying Sookshma dynamics  |  set={dyn.active_set}  "
          f"Coriolis={'on' if dyn.include_coriolis else 'off'}")
    print("=" * 70)

    check_equilibrium(dyn)
    check_coriolis_workless(dyn)
    check_symmetry(dyn, thr)
    check_dissipativity(dyn)
    check_principal_signs(dyn)
    check_integrator(dyn, thr)
    check_free_decay(dyn)
    check_set_divergence(cfg, config_dir)

    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_warn = sum(1 for _, s, _ in RESULTS if s == "WARN")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    print("=" * 70)
    print(f" SUMMARY: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL")
    print("   PASS/FAIL = implementation faithfulness (code vs equations).")
    print("   WARN      = physical / model-quality flags to review.")
    print("   None of these prove the coefficients match the real vessel;")
    print("   that requires validation against withheld measured trajectories.")
    print("=" * 70)


if __name__ == "__main__":
    main()
