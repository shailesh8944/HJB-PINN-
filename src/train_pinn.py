"""
train_pinn.py — Physics-informed network solver for the HJB reachability
variational inequality (pursuer = HJB, evader = APF).

Pipeline:
  1. Math check: torch drift / Hamiltonian / terminal are compared against the
     NumPy reference (relative_dynamics.py, hjb.py) on random states.
  2. Gradient check: autograd dV/d(inputs) vs finite differences on the SIREN.
  3. Train V_theta on the variational-inequality residual with a soft terminal
     condition and a backward-in-time curriculum.
  4. Diagnostics: loss history, a value-function slice and the backward reachable
     tube boundary at t = -T, saved to outputs/train_pinn.png.

Note: a 9-state reachability network needs a GPU and long training (DeepReach
reports 16-25 h) to fully converge. On CPU this run is a correctness/behaviour
demonstration: the math checks must pass and the loss must fall.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hjb_demo import build                              # noqa: E402  (numpy reference)
from hjb_torch import HJBTorch                          # noqa: E402
from pinn import SIREN, value_and_grads                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)


def math_check(Bnp, phys, n=300, seed=0):
    """Compare torch physics against the numpy reference on random states."""
    rng = np.random.default_rng(seed)
    rel, hjb = Bnp["rel"], Bnp["hjb"]
    Z = rng.uniform(low=[-14, -14, -np.pi, -0.2, -0.6, -0.2, -0.6, -1.5, -1.5],
                    high=[14, 14, np.pi, 1.2, 0.6, 1.2, 0.6, 1.5, 1.5], size=(n, 9))
    # push away from the exact origin so atan2 is well conditioned
    Z[np.hypot(Z[:, 0], Z[:, 1]) < 1.0, 0] += 3.0
    P = rng.normal(size=(n, 9))
    Zt = torch.tensor(Z, dtype=torch.float64)
    Pt = torch.tensor(P, dtype=torch.float64)

    drift_t = phys.drift(Zt).numpy()
    ham_t = phys.hamiltonian(Zt, Pt).numpy()
    term_t = phys.terminal(Zt).numpy()

    drift_n = np.array([rel.drift(Z[k]) for k in range(n)])
    ham_n = np.array([hjb.hamiltonian(Z[k], P[k]) for k in range(n)])
    term_n = np.array([hjb.terminal(Z[k]) for k in range(n)])

    e_drift = np.max(np.abs(drift_t - drift_n))
    e_ham = np.max(np.abs(ham_t - ham_n))
    e_term = np.max(np.abs(term_t - term_n))
    return e_drift, e_ham, e_term


def grad_check(net, seed=0):
    """Finite-difference vs autograd gradient of the SIREN value."""
    torch.manual_seed(seed)
    zeta = torch.zeros(1, 9, dtype=torch.float64); zeta[0, 0] = 6.0; zeta[0, 1] = 2.0
    t = torch.tensor([-3.0], dtype=torch.float64)
    V, dVdt, gradV = value_and_grads(net, zeta, t)
    # finite difference on X (index 0) and on t
    eps = 1e-5
    def val(z, tt):
        with torch.no_grad():
            return net(torch.cat([z, tt.reshape(-1, 1)], 1)).item()
    zp = zeta.clone(); zp[0, 0] += eps; zm = zeta.clone(); zm[0, 0] -= eps
    dX_fd = (val(zp, t) - val(zm, t)) / (2 * eps)
    tp = t + eps; tm = t - eps
    dt_fd = (val(zeta, tp) - val(zeta, tm)) / (2 * eps)
    return abs(gradV[0, 0].item() - dX_fd), abs(dVdt[0].item() - dt_fd)


def continuity_check(net, lo, hi, n=200, seed=1, dtype=torch.float64):
    """Value must be identical at psi_rel = -pi and +pi (same everything else),
    because heading is periodic. With the (cos, sin) encoding this holds by
    construction; this test confirms it."""
    rng = np.random.default_rng(seed)
    lo9 = np.array(lo[:9]); hi9 = np.array(hi[:9])
    Z = lo9 + (hi9 - lo9) * rng.random((n, 9))
    t = -rng.random((n, 1)) * (-lo[9])              # t in [lo_t, 0]
    Zm = Z.copy(); Zm[:, 2] = -np.pi
    Zp = Z.copy(); Zp[:, 2] = np.pi
    with torch.no_grad():
        Vm = net(torch.tensor(np.hstack([Zm, t]), dtype=dtype)).numpy()
        Vp = net(torch.tensor(np.hstack([Zp, t]), dtype=dtype)).numpy()
    return float(np.max(np.abs(Vm - Vp)))


def train(phys, lo, hi, iters=1200, pretrain=300, batch=1024, hidden=128,
          lr=2e-4, w_term=10.0, w_pde=1.0, curric_frac=0.6, seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    net = SIREN(lo, hi, hidden=hidden, layers=3, dtype=dtype)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lo9 = torch.tensor(lo[:9], dtype=dtype); hi9 = torch.tensor(hi[:9], dtype=dtype)
    T = phys.T
    hist = []

    def sample_zeta(nb):
        return lo9 + (hi9 - lo9) * torch.rand(nb, 9, dtype=dtype)

    t0 = time.time()
    for it in range(iters):
        opt.zero_grad()
        # --- terminal condition (t = 0): V(zeta,0) = ell(zeta) ---
        zt = sample_zeta(batch)
        Vt = net(torch.cat([zt, torch.zeros(batch, 1, dtype=dtype)], 1)).squeeze(1)
        loss_term = ((Vt - phys.terminal(zt)) ** 2).mean()

        if it < pretrain:
            loss = loss_term                      # learn terminal first
            loss_pde = torch.tensor(0.0)
        else:
            # --- variational-inequality residual with backward-time curriculum ---
            frac = min(1.0, (it - pretrain) / max(1.0, curric_frac * iters))
            t_cur = T * frac
            zc = sample_zeta(batch)
            tc = -t_cur * torch.rand(batch, dtype=dtype)
            V, dVdt, gradV = value_and_grads(net, zc, tc)
            res = phys.vi_residual(zc, V, dVdt, gradV)
            loss_pde = (res ** 2).mean()
            loss = w_term * loss_term + w_pde * loss_pde

        loss.backward()
        opt.step()
        if it % 50 == 0 or it == iters - 1:
            hist.append((it, float(loss_term.detach()), float(loss_pde.detach())))
    dt = time.time() - t0
    return net, hist, dt


def diagnostics(net, phys, lo, hi, path, dtype=torch.float32):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # value slice over (X,Y) at t=-T, other states = 0
    ng = 120
    xs = np.linspace(lo[0], hi[0], ng); ys = np.linspace(lo[1], hi[1], ng)
    XX, YY = np.meshgrid(xs, ys)
    Z = np.zeros((ng * ng, 9), np.float32)
    Z[:, 0] = XX.ravel(); Z[:, 1] = YY.ravel()
    tt = np.full((ng * ng, 1), -phys.T, np.float32)
    with torch.no_grad():
        Vg = net(torch.cat([torch.tensor(Z), torch.tensor(tt)], 1)).numpy().reshape(ng, ng)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    pc = ax[0].contourf(YY, XX, Vg, levels=30, cmap="viridis")
    ax[0].contour(YY, XX, Vg, levels=[0.0], colors="r", linewidths=2)
    th = np.linspace(0, 2 * np.pi, 100)
    ax[0].plot(phys.rho * np.sin(th), phys.rho * np.cos(th), "w--", lw=1)
    ax[0].plot(0, 0, "ks"); ax[0].axis("equal")
    ax[0].set_xlabel("Y rel [m]"); ax[0].set_ylabel("X rel [m]")
    ax[0].set_title(r"$V_\theta(\zeta,-T)$ slice (velocities=0); red = BRT boundary $V=0$")
    fig.colorbar(pc, ax=ax[0])

    return fig, ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vessel", default=os.path.join(ROOT, "config", "vessel_config.yaml"))
    ap.add_argument("--game", default=os.path.join(ROOT, "config", "game_config.yaml"))
    ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--pretrain", type=int, default=300)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=128)
    args = ap.parse_args()

    with open(args.vessel) as f: vcfg = yaml.safe_load(f)
    with open(args.game) as f: gcfg = yaml.safe_load(f)

    Bnp = build(args.vessel, args.game)
    phys64 = HJBTorch(vcfg, gcfg, dtype=torch.float64)
    phys32 = HJBTorch(vcfg, gcfg, dtype=torch.float32)

    print("=" * 70)
    print(" STEP 1 — math check: torch physics vs numpy reference")
    e_drift, e_ham, e_term = math_check(Bnp, phys64)
    print(f"   max|drift_torch - drift_numpy|      = {e_drift:.2e}")
    print(f"   max|H_torch - H_numpy|              = {e_ham:.2e}")
    print(f"   max|terminal_torch - terminal_numpy| = {e_term:.2e}")
    ok_math = max(e_drift, e_ham, e_term) < 1e-8
    print(f"   -> {'PASS' if ok_math else 'FAIL'} (threshold 1e-8)")

    T = phys32.T
    lo = [-15, -15, -np.pi, -0.2, -0.6, -0.2, -0.6, -1.5, -1.5, -T]
    hi = [15, 15, np.pi, 1.2, 0.6, 1.2, 0.6, 1.5, 1.5, 0.0]

    print("-" * 70)
    print(" STEP 2 — gradient check: autograd vs finite difference on SIREN")
    net_probe = SIREN(lo, hi, hidden=64, layers=3, dtype=torch.float64)
    eX, et = grad_check(net_probe)
    print(f"   |dV/dX autograd - finite-diff| = {eX:.2e}")
    print(f"   |dV/dt autograd - finite-diff| = {et:.2e}")
    print(f"   -> {'PASS' if max(eX, et) < 1e-4 else 'FAIL'} (threshold 1e-4)")

    print("-" * 70)
    print(" STEP 2b — heading periodicity: |V(psi=-pi) - V(psi=+pi)| (untrained net)")
    e_wrap = continuity_check(net_probe, lo, hi)
    print(f"   max wrap-around gap = {e_wrap:.2e}")
    print(f"   -> {'PASS' if e_wrap < 1e-9 else 'FAIL'} (should be ~0 by construction)")

    print("-" * 70)
    print(f" STEP 3 — training ({args.iters} iters, hidden={args.hidden}, batch={args.batch}, CPU)")
    net, hist, dt = train(phys32, lo, hi, iters=args.iters, pretrain=args.pretrain,
                          batch=args.batch, hidden=args.hidden)
    print(f"   trained in {dt:.1f} s")
    print(f"   {'iter':>6} {'loss_term':>14} {'loss_pde':>14}")
    for it, lt, lp in hist:
        print(f"   {it:>6} {lt:>14.6e} {lp:>14.6e}")

    e_wrap_trained = continuity_check(net, lo, hi, dtype=torch.float32)
    print(f"   heading wrap-around gap after training: {e_wrap_trained:.2e} (stays ~0)")

    print("-" * 70)
    print(" STEP 4 — diagnostics -> outputs/train_pinn.png")
    import matplotlib.pyplot as plt
    fig, ax = diagnostics(net, phys32, lo, hi, None)
    its = [h[0] for h in hist]; lts = [h[1] for h in hist]; lps = [max(h[2], 1e-12) for h in hist]
    ax[1].semilogy(its, lts, label="terminal loss")
    ax[1].semilogy(its, lps, label="variational-inequality residual")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("loss (log scale)")
    ax[1].grid(True, which="both", alpha=0.3); ax[1].legend(); ax[1].set_title("Training loss")
    out = os.path.join(ROOT, "outputs"); os.makedirs(out, exist_ok=True)
    fig.tight_layout(); fig.savefig(os.path.join(out, "train_pinn.png"), dpi=130)
    torch.save(net.state_dict(), os.path.join(out, "pinn_value.pt"))
    print("   saved outputs/train_pinn.png and outputs/pinn_value.pt")
    print("=" * 70)


if __name__ == "__main__":
    main()
