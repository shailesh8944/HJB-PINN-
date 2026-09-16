"""
verify_hji.py — Verification harness for the two-player game value function.

Two independent checks that a *trained* value function actually solves the game
(neither is a training signal; both are run after training):

  A. Variational-inequality residual statistics over the domain — how close the
     network is to satisfying min{ dV/dt + H_game, ell - V } = 0.

  B. Closed-loop saddle-point rollout — synthesise BOTH controls from grad V
     (pursuer tau_i* = argmin, evader tau_e* = argmax), integrate the true game
     dynamics, and compare the realised closest approach min_s ell(zeta(s)) to
     the predicted value V(zeta_0, -T). Agreement is the strongest evidence the
     value function is correct; a state with V<0 should be captured (min ell<=0)
     and a state with V>0 should escape.

Run directly for a harness smoke test on an UNTRAINED network: it confirms the
machinery executes and the controls stay realisable. The numbers are meaningful
only once the network is trained.
"""

import os, sys, numpy as np, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hji_torch import GameHJITorch
from pinn import SIREN, ExactBCValue

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)


def grad_V(net, zeta, t):
    """V, dV/dt, grad_zeta V (no create_graph — for evaluation/rollout)."""
    inp = torch.cat([zeta, t.reshape(-1, 1)], 1).requires_grad_(True)
    V = net(inp).squeeze(1)
    g = torch.autograd.grad(V.sum(), inp)[0]
    return V.detach(), g[:, 9].detach(), g[:, :9].detach()


def residual_stats(net, phys, lo, hi, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    Z = torch.tensor(rng.uniform(lo[:9], hi[:9], size=(n, 9)), dtype=torch.float32)
    t = torch.tensor(-rng.random((n,)) * phys.T, dtype=torch.float32)
    V, dVdt, gradV = grad_V(net, Z, t)
    res = phys.game_vi_residual(Z, V, dVdt, gradV).abs().numpy()
    return dict(rms=float(np.sqrt((res ** 2).mean())), mx=float(res.max()),
                p90=float(np.percentile(res, 90)))


def rollout(net, phys, zeta0, dt=0.05, realizable_report=True):
    """Closed-loop saddle rollout from t=-T to 0. Returns realised min ell,
    predicted V(zeta0,-T), and the worst thruster-band violation seen."""
    T = phys.T
    zeta = torch.tensor(zeta0, dtype=torch.float32).reshape(1, 9)
    n = int(round(T / dt))
    d, Fmax = phys.arm, phys.Fmax
    Vpred, _, _ = grad_V(net, zeta, torch.tensor([-T], dtype=torch.float32))
    ell_min = float(phys.terminal(zeta))
    worst = 0.0

    def step_dyn(z, ti, te):
        return phys.zeta_dot(z, ti, te)

    for k in range(n):
        t = torch.tensor([-T + k * dt], dtype=torch.float32)
        _, _, gz = grad_V(net, zeta, t)
        ti, te = phys.optimal_controls(gz)
        if realizable_report:
            for tau in (ti, te):
                Fs = tau[0, 0] / 2 + tau[0, 1] / (2 * d); Fp = tau[0, 0] / 2 - tau[0, 1] / (2 * d)
                worst = max(worst, float(max(-min(Fs, Fp, 0.0), max(Fs - Fmax, Fp - Fmax, 0.0))))
        # rk4 with controls held constant over the step
        k1 = step_dyn(zeta, ti, te)
        k2 = step_dyn(zeta + 0.5 * dt * k1, ti, te)
        k3 = step_dyn(zeta + 0.5 * dt * k2, ti, te)
        k4 = step_dyn(zeta + dt * k3, ti, te)
        zeta = zeta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        ell_min = min(ell_min, float(phys.terminal(zeta)))
    return dict(realised_min_ell=ell_min, V_pred=float(Vpred), worst_band=worst)


def main():
    with open(os.path.join(ROOT, "config", "vessel_config.yaml")) as f: vcfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, "config", "game_config.yaml")) as f: gcfg = yaml.safe_load(f)
    phys = GameHJITorch(vcfg, gcfg, dtype=torch.float32)
    T = phys.T
    lo = [-15, -15, -np.pi, -0.2, -0.6, -0.2, -0.6, -1.5, -1.5, -T]
    hi = [15, 15, np.pi, 1.2, 0.6, 1.2, 0.6, 1.5, 1.5, 0.0]

    torch.manual_seed(0)
    net = ExactBCValue(SIREN(lo, hi, hidden=64, layers=3, dtype=torch.float32), phys.terminal)

    print("=" * 66)
    print(" HJI verification harness — SMOKE TEST (untrained net; numbers not")
    print(" meaningful, only mechanics + control realisability are checked)")
    print("=" * 66)
    st = residual_stats(net, phys, lo, hi)
    print(f" A. residual stats: rms={st['rms']:.3e}  p90={st['p90']:.3e}  max={st['mx']:.3e}")
    ro = rollout(net, phys, [8.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    print(f" B. rollout: realised min ell = {ro['realised_min_ell']:.3f} m,  "
          f"V_pred = {ro['V_pred']:.3f} m")
    print(f"    worst thruster-band violation during rollout = {ro['worst_band']:.2e}  "
          f"[{'PASS' if ro['worst_band'] < 1e-6 else 'FAIL'}]")
    print("=" * 66)
    print(" (After training: |realised_min_ell - V_pred| should be small, and")
    print("  V<0 states should be captured, V>0 states should escape.)")


if __name__ == "__main__":
    main()
