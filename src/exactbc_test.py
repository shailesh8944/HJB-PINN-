"""
exactbc_test.py — Unit tests for the exact-boundary-condition value ansatz
V(zeta,t) = ell(zeta) + t * softplus(N(zeta,t)), BEFORE any training.

Checks (all must hold for an untrained, random network, by construction):
  (1) V(zeta, 0) = ell(zeta)   exactly (terminal condition).
  (2) V(zeta, t) <= ell(zeta)  for all t in [-T, 0]  (tube property).
  (3) autograd dV/dzeta, dV/dt match finite differences (gradients usable).
"""

import os, sys, numpy as np, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hji_torch import GameHJITorch
from pinn import SIREN, ExactBCValue, value_and_grads

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)


def main():
    with open(os.path.join(ROOT, "config", "vessel_config.yaml")) as f: vcfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, "config", "game_config.yaml")) as f: gcfg = yaml.safe_load(f)
    phys = GameHJITorch(vcfg, gcfg, dtype=torch.float64)
    T = phys.T
    lo = [-15, -15, -np.pi, -0.2, -0.6, -0.2, -0.6, -1.5, -1.5, -T]
    hi = [15, 15, np.pi, 1.2, 0.6, 1.2, 0.6, 1.5, 1.5, 0.0]

    torch.manual_seed(0)
    siren = SIREN(lo, hi, hidden=64, layers=3, dtype=torch.float64)
    net = ExactBCValue(siren, phys.terminal)

    rng = np.random.default_rng(0)
    n = 400
    Z = torch.tensor(rng.uniform(lo[:9], hi[:9], size=(n, 9)))

    # (1) terminal condition at t = 0
    inp0 = torch.cat([Z, torch.zeros(n, 1, dtype=torch.float64)], 1)
    with torch.no_grad():
        V0 = net(inp0).squeeze(1)
    ell = phys.terminal(Z)
    e_term = float((V0 - ell).abs().max())

    # (2) tube property V <= ell for random t in [-T, 0]
    tneg = torch.tensor(-rng.random((n, 1)) * T)
    inpt = torch.cat([Z, tneg], 1)
    with torch.no_grad():
        Vt = net(inpt).squeeze(1)
    over = float((Vt - ell).max())          # should be <= 0

    # (3) gradient autograd vs finite difference
    zeta = torch.zeros(1, 9, dtype=torch.float64); zeta[0, 0] = 6.0; zeta[0, 1] = 2.0
    t = torch.tensor([-3.0], dtype=torch.float64)
    _, dVdt, gradV = value_and_grads(net, zeta, t)
    eps = 1e-6
    def val(z, tt):
        with torch.no_grad():
            return net(torch.cat([z, tt.reshape(-1, 1)], 1)).item()
    zp = zeta.clone(); zp[0, 0] += eps; zm = zeta.clone(); zm[0, 0] -= eps
    dX_fd = (val(zp, t) - val(zm, t)) / (2 * eps)
    dt_fd = (val(zeta, t + eps) - val(zeta, t - eps)) / (2 * eps)
    eX = abs(gradV[0, 0].item() - dX_fd); et = abs(dVdt[0].item() - dt_fd)

    print("=" * 62)
    print(" Exact-boundary-condition ansatz — unit tests")
    print("=" * 62)
    print(f" (1) max|V(.,0) - ell|            : {e_term:.2e}  [{'PASS' if e_term < 1e-10 else 'FAIL'}]")
    print(f" (2) max(V - ell) over t<=0        : {over:.2e}  [{'PASS' if over <= 1e-9 else 'FAIL'}]  (<=0 required)")
    print(f" (3) |dV/dX  autograd - finite|    : {eX:.2e}  [{'PASS' if eX < 1e-4 else 'FAIL'}]")
    print(f"     |dV/dt  autograd - finite|    : {et:.2e}  [{'PASS' if et < 1e-4 else 'FAIL'}]")
    print("=" * 62)


if __name__ == "__main__":
    main()
