"""Ad-hoc diagnostic: (1) compare V(z, t=-T_HORIZON) at fixed query points across
two checkpoints to see exactly how much the full-horizon read has moved between
them, and (2) sweep t from 0 to -T_HORIZON at those points on the LATEST
checkpoint to check the monotonic-tube-growth property (V should be
non-increasing as |t| grows -- more time remaining can only enlarge the danger
set, never shrink it). A violation here (V going back up as |t| grows) would
be a real bug; a violation only far past the current curriculum frontier
(t_cur) is expected mid-training noise, not yet meaningful.

usage: python3 check_monotonic.py <ckpt_earlier.pt> <ckpt_later.pt>
"""
import sys
import numpy as np
import torch

from config import IX, IY, IPSI, IUE, IUI, T_HORIZON
from value_network import SIREN, ExactBCValue

device = "cuda" if torch.cuda.is_available() else "cpu"


def load(path):
    lo = [-15, -15, -np.pi, -0.2, -0.6, -0.2, -0.6, -1.5, -1.5, -T_HORIZON]
    hi = [ 15,  15,  np.pi,  1.2,  0.6,  1.2,  0.6,  1.5,  1.5, 0.0]
    net = ExactBCValue(SIREN(lo, hi, 512, 3)).to(device)
    ck = torch.load(path, map_location=device)
    net.load_state_dict(ck["net"]); net.eval()
    return net, ck["iter"]


def t_cur_at(it, warmup=10000, iters=150000, curric_frac=0.5):
    frac = min(1.0, max(0.0, it - warmup) / max(1.0, curric_frac*iters))
    return 1e-3 + T_HORIZON*frac


def eval_V(net, X, Y, t):
    z = np.zeros((1, 9), np.float32)
    z[0, IX] = X; z[0, IY] = Y; z[0, IPSI] = np.pi
    z[0, IUE] = 0.8; z[0, IUI] = 0.8
    inp = torch.tensor(np.hstack([z, [[t]]]), dtype=torch.float32, device=device)
    with torch.no_grad():
        return float(net(inp).item())


if __name__ == "__main__":
    ck_a, ck_b = sys.argv[1], sys.argv[2]
    net_a, it_a = load(ck_a)
    net_b, it_b = load(ck_b)
    tc_a, tc_b = t_cur_at(it_a), t_cur_at(it_b)
    print(f"checkpoint A: {ck_a}  iter={it_a}  t_cur(trained horizon)={tc_a:.2f}s")
    print(f"checkpoint B: {ck_b}  iter={it_b}  t_cur(trained horizon)={tc_b:.2f}s")

    print("\n-- V(X=0, Y, t=-20s) at both checkpoints (Y sweep along approach axis) --")
    print(f"{'Y':>6} {'V@A':>12} {'V@B':>12} {'delta':>12}")
    for Y in [0, 3, 6, 8, 9, 10, 11, 12, 14]:
        va = eval_V(net_a, 0.0, Y, -T_HORIZON)
        vb = eval_V(net_b, 0.0, Y, -T_HORIZON)
        print(f"{Y:>6.1f} {va:>12.4f} {vb:>12.4f} {vb-va:>12.4f}")

    print(f"\n-- monotonic-tube-growth check on LATEST checkpoint (iter {it_b}) --")
    print("V(X=0,Y,t) swept over t=0..-20s ; should be non-increasing as |t| grows")
    for Y in [3, 6, 9, 12]:
        vals = [eval_V(net_b, 0.0, Y, -t) for t in np.linspace(0, T_HORIZON, 11)]
        worst_violation = max((vals[i+1]-vals[i]) for i in range(len(vals)-1))
        flag = "  <-- VIOLATION (V increased)" if worst_violation > 1e-3 else ""
        print(f"Y={Y:>4.1f}: " + " ".join(f"{v:6.2f}" for v in vals) + f"   max_increase={worst_violation:.4f}{flag}")
        print(f"        (t_cur={tc_b:.1f}s marks the trained frontier; values past that column are extrapolated)")
