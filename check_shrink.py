"""Shrink-to-disc sanity check (PROJECT_STATUS.md Sec 6c): as horizon -> 0,
the danger set {V<=0} must collapse to exactly the R=1m collision disc.
Evaluates the trained checkpoint at several |t| values, from near-zero up
to the full horizon, on the same head-on slice used by save_map().
"""
import os, sys, glob, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train_overnight import SIREN, ExactBCValue, IX, IY, IPSI, IUE, IUI, R_COLLIDE, T_HORIZON

# usage: python3 check_shrink.py [ckpt_path]   (defaults to the latest snapshot in out_hji/)
if len(sys.argv) > 1:
    ckpt_path = sys.argv[1]
else:
    cks = sorted(glob.glob(os.path.join("out_hji", "ckpt_*.pt")))
    if not cks:
        raise FileNotFoundError("no checkpoints found in out_hji/ -- pass a path explicitly")
    ckpt_path = cks[-1]

device = "cuda" if torch.cuda.is_available() else "cpu"
lo = [-15, -15, -np.pi, -0.2, -0.6, -0.2, -0.6, -1.5, -1.5, -T_HORIZON]
hi = [ 15,  15,  np.pi,  1.2,  0.6,  1.2,  0.6,  1.5,  1.5, 0.0]
net = ExactBCValue(SIREN(lo, hi, 512, 3)).to(device)
ck = torch.load(ckpt_path, map_location=device)
net.load_state_dict(ck["net"]); net.eval()
print(f"loaded {ckpt_path}  (iter {ck['iter']})")

ng, approach = 200, 0.8
xs = np.linspace(-15, 15, ng); ys = np.linspace(-15, 15, ng)
XX, YY = np.meshgrid(xs, ys)
t_values = [-1e-3, -0.05, -0.2, -0.5, -1.0, -2.0, -5.0, -T_HORIZON]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for ax, tval in zip(axes.flat, t_values):
    Z = np.zeros((ng*ng, 9), np.float32)
    Z[:, IX] = XX.ravel(); Z[:, IY] = YY.ravel()
    Z[:, IPSI] = np.pi
    Z[:, IUE] = approach; Z[:, IUI] = approach
    tt = np.full((ng*ng, 1), tval, np.float32)
    inp = torch.tensor(np.hstack([Z, tt]), device=device)
    with torch.no_grad():
        V = net(inp).float().cpu().numpy().reshape(ng, ng)
    ax.contourf(YY, XX, (V <= 0).astype(float), levels=[-.5, .5, 1.5], colors=["#c7e9c0", "#fb6a4a"])
    ax.contour(YY, XX, V, levels=[0.0], colors="k", linewidths=1.5)
    th = np.linspace(0, 2*np.pi, 100)
    ax.plot(R_COLLIDE*np.sin(th), R_COLLIDE*np.cos(th), "b--", lw=1.5, label=f"R={R_COLLIDE:.0f}m disc")
    ax.plot(0, 0, "ks", ms=5)
    ax.set_title(f"t = {tval:.3f}s  ({-tval:.3f}s to go)")
    ax.set_xlabel("Y rel [m]"); ax.set_ylabel("X rel [m]")
    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6) if -tval <= 2.0 else (ax.set_xlim(-15,15), ax.set_ylim(-15,15))
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)

fig.suptitle(f"Shrink-to-disc check @ iter {ck['iter']}  (head-on, approach={approach} m/s)")
fig.tight_layout()
out_path = os.path.join("out_hji", f"shrink_check_{ck['iter']:07d}.png")
fig.savefig(out_path, dpi=130)
print(f"saved {out_path}")
