"""Closed-loop Monte-Carlo rollout check (PROJECT_STATUS.md Sec 6d) -- the gold
standard. Simulates the real two-player game forward from many initial states
using the saddle-point controls read off the trained value gradient (pursuer
minimising, evader maximising), and compares the realised minimum separation
against the network's own {V<=0} danger prediction, on the same head-on slice
used by save_map(). Explicit-Euler integration with a small dt.

usage: python3 closed_loop_rollout.py [ckpt_path]   (defaults to latest in out_hji/)
"""
import os, sys, glob, time, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import IX, IY, IPSI, IUE, IVE, IUI, IVI, IRE, IRI, M11, M33, R_COLLIDE, T_HORIZON
from vessel_dynamics import drift0, optimal_controls
from value_network import SIREN, ExactBCValue

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

# ---- initial states: same head-on slice as danger_safe.png ----
ng, approach = 46, 0.8
xs = np.linspace(-15, 15, ng); ys = np.linspace(-15, 15, ng)
XX, YY = np.meshgrid(xs, ys)
n = ng * ng
Z0 = np.zeros((n, 9), np.float32)
Z0[:, IX] = XX.ravel(); Z0[:, IY] = YY.ravel()
Z0[:, IPSI] = np.pi
Z0[:, IUE] = approach; Z0[:, IUI] = approach
z = torch.tensor(Z0, device=device)

with torch.no_grad():
    inp0 = torch.cat([z, torch.full((n, 1), -T_HORIZON, device=device)], 1)
    V_pred = net(inp0).squeeze(1).cpu().numpy()
pred_danger = V_pred <= 0.0
print(f"predicted danger fraction: {pred_danger.mean()*100:.1f}%  ({pred_danger.sum()}/{n} points)")

# ---- rollout ----
dt = 0.02
steps = int(T_HORIZON / dt)
t_elapsed = torch.zeros(n, device=device)
min_sep = torch.sqrt(z[:, IX]**2 + z[:, IY]**2).clone()
max_abs_xy = max_abs_uvr = 0.0

t0 = time.time()
for step in range(steps):
    t_in = -(T_HORIZON - t_elapsed)
    zt = z.clone().requires_grad_(True)
    inp = torch.cat([zt, t_in.reshape(-1, 1)], 1)
    V = net(inp).squeeze(1)
    gradV = torch.autograd.grad(V.sum(), zt)[0].detach()
    with torch.no_grad():
        a0 = drift0(z)
        ti, te = optimal_controls(gradV)
        adot = a0.clone()
        adot[:, IUE] += te[:, 0] / M11
        adot[:, IRE] += te[:, 1] / M33
        adot[:, IUI] += ti[:, 0] / M11
        adot[:, IRI] += ti[:, 1] / M33
        z = z + dt * adot
        t_elapsed = t_elapsed + dt
        sep = torch.sqrt(z[:, IX]**2 + z[:, IY]**2 + 1e-12)
        min_sep = torch.minimum(min_sep, sep)
        max_abs_xy = max(max_abs_xy, z[:, IX].abs().max().item(), z[:, IY].abs().max().item())
        max_abs_uvr = max(max_abs_uvr, z[:, [IUE, IVE, IUI, IVI, IRE, IRI]].abs().max().item())
    if step % 200 == 0:
        print(f"  step {step:4d}/{steps}  wall {time.time()-t0:5.1f}s", flush=True)

min_sep = min_sep.cpu().numpy()
realised_danger = min_sep <= R_COLLIDE

false_safe = np.mean(realised_danger & ~pred_danger)   # predicted SAFE, actually collided -- the dangerous error
false_danger = np.mean(~realised_danger & pred_danger) # predicted DANGER, actually safe -- conservative error
agree = np.mean(realised_danger == pred_danger)

print("=" * 70)
print(f" grid points: {n}   rollout wall time: {time.time()-t0:.1f}s")
print(f" agreement: {agree*100:.2f}%")
print(f" false-SAFE rate  (predicted safe, actually collided): {false_safe*100:.2f}%   <-- dangerous error")
print(f" false-DANGER rate (predicted danger, actually safe):  {false_danger*100:.2f}%   <-- conservative error")
print(f" max |X or Y| reached during rollout: {max_abs_xy:.1f} m   (training domain: +-15 m)")
print(f" max |u,v,r| reached during rollout:  {max_abs_uvr:.3f}   (training caps: u<=1.2, v<=0.6, r<=1.5)")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
th = np.linspace(0, 2*np.pi, 100)
for ax, mask, title in zip(
        axes,
        [pred_danger, realised_danger, None],
        ["Predicted danger {V<=0}", "Realised danger (min sep <= R)", "Errors (red=false-safe, orange=false-danger)"]):
    if mask is not None:
        ax.contourf(YY, XX, mask.reshape(ng, ng).astype(float), levels=[-.5, .5, 1.5],
                    colors=["#c7e9c0", "#fb6a4a"])
    else:
        err = np.zeros(n)
        err[realised_danger & ~pred_danger] = 2
        err[~realised_danger & pred_danger] = 1
        ax.contourf(YY, XX, err.reshape(ng, ng), levels=[-.5, .5, 1.5, 2.5],
                    colors=["#c7e9c0", "#fdae6b", "#d62728"])
    ax.plot(R_COLLIDE*np.sin(th), R_COLLIDE*np.cos(th), "b--", lw=1.2)
    ax.plot(0, 0, "ks", ms=6)
    ax.set_xlabel("Y rel [m]"); ax.set_ylabel("X rel [m]")
    ax.set_title(title); ax.set_aspect("equal")
fig.suptitle(f"Closed-loop rollout check @ iter {ck['iter']}  "
             f"(agree={agree*100:.1f}%, false-safe={false_safe*100:.2f}%, false-danger={false_danger*100:.2f}%)")
fig.tight_layout()
out_path = os.path.join("out_hji", f"rollout_check_{ck['iter']:07d}.png")
fig.savefig(out_path, dpi=130)
print(f"saved {out_path}")
