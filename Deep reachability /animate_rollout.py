"""Closed-loop trajectory animation -- watch the trained saddle-point policy
actually play out. Rolls out ONE initial condition under the pursuer/evader
optimal controls read off gradV (same controls as closed_loop_rollout.py's
Monte-Carlo check), but instead of only comparing final danger/safe labels,
reconstructs BOTH vessels' poses in a shared inertial (world) frame and
animates them -- so steering/engagement mistakes are visible directly
instead of only inferred from a static barrier plot.

The trained state is relative (X,Y,psi_rel in the pursuer's body frame), so
there is no absolute frame in the state itself. We recover one with a fixed
gauge: the pursuer starts at the world origin heading 0, and its own
body-frame velocities (u_i,v_i,r_i) -- which ARE part of the state -- are
integrated forward with standard 3-DOF kinematics. The evader's world pose
is then recovered exactly from the relative state each step:
    x_e = x_i + X*cos(psi_i) - Y*sin(psi_i)
    y_e = y_i + X*sin(psi_i) + Y*cos(psi_i)
    psi_e = psi_i + psi_rel
(Differentiating this reproduces d(x_e)/dt = u_e*cos(psi_e) - v_e*sin(psi_e)
exactly, i.e. it is a faithful reconstruction, not an approximation.)

usage:
    python3 animate_rollout.py [--ckpt PATH] [--preset headon|crossing|overtake]
                                [--x0 X] [--y0 Y] [--psi0 RAD] ...
                                [--tmax SECONDS] [--out out_hji/rollout.gif]
"""
import os, sys, glob, argparse
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from config import (IX, IY, IPSI, IUE, IVE, IUI, IVI, IRE, IRI,
                     M11, M33, R_COLLIDE, T_HORIZON)
from vessel_dynamics import drift0, optimal_controls
from value_network import SIREN, ExactBCValue

PRESETS = {
    # X, Y, psi_rel, ue, ve, ui, vi, re, ri
    "headon":   dict(x0=8.0, y0=1.5, psi0=np.pi,      ue0=0.8, ui0=0.8),
    "crossing": dict(x0=6.0, y0=6.0, psi0=-np.pi/2,   ue0=0.6, ui0=0.8),
    "overtake": dict(x0=5.0, y0=0.0, psi0=0.0,        ue0=0.5, ui0=0.9),
}


def find_latest_ckpt(out_dir="out_hji"):
    cks = sorted(glob.glob(os.path.join(out_dir, "ckpt_*.pt")))
    if not cks:
        raise FileNotFoundError(f"no checkpoints found in {out_dir}/ -- pass --ckpt explicitly")
    return cks[-1]


def load_net(ckpt_path, device):
    lo = [-15, -15, -np.pi, -0.2, -0.6, -0.2, -0.6, -1.5, -1.5, -T_HORIZON]
    hi = [ 15,  15,  np.pi,  1.2,  0.6,  1.2,  0.6,  1.5,  1.5, 0.0]
    net = ExactBCValue(SIREN(lo, hi, 512, 3)).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ck["net"]); net.eval()
    return net, ck["iter"]


def rollout(net, device, z0, dt=0.02, tmax=T_HORIZON):
    """Integrate the relative state forward under saddle-point controls and
    simultaneously reconstruct both vessels' inertial-frame poses."""
    steps = int(tmax / dt)
    z = torch.tensor(z0, dtype=torch.float32, device=device).unsqueeze(0)
    t_elapsed = torch.zeros(1, device=device)
    xi = torch.zeros(1, device=device); yi = torch.zeros(1, device=device); psii = torch.zeros(1, device=device)

    hist = {k: [] for k in ("t", "xi", "yi", "psii", "xe", "ye", "psie", "V", "sep")}
    for step in range(steps):
        t_in = -(T_HORIZON - t_elapsed)
        zt = z.clone().requires_grad_(True)
        inp = torch.cat([zt, t_in.reshape(-1, 1)], 1)
        V = net(inp).squeeze(1)
        gradV = torch.autograd.grad(V.sum(), zt)[0].detach()
        with torch.no_grad():
            X, Y, psi_rel = z[:, IX], z[:, IY], z[:, IPSI]
            cpi, spi = torch.cos(psii), torch.sin(psii)
            xe = xi + X*cpi - Y*spi
            ye = yi + X*spi + Y*cpi
            psie = psii + psi_rel
            sep = torch.sqrt(X**2 + Y**2 + 1e-12)

            hist["t"].append(float(t_elapsed[0]))
            hist["xi"].append(float(xi[0])); hist["yi"].append(float(yi[0])); hist["psii"].append(float(psii[0]))
            hist["xe"].append(float(xe[0])); hist["ye"].append(float(ye[0])); hist["psie"].append(float(psie[0]))
            hist["V"].append(float(V[0])); hist["sep"].append(float(sep[0]))

            a0 = drift0(z)
            ti, te = optimal_controls(gradV)
            adot = a0.clone()
            adot[:, IUE] += te[:, 0] / M11
            adot[:, IRE] += te[:, 1] / M33
            adot[:, IUI] += ti[:, 0] / M11
            adot[:, IRI] += ti[:, 1] / M33

            ui, vi, ri = z[:, IUI], z[:, IVI], z[:, IRI]
            xi = xi + dt * (ui*cpi - vi*spi)
            yi = yi + dt * (ui*spi + vi*cpi)
            psii = psii + dt * ri

            z = z + dt * adot
            t_elapsed = t_elapsed + dt
            if sep[0] <= R_COLLIDE:
                break
    return hist


def make_animation(hist, ckpt_iter, out_path, fps=25, stride=2):
    t = np.array(hist["t"]); xi = np.array(hist["xi"]); yi = np.array(hist["yi"]); psii = np.array(hist["psii"])
    xe = np.array(hist["xe"]); ye = np.array(hist["ye"]); psie = np.array(hist["psie"])
    V = np.array(hist["V"]); sep = np.array(hist["sep"])
    idx = np.arange(0, len(t), stride)
    if idx[-1] != len(t) - 1:
        idx = np.append(idx, len(t) - 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.3, 1]})

    all_x = np.concatenate([xi, xe]); all_y = np.concatenate([yi, ye])
    pad = 2.0
    ax1.set_xlim(all_x.min()-pad, all_x.max()+pad)
    ax1.set_ylim(all_y.min()-pad, all_y.max()+pad)
    ax1.set_aspect("equal"); ax1.set_xlabel("X world [m]"); ax1.set_ylabel("Y world [m]")
    ax1.set_title("Closed-loop rollout (world frame)")

    trail_i, = ax1.plot([], [], "b-", lw=1, alpha=0.6)
    trail_e, = ax1.plot([], [], "r-", lw=1, alpha=0.6)
    pursuer_dot, = ax1.plot([], [], marker=(3, 0, 0), color="b", ms=14, ls="", label="pursuer")
    evader_dot,  = ax1.plot([], [], marker=(3, 0, 0), color="r", ms=14, ls="", label="evader")
    danger_circle = plt.Circle((0, 0), R_COLLIDE, fc="none", ec="k", ls="--", lw=1)
    ax1.add_patch(danger_circle)
    ax1.legend(loc="upper right")

    ax2.set_xlim(t.min(), max(t.max(), t.min()+1e-3))
    ax2.set_ylim(min(V.min(), 0)-0.5, max(V.max(), sep.max())+0.5)
    ax2.set_xlabel("t [s] (elapsed)"); ax2.set_title("V(t) and separation(t)")
    line_V, = ax2.plot([], [], "k-", label="V (network)")
    line_sep, = ax2.plot([], [], "g-", label="separation [m]")
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.legend(loc="upper right")
    now_line = ax2.axvline(t[0], color="orange", lw=1)

    fig.suptitle(f"Closed-loop rollout @ iter {ckpt_iter}  (final sep={sep[-1]:.2f} m, "
                 f"{'COLLISION' if sep[-1] <= R_COLLIDE else 'no collision'})")

    def update(k):
        i = idx[k]
        trail_i.set_data(xi[:i+1], yi[:i+1])
        trail_e.set_data(xe[:i+1], ye[:i+1])
        pursuer_dot.set_data([xi[i]], [yi[i]])
        pursuer_dot.set_marker((3, 0, np.degrees(psii[i]) - 90))
        evader_dot.set_data([xe[i]], [ye[i]])
        evader_dot.set_marker((3, 0, np.degrees(psie[i]) - 90))
        danger_circle.center = (xi[i], yi[i])
        line_V.set_data(t[:i+1], V[:i+1])
        line_sep.set_data(t[:i+1], sep[:i+1])
        now_line.set_xdata([t[i], t[i]])
        return trail_i, trail_e, pursuer_dot, evader_dot, danger_circle, line_V, line_sep, now_line

    anim = animation.FuncAnimation(fig, update, frames=len(idx), blit=False)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint path (default: latest in --ckpt-dir)")
    ap.add_argument("--ckpt-dir", default="out_hji")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--preset", choices=list(PRESETS), default="headon")
    ap.add_argument("--x0", type=float, default=None); ap.add_argument("--y0", type=float, default=None)
    ap.add_argument("--psi0", type=float, default=None)
    ap.add_argument("--ue0", type=float, default=None); ap.add_argument("--ve0", type=float, default=None)
    ap.add_argument("--ui0", type=float, default=None); ap.add_argument("--vi0", type=float, default=None)
    ap.add_argument("--re0", type=float, default=None); ap.add_argument("--ri0", type=float, default=None)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--tmax", type=float, default=T_HORIZON)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--stride", type=int, default=2, help="animate every Nth simulated step")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt_path = args.ckpt or find_latest_ckpt(args.ckpt_dir)
    net, ckpt_iter = load_net(ckpt_path, args.device)
    print(f"loaded {ckpt_path}  (iter {ckpt_iter})")

    p = dict(PRESETS[args.preset])
    overrides = dict(x0=args.x0, y0=args.y0, psi0=args.psi0, ue0=args.ue0, ve0=args.ve0,
                      ui0=args.ui0, vi0=args.vi0, re0=args.re0, ri0=args.ri0)
    for k, v in overrides.items():
        if v is not None:
            p[k] = v

    z0 = np.zeros(9, np.float32)
    z0[IX] = p.get("x0", 0.0); z0[IY] = p.get("y0", 0.0); z0[IPSI] = p.get("psi0", 0.0)
    z0[IUE] = p.get("ue0", 0.0); z0[IVE] = p.get("ve0", 0.0)
    z0[IUI] = p.get("ui0", 0.0); z0[IVI] = p.get("vi0", 0.0)
    z0[IRE] = p.get("re0", 0.0); z0[IRI] = p.get("ri0", 0.0)
    print(f"initial state: X={z0[IX]:.2f} Y={z0[IY]:.2f} psi_rel={z0[IPSI]:.2f} "
          f"ue={z0[IUE]:.2f} ui={z0[IUI]:.2f}")

    hist = rollout(net, args.device, z0, dt=args.dt, tmax=args.tmax)
    print(f"rollout: {len(hist['t'])} steps, final separation={hist['sep'][-1]:.3f} m "
          f"(collision disc R={R_COLLIDE} m)")

    out_path = args.out or os.path.join(args.ckpt_dir, f"rollout_anim_{ckpt_iter:07d}_{args.preset}.gif")
    make_animation(hist, ckpt_iter, out_path, fps=args.fps, stride=args.stride)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
