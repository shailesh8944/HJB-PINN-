"""
simulate.py — Run the Sookshma 3-DOF simulation from the dynamic config file.

Usage:
    python src/simulate.py                       # uses config/vessel_config.yaml
    python src/simulate.py --config path.yaml
    python src/simulate.py --set theta_boot      # override coefficient set
    python src/simulate.py --no-coriolis         # turn C(v) off

Outputs (in outputs/):
    trajectory.csv   full time history
    trajectory.png   XY track + state plots
"""

import os
import sys
import argparse
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dynamics import VesselDynamics          # noqa: E402
from thruster import Thruster                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def command_at(schedule, t):
    """Return (F_port, F_stbd) kgf active at time t from a piecewise schedule."""
    Fp, Fs = 0.0, 0.0
    for row in schedule:
        if t >= row[0]:
            Fp, Fs = row[1], row[2]
    return Fp, Fs


def rk4_step(dyn, s, tau, dt):
    k1 = dyn.state_dot(s, tau)
    k2 = dyn.state_dot(s + 0.5 * dt * k1, tau)
    k3 = dyn.state_dot(s + 0.5 * dt * k2, tau)
    k4 = dyn.state_dot(s + dt * k3, tau)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def euler_step(dyn, s, tau, dt):
    return s + dt * dyn.state_dot(s, tau)


def run(cfg, config_dir):
    dyn = VesselDynamics(cfg)
    thr = Thruster(cfg, config_dir)
    sim = cfg["simulation"]

    dt = float(sim["dt"])
    tf = float(sim["t_final"])
    step = rk4_step if sim.get("integrator", "rk4") == "rk4" else euler_step
    schedule = sim["command_schedule"]

    s0 = sim["initial_state"]
    s = np.array([s0["x"], s0["y"], s0["psi"], s0["u"], s0["v"], s0["r"]], float)

    n = int(round(tf / dt)) + 1
    log = []
    for k in range(n):
        t = k * dt
        Fp_cmd, Fs_cmd = command_at(schedule, t)
        tau, (Fp, Fs) = thr.allocate(Fp_cmd, Fs_cmd)
        x, y, psi, u, v, r = s
        log.append([t, x, y, psi, u, v, r, tau[0], tau[2], Fp, Fs])
        s = step(dyn, s, tau, dt)

    return np.array(log), dyn, thr


def save_csv(log, path):
    header = "t,x,y,psi,u,v,r,tau_u_N,tau_r_Nm,F_port_kgf,F_stbd_kgf"
    np.savetxt(path, log, delimiter=",", header=header, comments="")


def make_plot(log, dyn, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = log[:, 0]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    ax[0, 0].plot(log[:, 2], log[:, 1], "b-")          # y (East) vs x (North)
    ax[0, 0].plot(log[0, 2], log[0, 1], "go", label="start")
    ax[0, 0].plot(log[-1, 2], log[-1, 1], "rs", label="end")
    ax[0, 0].set_xlabel("y  East [m]"); ax[0, 0].set_ylabel("x  North [m]")
    ax[0, 0].set_title("Track (NED)"); ax[0, 0].axis("equal")
    ax[0, 0].legend(); ax[0, 0].grid(True)

    ax[0, 1].plot(t, log[:, 4], label="u surge")
    ax[0, 1].plot(t, log[:, 5], label="v sway")
    ax[0, 1].set_xlabel("t [s]"); ax[0, 1].set_ylabel("[m/s]")
    ax[0, 1].set_title("Body velocities"); ax[0, 1].legend(); ax[0, 1].grid(True)

    ax[0, 2].plot(t, np.rad2deg(log[:, 6]), "m")
    ax[0, 2].set_xlabel("t [s]"); ax[0, 2].set_ylabel("r [deg/s]")
    ax[0, 2].set_title("Yaw rate"); ax[0, 2].grid(True)

    ax[1, 0].plot(t, np.rad2deg(log[:, 3]), "c")
    ax[1, 0].set_xlabel("t [s]"); ax[1, 0].set_ylabel("psi [deg]")
    ax[1, 0].set_title("Heading"); ax[1, 0].grid(True)

    ax[1, 1].plot(t, log[:, 9], label="F_port")
    ax[1, 1].plot(t, log[:, 10], label="F_stbd")
    ax[1, 1].set_xlabel("t [s]"); ax[1, 1].set_ylabel("kgf")
    ax[1, 1].set_title("Thruster forces (applied)"); ax[1, 1].legend(); ax[1, 1].grid(True)

    ax[1, 2].plot(t, log[:, 7], label="tau_u [N]")
    ax[1, 2].plot(t, log[:, 8], label="tau_r [N.m]")
    ax[1, 2].set_xlabel("t [s]"); ax[1, 2].set_title("Generalised force")
    ax[1, 2].legend(); ax[1, 2].grid(True)

    fig.suptitle(f"Sookshma 3-DOF Fossen simulation  |  set={dyn.active_set}  "
                 f"Coriolis={'on' if dyn.include_coriolis else 'off'}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "vessel_config.yaml"))
    ap.add_argument("--set", dest="cset", default=None,
                    help="override active_coeff_set (theta_anneal|theta_boot)")
    ap.add_argument("--no-coriolis", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    config_dir = os.path.dirname(os.path.abspath(args.config))
    if args.cset:
        cfg["model"]["active_coeff_set"] = args.cset
    if args.no_coriolis:
        cfg["model"]["include_coriolis"] = False

    log, dyn, thr = run(cfg, config_dir)

    out = os.path.join(ROOT, "outputs")
    os.makedirs(out, exist_ok=True)
    save_csv(log, os.path.join(out, "trajectory.csv"))
    make_plot(log, dyn, os.path.join(out, "trajectory.png"))

    fmin_N, fmax_N = thr.limits_N()
    print("=" * 60)
    print(f" Sookshma 3-DOF simulation complete")
    print(f"   coefficient set : {dyn.active_set}")
    print(f"   Coriolis term   : {'on' if dyn.include_coriolis else 'off'}")
    print(f"   M = diag({dyn.M11:.3f}, {dyn.M22:.3f}, {dyn.M33:.3f})")
    print(f"   thruster band   : {thr.f_min_kgf}-{thr.f_max_kgf} kgf "
          f"({fmin_N:.2f}-{fmax_N:.2f} N)")
    print(f"   final pos (N,E) : ({log[-1,1]:.2f}, {log[-1,2]:.2f}) m")
    print(f"   final u,v,r     : {log[-1,4]:.3f} m/s, {log[-1,5]:.3f} m/s, "
          f"{np.rad2deg(log[-1,6]):.2f} deg/s")
    print(f"   max |u|         : {np.max(np.abs(log[:,4])):.3f} m/s")
    print(f"   outputs         : outputs/trajectory.csv, outputs/trajectory.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
