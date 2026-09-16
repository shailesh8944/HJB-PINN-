"""
track_waypoints.py — Closed-loop waypoint-tracking simulation for Sookshma.

Runs the LOS-guided PD/PI controller against the identified 3-DOF dynamics and
produces a track plot and a thruster plot.

Usage:
    python src/track_waypoints.py
    python src/track_waypoints.py --set theta_boot
    python src/track_waypoints.py --no-coriolis
"""

import os
import sys
import argparse
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dynamics import VesselDynamics          # noqa: E402
from thruster import Thruster                # noqa: E402
from controller import WaypointController    # noqa: E402
import simulate as sim                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def cross_track_error(p, wp_prev, wp_next):
    """Signed perpendicular distance of point p from the wp_prev->wp_next line."""
    a = np.array(wp_prev); b = np.array(wp_next); pp = np.array(p)
    ab = b - a
    L = np.hypot(*ab)
    if L < 1e-9:
        return np.hypot(*(pp - a))
    # 2D cross product (ab x ap) / |ab|
    return (ab[0] * (pp[1] - a[1]) - ab[1] * (pp[0] - a[0])) / L


def run(cfg, config_dir):
    dyn = VesselDynamics(cfg)
    thr = Thruster(cfg, config_dir)
    ctrl = WaypointController(cfg, thr)
    m = cfg["mission"]

    dt = float(m["dt"])
    t_max = float(m["t_max"])
    s0 = m["initial_state"]
    s = np.array([s0["x"], s0["y"], s0["psi"], s0["u"], s0["v"], s0["r"]], float)

    wps = [np.array(w, float) for w in m["waypoints"]]
    start = np.array([s0["x"], s0["y"]], float)

    log = []
    t = 0.0
    while t < t_max and not ctrl.done:
        prev_wp = start if ctrl.idx == 0 else wps[ctrl.idx - 1]
        next_wp = wps[ctrl.idx]
        tau_cmd, tau_app, (Fp, Fs), tele = ctrl.update(s, dt)
        xte = cross_track_error(s[:2], prev_wp, next_wp)
        x, y, psi, u, v, r = s
        log.append([t, x, y, psi, u, v, r,
                    tele["u_d"], np.rad2deg(psi), np.rad2deg(tele["psi_d"]),
                    xte, Fp, Fs, tau_app[0], tau_app[2],
                    tele["tau_u_cmd"], tele["tau_r_cmd"], tele["wp_idx"]])
        # integrate with the ACTUAL (saturated) force held over the step
        s = sim.rk4_step(dyn, s, tau_app, dt)
        t += dt

    return np.array(log), dyn, thr, ctrl, wps


COLS = ["t", "x", "y", "psi", "u", "v", "r", "u_d", "psi_deg", "psi_d_deg",
        "xte_m", "F_port_kgf", "F_stbd_kgf", "tau_u_N", "tau_r_Nm",
        "tau_u_cmd_N", "tau_r_cmd_Nm", "wp_idx"]


def save_csv(log, path):
    np.savetxt(path, log, delimiter=",", header=",".join(COLS), comments="")


def make_plots(log, dyn, thr, wps, start, track_path, thr_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = log[:, 0]
    C = {c: i for i, c in enumerate(COLS)}

    # ----- Track figure -----
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    a = ax[0, 0]
    a.plot(log[:, C["y"]], log[:, C["x"]], "b-", lw=1.6, label="track")
    wpx = [w[0] for w in wps]; wpy = [w[1] for w in wps]
    a.plot(wpy, wpx, "k--", lw=0.8, alpha=0.6)
    a.plot(wpy, wpx, "r^", ms=10, label="waypoints")
    for i, w in enumerate(wps):
        circ = plt.Circle((w[1], w[0]), thr and 1.5, color="r", fill=False,
                          ls=":", alpha=0.5)
        a.add_patch(circ)
        a.annotate(f"WP{i+1}", (w[1], w[0]), textcoords="offset points",
                   xytext=(8, 8))
    a.plot(start[1], start[0], "go", ms=10, label="start")
    a.plot(log[-1, C["y"]], log[-1, C["x"]], "ms", ms=9, label="end")
    a.set_xlabel("y  East [m]"); a.set_ylabel("x  North [m]")
    a.set_title("Waypoint track (NED)"); a.axis("equal"); a.grid(True); a.legend()

    a = ax[0, 1]
    a.plot(t, log[:, C["xte_m"]], "purple")
    a.axhline(0, color="k", lw=0.6)
    a.set_xlabel("t [s]"); a.set_ylabel("cross-track error [m]")
    a.set_title("Cross-track error"); a.grid(True)

    a = ax[1, 0]
    a.plot(t, log[:, C["psi_deg"]], label="heading psi")
    a.plot(t, log[:, C["psi_d_deg"]], "--", label="desired psi_d")
    a.set_xlabel("t [s]"); a.set_ylabel("[deg]")
    a.set_title("Heading tracking"); a.grid(True); a.legend()

    a = ax[1, 1]
    a.plot(t, log[:, C["u"]], label="surge u")
    a.plot(t, log[:, C["u_d"]], "--", label="desired u_d")
    a.plot(t, log[:, C["v"]], label="sway v", alpha=0.7)
    a.set_xlabel("t [s]"); a.set_ylabel("[m/s]")
    a.set_title("Speed tracking"); a.grid(True); a.legend()

    fig.suptitle(f"Sookshma waypoint tracking  |  set={dyn.active_set}  "
                 f"Coriolis={'on' if dyn.include_coriolis else 'off'}")
    fig.tight_layout(); fig.savefig(track_path, dpi=130); plt.close(fig)

    # ----- Thruster figure -----
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fmax = thr.f_max_kgf

    a = ax[0, 0]
    a.plot(t, log[:, C["F_port_kgf"]], label="F_port")
    a.plot(t, log[:, C["F_stbd_kgf"]], label="F_stbd")
    a.axhline(fmax, color="r", ls=":", label=f"limit {fmax} kgf")
    a.axhline(0, color="k", lw=0.6)
    a.set_xlabel("t [s]"); a.set_ylabel("thrust [kgf]")
    a.set_title("Thruster forces (applied)"); a.grid(True); a.legend()

    a = ax[0, 1]
    # convert applied forces to PWM via the T200 16 V curve
    pwm_p = np.array([thr.pwm_from_force_kgf(f) for f in log[:, C["F_port_kgf"]]])
    pwm_s = np.array([thr.pwm_from_force_kgf(f) for f in log[:, C["F_stbd_kgf"]]])
    a.plot(t, pwm_p, label="PWM port")
    a.plot(t, pwm_s, label="PWM stbd")
    a.set_xlabel("t [s]"); a.set_ylabel("PWM [us]  (T200 @ 16 V)")
    a.set_title("Thruster PWM command"); a.grid(True); a.legend()

    a = ax[1, 0]
    a.plot(t, log[:, C["tau_u_cmd_N"]], "--", label="tau_u commanded")
    a.plot(t, log[:, C["tau_u_N"]], label="tau_u applied")
    a.set_xlabel("t [s]"); a.set_ylabel("surge force [N]")
    a.set_title("Surge force: command vs applied (saturation)"); a.grid(True); a.legend()

    a = ax[1, 1]
    a.plot(t, log[:, C["tau_r_cmd_Nm"]], "--", label="tau_r commanded")
    a.plot(t, log[:, C["tau_r_Nm"]], label="tau_r applied")
    a.set_xlabel("t [s]"); a.set_ylabel("yaw moment [N.m]")
    a.set_title("Yaw moment: command vs applied (saturation)"); a.grid(True); a.legend()

    fig.suptitle(f"Sookshma thruster activity  |  set={dyn.active_set}")
    fig.tight_layout(); fig.savefig(thr_path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "vessel_config.yaml"))
    ap.add_argument("--set", dest="cset", default=None)
    ap.add_argument("--no-coriolis", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    config_dir = os.path.dirname(os.path.abspath(args.config))
    if args.cset:
        cfg["model"]["active_coeff_set"] = args.cset
    if args.no_coriolis:
        cfg["model"]["include_coriolis"] = False

    log, dyn, thr, ctrl, wps = run(cfg, config_dir)
    start = np.array([cfg["mission"]["initial_state"]["x"],
                      cfg["mission"]["initial_state"]["y"]], float)

    out = os.path.join(ROOT, "outputs")
    os.makedirs(out, exist_ok=True)
    save_csv(log, os.path.join(out, "waypoint_track.csv"))
    make_plots(log, dyn, thr, wps, start,
               os.path.join(out, "waypoint_track.png"),
               os.path.join(out, "waypoint_thrusters.png"))

    C = {c: i for i, c in enumerate(COLS)}
    reached = int(log[-1, C["wp_idx"]]) + (1 if ctrl.done else 0)
    print("=" * 62)
    print(" Sookshma waypoint-tracking run complete")
    print(f"   coefficient set : {dyn.active_set}  (Coriolis "
          f"{'on' if dyn.include_coriolis else 'off'})")
    print(f"   waypoints       : {len(wps)}  |  reached: {reached}  "
          f"|  mission {'COMPLETE' if ctrl.done else 'timed out'}")
    print(f"   duration        : {log[-1,0]:.1f} s")
    print(f"   RMS cross-track : {np.sqrt(np.mean(log[:,C['xte_m']]**2)):.3f} m")
    print(f"   max cross-track : {np.max(np.abs(log[:,C['xte_m']])):.3f} m")
    print(f"   mean surge speed: {np.mean(log[:,C['u']]):.3f} m/s")
    sat = np.mean((log[:, C['F_port_kgf']] >= thr.f_max_kgf - 1e-3) |
                  (log[:, C['F_stbd_kgf']] >= thr.f_max_kgf - 1e-3)) * 100
    print(f"   thrust-saturated: {sat:.1f}% of the time")
    print(f"   outputs         : outputs/waypoint_track.png, "
          f"outputs/waypoint_thrusters.png, outputs/waypoint_track.csv")
    print("=" * 62)


if __name__ == "__main__":
    main()
