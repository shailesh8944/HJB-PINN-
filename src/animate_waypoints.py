"""
animate_waypoints.py — animate the Sookshma waypoint-tracking run.

Reads outputs/waypoint_track.csv (produced by track_waypoints.py) and the mission
waypoints from the config, and renders the vessel (drawn as a small boat, oriented
by heading) tracing the mission, with a fading trail and a heads-up display.

Saves outputs/waypoint_anim.gif (and .mp4 if ffmpeg is present).
Usage: python src/animate_waypoints.py
"""

import os, sys, argparse, numpy as np, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)


def boat_polygon(x, y, psi, L=2.0, B=0.8):
    """Boat outline (pointed bow) at NED position (x=N, y=E) heading psi,
    returned as plot coords (East, North)."""
    body = np.array([[0.6*L, 0.0], [0.2*L, 0.5*B], [-0.5*L, 0.5*B],
                     [-0.5*L, -0.5*B], [0.2*L, -0.5*B]])          # (x_fwd, y_stbd)
    c, s = np.cos(psi), np.sin(psi)
    R = np.array([[c, -s], [s, c]])
    NE = (R @ body.T).T + np.array([x, y])                        # (North, East)
    return np.column_stack([NE[:, 1], NE[:, 0]])                  # (East, North) for plotting


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "outputs", "waypoint_track.csv"))
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "vessel_config.yaml"))
    ap.add_argument("--stride", type=int, default=10, help="use every Nth sample as a frame")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    data = np.genfromtxt(args.csv, delimiter=",", names=True)
    cfg = yaml.safe_load(open(args.config))
    m = cfg["mission"]
    wps = [np.array(w, float) for w in m["waypoints"]]          # [N, E]
    acc = float(m["acceptance_radius_m"])
    start = np.array([m["initial_state"]["x"], m["initial_state"]["y"]], float)

    t = data["t"]; xN = data["x"]; yE = data["y"]; psi = data["psi"]; u = data["u"]
    xte = data["xte_m"]; wpix = data["wp_idx"]
    idx = np.arange(0, len(t), args.stride)

    # figure limits from track + waypoints
    allN = np.r_[xN, [w[0] for w in wps], start[0]]
    allE = np.r_[yE, [w[1] for w in wps], start[1]]
    padN = 0.1*(allN.max()-allN.min()+1); padE = 0.1*(allE.max()-allE.min()+1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xlim(allE.min()-padE, allE.max()+padE)
    ax.set_ylim(allN.min()-padN, allN.max()+padN)
    ax.set_xlabel("y  East [m]"); ax.set_ylabel("x  North [m]")
    ax.set_title("Sookshma waypoint tracking (coupled 3-DOF dynamics)")
    ax.grid(True, alpha=0.3)

    # static waypoints + acceptance circles + full route
    wpN = [w[0] for w in wps]; wpE = [w[1] for w in wps]
    ax.plot(wpE, wpN, "k--", lw=0.7, alpha=0.5)
    ax.plot(wpE, wpN, "r^", ms=11, label="waypoints")
    for i, w in enumerate(wps):
        ax.add_patch(Circle((w[1], w[0]), acc, fill=False, ls=":", ec="r", alpha=0.5))
        ax.annotate(f"WP{i+1}", (w[1], w[0]), textcoords="offset points", xytext=(9, 9))
    ax.plot(start[1], start[0], "go", ms=10, label="start")
    ax.plot(yE, xN, "-", color="0.8", lw=1.0, zorder=1)         # faint full path
    trail, = ax.plot([], [], "-", color="tab:blue", lw=2.0, zorder=3)
    boat = Polygon(boat_polygon(start[0], start[1], 0.0), closed=True,
                   fc="tab:blue", ec="navy", zorder=5)
    ax.add_patch(boat)
    hud = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                  fontsize=10, family="monospace",
                  bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.legend(loc="lower right")

    def update(k):
        j = idx[k]
        trail.set_data(yE[:j+1], xN[:j+1])
        boat.set_xy(boat_polygon(xN[j], yE[j], psi[j]))
        hud.set_text(f"t = {t[j]:5.1f} s\nspeed = {u[j]:4.2f} m/s\n"
                     f"heading = {np.rad2deg(psi[j]):6.1f} deg\n"
                     f"cross-track = {xte[j]:+4.2f} m\nheading to WP{int(wpix[j])+1}")
        return trail, boat, hud

    anim = FuncAnimation(fig, update, frames=len(idx), interval=1000/args.fps, blit=True)

    out = os.path.join(ROOT, "outputs")
    gif = os.path.join(out, "waypoint_anim.gif")
    anim.save(gif, writer=PillowWriter(fps=args.fps))
    print("saved", gif)
    try:
        mp4 = os.path.join(out, "waypoint_anim.mp4")
        anim.save(mp4, writer="ffmpeg", fps=args.fps, dpi=110)
        print("saved", mp4)
    except Exception as ex:
        print("mp4 skipped:", ex)


if __name__ == "__main__":
    main()
