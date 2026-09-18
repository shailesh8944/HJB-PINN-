"""
animate_maneuvers.py — animate the turning-circle and zig-zag runs
(from maneuvers.py CSVs). Saves GIF + MP4 per manoeuvre.
"""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from animate_waypoints import boat_polygon
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")


def animate(csv, title, gif, stride=8, fps=25):
    d = np.genfromtxt(csv, delimiter=",", names=True)
    xN, yE, psi = d["x"], d["y"], d["psi"]
    idx = np.arange(0, len(xN), stride)
    padx = 0.15*(xN.max()-xN.min()+2); pady = 0.15*(yE.max()-yE.min()+2)
    fig, ax = plt.subplots(figsize=(7, 7)); ax.set_aspect("equal")
    ax.set_xlim(yE.min()-pady, yE.max()+pady); ax.set_ylim(xN.min()-padx, xN.max()+padx)
    ax.set_xlabel("y East [m]"); ax.set_ylabel("x North [m]"); ax.set_title(title); ax.grid(True, alpha=.3)
    ax.plot(yE, xN, "-", color="0.85", lw=1)
    trail, = ax.plot([], [], "b-", lw=2)
    boat = Polygon(boat_polygon(xN[0], yE[0], psi[0]), closed=True, fc="tab:blue", ec="navy")
    ax.add_patch(boat)
    hud = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", family="monospace",
                  fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    t = d["t"]

    def upd(k):
        j = idx[k]
        trail.set_data(yE[:j+1], xN[:j+1])
        boat.set_xy(boat_polygon(xN[j], yE[j], psi[j]))
        hud.set_text(f"t={t[j]:5.1f}s   heading={np.rad2deg(psi[j]):6.1f} deg")
        return trail, boat, hud

    an = FuncAnimation(fig, upd, frames=len(idx), interval=1000/fps, blit=True)
    an.save(os.path.join(OUT, gif+".gif"), writer=PillowWriter(fps=fps))
    try: an.save(os.path.join(OUT, gif+".mp4"), writer="ffmpeg", fps=fps, dpi=110)
    except Exception as e: print("mp4 skip:", e)
    plt.close(fig); print("saved", gif)


if __name__ == "__main__":
    animate(os.path.join(OUT,"manoeuvre_turning_circle.csv"),
            "Turning circle (dF=0.5 kgf) — coupled dynamics", "manoeuvre_turning_anim", stride=6)
    animate(os.path.join(OUT,"manoeuvre_zigzag.csv"),
            "Zig-zag 20/20 — overshoot grows as model leaves valid range", "manoeuvre_zigzag_anim", stride=8)
