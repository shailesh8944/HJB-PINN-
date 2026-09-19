"""
maneuvers.py — standard manoeuvring tests for the Sookshma coupled 3-DOF model,
to verify the dynamics: turning circle, Kempf zig-zag, Dieudonne spiral, plus the
thruster actuator mapping and the forces during a turn.

Control analogue: with twin forward-only thrusters the "rudder" is the thrust
DIFFERENTIAL dF (kgf). With a base throttle T0 on both:
    F_port = clip(T0 - dF/2, 0, Fmax) ,  F_stbd = clip(T0 + dF/2, 0, Fmax)
so dF > 0 turns the bow to starboard (positive yaw).

MODEL: M v_dot = tau - C(v)v - D(v)v, with D = theta_anneal representing
the identified damping vector only. Coriolis is enabled from vessel_config.yaml.

Outputs: outputs/manoeuvre_*.png and CSVs.
"""

import os, sys, numpy as np, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPoly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dynamics import VesselDynamics
from thruster import Thruster

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs"); os.makedirs(OUT, exist_ok=True)

cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "vessel_config.yaml")))
dyn = VesselDynamics(cfg)
thr = Thruster(cfg, os.path.join(ROOT, "config"))
FMAXk = thr.f_max_kgf; ARM = thr.arm; G = thr.g
FMAXn = thr.kgf_to_N(FMAXk)


def rk4(s, tau, dt):
    k1 = dyn.state_dot(s, tau); k2 = dyn.state_dot(s + .5*dt*k1, tau)
    k3 = dyn.state_dot(s + .5*dt*k2, tau); k4 = dyn.state_dot(s + dt*k3, tau)
    return s + dt/6*(k1 + 2*k2 + 2*k3 + k4)


def cmd(T0, dF):
    return np.clip(T0 - dF/2, 0, FMAXk), np.clip(T0 + dF/2, 0, FMAXk)


# ---------------------------------------------------------------------------
# 1. TURNING CIRCLE
# ---------------------------------------------------------------------------
def turning_circle(T0=0.9, dF=1.0, straight=5.0, T=35.0, dt=0.02):
    Fp, Fs = cmd(T0, dF)
    s = np.zeros(6); log = []
    n = int(T/dt)
    for k in range(n):
        t = k*dt
        tau, (fp, fs) = thr.allocate(*( (T0, T0) if t < straight else (Fp, Fs) ))
        log.append([t, s[0], s[1], s[2], s[3], s[4], s[5], fp, fs])
        s = rk4(s, tau, dt)
    return np.array(log)


# ---------------------------------------------------------------------------
# 2. KEMPF ZIG-ZAG (psi0 deg heading, dF differential)
# ---------------------------------------------------------------------------
def zigzag(T0=0.9, dF=0.8, psi0_deg=20.0, T=70.0, dt=0.02):
    psi0 = np.deg2rad(psi0_deg)
    s = np.zeros(6); log = []; sign = +1.0
    for k in range(int(T/dt)):
        t = k*dt; psi = s[2]
        if sign > 0 and psi >= psi0:  sign = -1.0
        elif sign < 0 and psi <= -psi0: sign = +1.0
        Fp, Fs = cmd(T0, sign*dF)
        tau, (fp, fs) = thr.allocate(Fp, Fs)
        log.append([t, s[0], s[1], s[2], s[5], sign*dF, fp, fs])
        s = rk4(s, tau, dt)
    return np.array(log)


# ---------------------------------------------------------------------------
# 3. DIEUDONNE SPIRAL (steady yaw rate vs differential; sweep down then up)
# ---------------------------------------------------------------------------
def spiral(T0=0.9, dFmax=1.4, step=0.2, hold=22.0, dt=0.02):
    seq = list(np.arange(dFmax, -dFmax-1e-9, -step)) + list(np.arange(-dFmax, dFmax+1e-9, step))
    branch = ["down"]*len(np.arange(dFmax, -dFmax-1e-9, -step)) + \
             ["up"]*len(np.arange(-dFmax, dFmax+1e-9, step))
    s = np.zeros(6)
    # settle straight first
    tau0, _ = thr.allocate(T0, T0)
    for _ in range(int(6/dt)): s = rk4(s, tau0, dt)
    rows = []
    for dF, br in zip(seq, branch):
        Fp, Fs = cmd(T0, dF)
        tau, _ = thr.allocate(Fp, Fs)
        for _ in range(int(hold/dt)): s = rk4(s, tau, dt)
        rows.append([dF, s[3], s[4], np.rad2deg(s[5]), br])
    return rows


def main():
    # ---- run (differentials kept inside the valid envelope dF <= ~0.7) ----
    tc = turning_circle(dF=1.0)
    zz = zigzag(dF=0.6)
    sp = spiral(dFmax=1.5, step=0.2, hold=20.0)

    # ============ FIG 1: turning circle ============
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].plot(tc[:,2], tc[:,1], "b-"); ax[0].plot(tc[0,2], tc[0,1], "go")
    ax[0].axis("equal"); ax[0].grid(True, alpha=.3)
    ax[0].set_xlabel("y East [m]"); ax[0].set_ylabel("x North [m]")
    # steady radius from last third
    u,v,r = tc[-1,4], tc[-1,5], tc[-1,6]; U=np.hypot(u,v); R=U/abs(r) if abs(r)>1e-6 else np.inf
    ax[0].set_title(f"Turning circle (dF=1.0 kgf)\nsteady R={R:.2f} m, r={np.rad2deg(r):.1f} deg/s")
    ax[1].plot(tc[:,0], np.rad2deg(tc[:,6]), label="yaw rate r [deg/s]")
    ax[1].plot(tc[:,0], tc[:,4], label="surge u [m/s]")
    ax[1].plot(tc[:,0], tc[:,5], label="sway v [m/s]")
    ax[1].set_xlabel("t [s]"); ax[1].grid(True, alpha=.3); ax[1].legend(); ax[1].set_title("States")
    drift = np.rad2deg(np.arctan2(-tc[:,5], np.maximum(tc[:,4],1e-3)))
    ax[2].plot(tc[:,0], drift, "m"); ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("drift angle [deg]")
    ax[2].grid(True, alpha=.3); ax[2].set_title("Drift angle (should stay small)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"manoeuvre_turning_circle.png"), dpi=130); plt.close(fig)

    # ============ FIG 2: zig-zag ============
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(zz[:,0], np.rad2deg(zz[:,3]), "b", lw=1.6, label="heading psi [deg]")
    ax[0].plot(zz[:,0], np.rad2deg(zz[:,4]), "g", alpha=.6, label="yaw rate [deg/s]")
    # command scaled to +/-20 to overlay on heading
    ax[0].plot(zz[:,0], np.sign(zz[:,5])*20.0, "r--", lw=0.9, label="turn command (+/-20 band)")
    ax[0].axhline(20, color="0.7", lw=.6); ax[0].axhline(-20, color="0.7", lw=.6)
    ax[0].set_xlabel("t [s]"); ax[0].grid(True, alpha=.3); ax[0].legend()
    ax[0].set_title("Zig-zag 20/20 (twin-thruster differential)")
    ax[1].plot(zz[:,0], zz[:,6], "C0", label="F_port [kgf]")
    ax[1].plot(zz[:,0], zz[:,7], "C1", label="F_stbd [kgf]")
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("thrust [kgf]"); ax[1].grid(True, alpha=.3)
    ax[1].legend(); ax[1].set_title("Zig-zag thruster forces")
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"manoeuvre_zigzag.png"), dpi=130); plt.close(fig)

    # ============ FIG 3: spiral ============
    fig, ax = plt.subplots(figsize=(7.5,6))
    down=[(d,r) for d,uu,vv,r,b in sp if b=="down"]; up=[(d,r) for d,uu,vv,r,b in sp if b=="up"]
    ax.plot([d for d,r in up],[r for d,r in up], "s-", color="tab:green", ms=4, label="sweep up")
    ax.plot([d for d,r in down],[r for d,r in down], "o--", color="tab:blue", ms=4, label="sweep down")
    ax.axhline(0,color="k",lw=.6); ax.axvline(0,color="k",lw=.6)
    ax.set_xlabel("thrust differential dF [kgf]  (rudder analogue)")
    ax.set_ylabel("steady yaw rate r [deg/s]")
    ax.set_title("Dieudonne spiral (full M v_dot = tau - C(v)v - D(v)v)")
    ax.grid(True, alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"manoeuvre_spiral.png"), dpi=130); plt.close(fig)

    # ============ FIG 4: thruster mapping ============
    fig, ax = plt.subplots(1, 2, figsize=(14,5.5))
    # actuator set: diamond (physical) + inscribed ellipse (used in HJI)
    d=ARM
    verts=np.array([[0,0],[2*FMAXn,0],[FMAXn, d*FMAXn],[0,0]])  # not used directly
    Fp=np.array([0,FMAXn,FMAXn,0,0]); Fs=np.array([0,0,FMAXn,FMAXn,0])
    tu=Fp+Fs; tr=d*(Fs-Fp)
    ax[0].plot(tu,tr,"k-",label="physical diamond")
    th=np.linspace(0,2*np.pi,200)
    au=FMAXn/np.sqrt(2); ar=d*FMAXn/np.sqrt(2)
    ax[0].plot(FMAXn+au*np.cos(th), ar*np.sin(th), "b--", label="inscribed ellipse (HJI)")
    ax[0].set_xlabel("tau_u  surge force [N]"); ax[0].set_ylabel("tau_r  yaw moment [N.m]")
    ax[0].set_title("Thruster actuator map"); ax[0].grid(True,alpha=.3); ax[0].legend(); ax[0].axis("equal")
    # force <-> PWM curve from the T200 16V table
    pwm=np.linspace(1500,1700,50); f=[thr.force_N_from_pwm(p) for p in pwm]
    ax[1].plot(pwm, f, "b"); ax[1].axhline(FMAXn, color="r", ls=":", label=f"Fmax={FMAXn:.1f} N")
    ax[1].set_xlabel("PWM [us] (T200 @16V)"); ax[1].set_ylabel("thrust [N]")
    ax[1].set_title("Thruster force vs PWM (operating band)"); ax[1].grid(True,alpha=.3); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"manoeuvre_thruster_map.png"), dpi=130); plt.close(fig)

    # save CSVs
    np.savetxt(os.path.join(OUT,"manoeuvre_turning_circle.csv"), tc, delimiter=",",
               header="t,x,y,psi,u,v,r,Fport,Fstbd", comments="")
    np.savetxt(os.path.join(OUT,"manoeuvre_zigzag.csv"), zz, delimiter=",",
               header="t,x,y,psi,r,dF,Fport,Fstbd", comments="")

    # ---- console summary (honest numbers) ----
    print("TURNING CIRCLE (dF=1.0): steady R=%.2f m, r=%.1f deg/s, u=%.3f, drift=%.1f deg"%(
        R, np.rad2deg(r), u, drift[-1]))
    over=[]
    psi=np.rad2deg(zz[:,3]); sgn=zz[:,5]
    for k in range(1,len(sgn)):
        if sgn[k]!=sgn[k-1]:
            seg=psi[k:k+int(8/0.02)]
            over.append(abs(seg).max()-20 if len(seg) else np.nan)
    print("ZIG-ZAG 20/20: overshoot angles (deg):", [f"{o:.1f}" for o in over[:6]])
    print("SPIRAL down vs up max |r| gap:",
          max(abs(dr-ur) for (_,dr),(_,ur) in zip(sorted(down),sorted(up))))
    return tc, zz


if __name__ == "__main__":
    main()
