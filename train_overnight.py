"""
train_overnight.py — SELF-CONTAINED two-player Hamilton-Jacobi-Isaacs (HJI)
reachability trainer for the Sookshma head-on collision game.

One file, no local imports. All physics constants are baked in with the final
values agreed in design (diagonal mass matrix, theta_anneal damping, static
thruster ellipse). Just needs:  pip install torch numpy pyyaml matplotlib

Pursuer (minimiser) tries to force separation <= R; evader (maximiser) avoids.
Value V(zeta,t): {V<=0} = DANGER (collision unavoidable), {V>0} = SAFE.
State  zeta = [X, Y, psi_rel, u_e, v_e, u_i, v_i, r_e, r_i]  in R^9.

Typical overnight run on an RTX A4000 (16 GB):
    nohup python3 train_overnight.py --device cuda --iters 150000 \
        --hidden 512 --layers 3 --batch 65536 --lr 2e-5 --warmup 10000 \
        --save-every 5000 --out out_hji > train.log 2>&1 &
    tail -f train.log
Resume after an interruption: add --resume (loads out_hji/ckpt.pt).
If you hit a GPU out-of-memory error, drop --batch to 32768 or 16384.
"""

import os, time, math, argparse, numpy as np, torch, torch.nn as nn

# ============================================================================
#  BAKED-IN PHYSICS CONSTANTS  (final agreed values)
# ============================================================================
G          = 9.80665
MASS       = 20.0
# COUPLED effective mass matrix (body origin at centre of mass):
#   M = [[M11, 0,   0  ],
#        [ 0,  M22, MC ],
#        [ 0,  MC,  M33]]
# M11 = m - X_ud = 21.72 ; M22 = m - Y_vd = 29.214 ;
# M33 = Izz - N_rd = 2.44(CAD) + 0.4232 = 2.8632 ;
# MC  = -Y_rd = -N_vd ~= 0.76  (sway-yaw added-mass coupling)
M11, M22, M33 = 21.7200, 29.2140, 2.8632
MC         = 0.76
DELTA      = M22 * M33 - MC * MC                 # determinant of the sway-yaw 2x2 block
ARM        = 0.21                                # thruster lateral arm [m] (CAD: y = +/-0.21)
FMAX       = 1.82 * G                            # 17.848 N per thruster (static)
ALPHA_U    = FMAX / math.sqrt(2.0)              # inscribed-ellipse semi-axes
ALPHA_R    = ARM * FMAX / math.sqrt(2.0)
SIG0, SIG1 = ALPHA_U ** 2, ALPHA_R ** 2
EPS_H      = 1.0e-6                              # rounds the Hamiltonian norm corner
R_COLLIDE  = 1.0                                 # collision radius [m]
T_HORIZON  = 20.0                                # reachability horizon [s]
DAMP_SIGN  = -1.0                                # D(v)v subtracted

# theta_anneal damping (dimensional), identical for both vessels
TA = dict(
    Dx_u=28.5721, Dx_u2=-15.982, Dx_v2=73.3826, Dx_r2=20.4587, Dx_vr=16.4984,
    Dx_u3=55.7021, Dx_uv2=-36.2765, Dx_ur2=-44.7408, Dx_uvr=-82.1422, Dx_absu_u=-44.2545,
    Dy_v=32.837, Dy_r=-6.80627, Dy_uv=-28.4202, Dy_ur=-16.6802, Dy_v3=66.2334,
    Dy_r3=8.26366, Dy_u2v=2.75094, Dy_u2r=-1.0423, Dy_v2r=148.983, Dy_vr2=-4.43673,
    Dy_absv_v=-8.63508, Dy_u_absv=5.81447,
    Dn_v=6.74496, Dn_r=0.974895, Dn_uv=-11.1648, Dn_ur=-9.47937, Dn_v3=43.9947,
    Dn_r3=16.8805, Dn_u2v=13.2957, Dn_u2r=9.21704, Dn_v2r=62.4145, Dn_vr2=-8.60609,
    Dn_absr_r=-6.27807, Dn_u_absr=-0.209728,
)

# state indices
IX, IY, IPSI, IUE, IVE, IUI, IVI, IRE, IRI = range(9)


# ============================================================================
#  PHYSICS
# ============================================================================
def damping(u, v, r, c):
    au, av, ar = u.abs(), v.abs(), r.abs()
    X = (c["Dx_u"]*u + c["Dx_u2"]*u*u + c["Dx_v2"]*v*v + c["Dx_r2"]*r*r + c["Dx_vr"]*v*r
         + c["Dx_u3"]*u**3 + c["Dx_uv2"]*u*v*v + c["Dx_ur2"]*u*r*r + c["Dx_uvr"]*u*v*r
         + c["Dx_absu_u"]*au*u)
    Y = (c["Dy_v"]*v + c["Dy_r"]*r + c["Dy_uv"]*u*v + c["Dy_ur"]*u*r + c["Dy_v3"]*v**3
         + c["Dy_r3"]*r**3 + c["Dy_u2v"]*u*u*v + c["Dy_u2r"]*u*u*r + c["Dy_v2r"]*v*v*r
         + c["Dy_vr2"]*v*r*r + c["Dy_absv_v"]*av*v + c["Dy_u_absv"]*u*av)
    N = (c["Dn_v"]*v + c["Dn_r"]*r + c["Dn_uv"]*u*v + c["Dn_ur"]*u*r + c["Dn_v3"]*v**3
         + c["Dn_r3"]*r**3 + c["Dn_u2v"]*u*u*v + c["Dn_u2r"]*u*u*r + c["Dn_v2r"]*v*v*r
         + c["Dn_vr2"]*v*r*r + c["Dn_absr_r"]*ar*r + c["Dn_u_absr"]*u*ar)
    return X, Y, N


def nu_dot0(u, v, r):
    """Control-independent body acceleration (tau=0), coupled sway-yaw.
    Solves M v_dot = -D(v)v (no separate Coriolis) with the coupled M."""
    X, Y, N = damping(u, v, r, TA)
    # NO separate Coriolis: theta_anneal (CoG frame) is the total velocity-dependent
    # reaction and already contains the vr/ur/uv coupling.  M v_dot = tau - D(v).
    b_u = DAMP_SIGN * X
    b_v = DAMP_SIGN * Y
    b_r = DAMP_SIGN * N
    du = b_u / M11
    dv = (M33 * b_v - MC * b_r) / DELTA           # M^{-1} b  (sway-yaw block)
    dr = (-MC * b_v + M22 * b_r) / DELTA
    return du, dv, dr


def drift0(z):
    """Pure drift a0(zeta): relative kinematics + both vessels' control-independent
    velocity dynamics (no thrust). Shape (B,9)."""
    X, Y, psi = z[:, IX], z[:, IY], z[:, IPSI]
    ue, ve, re = z[:, IUE], z[:, IVE], z[:, IRE]
    ui, vi, ri = z[:, IUI], z[:, IVI], z[:, IRI]
    cp, sp = torch.cos(psi), torch.sin(psi)
    due, dve, dre = nu_dot0(ue, ve, re)
    dui, dvi, dri = nu_dot0(ui, vi, ri)
    a = torch.empty_like(z)
    a[:, IX]  = ue*cp - ve*sp - ui + ri*Y
    a[:, IY]  = ue*sp + ve*cp - vi - ri*X
    a[:, IPSI] = re - ri
    a[:, IUE] = due; a[:, IVE] = dve; a[:, IRE] = dre
    a[:, IUI] = dui; a[:, IVI] = dvi; a[:, IRI] = dri
    return a


def terminal(z):
    """ell(zeta) = sqrt(X^2+Y^2) - R  (signed distance to collision disc)."""
    return torch.sqrt(z[:, IX]**2 + z[:, IY]**2 + 1e-12) - R_COLLIDE


def _R(c0, c1):
    return torch.sqrt(SIG0 * c0**2 + SIG1 * c1**2 + EPS_H)


def _cdirs(p):
    """Control costate directions with the COUPLED input matrix G:
    tau_u acts on surge (1/M11); tau_r acts on yaw (M22/DELTA) AND sway (-MC/DELTA)."""
    ci0 = p[:, IUI] / M11
    ci1 = (M22 * p[:, IRI] - MC * p[:, IVI]) / DELTA
    ce0 = p[:, IUE] / M11
    ce1 = (M22 * p[:, IRE] - MC * p[:, IVE]) / DELTA
    return ci0, ci1, ce0, ce1


def game_hamiltonian(z, p):
    """Isaacs H = <p,a0> + (Fmax*ci0 - Ri) + (Fmax*ce0 + Re)."""
    a = drift0(z).detach()
    drift_term = (p * a).sum(dim=1)
    ci0, ci1, ce0, ce1 = _cdirs(p)
    return drift_term + (FMAX*ci0 - _R(ci0, ci1)) + (FMAX*ce0 + _R(ce0, ce1))


def game_vi_residual(z, V, dVdt, gradV):
    return torch.minimum(dVdt + game_hamiltonian(z, gradV), terminal(z) - V)


def optimal_controls(p):
    ci0, ci1, ce0, ce1 = _cdirs(p)
    Ri, Re = _R(ci0, ci1), _R(ce0, ce1)
    ti = torch.stack([FMAX - SIG0*ci0/Ri, -SIG1*ci1/Ri], 1)
    te = torch.stack([FMAX + SIG0*ce0/Re,  SIG1*ce1/Re], 1)
    return ti, te


# ============================================================================
#  VALUE NETWORK (SIREN + exact boundary condition)
# ============================================================================
class Sine(nn.Module):
    def __init__(self, w0=1.0): super().__init__(); self.w0 = w0
    def forward(self, x): return torch.sin(self.w0 * x)


class SIREN(nn.Module):
    def __init__(self, lo, hi, hidden=256, layers=3, w0=30.0, dtype=torch.float32):
        super().__init__()
        self.register_buffer("lo", torch.as_tensor(lo, dtype=dtype))
        self.register_buffer("hi", torch.as_tensor(hi, dtype=dtype))
        din = len(lo) + 1                       # psi -> (cos,sin) adds one
        net = [nn.Linear(din, hidden), Sine(w0)]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), Sine(w0)]
        net += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*net).to(dtype)
        with torch.no_grad():
            lins = [m for m in self.net if isinstance(m, nn.Linear)]
            lins[0].weight.uniform_(-1.0/din, 1.0/din)
            for lin in lins[1:]:
                b = math.sqrt(6.0/lin.weight.shape[1]) / w0
                lin.weight.uniform_(-b, b)

    def _feat(self, x):
        xs = 2.0*(x - self.lo)/(self.hi - self.lo) - 1.0
        cols = []
        for i in range(x.shape[1]):
            if i == IPSI:
                cols += [torch.cos(x[:, i:i+1]), torch.sin(x[:, i:i+1])]
            else:
                cols.append(xs[:, i:i+1])
        return torch.cat(cols, 1)

    def forward(self, x): return self.net(self._feat(x))


class ExactBCValue(nn.Module):
    """V(zeta,t) = ell(zeta) + t*softplus(N):  exact terminal V(.,0)=ell and
    tube property V<=ell by construction."""
    def __init__(self, siren): super().__init__(); self.siren = siren
    def forward(self, inp):
        z, t = inp[:, :9], inp[:, 9]
        N = self.siren(inp).squeeze(1)
        return (terminal(z) + t * torch.nn.functional.softplus(N)).unsqueeze(1)


def value_and_grads(net, z, t):
    inp = torch.cat([z, t.reshape(-1, 1)], 1).requires_grad_(True)
    V = net(inp).squeeze(1)
    g = torch.autograd.grad(V.sum(), inp, create_graph=True)[0]
    return V, g[:, 9], g[:, :9]


# ============================================================================
#  ISAACS MATH CHECKS
# ============================================================================
def isaacs_checks(device, n=200, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.uniform([-14,-14,-np.pi,-0.2,-0.6,-0.2,-0.6,-1.5,-1.5],
                    [14,14,np.pi,1.2,0.6,1.2,0.6,1.5,1.5], size=(n,9))
    Z[np.hypot(Z[:,0], Z[:,1]) < 1.0, 0] += 3.0
    P = rng.normal(size=(n,9))
    Zt = torch.tensor(Z, dtype=torch.float64, device=device)
    Pt = torch.tensor(P, dtype=torch.float64, device=device)
    global M11, M22, M33  # (unchanged; float math is dtype-agnostic here)
    Hcf = game_hamiltonian(Zt, Pt).cpu().numpy()
    a0 = drift0(Zt).cpu().numpy()
    th = np.linspace(0, 2*np.pi, 3000)
    cu = FMAX + ALPHA_U*np.cos(th); cr = ALPHA_R*np.sin(th)
    e_ham = e_is = 0.0
    for k in range(n):
        base = float(P[k] @ a0[k])
        ci0 = P[k,IUI]/M11; ci1 = (M22*P[k,IRI] - MC*P[k,IVI])/DELTA
        ce0 = P[k,IUE]/M11; ce1 = (M22*P[k,IRE] - MC*P[k,IVE])/DELTA
        mn = (ci0*cu + ci1*cr).min(); mx = (ce0*cu + ce1*cr).max()
        e_ham = max(e_ham, abs(Hcf[k] - (base+mn+mx)))
        e_is = max(e_is, abs((base+mn+mx) - (base+mx+mn)))
    ti, te = optimal_controls(Pt)
    def outside(tau):
        Fs = tau[:,0]/2 + tau[:,1]/(2*ARM); Fp = tau[:,0]/2 - tau[:,1]/(2*ARM)
        return float(torch.maximum(torch.clamp(-torch.minimum(Fs,Fp),min=0),
                                   torch.clamp(torch.maximum(Fs,Fp)-FMAX,min=0)).max())
    worst = max(outside(ti), outside(te))
    return e_ham, e_is, worst


# ============================================================================
#  DIAGNOSTIC:  head-on danger / safe map
# ============================================================================
def save_map(net, path, device, approach=0.8, ng=160, hist=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = -15.0, 15.0
    xs = np.linspace(lo, hi, ng); ys = np.linspace(lo, hi, ng)
    XX, YY = np.meshgrid(xs, ys)
    Z = np.zeros((ng*ng, 9), np.float32)
    Z[:, IX] = XX.ravel(); Z[:, IY] = YY.ravel()
    Z[:, IPSI] = np.pi                     # head-on
    Z[:, IUE] = approach; Z[:, IUI] = approach
    tt = np.full((ng*ng, 1), -T_HORIZON, np.float32)
    inp = torch.tensor(np.hstack([Z, tt]), device=device)
    with torch.no_grad():
        V = net(inp).float().cpu().numpy().reshape(ng, ng)
    fig, ax = plt.subplots(1, 2 if hist else 1, figsize=(13, 5.3) if hist else (6.5, 5.3))
    a0 = ax[0] if hist else ax
    a0.contourf(YY, XX, (V <= 0).astype(float), levels=[-.5,.5,1.5], colors=["#c7e9c0","#fb6a4a"])
    a0.contour(YY, XX, V, levels=[0.0], colors="k", linewidths=2)
    th = np.linspace(0, 2*np.pi, 100)
    a0.plot(R_COLLIDE*np.sin(th), R_COLLIDE*np.cos(th), "b--", lw=1.2, label=f"collision R={R_COLLIDE:.0f} m")
    a0.plot(0, 0, "ks", ms=7, label="pursuer"); a0.axis("equal")
    a0.set_xlabel("Y rel [m]"); a0.set_ylabel("X rel [m]")
    a0.set_title("Head-on: danger (red) / safe (green), V=0 barrier"); a0.legend(loc="upper right")
    if hist:
        its = [h[0] for h in hist]; ls = [max(h[1],1e-12) for h in hist]
        ax[1].semilogy(its, ls); ax[1].set_xlabel("iter"); ax[1].set_ylabel("HJI residual (log)")
        ax[1].grid(True, which="both", alpha=0.3); ax[1].set_title("Training loss")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ============================================================================
#  TRAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters", type=int, default=150000)
    ap.add_argument("--warmup", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--curric-frac", type=float, default=0.5)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--out", default="out_hji")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("cuda not available -> cpu"); device = "cpu"
    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "ckpt.pt")
    dtype = torch.float32
    torch.manual_seed(args.seed)

    print("="*70)
    print(f" device={device}  M=[[{M11},0,0],[0,{M22},{MC}],[0,{MC},{M33}]]  "
          f"arm={ARM}  Fmax={FMAX:.3f}N  R={R_COLLIDE}  T={T_HORIZON}")
    eh, ei, w = isaacs_checks(device)
    print(f" Isaacs: |H-minmax|={eh:.2e}[{'PASS' if eh<1e-3 else 'FAIL'}]  "
          f"gap={ei:.2e}[{'PASS' if ei<1e-9 else 'FAIL'}]  "
          f"saddle_outside={w:.2e}[{'PASS' if w<1e-6 else 'FAIL'}]")
    print("="*70, flush=True)

    T = T_HORIZON
    lo = [-15,-15,-np.pi,-0.2,-0.6,-0.2,-0.6,-1.5,-1.5,-T]
    hi = [ 15, 15, np.pi, 1.2, 0.6, 1.2, 0.6, 1.5, 1.5, 0.0]
    net = ExactBCValue(SIREN(lo, hi, args.hidden, args.layers, dtype=dtype)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    lo9 = torch.tensor(lo[:9], dtype=dtype, device=device)
    hi9 = torch.tensor(hi[:9], dtype=dtype, device=device)

    start, hist = 0, []
    if args.resume and os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location=device)
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
        start = int(ck["iter"]) + 1; hist = ck.get("hist", [])
        print(f" resumed at iter {start}", flush=True)

    t0 = time.time()
    for it in range(start, args.iters):
        opt.zero_grad()
        frac = min(1.0, max(0.0, it - args.warmup) / max(1.0, args.curric_frac*args.iters))
        t_cur = 1e-3 + T*frac
        z = lo9 + (hi9 - lo9) * torch.rand(args.batch, 9, dtype=dtype, device=device)
        tc = -t_cur * torch.rand(args.batch, dtype=dtype, device=device)
        V, dVdt, gradV = value_and_grads(net, z, tc)
        loss = (game_vi_residual(z, V, dVdt, gradV) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if it % 100 == 0 or it == args.iters - 1:
            hist.append((it, float(loss.detach())))
        if args.save_every and (it % args.save_every == 0 or it == args.iters - 1) and it > start:
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                        "iter": it, "hist": hist}, ckpt)
            save_map(net, os.path.join(args.out, "danger_safe.png"), device, hist=hist)
            print(f" [ckpt] iter {it:>7}  loss {float(loss.detach()):.4e}  "
                  f"horizon {t_cur:5.1f}s  elapsed {(time.time()-t0)/60:.1f} min", flush=True)

    torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                "iter": args.iters-1, "hist": hist}, ckpt)
    save_map(net, os.path.join(args.out, "danger_safe.png"), device, hist=hist)
    print(f" DONE in {(time.time()-t0)/60:.1f} min. Results in {args.out}/", flush=True)


if __name__ == "__main__":
    main()
