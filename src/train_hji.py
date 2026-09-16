"""
train_hji.py — Physics-informed neural-network solver for the TWO-PLAYER
Hamilton-Jacobi-Isaacs (HJI) reachability game.

Pursuer (minimiser) tries to force the relative distance <= R (collision);
evader (maximiser) tries to keep it above R. The value function's zero level
set separates DANGER {V <= 0} (collision unavoidable) from SAFE {V > 0}.

Differences from the single-player train_pinn.py:
  * physics = GameHJITorch (Isaacs Hamiltonian, no APF), residual = game_vi_residual
  * value = ExactBCValue: V(zeta,t) = ell(zeta) + t*softplus(N(zeta,t)), which makes
    the terminal condition V(zeta,0)=ell exact and the tube property V<=ell hold by
    construction, so training is purely on the HJI residual (no soft terminal loss)
  * STEP 1 runs the Isaacs math checks (closed-form H vs brute-force min-max,
    min-max == max-min, saddle controls realisable) instead of an APF comparison
  * diagnostics draw the HEAD-ON danger/safe map (psi_rel = pi, both vessels closing)

On CPU this is a correctness/behaviour run; a converged 9-D tube needs a GPU and
long training (DeepReach reports 16-25 h). Use --iters/--hidden to scale.
"""

import os, sys, time, argparse, numpy as np, torch, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hji_torch import GameHJITorch
from hjb_torch import IUE, IUI, IRE, IRI
from pinn import SIREN, ExactBCValue, value_and_grads

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)


# ---------------------------------------------------------------------------
# STEP 1 — Isaacs math checks (before any training)
# ---------------------------------------------------------------------------
def isaacs_checks(phys, n=200, seed=0):
    M11, M33, Fmax = phys.M11, phys.M33, phys.Fmax
    au, ar = phys.alpha_u, phys.alpha_r
    rng = np.random.default_rng(seed)
    Z = rng.uniform([-14, -14, -np.pi, -0.2, -0.6, -0.2, -0.6, -1.5, -1.5],
                    [14, 14, np.pi, 1.2, 0.6, 1.2, 0.6, 1.5, 1.5], size=(n, 9))
    Z[np.hypot(Z[:, 0], Z[:, 1]) < 1.0, 0] += 3.0
    P = rng.normal(size=(n, 9))
    Zt, Pt = torch.tensor(Z), torch.tensor(P)

    Hcf = phys.game_hamiltonian(Zt, Pt).numpy()
    a0 = phys.drift0(Zt).numpy()
    th = np.linspace(0, 2 * np.pi, 3000)
    cu = Fmax + au * np.cos(th); cr = ar * np.sin(th)     # a point on the ellipse
    e_ham = e_isaacs = 0.0
    for k in range(n):
        base = float(P[k] @ a0[k])
        ci0, ci1 = P[k, IUI] / M11, P[k, IRI] / M33
        ce0, ce1 = P[k, IUE] / M11, P[k, IRE] / M33
        inner_i = ci0 * cu + ci1 * cr
        inner_e = ce0 * cu + ce1 * cr
        minmax = base + inner_i.min() + inner_e.max()
        maxmin = base + inner_e.max() + inner_i.min()
        e_ham = max(e_ham, abs(Hcf[k] - minmax))
        e_isaacs = max(e_isaacs, abs(minmax - maxmin))

    ti, te = phys.optimal_controls(Pt)
    d = phys.arm
    def outside(tau):
        Fs = tau[:, 0] / 2 + tau[:, 1] / (2 * d)
        Fp = tau[:, 0] / 2 - tau[:, 1] / (2 * d)
        return float(torch.maximum(torch.clamp(-torch.minimum(Fs, Fp), min=0),
                                   torch.clamp(torch.maximum(Fs, Fp) - Fmax, min=0)).max())
    worst = max(outside(ti), outside(te))
    return e_ham, e_isaacs, worst


# ---------------------------------------------------------------------------
# STEP 2 — autograd gradient check on the value network
# ---------------------------------------------------------------------------
def grad_check(net, seed=0):
    torch.manual_seed(seed)
    zeta = torch.zeros(1, 9, dtype=torch.float64); zeta[0, 0] = 6.0; zeta[0, 1] = 2.0
    t = torch.tensor([-3.0], dtype=torch.float64)
    V, dVdt, gradV = value_and_grads(net, zeta, t)
    eps = 1e-5
    def val(z, tt):
        with torch.no_grad():
            return net(torch.cat([z, tt.reshape(-1, 1)], 1)).item()
    zp = zeta.clone(); zp[0, 0] += eps; zm = zeta.clone(); zm[0, 0] -= eps
    dX_fd = (val(zp, t) - val(zm, t)) / (2 * eps)
    dt_fd = (val(zeta, t + eps) - val(zeta, t - eps)) / (2 * eps)
    return abs(gradV[0, 0].item() - dX_fd), abs(dVdt[0].item() - dt_fd)


# ---------------------------------------------------------------------------
# STEP 3 — training on the HJI residual (exact BC, backward-time curriculum)
# ---------------------------------------------------------------------------
def make_value(phys, lo, hi, hidden, layers, dtype):
    siren = SIREN(lo, hi, hidden=hidden, layers=layers, dtype=dtype)
    return ExactBCValue(siren, phys.terminal)


def train(phys, lo, hi, iters=1500, warmup=200, batch=2048, hidden=128, layers=3,
          lr=2e-4, curric_frac=0.6, seed=0, dtype=torch.float32, device="cpu",
          ckpt_path=None, save_every=0, resume=False, out_png=None):
    torch.manual_seed(seed)
    net = make_value(phys, lo, hi, hidden, layers, dtype).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lo9 = torch.tensor(lo[:9], dtype=dtype, device=device)
    hi9 = torch.tensor(hi[:9], dtype=dtype, device=device)
    T = phys.T
    hist = []
    start_iter = 0

    # resume from a checkpoint if asked and present
    if resume and ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
        start_iter = int(ck["iter"]) + 1; hist = ck.get("hist", [])
        print(f"   resumed from {ckpt_path} at iter {start_iter}")

    def sample_zeta(nb):
        return lo9 + (hi9 - lo9) * torch.rand(nb, 9, dtype=dtype, device=device)

    def save_ckpt(it):
        if ckpt_path:
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                        "iter": it, "hist": hist}, ckpt_path)
        if out_png:
            try:
                diagnostics(net, phys, lo, hi, hist, out_png, device=device)
            except Exception as ex:
                print("   (snapshot skipped:", ex, ")")

    t0 = time.time()
    for it in range(start_iter, iters):
        opt.zero_grad()
        # backward-time curriculum: grow the horizon from 0 to T
        frac = min(1.0, max(0.0, (it - warmup)) / max(1.0, curric_frac * iters))
        t_cur = 1e-3 + T * frac
        zc = sample_zeta(batch)
        tc = -t_cur * torch.rand(batch, dtype=dtype, device=device)
        V, dVdt, gradV = value_and_grads(net, zc, tc)
        res = phys.game_vi_residual(zc, V, dVdt, gradV)
        loss = (res ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if it % 50 == 0 or it == iters - 1:
            hist.append((it, float(loss.detach()), t_cur))
        if save_every and (it % save_every == 0 or it == iters - 1) and it > start_iter:
            save_ckpt(it)
            print(f"   [ckpt] iter {it}  loss {float(loss.detach()):.4e}  horizon {t_cur:.1f}s "
                  f"  elapsed {time.time()-t0:.0f}s", flush=True)
    return net, hist, time.time() - t0


# ---------------------------------------------------------------------------
# STEP 4 — head-on danger / safe map
# ---------------------------------------------------------------------------
def headon_slice(net, phys, lo, hi, approach=0.8, ng=140, dtype=torch.float32, device="cpu"):
    """V over the relative (X,Y) plane with psi_rel = pi and both vessels closing
    at surge = approach (v, r = 0), at t = -T. Returns grids for contouring."""
    xs = np.linspace(lo[0], hi[0], ng); ys = np.linspace(lo[1], hi[1], ng)
    XX, YY = np.meshgrid(xs, ys)
    Z = np.zeros((ng * ng, 9), np.float32)
    Z[:, 0] = XX.ravel(); Z[:, 1] = YY.ravel()
    Z[:, 2] = np.pi                       # head-on: evader points back at pursuer
    Z[:, IUE] = approach                  # evader surge
    Z[:, IUI] = approach                  # pursuer surge (closing)
    tt = np.full((ng * ng, 1), -phys.T, np.float32)
    inp = torch.tensor(np.hstack([Z, tt]), device=device)
    with torch.no_grad():
        V = net(inp).cpu().numpy().reshape(ng, ng)
    return XX, YY, V


def diagnostics(net, phys, lo, hi, hist, path, device="cpu"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    XX, YY, V = headon_slice(net, phys, lo, hi, device=device)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.3))
    # danger (V<=0) vs safe (V>0)
    ax[0].contourf(YY, XX, (V <= 0).astype(float), levels=[-.5, .5, 1.5],
                   colors=["#c7e9c0", "#fb6a4a"])   # green safe, red danger
    ax[0].contour(YY, XX, V, levels=[0.0], colors="k", linewidths=2)
    th = np.linspace(0, 2 * np.pi, 100)
    ax[0].plot(phys.rho * np.sin(th), phys.rho * np.cos(th), "b--", lw=1.2,
               label=f"collision disc R={phys.rho:.0f} m")
    ax[0].plot(0, 0, "ks", ms=7, label="pursuer")
    ax[0].axis("equal"); ax[0].set_xlabel("Y rel [m]"); ax[0].set_ylabel("X rel [m]")
    ax[0].set_title("Head-on danger (red) / safe (green), V=0 barrier"); ax[0].legend(loc="upper right")

    its = [h[0] for h in hist]; ls = [max(h[1], 1e-12) for h in hist]
    ax[1].semilogy(its, ls)
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("HJI residual loss (log)")
    ax[1].grid(True, which="both", alpha=0.3); ax[1].set_title("Training loss")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vessel", default=os.path.join(ROOT, "config", "vessel_config.yaml"))
    ap.add_argument("--game", default=os.path.join(ROOT, "config", "game_config.yaml"))
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "outputs", "hji_ckpt.pt"))
    ap.add_argument("--save-every", type=int, default=0, help="checkpoint + snapshot every N iters (0=off)")
    ap.add_argument("--resume", action="store_true", help="resume from --ckpt if it exists")
    args = ap.parse_args()

    with open(args.vessel) as f: vcfg = yaml.safe_load(f)
    with open(args.game) as f: gcfg = yaml.safe_load(f)

    phys64 = GameHJITorch(vcfg, gcfg, dtype=torch.float64)
    phys32 = GameHJITorch(vcfg, gcfg, dtype=torch.float32)
    T = phys32.T

    print("=" * 72)
    print(" STEP 1 — Isaacs math checks")
    e_ham, e_isaacs, worst = isaacs_checks(phys64)
    print(f"   closed-form H vs brute-force min-max : {e_ham:.2e}  [{'PASS' if e_ham<1e-3 else 'FAIL'}]")
    print(f"   Isaacs gap |min-max - max-min|       : {e_isaacs:.2e}  [{'PASS' if e_isaacs<1e-9 else 'FAIL'}]")
    print(f"   saddle controls outside thrust band  : {worst:.2e}  [{'PASS' if worst<1e-6 else 'FAIL'}]")

    lo = [-15, -15, -np.pi, -0.2, -0.6, -0.2, -0.6, -1.5, -1.5, -T]
    hi = [15, 15, np.pi, 1.2, 0.6, 1.2, 0.6, 1.5, 1.5, 0.0]

    print("-" * 72)
    print(" STEP 2 — autograd vs finite-difference gradient (value net)")
    probe = make_value(phys64, lo, hi, hidden=64, layers=3, dtype=torch.float64)
    eX, et = grad_check(probe)
    print(f"   |dV/dX err|={eX:.2e}  |dV/dt err|={et:.2e}  [{'PASS' if max(eX,et)<1e-4 else 'FAIL'}]")

    print("-" * 72)
    print(f" STEP 3 — training HJI residual ({args.iters} iters, hidden={args.hidden}, batch={args.batch})")
    dev = args.device
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("   (cuda requested but not available; falling back to cpu)"); dev = "cpu"
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    net, hist, dt = train(phys32, lo, hi, iters=args.iters, warmup=args.warmup,
                          batch=args.batch, hidden=args.hidden, layers=args.layers,
                          lr=args.lr, device=dev, ckpt_path=args.ckpt,
                          save_every=args.save_every, resume=args.resume,
                          out_png=os.path.join(ROOT, "outputs", "train_hji.png"))
    print(f"   trained in {dt:.1f} s")
    print(f"   {'iter':>6} {'residual_loss':>16} {'horizon_t':>10}")
    for it, l, tc in hist:
        print(f"   {it:>6} {l:>16.6e} {tc:>10.3f}")

    print("-" * 72)
    print(" STEP 4 — diagnostics -> outputs/train_hji.png")
    out = os.path.join(ROOT, "outputs"); os.makedirs(out, exist_ok=True)
    diagnostics(net, phys32, lo, hi, hist, os.path.join(out, "train_hji.png"), device=dev)
    torch.save(net.state_dict(), os.path.join(out, "hji_value.pt"))
    print("   saved outputs/train_hji.png and outputs/hji_value.pt")
    print("=" * 72)


if __name__ == "__main__":
    main()
