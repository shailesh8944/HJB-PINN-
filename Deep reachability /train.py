"""Training loop: samples random (state,time) points inside the curriculum
horizon, minimises the squared HJI variational-inequality residual, and
checkpoints the network + a diagnostic plot every --save-every iterations
(numbered snapshots, never overwritten)."""
import os, glob, shutil, time
import torch

from config import STATE_LO, STATE_HI, T_HORIZON
from vessel_dynamics import game_vi_residual
from value_network import value_and_grads
from diagnostics import save_map


def train(net, opt, args, device, dtype=torch.float32):
    T = T_HORIZON
    lo9 = torch.tensor(STATE_LO, dtype=dtype, device=device)
    hi9 = torch.tensor(STATE_HI, dtype=dtype, device=device)

    start, hist = 0, []
    if args.resume:
        prior = sorted(glob.glob(os.path.join(args.out, "ckpt_*.pt")))
        if prior:
            latest = prior[-1]
            ck = torch.load(latest, map_location=device)
            net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
            start = int(ck["iter"]) + 1; hist = ck.get("hist", [])
            print(f" resumed from {latest} at iter {start}", flush=True)

    print(f" {'iter':>7} {'loss':>12} {'horizon':>8} {'it/s':>7} {'elapsed':>9} {'ETA':>9}", flush=True)
    t0 = time.time()
    t_last, it_last = t0, start
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
        if args.log_every and (it % args.log_every == 0 or it == args.iters - 1):
            now = time.time()
            rate = (it - it_last) / max(1e-9, now - t_last)
            t_last, it_last = now, it
            remaining = args.iters - 1 - it
            eta_s = remaining / rate if rate > 0 else float("inf")
            print(f" {it:>7} {float(loss.detach()):>12.4e} {t_cur:>7.1f}s "
                  f"{rate:>6.1f}/s {(now-t0)/60:>8.1f}m {eta_s/60:>8.1f}m", flush=True)
        if args.save_every and (it % args.save_every == 0 or it == args.iters - 1) and it > start:
            ckpt_path = os.path.join(args.out, f"ckpt_{it:07d}.pt")
            map_path = os.path.join(args.out, f"danger_safe_{it:07d}.png")
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                        "iter": it, "hist": hist}, ckpt_path)
            save_map(net, map_path, device, hist=hist)
            shutil.copyfile(map_path, os.path.join(args.out, "danger_safe.png"))  # convenience "latest" alias
            print(f" [ckpt saved] iter {it:>7}  -> {ckpt_path}, {map_path}", flush=True)

    print(f" DONE in {(time.time()-t0)/60:.1f} min. Results in {args.out}/  "
          f"({len(glob.glob(os.path.join(args.out, 'ckpt_*.pt')))} snapshots kept)", flush=True)
