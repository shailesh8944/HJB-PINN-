"""DeepReach training: terminal-value pretraining followed by HJI curriculum.

This follows Eq. (14) and the training procedure in Bansal & Tomlin (2021):
uniform state samples, a direct sinusoidal value network, terminal L1 loss,
and HJI-VI residual loss while the time interval expands from terminal time.
"""
import os, glob, shutil, time
import torch

from config import STATE_LO, STATE_HI, T_HORIZON
from vessel_dynamics import game_vi_residual, terminal
from value_network import value_and_grads
from diagnostics import save_map


def uniform_states(batch, lo, hi):
    return lo + (hi - lo) * torch.rand(batch, 9, dtype=lo.dtype, device=lo.device)


def paper_loss_weight(net, args, lo, hi, device, dtype):
    """Set lambda from the untrained network, before terminal pretraining.

    The paper balances the two Eq. (14) components at the beginning of
    training.  Calibrating after terminal pretraining incorrectly suppresses
    the HJI term because the terminal loss is already small then.
    """
    z = uniform_states(args.batch, lo, hi)
    tc = -1e-3 * torch.rand(args.batch, dtype=dtype, device=device)
    z_terminal = uniform_states(args.batch, lo, hi)
    V, dVdt, gradV = value_and_grads(net, z, tc)
    vi_l1 = game_vi_residual(z, V, dVdt, gradV).abs().mean().detach()
    terminal_l1 = (net(torch.cat([z_terminal, torch.zeros(args.batch, 1, dtype=dtype, device=device)], 1)).squeeze(1)
                   - terminal(z_terminal)).abs().mean().detach()
    return float(terminal_l1 / vi_l1.clamp_min(1e-12))


def terminal_pretrain(net, opt, args, lo, hi, device, dtype):
    """Paper pretraining phase: learn V_theta(z, 0)=ell(z) with L1 loss."""
    print(f" terminal-value pretraining for {args.pretrain_iters} iterations", flush=True)
    for it in range(args.pretrain_iters):
        opt.zero_grad()
        terminal_l1 = 0.0
        for _ in range(args.accum_steps):
            z = uniform_states(args.batch, lo, hi)
            inp = torch.cat([z, torch.zeros(args.batch, 1, dtype=dtype, device=device)], 1)
            loss = (net(inp).squeeze(1) - terminal(z)).abs().mean()
            (loss / args.accum_steps).backward()
            terminal_l1 += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if it % max(1, args.log_every) == 0 or it == args.pretrain_iters - 1:
            print(f" pretrain {it:>7}/{args.pretrain_iters}  terminal_L1={terminal_l1/args.accum_steps:.4e}", flush=True)


def train(net, opt, args, device, dtype=torch.float32):
    T = T_HORIZON
    prior = sorted(glob.glob(os.path.join(args.out, "ckpt_*.pt")))
    if prior and not args.resume:
        raise ValueError("output already contains checkpoints; choose a new --out directory")

    lo = torch.tensor(STATE_LO, dtype=dtype, device=device)
    hi = torch.tensor(STATE_HI, dtype=dtype, device=device)
    start, hist, pretrain_complete, pde_weight = 0, [], False, None
    if args.resume:
        if not prior:
            raise FileNotFoundError("--resume requested but no checkpoints exist in --out")
        if prior:
            latest = prior[-1]
            ck = torch.load(latest, map_location=device)
            net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
            start = int(ck["iter"]) + 1; hist = ck.get("hist", [])
            pretrain_complete = bool(ck.get("pretrain_complete", False))
            pde_weight = ck.get("pde_weight")
            if not pretrain_complete:
                raise ValueError("checkpoint predates paper-style terminal pretraining; start a fresh run")
            print(f" resumed from {latest} at iter {start}", flush=True)

    if not pretrain_complete:
        pde_weight = paper_loss_weight(net, args, lo, hi, device, dtype)
        print(f" calibrated paper loss weight lambda={pde_weight:.4e} before terminal pretraining", flush=True)
        terminal_pretrain(net, opt, args, lo, hi, device, dtype)
        pretrain_complete = True
    print(f" {'iter':>7} {'loss':>12} {'term L1':>12} {'VI L1':>12} {'horizon':>8} {'it/s':>7} {'elapsed':>9} {'ETA':>9}", flush=True)
    t0 = time.time()
    t_last, it_last = t0, start
    for it in range(start, args.iters):
        opt.zero_grad()
        frac = min(1.0, max(0.0, it - args.warmup) / max(1.0, args.curric_frac*args.iters))
        t_cur = min(T, 1e-3 + T*frac)
        terminal_l1 = vi_l1 = loss = 0.0
        for _ in range(args.accum_steps):
            z = uniform_states(args.batch, lo, hi)
            tc = -t_cur * torch.rand(args.batch, dtype=dtype, device=device)
            z_terminal = uniform_states(args.batch, lo, hi)
            terminal_time = torch.zeros(args.batch, dtype=dtype, device=device)
            V, dVdt, gradV = value_and_grads(net, z, tc)
            vi_l1_part = game_vi_residual(z, V, dVdt, gradV).abs().mean()
            terminal_l1_part = (net(torch.cat([z_terminal, terminal_time[:, None]], 1)).squeeze(1)
                                - terminal(z_terminal)).abs().mean()
            loss_part = terminal_l1_part + pde_weight * vi_l1_part
            (loss_part / args.accum_steps).backward()
            terminal_l1 += float(terminal_l1_part.detach())
            vi_l1 += float(vi_l1_part.detach())
            loss += float(loss_part.detach())
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        terminal_l1 /= args.accum_steps
        vi_l1 /= args.accum_steps
        loss /= args.accum_steps
        if it % 100 == 0 or it == args.iters - 1:
            hist.append((it, loss))
        if args.log_every and (it % args.log_every == 0 or it == args.iters - 1):
            now = time.time()
            rate = (it - it_last) / max(1e-9, now - t_last)
            t_last, it_last = now, it
            remaining = args.iters - 1 - it
            eta_s = remaining / rate if rate > 0 else float("inf")
            print(f" {it:>7} {loss:>12.4e} {terminal_l1:>12.4e} {vi_l1:>12.4e} {t_cur:>7.1f}s "
                  f"{rate:>6.1f}/s {(now-t0)/60:>8.1f}m {eta_s/60:>8.1f}m", flush=True)
        if args.save_every and (it % args.save_every == 0 or it == args.iters - 1) and it > start:
            ckpt_path = os.path.join(args.out, f"ckpt_{it:07d}.pt")
            map_path = os.path.join(args.out, f"danger_safe_{it:07d}.png")
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                        "iter": it, "hist": hist, "pretrain_complete": pretrain_complete,
                        "pde_weight": pde_weight,
                        "run": args.run_metadata}, ckpt_path)
            save_map(net, map_path, device, hist=hist, time_remaining=t_cur)
            shutil.copyfile(map_path, os.path.join(args.out, "danger_safe.png"))  # convenience "latest" alias
            print(f" [ckpt saved] iter {it:>7}  -> {ckpt_path}, {map_path}", flush=True)

    print(f" DONE in {(time.time()-t0)/60:.1f} min. Results in {args.out}/  "
          f"({len(glob.glob(os.path.join(args.out, 'ckpt_*.pt')))} snapshots kept)", flush=True)
