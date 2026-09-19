"""Entry point: parses CLI args, builds the value network + optimizer, runs
the Isaacs self-consistency checks (fails loudly before burning GPU-hours on
a broken Hamiltonian), then launches training.

For the RTX 3050 6 GB laptop GPU, use run_reachability.ps1. It retains the
paper's direct-SIREN/terminal-pretraining/curriculum method while reducing the
physical batch size to fit GPU memory. If it runs out of memory, use batch 2048.
"""
import argparse
import os
import torch

from config import M11, M22, M33, M23, M32, ARM, FMAX, R_COLLIDE, T_HORIZON, STATE_LO, STATE_HI
from vessel_dynamics import isaacs_checks
from value_network import SIREN
from train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters", type=int, default=100000,
                    help="paper curriculum-training iterations for the 9D case")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4096,
                    help="safe default for the RTX 3050 6 GB laptop GPU")
    ap.add_argument("--accum-steps", type=int, default=16,
                    help="microbatches per Adam update; 16*4096 reproduces the paper's 65k sample update")
    ap.add_argument("--hidden", type=int, default=512,
                    help="paper hidden-layer width")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="paper Adam learning rate")
    ap.add_argument("--curric-frac", type=float, default=1.0,
                    help="fraction of curriculum iterations over which time expands to the full horizon")
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--out", default="out_hji")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pretrain-iters", type=int, default=60000,
                    help="paper terminal-value pretraining iterations for the 9D case")
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("cuda not available -> cpu"); device = "cpu"
    if (args.pretrain_iters < 0 or args.warmup < 0 or args.accum_steps < 1
            or not 0 < args.curric_frac <= 1):
        raise ValueError("--pretrain-iters and --warmup must be non-negative; --accum-steps must be positive; --curric-frac must be in (0, 1]")
    os.makedirs(args.out, exist_ok=True)
    dtype = torch.float32
    torch.manual_seed(args.seed)

    print("="*70)
    print(f" device={device}  M=[[{M11},0,0],[0,{M22},{M23}],[0,{M32},{M33}]]  "
          f"arm={ARM}  Fmax={FMAX:.3f}N  R={R_COLLIDE}  T={T_HORIZON}")
    print(f" lr={args.lr}  terminal-pretrain={args.pretrain_iters}  batch={args.batch}x{args.accum_steps} "
          f"(effective={args.batch*args.accum_steps})  hidden={args.hidden}x{args.layers} "
          f"direct-SIREN  uniform-sampling  curric_frac={args.curric_frac}")
    eh, ei, w = isaacs_checks(device)
    print(f" Isaacs: |H-minmax|={eh:.2e}[{'PASS' if eh<1e-3 else 'FAIL'}]  "
          f"gap={ei:.2e}[{'PASS' if ei<1e-9 else 'FAIL'}]  "
          f"saddle_outside={w:.2e}[{'PASS' if w<1e-6 else 'FAIL'}]")
    print("="*70, flush=True)

    lo = STATE_LO + [-T_HORIZON]
    hi = STATE_HI + [0.0]
    net = SIREN(lo, hi, args.hidden, args.layers, dtype=dtype).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    args.run_metadata = dict(version=3, methodology="DeepReach-paper",
                             pretrain_iters=args.pretrain_iters, warmup=args.warmup,
                             batch=args.batch, accum_steps=args.accum_steps,
                             sampling="uniform", seed=args.seed,
                             state_lo=STATE_LO, state_hi=STATE_HI,
                             mass_matrix=[[M11, 0., 0.], [0., M22, M23], [0., M32, M33]],
                             arm_m=ARM, fmax_N=FMAX, horizon_s=T_HORIZON)

    train(net, opt, args, device, dtype=dtype)


if __name__ == "__main__":
    main()
