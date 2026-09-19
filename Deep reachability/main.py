"""Entry point: parses CLI args, builds the value network + optimizer, runs
the Isaacs self-consistency checks (fails loudly before burning GPU-hours on
a broken Hamiltonian), then launches training.

Typical overnight run on an RTX A4000 (16 GB):
    nohup python3 main.py --device cuda --iters 150000 \
        --hidden 512 --layers 3 --batch 65536 --lr 2e-5 --warmup 10000 \
        --save-every 5000 --out out_hji > train.log 2>&1 &
    tail -f train.log
Resume after an interruption: add --resume (loads the latest out_hji/ckpt_*.pt).
If you hit a GPU out-of-memory error, drop --batch to 32768 or 16384.
"""
import argparse
import os
import torch

from config import M11, M22, M33, MC, ARM, FMAX, R_COLLIDE, T_HORIZON, STATE_LO, STATE_HI
from vessel_dynamics import isaacs_checks
from value_network import SIREN, ExactBCValue
from train import train


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
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--out", default="out_hji")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("cuda not available -> cpu"); device = "cpu"
    os.makedirs(args.out, exist_ok=True)
    dtype = torch.float32
    torch.manual_seed(args.seed)

    print("="*70)
    print(f" device={device}  M=[[{M11},0,0],[0,{M22},{MC}],[0,{MC},{M33}]]  "
          f"arm={ARM}  Fmax={FMAX:.3f}N  R={R_COLLIDE}  T={T_HORIZON}")
    print(f" lr={args.lr}  warmup={args.warmup}  batch={args.batch}  "
          f"hidden={args.hidden}x{args.layers}  curric_frac={args.curric_frac}")
    eh, ei, w = isaacs_checks(device)
    print(f" Isaacs: |H-minmax|={eh:.2e}[{'PASS' if eh<1e-3 else 'FAIL'}]  "
          f"gap={ei:.2e}[{'PASS' if ei<1e-9 else 'FAIL'}]  "
          f"saddle_outside={w:.2e}[{'PASS' if w<1e-6 else 'FAIL'}]")
    print("="*70, flush=True)

    lo = STATE_LO + [-T_HORIZON]
    hi = STATE_HI + [0.0]
    net = ExactBCValue(SIREN(lo, hi, args.hidden, args.layers, dtype=dtype)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    train(net, opt, args, device, dtype=dtype)


if __name__ == "__main__":
    main()
