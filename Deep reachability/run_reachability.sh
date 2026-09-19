#!/usr/bin/env bash
set -euo pipefail

# Ubuntu launcher for the paper-method DeepReach ASV run.
# Override for a larger/smaller GPU while keeping BATCH * ACCUM_STEPS = 65536:
#   BATCH=16384 ACCUM_STEPS=4 ./run_reachability.sh
# Resume an interrupted run:
#   RESUME=1 ./run_reachability.sh

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

python_bin="${PYTHON_BIN:-python3}"
batch="${BATCH:-8192}"
accum_steps="${ACCUM_STEPS:-8}"
resume="${RESUME:-0}"
output_dir="${OUTPUT_DIR:-output/out_hji_deepreach_paper_v2}"

if (( batch < 128 || accum_steps < 1 )); then
    echo "BATCH must be at least 128 and ACCUM_STEPS must be positive." >&2
    exit 2
fi
if (( batch * accum_steps != 65536 )); then
    echo "BATCH * ACCUM_STEPS must equal 65536 to preserve the paper-scale update." >&2
    exit 2
fi

"$python_bin" -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('PyTorch', torch.__version__); print('CUDA', torch.version.cuda); print('GPU', torch.cuda.get_device_name(0))"

mkdir -p "$output_dir"
shopt -s nullglob
checkpoints=("$output_dir"/ckpt_*.pt)
shopt -u nullglob

if (( ${#checkpoints[@]} > 0 )) && [[ "$resume" != "1" ]]; then
    echo "'$output_dir' already contains checkpoints. Set RESUME=1 to continue it." >&2
    exit 2
fi
if (( ${#checkpoints[@]} == 0 )) && [[ "$resume" == "1" ]]; then
    echo "RESUME=1 was requested but '$output_dir' has no checkpoint." >&2
    exit 2
fi

args=(
    main.py --device cuda --iters 100000
    --hidden 512 --layers 3 --batch "$batch" --accum-steps "$accum_steps"
    --lr 1e-4 --pretrain-iters 60000 --curric-frac 1.0
    --save-every 5000 --out "$output_dir"
)
if [[ "$resume" == "1" ]]; then
    args+=(--resume)
fi

echo "Starting DeepReach paper run: microbatch=$batch, accumulation=$accum_steps, effective batch=65536"
"$python_bin" "${args[@]}" 2>&1 | tee "$output_dir/train.log"
status=${PIPESTATUS[0]}
if (( status != 0 )); then
    echo "Training failed with exit code $status. See $output_dir/train.log" >&2
    exit "$status"
fi

echo "Training completed. Results: $output_dir"
