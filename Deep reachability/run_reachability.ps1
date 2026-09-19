<#
Launch the paper-method DeepReach ASV reachability run from this directory.

Examples:
  .\run_reachability.ps1
  .\run_reachability.ps1 -Resume

The script checks CUDA first, never overwrites a checkpointed fresh run, and
saves the training log. It runs the paper's 60k terminal-value pretraining and
100k uniform-sampling curriculum iterations. Sixteen 4,096-sample GPU
microbatches form each 65,536-sample paper-scale Adam update.
#>
[CmdletBinding()]
param(
    [int]$Batch = 4096,
    [switch]$Resume,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
if ($Batch -lt 128) { throw '-Batch must be at least 128.' }

$root = $PSScriptRoot
Set-Location -LiteralPath $root

& $Python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable in this Python environment'; print('PyTorch', torch.__version__); print('CUDA', torch.version.cuda); print('GPU', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed. Activate the Python environment containing the CUDA-enabled PyTorch installation, then retry.' }

$iterations = 100000
$pretrainIterations = 60000
$saveEvery = 5000
$outputName = 'output/out_hji_deepreach_paper_v2'

$checkpointPattern = Join-Path $outputName 'ckpt_*.pt'
$existing = @(Get-ChildItem -Path $checkpointPattern -ErrorAction SilentlyContinue)
if ($existing.Count -gt 0 -and -not $Resume) {
    throw "'$outputName' already has checkpoints. Use -Resume to continue it, or choose the other mode/output directory."
}
if ($Resume -and $existing.Count -eq 0) {
    throw "-Resume was requested but '$outputName' has no checkpoint."
}

$trainArgs = @(
    'main.py', '--device', 'cuda', '--iters', $iterations,
    '--hidden', '512', '--layers', '3', '--batch', $Batch, '--accum-steps', '16',
    '--lr', '1e-4', '--pretrain-iters', $pretrainIterations,
    '--curric-frac', '1.0', '--save-every', $saveEvery,
    '--out', $outputName
)
if ($Resume) { $trainArgs += '--resume' }

New-Item -ItemType Directory -Force -Path $outputName | Out-Null
$log = Join-Path $outputName 'train.log'
Write-Host "Starting paper-method DeepReach training in $outputName (batch $Batch)."
& $Python @trainArgs 2>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) { throw "Training failed; see $log" }

$latest = Get-ChildItem -Path $checkpointPattern | Sort-Object Name | Select-Object -Last 1
if ($null -eq $latest) { throw "Training ended without a checkpoint; see $log" }
Write-Host "Training completed. Final checkpoint: $($latest.FullName)"
