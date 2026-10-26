param([int]$MaxRuntimeSeconds = 600)
Set-Location (Split-Path -Parent $PSScriptRoot)
python train.py --resume latest --max-runtime-seconds $MaxRuntimeSeconds
