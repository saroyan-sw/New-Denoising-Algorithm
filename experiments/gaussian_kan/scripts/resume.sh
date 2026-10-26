#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
python train.py --resume latest --max-runtime-seconds "${1:-600}"
