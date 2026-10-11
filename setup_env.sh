#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ""
echo "Environment ready."
echo "Run:"
echo "source .venv/bin/activate"
echo "python src/train_ccld_kvasir.py --config configs/kvasir_real.json"
