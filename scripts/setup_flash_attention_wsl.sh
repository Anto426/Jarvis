#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pip install --upgrade pip wheel "setuptools<82"
python -m pip install ninja packaging psutil

export MAX_JOBS="${MAX_JOBS:-4}"
python -m pip install flash-attn --no-build-isolation

python - <<'PY'
import flash_attn
print("flash_attn OK", getattr(flash_attn, "__version__", "unknown"))
PY
