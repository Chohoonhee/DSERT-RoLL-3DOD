#!/usr/bin/env bash
#
# Build & register the local Python packages and their CUDA extensions.
# Run from the repository root, AFTER you have:
#   1. created/activated the conda env and installed PyTorch (Step 1)
#   2. installed cmake + spconv + waymo eval (Step 2)
#   3. pip install -r requirements.txt              (Step 3)
#   4. installed timm<0.6, mmcv-full==1.4.0, and run swin_model setup.py (Step 4)
# See README.md > Installation for full context.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$ROOT/utils" ] || [ ! -d "$ROOT/detection" ]; then
    echo "ERROR: setup_py.sh must live next to utils/ and detection/." >&2
    exit 1
fi

echo "==> [1/3] Building al3d_utils ..."
( cd "$ROOT/utils" && python setup.py develop )

echo "==> [2/3] Building al3d_det ..."
( cd "$ROOT/detection" && python setup.py develop )

echo "==> [3/3] Building Deformable-attention DCN op ..."
( cd "$ROOT/detection/al3d_det/models/ops" && python setup.py develop )

echo ""
echo "All extensions built. Quick import check:"
python -c "
from al3d_det.datasets.dsert.dsert_dataset import DSERTTrainingDataset
from al3d_det.models import build_network
print('  Install OK')
"
