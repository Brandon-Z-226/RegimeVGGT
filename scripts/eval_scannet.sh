#!/usr/bin/env bash
# RegimeVGGT — ScanNet-50 long-sequence evaluation.
#
# Reproduces Table 3 (ScanNet-50, dense reconstruction CD across
# 1000 / 500 / 300 / 100 frame counts) and Table 9 (ScanNet-50 pose).
#
# Required env vars:
#   SCANNET_DATA   — root of processed ScanNet-50 frames (50 scenes,
#                    each with color/, depth/, intrinsic/, pose/).
#   SCANNET_GT_PLY — directory of ScanNet ground-truth scene PLYs
#                    (one <scene>_vh_clean_2.ply per scene).
#   CKPT           — path to model_tracker_fixed_e20.pt.
#
# Optional env vars:
#   FRAMES         — input frame count (default: runs the full
#                    1000/500/300/100 sweep).
#
# Usage:
#   SCANNET_DATA=/data/scannet/processed \
#   SCANNET_GT_PLY=/data/scannet/scans \
#   CKPT=ckpt/model_tracker_fixed_e20.pt \
#       bash scripts/eval_scannet.sh
#
# Paper: ScanNet N=1000 CD 0.472 / 71.7s.

set -euo pipefail

: "${SCANNET_DATA:?Set SCANNET_DATA to your processed ScanNet-50 frames root.}"
: "${SCANNET_GT_PLY:?Set SCANNET_GT_PLY to your ScanNet ground-truth PLY root.}"
: "${CKPT:?Set CKPT to your VGGT-1B checkpoint path.}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

OUT_DIR="${REPO_ROOT}/results/regimevggt/scannet"
mkdir -p "${OUT_DIR}"

EXTRA_ARGS=()
if [[ -n "${FRAMES:-}" ]]; then
    EXTRA_ARGS+=(--num_frames "${FRAMES}")
fi

python "${REPO_ROOT}/eval/eval_scannet50.py" \
    --data_dir       "${SCANNET_DATA}" \
    --gt_ply_dir     "${SCANNET_GT_PLY}" \
    --output_path    "${OUT_DIR}" \
    --method         regimevggt \
    "${EXTRA_ARGS[@]}"

echo "Results written to ${OUT_DIR}"
