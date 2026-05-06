#!/usr/bin/env bash
# RegimeVGGT — 7Scenes + NRGBD dense reconstruction evaluation.
#
# Reproduces Tables 1 (7Scenes) and 2 (NRGBD) from the paper.
#
# Required env vars:
#   SEVEN_ROOT  — root of the 7Scenes dataset.
#   NRGBD_ROOT  — root of the NRGBD dataset.
#   CKPT        — path to model_tracker_fixed_e20.pt (see README.md).
#
# Optional env vars:
#   KF          — keyframe stride (default 10; paper also reports 3).
#
# Usage:
#   SEVEN_ROOT=/data/7scenes \
#   NRGBD_ROOT=/data/nrgbd \
#   CKPT=ckpt/model_tracker_fixed_e20.pt \
#       bash scripts/eval_7andn.sh

set -euo pipefail

KF="${KF:-10}"

: "${SEVEN_ROOT:?Set SEVEN_ROOT to your 7Scenes dataset root.}"
: "${NRGBD_ROOT:?Set NRGBD_ROOT to your NRGBD dataset root.}"
: "${CKPT:?Set CKPT to your VGGT-1B checkpoint path.}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

OUT_DIR="${REPO_ROOT}/results/regimevggt/7andn_kf${KF}"
mkdir -p "${OUT_DIR}"

# AB2 canonical: Line-A merge + Line-B uniform sigma_b=1.5 + frame-0 anchor.
python "${REPO_ROOT}/eval/eval_7andN.py" \
    --datasets       7scenes NRGBD \
    --7scenes_root   "${SEVEN_ROOT}" \
    --nrgbd_root     "${NRGBD_ROOT}" \
    --kf             "${KF}" \
    --output_dir     "${OUT_DIR}" \
    --method                regimevggt \
    --shallow_merge_ratio   0.99 \
    --merge_ratio_ours      0.50 \
    --deep_merge_ratio      0.99 \
    --cache_band_starts     0 10 18 \
    --no_cache_layers       10 11 12 13 \
    --hyb_sigma_sub         1.3 \
    --hyb_sigma_shallow     1.5 \
    --hyb_sigma_deep        1.7 \
    --hyb_use_phase_shift \
    --hyb_anchor_frame_mode first \
    --hyb_protect_middle \
    --hyb_middle_range      10 14

echo "Results written to ${OUT_DIR}"
