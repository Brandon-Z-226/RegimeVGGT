#!/usr/bin/env bash
# RegimeVGGT — Tanks & Temples 6-scene pose evaluation.
#
# Reproduces Table 4 (camera pose on T&T) from the paper.
#
# Required env vars:
#   TNT_DIR         — root of T&T training set, contains scene subfolders
#                     (Barn, Caterpillar, Courthouse, Ignatius,
#                      Meetingroom, Truck), each holding RGB frames.
#   TNT_COLMAP_DIR  — directory of T&T ground-truth COLMAP poses,
#                     organised the same way (one subdir per scene).
#   CKPT            — path to VGGT-1B checkpoint (see README.md).
#
# Usage:
#   TNT_DIR=/data/tnt/training \
#   TNT_COLMAP_DIR=/data/tnt/poses \
#   CKPT=ckpt/model_tracker_fixed_e20.pt \
#       bash scripts/eval_tnt.sh
#
# Paper Table 4: AUC@30 0.9105 / 111s / 5.01x.

set -euo pipefail

: "${TNT_DIR:?Set TNT_DIR to your T&T training-set root.}"
: "${TNT_COLMAP_DIR:?Set TNT_COLMAP_DIR to your T&T COLMAP pose root.}"
: "${CKPT:?Set CKPT to your VGGT-1B checkpoint path.}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

OUT_DIR="${REPO_ROOT}/results/regimevggt/tnt"
mkdir -p "${OUT_DIR}"

# AB2 canonical: Line-A merge (rho=0.99/0.50/0.99, cache=(0,10,18))
# composed with Line-B uniform sigma_b=1.5 + frame-0 anchor.
python "${REPO_ROOT}/eval/eval_pose_tnt.py" \
    --tnt_dir       "${TNT_DIR}" \
    --colmap_dir    "${TNT_COLMAP_DIR}" \
    --ckpt          "${CKPT}" \
    --output_dir    "${OUT_DIR}" \
    --kf_every      1 \
    --method                regimevggt \
    --shallow_merge_ratio   0.99 \
    --merge_ratio_ours      0.50 \
    --deep_merge_ratio      0.99 \
    --cache_band_starts     0 10 18 \
    --hyb_sigma_sub         1.3 \
    --hyb_sigma_shallow     1.5 \
    --hyb_sigma_deep        1.7 \
    --hyb_use_phase_shift \
    --hyb_anchor_frame_mode first \
    --hyb_protect_middle \
    --hyb_middle_range      10 14

echo "Results written to ${OUT_DIR}"
