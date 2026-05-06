# RegimeVGGT

**Training-free acceleration for VGGT via a layer-wise rank-band prior.**

> Anonymous code release for NeurIPS 2026 submission. Author information
> and original repository links are intentionally omitted for double-blind
> review.

---

## What's here

A single training-free acceleration of VGGT-1B model_tracker's global self-attention.
The method composes two orthogonal compressions of the same global block:

- **Token merging** on the token-count axis — three-band ratio
  ($\rho_s, \rho_m, \rho_d$) with band-aligned index cache and a
  DINOv2-saliency protection set;
- **K/V phase-shifted subsampling** on the K/V-set axis — U-shape
  per-band $\sigma$ with a frame-0 anchor preserving long-range
  coordinate consistency;
- with the rank-peak sub-band L10–13 doing **merge-only** (no subsample
  on top), to protect the cross-view alignment regime.

All on top of stock VGGT-1B model_tracker weights, **no fine-tuning**.

| Benchmark                       | Metric          | Result               |
|---------------------------------|-----------------|----------------------|
| Tanks & Temples (kf=1, 6 scenes)| AUC@30 / time   | 0.9105 / 111s (5.01x)|
| ScanNet-50 (1000 input frames)  | Chamfer / time  | 0.472 / 71.7s        |

---

## Installation

```bash
pip install -r requirements.txt
```

Tested on a single NVIDIA H800 80 GB with PyTorch 2.1+, CUDA 11.8+, Python 3.10+.
A GPU with ≥ 24 GB VRAM suffices for short sequences; ScanNet-1000 needs ≥ 80 GB.

## Checkpoint

RegimeVGGT uses the **VGGT-1B model_tracker** checkpoint with no
fine-tuning — the upstream weights with the tracking head attached,
distributed by Meta on Hugging Face.

```bash
# create a folder and download the model_tracker_fixed_e20.pt checkpoint
mkdir -p ckpt
wget -P ckpt \
    https://huggingface.co/facebook/VGGT_tracker_fixed/resolve/main/model_tracker_fixed_e20.pt
```

You can also point `CKPT` directly at any path that holds the file
(e.g., a shared cache):

```bash
export CKPT=/shared/cache/vggt-1b/model_tracker_fixed_e20.pt
```

The checkpoint is subject to the original VGGT license; see
<https://huggingface.co/facebook/VGGT_tracker_fixed>.

---


## Reproduce paper numbers

Three reproduction entry points live in `scripts/`. Each reads dataset
paths and `CKPT` from environment variables — no hardcoded paths,
no CLI argument needed.

```bash
# Tanks & Temples 6-scene pose (Table 4)
TNT_DIR=/path/to/tnt/training \
TNT_COLMAP_DIR=/path/to/tnt/poses \
CKPT=ckpt/model_tracker_fixed_e20.pt \
    bash scripts/eval_tnt.sh

# 7Scenes + NRGBD dense reconstruction (Tables 1 & 2)
SEVEN_ROOT=/path/to/7scenes \
NRGBD_ROOT=/path/to/nrgbd \
CKPT=ckpt/model_tracker_fixed_e20.pt \
    bash scripts/eval_7andn.sh

# ScanNet-50 long-sequence (Tables 3 & 9)
SCANNET_DATA=/path/to/scannet/processed \
SCANNET_GT_PLY=/path/to/scannet/scans \
CKPT=ckpt/model_tracker_fixed_e20.pt \
    bash scripts/eval_scannet.sh
```

Each run writes `summary.json` and `logs.txt` under
`results/regimevggt/<dataset>/`.

To compare against uncompressed VGGT*, pass `--method baseline` to the
underlying `eval/*.py` script (the shell wrappers default to RegimeVGGT).

---

## Repository layout

```
RegimeVGGT/
├── README.md                  # this file
├── LICENSE
├── requirements.txt
├── demo.py                    # minimal single-scene smoke test
├── methods/
│   ├── regimevggt.py          # main aggregator: token merge x K/V phase-shift
│   └── fastvggt_merge.py      # ToMe bipartite-2d primitive
├── vggt/                      # forked VGGT-1B model_tracker code (aggregator + heads)
├── eval/
│   ├── eval_pose_tnt.py       # T&T 6-scene pose
│   ├── eval_7andN.py          # 7Scenes + NRGBD dense reconstruction
│   └── eval_scannet50.py      # ScanNet-50 long-sequence (CD + pose)
└── scripts/
    ├── eval_tnt.sh            # T&T pose, env-var-driven
    ├── eval_7andn.sh          # 7Scenes + NRGBD dense recon
    └── eval_scannet.sh        # ScanNet-50
```

---

## Method configuration (canonical)

The single canonical recipe across all benchmarks is defined in three
places (kept in sync): `eval/eval_scannet50.py:HYBRID_CONFIG`, the three
`scripts/eval_*.sh` wrappers, and `demo.py`.

**Token-merge axis** (Line A):

| Parameter             | Value          | Notes                             |
|-----------------------|----------------|-----------------------------------|
| `shallow_merge_ratio` | 0.99           | layers L0–9                       |
| `merge_ratio`         | 0.50           | layers L10–17                     |
| `deep_merge_ratio`    | 0.99           | layers L18+                       |
| `cache_band_starts`   | (0, 10, 18)    | three-band index cache            |
| `importance_method`   | `dino_attn`    | DINOv2 [CLS] saliency (Ψ)         |


**K/V-subsample axis** (Line B):

| Parameter             | Value          | Notes                             |
|-----------------------|----------------|-----------------------------------|
| `sigma_shallow`       | 1.5            | shallow band σ                    |
| `sigma_sub`           | 1.3            | middle band σ                     |
| `sigma_deep`          | 1.7            | deep band σ                       |
| `use_phase_shift`     | True           | per-frame phase rotation          |
| `anchor_frame_idx`    | 0              | frame 0 kept full density         |
| `protect_middle`      | True           | L10–13 do merge-only              |
| `middle_range`        | (10, 14)       | half-open, covers L10–13          |

---

## License

Apache-2.0. See `LICENSE`. The VGGT-1B model_tracker weights and the
original VGGT model code under `vggt/` are subject to the upstream VGGT
license.

---

## Citation

```bibtex
@inproceedings{regimevggt2026,
  title     = {(anonymous title for NeurIPS 2026 submission)},
  author    = {Anonymous},
  booktitle = {Submitted to NeurIPS 2026},
  year      = {2026}
}
```

Citation will be updated upon paper acceptance.
