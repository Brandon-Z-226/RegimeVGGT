"""ScanNet-50 quantitative evaluation for VGGT* and RegimeVGGT.

Reconstruction (Chamfer Distance) and pose (ATE / ARE / RPE) on
ScanNet-50 across 1000 / 500 / 300 / 100 input frames.

Usage:
    python eval_scannet50.py --method baseline       # uncompressed VGGT*
    python eval_scannet50.py --method regimevggt     # RegimeVGGT (paper config)
    python eval_scannet50.py --method all            # both
"""

import os
import sys
import argparse
import time
from pathlib import Path
from collections import defaultdict

import gc
import numpy as np
import torch

# ── Path setup: FastVGGT as primary vggt package ──
EVAL_DIR = Path(__file__).resolve().parent
ADAVGGT_ROOT = EVAL_DIR / ".."
# Resolve FastVGGT location: post-reorg layout uses other_method/FastVGGT/,
# AutoDL legacy layout keeps FastVGGT/ at the repo root. Try both.
def _resolve_fastvggt_root(eval_dir):
    for candidate in (
        eval_dir / ".." / ".." / "other_method" / "FastVGGT",
        eval_dir / ".." / ".." / "FastVGGT",
    ):
        if (candidate / "eval" / "scannet_50.yaml").exists():
            return candidate.resolve()
    # Fall back to the canonical post-reorg path; downstream open() will raise
    # a clear FileNotFoundError pointing at the missing yaml.
    return (eval_dir / ".." / ".." / "other_method" / "FastVGGT").resolve()


FASTVGGT_ROOT = _resolve_fastvggt_root(EVAL_DIR)

# FastVGGT first (model + eval_utils)
sys.path.insert(0, str(FASTVGGT_ROOT))
# RegimeVGGT methods (importance_merging.py)
if str(ADAVGGT_ROOT) not in sys.path:
    sys.path.append(str(ADAVGGT_ROOT))

from vggt.models.vggt import VGGT
from vggt.utils.eval_utils import (
    load_poses,
    get_vgg_input_imgs,
    get_sorted_image_paths,
    build_frame_selection,
    load_images_rgb,
    infer_vggt_and_reconstruct,
    evaluate_scene_and_save,
    compute_average_metrics_and_save,
)
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


# ── RegimeVGGT canonical = AB2 (Line-A merge + Line-B subsample) ──
# Single recipe across all benchmarks.
# Paper: T&T AUC@30 0.9105 / 5.01x; ScanNet N=1000 CD 0.472 / 71.7s.
HYBRID_CONFIG = dict(
    shallow_merge_ratio=0.99,
    merge_ratio=0.50,
    deep_merge_ratio=0.99,
    middle_merge_start=10,
    deep_merge_start=18,
    cache_band_starts=(0, 10, 18),
    no_cache_layers=(10, 11, 12, 13),           # rank-peak: recompute every layer
    importance_method="dino_attn",
    protect_last=False,
    sigma_sub=1.3,                              # middle-band sigma (U-shape)
    sigma_shallow=1.5,                          # shallow-band sigma
    sigma_deep=1.7,                             # deep-band sigma
    use_phase_shift=True,
    use_avggt_mean_fill=False,
    use_avggt_full=False,
    protect_middle=True,                        # L10-13 do merge only (no subsample)
    middle_range=(10, 14),
    anchor_frame_idx=0,
)


@torch.no_grad()
def infer_regimevggt_and_reconstruct(model, vgg_input, dtype, depth_conf_thresh):
    """Run regimevggt (Line A merge + Line B phase-shift + anchor) and reconstruct."""
    from methods.regimevggt import run_aggregator_regimevggt

    torch.cuda.synchronize()
    start = time.time()
    with torch.cuda.amp.autocast(dtype=dtype):
        imgs = vgg_input.cuda().to(torch.bfloat16)
        if imgs.dim() == 4:
            imgs = imgs.unsqueeze(0)

        output_list, psi = run_aggregator_regimevggt(model, imgs, **HYBRID_CONFIG)

        pose_enc_list = model.camera_head(output_list)
        pose_enc = pose_enc_list[-1]
        depth, depth_conf = model.depth_head(
            output_list, images=imgs, patch_start_idx=psi,
        )

    torch.cuda.synchronize()
    inference_time_ms = (time.time() - start) * 1000.0

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        pose_enc, (vgg_input.shape[2], vgg_input.shape[3])
    )
    extrinsic_np = extrinsic[0].detach().float().cpu().numpy()
    intrinsic_np = intrinsic[0].detach().float().cpu().numpy()

    depth_np = depth[0].detach().float().cpu().numpy()
    depth_conf_np = depth_conf[0].detach().float().cpu().numpy()
    depth_np[depth_conf_np < depth_conf_thresh] = np.nan

    world_points = unproject_depth_map_to_point_map(
        depth_np, extrinsic_np, intrinsic_np
    )

    all_points, all_colors = [], []
    vgg_np = vgg_input.detach().float().cpu().numpy()
    for frame_idx in range(world_points.shape[0]):
        pts = world_points[frame_idx].reshape(-1, 3)
        valid = ~np.isnan(pts).any(axis=1) & ~np.isinf(pts).any(axis=1)
        if valid.sum() > 0:
            all_points.append(pts[valid])
            img_hwc = (
                (np.transpose(vgg_np[frame_idx], (1, 2, 0)) * 255.0)
                .clip(0, 255).astype(np.uint8)
            )
            all_colors.append(img_hwc.reshape(-1, 3)[valid])
    all_c2w = [extrinsic_np[i] for i in range(extrinsic_np.shape[0])]
    return extrinsic_np, intrinsic_np, all_points, all_colors, all_c2w, inference_time_ms


def load_model(method, ckpt_path, merge_ratio=0.9):
    """Load VGGT model with appropriate config for each method."""
    if method == "baseline":
        model = VGGT(merging=None)
    elif method == "regimevggt":
        # RegimeVGGT drives merge from the aggregator wrapper, not VGGT.merging.
        model = VGGT(merging=0, merge_ratio=HYBRID_CONFIG["merge_ratio"])
    else:
        raise ValueError(f"Unknown method: {method}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    del ckpt
    model = model.cuda().eval().to(torch.bfloat16)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ScanNet-50 evaluation")
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Root of the processed ScanNet-50 frames "
                             "(50 scene subdirectories, each holding "
                             "color/<idx>.jpg, depth/<idx>.png, "
                             "intrinsic/intrinsic_color.txt, pose/<idx>.txt).")
    parser.add_argument("--gt_ply_dir", type=Path, required=True,
                        help="Directory of ScanNet ground-truth scene PLY files "
                             "(one <scene>_vh_clean_2.ply per scene).")
    parser.add_argument("--output_path", type=Path, default="./results/main")
    parser.add_argument("--method", type=str, default="regimevggt",
                        choices=["baseline", "regimevggt", "all"],
                        help="'baseline' is uncompressed VGGT* for reference; "
                             "'all' runs both baseline and regimevggt.")
    parser.add_argument("--depth_conf_thresh", type=float, default=1.0)
    parser.add_argument("--chamfer_max_dist", type=float, default=0.5)
    parser.add_argument("--input_frame", type=int, default=1000)
    parser.add_argument("--num_scenes", type=int, default=None)
    parser.add_argument("--ckpt_path", type=str,
                        default=str(EVAL_DIR / ".." / ".." / "ckpt" / "model_tracker_fixed_e20.pt"))
    parser.add_argument("--merge_ratio", type=float, default=0.9,
                        help="(legacy) baseline VGGT merge ratio")
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional output-subdir tag.")
    args = parser.parse_args()

    torch.manual_seed(33)
    np.random.seed(0)
    dtype = torch.bfloat16

    # Scene list
    if args.num_scenes is not None:
        from vggt.utils.eval_utils import get_all_scenes
        scannet_scenes = get_all_scenes(args.data_dir, args.num_scenes)
    else:
        yaml_path = FASTVGGT_ROOT / "eval" / "scannet_50.yaml"
        with open(yaml_path) as f:
            scannet_scenes = [line.strip() for line in f if line.strip()]
    print(f"Evaluate {len(scannet_scenes)} scenes")

    # Methods to run
    methods = ["baseline", "regimevggt"] if args.method == "all" else [args.method]

    # Print RegimeVGGT config if running regimevggt
    if "regimevggt" in methods:
        print(f"RegimeVGGT config: {HYBRID_CONFIG}")
        if args.tag:
            print(f"Ablation tag: {args.tag}")

    for method in methods:
        # Use tag as method name for output path if provided
        method_label = args.tag if (args.tag and method == "regimevggt") else method
        print(f"\n{'='*60}")
        print(f"  Method: {method_label}")
        print(f"{'='*60}")

        output_base = args.output_path / method_label / "scannet"
        all_scenes_metrics = {"scenes": {}, "average": {}}
        scene_fps = defaultdict(list)

        model = load_model(method, args.ckpt_path, args.merge_ratio)

        for scene in scannet_scenes:
            scene_dir = args.data_dir / scene
            output_scene_dir = output_base / f"input_frame_{args.input_frame}" / scene
            if (output_scene_dir / "metrics.json").exists():
                print(f"  {scene}: already done, skip")
                continue

            images_dir = scene_dir / "color"
            pose_path = scene_dir / "pose"
            image_paths = get_sorted_image_paths(images_dir)
            poses_gt, first_gt_pose, available_pose_frame_ids = load_poses(pose_path)
            if poses_gt is None or first_gt_pose is None:
                print(f"  {scene}: no pose data, skip")
                continue

            selected_frame_ids, selected_image_paths, selected_pose_indices = (
                build_frame_selection(image_paths, available_pose_frame_ids, args.input_frame)
            )
            c2ws = poses_gt[selected_pose_indices]
            image_paths = selected_image_paths

            if len(image_paths) < 3:
                print(f"  {scene}: insufficient images, skip")
                continue

            print(f"  {scene}: {len(image_paths)} images", end=" ", flush=True)

            try:
                images = load_images_rgb(image_paths)
                if not images or len(images) < 3:
                    print("- insufficient valid images, skip")
                    continue

                images_array = np.stack(images)
                vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)
                model.update_patch_dimensions(patch_width, patch_height)

                torch.cuda.reset_peak_memory_stats()

                if method == "baseline":
                    (
                        extrinsic_np, intrinsic_np, all_world_points,
                        all_point_colors, all_cam_to_world_mat, inference_time_ms,
                    ) = infer_vggt_and_reconstruct(
                        model, vgg_input, dtype, args.depth_conf_thresh, image_paths,
                    )
                elif method == "regimevggt":
                    (
                        extrinsic_np, intrinsic_np, all_world_points,
                        all_point_colors, all_cam_to_world_mat, inference_time_ms,
                    ) = infer_regimevggt_and_reconstruct(
                        model, vgg_input, dtype, args.depth_conf_thresh,
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")

                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                fps = len(image_paths) / (inference_time_ms / 1000.0)
                print(f"- {inference_time_ms/1000:.1f}s, {fps:.1f} FPS, {peak_vram_mb:.0f}MB VRAM")

                if not all_cam_to_world_mat or not all_world_points:
                    print(f"    no valid outputs, skip")
                    continue

                metrics = evaluate_scene_and_save(
                    scene, c2ws, first_gt_pose, selected_frame_ids,
                    all_cam_to_world_mat, all_world_points,
                    output_scene_dir, args.gt_ply_dir,
                    args.chamfer_max_dist, inference_time_ms, args.plot,
                )
                if metrics is not None:
                    all_scenes_metrics["scenes"][scene] = {
                        k: float(v) for k, v in metrics.items()
                        if k in ["chamfer_distance", "ate", "are", "rpe_rot", "rpe_trans", "inference_time_ms"]
                    }
                    all_scenes_metrics["scenes"][scene]["fps"] = float(fps)
                    all_scenes_metrics["scenes"][scene]["vram_mb"] = float(peak_vram_mb)
                    scene_fps[scene].append(float(fps))

            except Exception as e:
                print(f"- ERROR: {e}")
                import traceback
                traceback.print_exc()

            # Free CPU/GPU RAM to prevent OOM kill across scenes
            images = images_array = vgg_input = None
            all_world_points = all_point_colors = all_cam_to_world_mat = None
            extrinsic_np = intrinsic_np = None
            gc.collect()
            torch.cuda.empty_cache()

        # Summary
        vram_vals = [s["vram_mb"] for s in all_scenes_metrics["scenes"].values() if "vram_mb" in s]
        if vram_vals:
            print(f"\n  VRAM: min={min(vram_vals):.0f}MB, max={max(vram_vals):.0f}MB, mean={np.mean(vram_vals):.0f}MB")
            all_scenes_metrics["average"]["vram_min_mb"] = float(min(vram_vals))
            all_scenes_metrics["average"]["vram_max_mb"] = float(max(vram_vals))
            all_scenes_metrics["average"]["vram_mean_mb"] = float(np.mean(vram_vals))

        fps_vals = [s["fps"] for s in all_scenes_metrics["scenes"].values() if "fps" in s]
        if fps_vals:
            all_scenes_metrics["average"]["fps_mean"] = float(np.mean(fps_vals))

        compute_average_metrics_and_save(all_scenes_metrics, output_base, args.input_frame)

        # Free model before loading next
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("  All done.")
    print(f"{'='*60}")
