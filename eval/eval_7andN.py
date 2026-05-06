import os
import sys

# Ensure project root is on sys.path for absolute imports like `vggt.*`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import time
import torch
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from collections import defaultdict
import torchvision.transforms as transforms


# ── RegimeVGGT: helper functions for our methods ──────────────────────

def _heads_to_preds(model, output_list, psi, imgs_tensor):
    """Call model heads on aggregator output -> standard preds dict."""
    preds = {}
    pts3d, pts3d_conf = model.point_head(output_list, images=imgs_tensor, patch_start_idx=psi)
    preds["world_points"] = pts3d
    preds["world_points_conf"] = pts3d_conf

    pose_enc_list = model.camera_head(output_list)
    preds["pose_enc"] = pose_enc_list[-1]

    depth, depth_conf = model.depth_head(output_list, images=imgs_tensor, patch_start_idx=psi)
    preds["depth"] = depth
    preds["depth_conf"] = depth_conf

    preds["images"] = imgs_tensor
    return preds


def _run_regimevggt(model, imgs_tensor, args):
    """Hybrid: Line A merge + Line B phase-shift subsample (+ optional anchor)."""
    from methods.regimevggt import run_aggregator_regimevggt
    imgs = imgs_tensor.unsqueeze(0) if imgs_tensor.dim() == 4 else imgs_tensor
    output_list, psi = run_aggregator_regimevggt(
        model, imgs,
        shallow_merge_ratio=args.shallow_merge_ratio,
        merge_ratio=args.merge_ratio_ours,
        deep_merge_ratio=args.deep_merge_ratio,
        middle_merge_start=args.middle_merge_start,
        deep_merge_start=args.deep_merge_start,
        cache_band_starts=(args.cache_band_starts if args.cache_band_starts else None),
        no_cache_layers=(args.no_cache_layers if args.no_cache_layers else None),
        importance_method=args.importance,
        protect_last=(not args.no_protect_last),
        sigma_sub=args.hyb_sigma_sub,
        sigma_shallow=args.hyb_sigma_shallow,
        sigma_deep=args.hyb_sigma_deep,
        use_phase_shift=args.hyb_use_phase_shift,
        use_avggt_mean_fill=args.hyb_avggt_mean_fill,
        use_avggt_full=args.hyb_avggt_full,
        protect_middle=args.hyb_protect_middle,
        middle_range=tuple(args.hyb_middle_range),
        anchor_frame_idx=(args.hyb_anchor_frame
                          if args.hyb_anchor_frame >= 0 else None),
    )
    return _heads_to_preds(model, output_list, psi, imgs)


# ── Arg parser ──────────────────────────────────────────────────────

def get_args_parser():
    parser = argparse.ArgumentParser("3D Reconstruction evaluation", add_help=False)
    parser.add_argument(
        "--ckpt_path", type=str,
        default=osp.join(ROOT_DIR, "..", "ckpt", "model_tracker_fixed_e20.pt"),
        help="checkpoint path",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_name", type=str, default="VGGT")
    parser.add_argument("--conf_thresh", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="./results/main")
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--revisit", type=int, default=1)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--use_proj", action="store_true")
    parser.add_argument("--merging", type=int, default=0)
    parser.add_argument("--merge_ratio", type=float, default=0.9)
    parser.add_argument("--kf", type=int, default=10, help="keyframe stride (paper uses 3 or 10)")

    # --- RegimeVGGT additions ---
    parser.add_argument("--method", type=str, default="regimevggt",
                        choices=["baseline", "regimevggt"],
                        help="'baseline' is uncompressed VGGT* for reference.")
    parser.add_argument("--datasets", type=str, nargs="*",
                        default=["7scenes", "NRGBD"])
    parser.add_argument("--7scenes_root", type=str, default=None,
                        help="Root of the 7Scenes dataset (required for "
                             "--datasets 7scenes).")
    parser.add_argument("--nrgbd_root", type=str, default=None,
                        help="Root of the NRGBD dataset (required for "
                             "--datasets NRGBD).")
    parser.add_argument("--skip_layers", type=int, nargs="*",
                        default=[])
    parser.add_argument("--g2f_layers", type=int, nargs="*",
                        default=None,
                        help="layers to run Global-to-Frame (per-frame global, no cross-view). "
                             "Overrides skip_layers for those layers.")
    parser.add_argument("--shallow_merge_ratio", type=float, default=0.99,
                        help="Ratio for layers < middle_merge_start (default 0.99)")
    parser.add_argument("--merge_ratio_ours", type=float, default=0.95,
                        help="Middle-band ratio (default 0.95)")
    parser.add_argument("--deep_merge_ratio", type=float, default=0.99,
                        help="Ratio for layers >= deep_merge_start (default 0.99)")
    parser.add_argument("--middle_merge_start", type=int, default=10,
                        help="First layer of middle band (shallow->middle boundary)")
    parser.add_argument("--deep_merge_start", type=int, default=18,
                        help="First layer to apply deep_merge_ratio")
    parser.add_argument("--no_protect_last", action="store_true",
                        help="Remove L23 hard-coded full-no-merge protection.")
    parser.add_argument("--global_start_layer", type=int, default=10,
                        help="first layer to enable global attention (shallow_skip_global method)")
    parser.add_argument("--cache_interval", type=int, default=4,
                        help="recompute merge indices every N layers (importance_merging)")
    parser.add_argument("--no_cache_layers", type=int, nargs="*", default=[],
                        help="Layer indices where merge cache is force-disabled.")
    parser.add_argument("--cache_band_starts", type=int, nargs="*", default=[],
                        help="Band-based caching: one cache compute per band. "
                             "Overrides cache_interval when set.")
    parser.add_argument("--importance", type=str, default="dino_attn",
                        choices=["sobel", "dino_attn", "token_norm", "frame_attn", "geom_probe_l0", "geom_probe_l9"],
                        help="token importance scoring method for importance_merging")
    parser.add_argument("--merge_strategy", type=str, default="bipartite",
                        choices=["bipartite", "importance", "random", "spatial"],
                        help="token merge strategy for importance_merging")
    # Hybrid (Line A merge + Line B phase-shift subsample) flags.
    # ── Line B (phase_shift) flags ──
    parser.add_argument("--ps_sigma_s", type=float, default=1.5,
                        help="phase_shift: shallow band sigma (default 1.5).")
    parser.add_argument("--ps_sigma_m", type=float, default=1.5,
                        help="phase_shift: middle band sigma (default 1.5).")
    parser.add_argument("--ps_sigma_d", type=float, default=1.5,
                        help="phase_shift: deep band sigma (default 1.5).")
    parser.add_argument("--ps_band_boundaries", type=int, nargs=3, default=[0, 10, 18],
                        help="phase_shift: (b0, b1, b2) band starts; layers [b0,b1) shallow, "
                             "[b1,b2) middle, [b2,depth) deep.")
    parser.add_argument("--ps_use_phase_shift", action="store_true",
                        help="phase_shift: enable phase shift (vs fixed grid).")
    parser.add_argument("--ps_anchor_frame", type=int, default=-1,
                        help="phase_shift: keep ALL K/V on this frame (-1 = no anchor).")
    parser.add_argument("--ps_avggt_mean_fill", action="store_true",
                        help="phase_shift: AVGGT mean-fill on dropped K/V.")
    parser.add_argument("--ps_avggt_full", action="store_true",
                        help="phase_shift: full AVGGT (selected + diagonal + mean-fill).")

    parser.add_argument("--hyb_use_phase_shift", action="store_true",
                        help="regimevggt: enable phase-shift on merged K/V.")
    parser.add_argument("--hyb_sigma_sub", type=float, default=2.0,
                        help="regimevggt: per-band sigma fallback (middle band uses this).")
    parser.add_argument("--hyb_sigma_shallow", type=float, default=None,
                        help="regimevggt: shallow band sigma; defaults to hyb_sigma_sub.")
    parser.add_argument("--hyb_sigma_deep", type=float, default=None,
                        help="regimevggt: deep band sigma; defaults to hyb_sigma_sub.")
    parser.add_argument("--hyb_avggt_mean_fill", action="store_true",
                        help="regimevggt: AVGGT mean-fill on merged K/V (no diagonal).")
    parser.add_argument("--hyb_avggt_full", action="store_true",
                        help="regimevggt: full AVGGT (selected + diagonal + mean-fill).")
    parser.add_argument("--hyb_protect_middle", action="store_true",
                        help="regimevggt: middle-band layers do merge only (no subsample).")
    parser.add_argument("--hyb_middle_range", type=int, nargs=2, default=[10, 18],
                        help="regimevggt: [start, end) range treated as middle band.")
    parser.add_argument("--hyb_anchor_frame", type=int, default=-1,
                        help="regimevggt: keep ALL merged tokens whose parent lies in "
                             "this frame (Line B C2 anchor). -1 = no anchor.")
    return parser


# ── Main ────────────────────────────────────────────────────────────

def main(args):
    from data import SevenScenes, NRGBD
    from utils import accuracy, completion

    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    elif args.size == 518:
        resolution = (518, 392)
    else:
        raise NotImplementedError

    datasets_all = {}
    if "7scenes" in args.datasets:
        datasets_all["7scenes"] = SevenScenes(
            split="test",
            ROOT=getattr(args, '7scenes_root', None),
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=args.kf,
        )
    if "NRGBD" in args.datasets:
        datasets_all["NRGBD"] = NRGBD(
            split="test",
            ROOT=args.nrgbd_root,
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=args.kf,
        )

    device = args.device
    model_name = args.model_name

    from utils.criterion import Regr3D_t_ScaleShiftInv, L21

    # Load model.
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    if args.method == "regimevggt":
        # RegimeVGGT drives merge from the aggregator wrapper, not via the
        # VGGT.merging path - keep merging=0 here.
        model = VGGT(merging=0, merge_ratio=args.merge_ratio_ours, enable_point=True)
    else:
        model = VGGT(merging=24, merge_ratio=0.0, enable_point=True)

    ckpt = torch.load(args.ckpt_path, map_location="cpu", mmap=False)
    model.load_state_dict(ckpt, strict=False)
    del ckpt
    model = model.cuda().eval().to(torch.bfloat16)

    method_dir = osp.join(args.output_dir, args.method, f"{args.kf}")
    os.makedirs(method_dir, exist_ok=True)

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    with torch.no_grad():
        for name_data, dataset in datasets_all.items():
            save_path = osp.join(method_dir, name_data)
            os.makedirs(save_path, exist_ok=True)
            log_file = osp.join(save_path, "logs.txt")

            acc_all = 0
            acc_all_med = 0
            comp_all = 0
            comp_all_med = 0
            nc1_all = 0
            nc1_all_med = 0
            nc2_all = 0
            nc2_all_med = 0
            scene_infer_times = defaultdict(list)

            for data_idx in tqdm(range(len(dataset))):
                batch = default_collate([dataset[data_idx]])
                ignore_keys = set(
                    [
                        "depthmap",
                        "dataset",
                        "label",
                        "instance",
                        "idx",
                        "true_shape",
                        "rng",
                    ]
                )
                for view in batch:
                    for name in view.keys():
                        if name in ignore_keys:
                            continue
                        if isinstance(view[name], tuple) or isinstance(
                            view[name], list
                        ):
                            view[name] = [
                                x.to(device, non_blocking=True) for x in view[name]
                            ]
                        else:
                            view[name] = view[name].to(device, non_blocking=True)

                pts_all = []
                pts_gt_all = []
                images_all = []
                masks_all = []
                conf_all = []
                in_camera1 = None

                dtype = (
                    torch.bfloat16
                    if torch.cuda.get_device_capability()[0] >= 8
                    else torch.float16
                )
                with torch.cuda.amp.autocast(dtype=dtype):
                    if isinstance(batch, dict) and "img" in batch:
                        batch["img"] = (batch["img"] + 1.0) / 2.0
                    elif isinstance(batch, list) and all(
                        isinstance(v, dict) and "img" in v for v in batch
                    ):
                        for view in batch:
                            view["img"] = (view["img"] + 1.0) / 2.0
                        imgs_tensor = torch.cat([v["img"] for v in batch], dim=0)

                with torch.cuda.amp.autocast(dtype=dtype):
                    with torch.no_grad():
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                        start = time.time()

                        # ── Method dispatch ──
                        if args.method == "baseline":
                            preds = model(imgs_tensor)
                        elif args.method == "regimevggt":
                            preds = _run_regimevggt(model, imgs_tensor, args)
                        else:
                            raise ValueError(f"Unknown method: {args.method}")

                        torch.cuda.synchronize()
                        end = time.time()
                        inference_time_ms = (end - start) * 1000
                        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                        print(f"Inference time: {inference_time_ms:.2f}ms, Peak VRAM: {peak_vram_mb:.1f}MB")

                    # Wrap model outputs per-view to align with batch later
                    predictions = preds
                    views = batch
                    if "pose_enc" in predictions:
                        B, S = predictions["pose_enc"].shape[:2]
                    elif "world_points" in predictions:
                        B, S = predictions["world_points"].shape[:2]
                    else:
                        raise KeyError(
                            "predictions is missing a key to infer sequence length"
                        )

                    ress = []
                    for s in range(S):
                        res = {
                            "pts3d_in_other_view": predictions["world_points"][:, s],
                            "conf": predictions["world_points_conf"][:, s],
                            "depth": predictions["depth"][:, s],
                            "depth_conf": predictions["depth_conf"][:, s],
                            "camera_pose": predictions["pose_enc"][:, s, :],
                        }
                        if (
                            isinstance(views, list)
                            and s < len(views)
                            and "valid_mask" in views[s]
                        ):
                            res["valid_mask"] = views[s]["valid_mask"]
                        if "track" in predictions:
                            res.update(
                                {
                                    "track": predictions["track"][:, s],
                                    "vis": (
                                        predictions.get("vis", None)[:, s]
                                        if "vis" in predictions
                                        else None
                                    ),
                                    "track_conf": (
                                        predictions.get("conf", None)[:, s]
                                        if "conf" in predictions
                                        else None
                                    ),
                                }
                            )
                        ress.append(res)

                    preds = ress

                    valid_length = len(preds) // args.revisit
                    if args.revisit > 1:
                        preds = preds[-valid_length:]
                        batch = batch[-valid_length:]

                    # Evaluation
                    print(f"Evaluation for {name_data} {data_idx+1}/{len(dataset)}")
                    gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
                        criterion.get_all_pts3d_t(batch, preds)
                    )

                    in_camera1 = None
                    pts_all = []
                    pts_gt_all = []
                    images_all = []
                    masks_all = []
                    conf_all = []

                    for j, view in enumerate(batch):
                        if in_camera1 is None:
                            in_camera1 = view["camera_pose"][0].cpu()

                        image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                        mask = view["valid_mask"].cpu().numpy()[0]

                        pts = pred_pts[j].cpu().numpy()[0]
                        conf = preds[j]["conf"].cpu().data.numpy()[0]

                        pts_gt = gt_pts[j].detach().cpu().numpy()[0]

                        H, W = image.shape[:2]
                        cx = W // 2
                        cy = H // 2
                        l, t = cx - 112, cy - 112
                        r, b = cx + 112, cy + 112
                        image = image[t:b, l:r]
                        mask = mask[t:b, l:r]
                        pts = pts[t:b, l:r]
                        pts_gt = pts_gt[t:b, l:r]

                        images_all.append(image[None, ...])
                        pts_all.append(pts[None, ...])
                        pts_gt_all.append(pts_gt[None, ...])
                        masks_all.append(mask[None, ...])
                        conf_all.append(conf[None, ...])

                images_all = np.concatenate(images_all, axis=0)
                pts_all = np.concatenate(pts_all, axis=0)
                pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                masks_all = np.concatenate(masks_all, axis=0)

                scene_id = view["label"][0].rsplit("/", 1)[0]
                try:
                    scene_infer_times[scene_id].append(float(inference_time_ms))
                except Exception:
                    pass

                save_params = {}
                save_params["images_all"] = images_all
                save_params["pts_all"] = pts_all
                save_params["pts_gt_all"] = pts_gt_all
                save_params["masks_all"] = masks_all

                pts_all_masked = pts_all[masks_all > 0]
                pts_gt_all_masked = pts_gt_all[masks_all > 0]
                images_all_masked = images_all[masks_all > 0]

                mask = np.isfinite(pts_all_masked)
                pts_all_masked = pts_all_masked[mask]

                mask_gt = np.isfinite(pts_gt_all_masked)
                pts_gt_all_masked = pts_gt_all_masked[mask_gt]
                images_all_masked = images_all_masked[mask]

                pts_all_masked = pts_all_masked.reshape(-1, 3)
                pts_gt_all_masked = pts_gt_all_masked.reshape(-1, 3)
                images_all_masked = images_all_masked.reshape(-1, 3)

                if pts_all_masked.shape[0] > 999999:
                    sample_indices = np.random.choice(
                        pts_all_masked.shape[0], 999999, replace=False
                    )
                    pts_all_masked = pts_all_masked[sample_indices]
                    images_all_masked = images_all_masked[sample_indices]

                if pts_gt_all_masked.shape[0] > 999999:
                    sample_indices_gt = np.random.choice(
                        pts_gt_all_masked.shape[0], 999999, replace=False
                    )
                    pts_gt_all_masked = pts_gt_all_masked[sample_indices_gt]

                if args.use_proj:

                    def umeyama_alignment(
                        src: np.ndarray, dst: np.ndarray, with_scale: bool = True
                    ):
                        assert src.shape == dst.shape
                        N, dim = src.shape
                        mu_src = src.mean(axis=0)
                        mu_dst = dst.mean(axis=0)
                        src_c = src - mu_src
                        dst_c = dst - mu_dst
                        Sigma = dst_c.T @ src_c / N
                        U, D, Vt = np.linalg.svd(Sigma)
                        S = np.eye(dim)
                        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
                            S[-1, -1] = -1
                        R = U @ S @ Vt
                        if with_scale:
                            var_src = (src_c**2).sum() / N
                            s = (D * S.diagonal()).sum() / var_src
                        else:
                            s = 1.0
                        t = mu_dst - s * R @ mu_src
                        return s, R, t

                    pts_all_masked = pts_all_masked.reshape(-1, 3)
                    pts_gt_all_masked = pts_gt_all_masked.reshape(-1, 3)
                    s, R, t = umeyama_alignment(
                        pts_all_masked, pts_gt_all_masked, with_scale=True
                    )
                    pts_all_aligned = (s * (R @ pts_all_masked.T)).T + t
                    pts_all_masked = pts_all_aligned

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts_all_masked)
                pcd.colors = o3d.utility.Vector3dVector(images_all_masked)

                pcd_gt = o3d.geometry.PointCloud()
                pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_all_masked)
                pcd_gt.colors = o3d.utility.Vector3dVector(images_all_masked)

                trans_init = np.eye(4)

                threshold = 0.1
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    pcd,
                    pcd_gt,
                    threshold,
                    trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                )

                transformation = reg_p2p.transformation

                pcd = pcd.transform(transformation)
                pcd.estimate_normals()
                pcd_gt.estimate_normals()

                gt_normal = np.asarray(pcd_gt.normals)
                pred_normal = np.asarray(pcd.normals)

                acc, acc_med, nc1, nc1_med = accuracy(
                    pcd_gt.points, pcd.points, gt_normal, pred_normal
                )
                comp, comp_med, nc2, nc2_med = completion(
                    pcd_gt.points, pcd.points, gt_normal, pred_normal
                )
                log_line = (
                    f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2}"
                    f" - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
                    f", Time_ms: {inference_time_ms:.2f}, VRAM_MB: {peak_vram_mb:.1f}"
                )
                print(log_line)
                print(log_line, file=open(log_file, "a"))

                acc_all += acc
                comp_all += comp
                nc1_all += nc1
                nc2_all += nc2

                acc_all_med += acc_med
                comp_all_med += comp_med
                nc1_all_med += nc1_med
                nc2_all_med += nc2_med

                torch.cuda.empty_cache()

            # Summary
            to_write = ""
            if os.path.exists(osp.join(save_path, "logs.txt")):
                with open(osp.join(save_path, "logs.txt"), "r") as f_sub:
                    to_write += f_sub.read()

            with open(osp.join(save_path, f"logs_all.txt"), "w") as f:
                log_data = to_write
                metrics = defaultdict(list)
                for line in log_data.strip().split("\n"):
                    match = regex.match(line)
                    if match:
                        data = match.groupdict()
                        for key, value in data.items():
                            if key == "scene_id" or value is None:
                                continue
                            metrics[key].append(float(value))
                        metrics["nc"].append(
                            (float(data["nc1"]) + float(data["nc2"])) / 2
                        )
                        metrics["nc_med"].append(
                            (float(data["nc1_med"]) + float(data["nc2_med"])) / 2
                        )
                mean_metrics = {
                    metric: sum(values) / len(values)
                    for metric, values in metrics.items()
                }

                c_name = "mean"
                print_str = f"{c_name.ljust(20)}: "
                for m_name in mean_metrics:
                    print_num = np.mean(mean_metrics[m_name])
                    print_str = print_str + f"{m_name}: {print_num:.3f} | "
                # VRAM summary: min/max/mean
                if "vram_mb" in metrics and metrics["vram_mb"]:
                    vram_vals = metrics["vram_mb"]
                    print_str += (
                        f"VRAM_min: {min(vram_vals):.1f} | "
                        f"VRAM_max: {max(vram_vals):.1f} | "
                        f"VRAM_mean: {np.mean(vram_vals):.1f} | "
                    )
                print_str = print_str + "\n"
                time_lines = []
                for sid, times in scene_infer_times.items():
                    if len(times) > 0:
                        time_lines.append(
                            f"Idx: {sid}, Time_avg_ms: {np.mean(times):.2f}"
                        )
                time_block = "\n".join(time_lines) + (
                    "\n" if len(time_lines) > 0 else ""
                )

                f.write(to_write + time_block + print_str)


from collections import defaultdict
import re

pattern = r"""
    Idx:\s*(?P<scene_id>[^,]+),\s*
    Acc:\s*(?P<acc>[^,]+),\s*
    Comp:\s*(?P<comp>[^,]+),\s*
    NC1:\s*(?P<nc1>[^,]+),\s*
    NC2:\s*(?P<nc2>[^,]+)\s*-\s*
    Acc_med:\s*(?P<acc_med>[^,]+),\s*
    Compc_med:\s*(?P<comp_med>[^,]+),\s*
    NC1c_med:\s*(?P<nc1_med>[^,]+),\s*
    NC2c_med:\s*(?P<nc2_med>[^,]+)
    (?:,\s*Time_ms:\s*(?P<time_ms>[^\s,]+))?
    (?:,\s*VRAM_MB:\s*(?P<vram_mb>[^\s]+))?
"""

regex = re.compile(pattern, re.VERBOSE)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
