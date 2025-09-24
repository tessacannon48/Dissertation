# 3d_modeling.py

# Import packages

import os, sys, json, glob, time, argparse
from math import ceil
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.enums import Resampling
from rasterio.warp import reproject
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Import modules
from src.data.dataset import LidarS2Dataset
from src.model.unet import ConditionalUNet
from src.diffusion.scheduler import LinearDiffusionScheduler, CosineDiffusionScheduler
from src.diffusion.sampling import p_sample_loop_ddpm, p_sample_loop_ddim, p_sample_loop_plms

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", None)
    state = ckpt["model_state_dict"]
    return state, cfg, ckpt

def find_stats_file(s2_dir):
    cand = os.path.join(s2_dir, "s2_stats_24.pt")
    return cand if os.path.exists(cand) else None

def list_all_patch_ids(s2_dir):
    return sorted([
        os.path.basename(p).split('_')[-1]
        for p in glob.glob(os.path.join(s2_dir, "s2_patch_*"))
        if os.path.isdir(p)
    ])

def find_lidar_patch(lidar_dir, tile_id):
    cands = glob.glob(os.path.join(lidar_dir, f"*{tile_id}*.tif"))
    if not cands:
        raise FileNotFoundError(f"No LiDAR patch found for tile_id={tile_id} in {lidar_dir}")
    ones = [c for c in cands if "1m" in os.path.basename(c)]
    return (ones[0] if ones else cands[0])

def get_patch_ids_subset(s2_dir, region_ids=None, max_tiles=None, seed=42, deterministic_order=True):
    pids = list_all_patch_ids(s2_dir)
    if region_ids is not None:
        region_ids = set(region_ids)
        filtered = []
        for pid in pids:
            rj = os.path.join(s2_dir, f"s2_patch_{pid}", "region.json")
            try:
                with open(rj, "r") as f:
                    rid = json.load(f).get("region_id", None)
                if rid in region_ids:
                    filtered.append(pid)
            except Exception:
                pass
        pids = filtered
    if (max_tiles is not None) and (len(pids) > max_tiles):
        if deterministic_order:
            pids = pids[:max_tiles]
        else:
            rng = np.random.default_rng(seed)
            pids = list(rng.choice(pids, size=max_tiles, replace=False))
    return pids

def write_tif_like(ref_tif, out_path, array_2d_float32):
    with rasterio.open(ref_tif) as ref:
        prof = ref.profile.copy()
    prof.update(
        dtype="float32", count=1, compress="deflate", predictor=3, tiled=True,
        blockxsize=min(256, prof["width"]), blockysize=min(256, prof["height"]),
        BIGTIFF="IF_SAFER"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(array_2d_float32.astype(np.float32), 1)

def mosaic_average_safe(tif_list, out_path, compress=None):
    assert len(tif_list) > 0, "No input tiles to mosaic."
    srcs = [rasterio.open(fp) for fp in tif_list]
    try:
        nodatas = [s.nodata for s in srcs]
        merge_kwargs = {}
        if all(nd == nodatas[0] for nd in nodatas) and (nodatas[0] is not None):
            merge_kwargs["nodata"] = nodatas[0]
        sum_arr, transform = rio_merge(srcs, method="sum", **merge_kwargs)
        cnt_arr, _         = rio_merge(srcs, method="count", **merge_kwargs)
        denom = np.maximum(cnt_arr.astype(np.float32), 1.0)
        avg2d = (sum_arr.astype(np.float32) / denom)[0]
        ref = srcs[0]
        prof = {
            "driver": "GTiff",
            "height": int(avg2d.shape[0]),
            "width":  int(avg2d.shape[1]),
            "count":  1,
            "dtype":  "float32",
            "crs":    ref.crs,
            "transform": transform,
            "tiled": False,
        }
        if merge_kwargs.get("nodata", None) is not None:
            prof["nodata"] = merge_kwargs["nodata"]
        if compress:
            prof["compress"] = compress
        # overwrite-safe
        try:
            from rasterio.shutil import delete as rio_delete
            if os.path.exists(out_path):
                rio_delete(out_path)
        except Exception:
            if os.path.exists(out_path):
                os.remove(out_path)
        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(avg2d, 1)
    finally:
        for s in srcs:
            s.close()
    return out_path


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def subsample(arr, step):
    return arr[::step, ::step]

def plot_single_3d_surface(ax, lidar_array, title="3D LiDAR Surface Plot",
                           cmap='terrain', z_label='Elevation Deviations (m)',
                           vmin=None, vmax=None):
    # treat zeros as NaN for visualization
    lidar_array = np.where(lidar_array == 0, np.nan, lidar_array)
    h, w = lidar_array.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    ax.set_zlim(-1.5, 1.5)
    surf = ax.plot_surface(
        X, Y, lidar_array,
        cmap=cmap, alpha=0.9, edgecolor='none',
        rstride=1, cstride=1, vmin=vmin, vmax=vmax
    )
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel(z_label)
    ax.set_title(title)
    ax.view_init(elev=15, azim=70)
    return surf

def plot_all_three_3d_surfaces(gt_array, pred_array, diff_array,
                               step=4, out_path=None, plot_title="Combined 3D Plots",
                               y_gt_x=True):
    gt_s   = subsample(gt_array, step)
    pr_s   = subsample(pred_array, step)
    diff_s = subsample(diff_array, step)

    fig = plt.figure(figsize=(24, 8))
    fig.suptitle(plot_title, fontsize=16)

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    s1 = plot_single_3d_surface(ax1, gt_s,  "Ground Truth LiDAR Elevation",
                                cmap='terrain', z_label='Elevation (m)',
                                vmin=-0.1, vmax=1.1)
    fig.colorbar(s1, ax=ax1, shrink=0.5, aspect=5, label='Elevation (m)')

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    s2 = plot_single_3d_surface(ax2, pr_s,  "Predicted LiDAR Elevation",
                                cmap='terrain', z_label='Elevation (m)',
                                vmin=-0.1, vmax=1.1)
    fig.colorbar(s2, ax=ax2, shrink=0.5, aspect=5, label='Elevation (m)')

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    s3 = plot_single_3d_surface(ax3, diff_s, "Prediction - Ground Truth (Error)",
                                cmap='RdBu', z_label='Difference (m)',
                                vmin=-0.15, vmax=0.15)
    fig.colorbar(s3, ax=ax3, shrink=0.5, aspect=5, label='Difference (m)')

    # manually set limits for regions
    for a in (ax1, ax2, ax3):
        if y_gt_x:
            a.set_xlim(0, 275)
            a.set_ylim(400, 1500)
        else:
            a.set_xlim(700, 1400)
            a.set_ylim(0, 800)
        a.set_zlim(-1.5, 1.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        print(f"Saved 3D plots to {out_path}")
    plt.close(fig)

# =============================================================================
# PIPELINE
# =============================================================================
@torch.no_grad()
def run_predictions_and_mosaics(ckpt_path, config_yaml, out_dir,
                                sampler_name="ddpm", batch_size=8, num_workers=4, device="cuda",
                                region_ids=None, max_tiles=None, seed=42, deterministic_order=True):
    """Runs per-tile predictions and builds pred/gt mosaics; returns paths."""
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    state, cfg_from_ckpt, _ = load_checkpoint(ckpt_path, device)
    if config_yaml:
        with open(config_yaml, "r") as f:
            cfg = yaml.safe_load(f)
            print("Loaded config from", config_yaml)
    else:
        cfg = cfg_from_ckpt

    s2_dir       = "/cs/student/projects2/aisd/2024/tcannon/dissertation/Dissertation/input_data/s2_patches"
    lidar_dir    = "/cs/student/projects2/aisd/2024/tcannon/dissertation/Dissertation/input_data/lidar_patches"
    context_k    = cfg["training"]["context_k"]
    noise_sched  = cfg["training"]["noise_schedule"]
    timesteps    = cfg["training"]["timesteps"]
    base_channels = cfg["model"]["base_channels"]
    embed_dim     = cfg["model"]["embed_dim"]
    unet_depth    = cfg["model"]["unet_depth"]
    attention_variant = cfg["model"]["attention_variant"]

    print("\n=== Inference Config ===")
    print(f"S2 dir:    {s2_dir}")
    print(f"LiDAR dir: {lidar_dir}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Sampler: {sampler_name} | Timesteps: {timesteps} | Schedule: {noise_sched}")
    print(f"Model: UNet depth={unet_depth}, base={base_channels}, attn={attention_variant}, embed_dim={embed_dim}")
    print(f"Context k: {context_k}\n")

    scheduler = LinearDiffusionScheduler(timesteps=timesteps, device=device) if noise_sched == "linear" \
        else CosineDiffusionScheduler(timesteps=timesteps, device=device)

    model = ConditionalUNet(
        in_channels=1,
        cond_channels=4 * context_k,
        attr_dim=8 * context_k,
        base_channels=base_channels,
        embed_dim=embed_dim,
        unet_depth=unet_depth,
        attention_variant=attention_variant
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    stats_path = find_stats_file(s2_dir)
    if stats_path:
        stats = torch.load(stats_path, map_location="cpu")
        s2_means, s2_stds = stats["mean"], stats["std"]
    else:
        raise FileNotFoundError(f"Could not find S2 stats file at {os.path.join(s2_dir, 's2_stats_24.pt')}.")

    subset_pids = get_patch_ids_subset(
        s2_dir=s2_dir, region_ids=region_ids, max_tiles=max_tiles,
        seed=seed, deterministic_order=deterministic_order
    )
    print(f"Using {len(subset_pids)} patch(es). Example: {subset_pids[:5]} ...")

    dataset = LidarS2Dataset(
        lidar_dir=lidar_dir, s2_dir=s2_dir,
        s2_means=s2_means, s2_stds=s2_stds,
        context_k=context_k, randomize_context=False,
        augment=False, debug=False, split_pids=subset_pids, split="val"
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    samplers = {
        "ddpm": lambda m, s, c, a, d: p_sample_loop_ddpm(m, scheduler, s, c, a, d),
        "ddim": lambda m, s, c, a, d: p_sample_loop_ddim(m, scheduler, s, c, a, d),
        "plms": lambda m, s, c, a, d: p_sample_loop_plms(m, scheduler, s, c, a, d),
    }
    if sampler_name not in samplers:
        raise ValueError(f"Unknown sampler: {sampler_name}")
    sampler = samplers[sampler_name]

    pred_tiles_dir = os.path.join(out_dir, "pred_tiles")
    os.makedirs(pred_tiles_dir, exist_ok=True)

    pred_tifs, gt_tifs = [], []
    start = time.perf_counter()
    for batch in tqdm(loader, desc="Predicting tiles"):
        s2     = batch["s2"].to(device)
        attrs  = batch["attrs"].to(device)
        lidar  = batch["lidar"].to(device)
        tile_ids_batch = batch["tile_id"]

        pred = sampler(model, lidar.shape, s2, attrs, device).float().cpu().numpy()

        B = pred.shape[0]
        for i in range(B):
            tile_id = tile_ids_batch[i]
            gt_lidar_tif = find_lidar_patch(lidar_dir, tile_id)
            out_tif = os.path.join(pred_tiles_dir, f"pred_{tile_id}.tif")
            write_tif_like(gt_lidar_tif, out_tif, pred[i, 0])
            pred_tifs.append(out_tif)
            gt_tifs.append(gt_lidar_tif)

    elapsed = time.perf_counter() - start
    print(f"\nFinished per-tile predictions for {len(pred_tifs)} tiles in {elapsed/60:.1f} min.")

    pred_mosaic_path = os.path.join(out_dir, "pred_mosaic.tif")
    print("Mosaicking predictions →", pred_mosaic_path)
    mosaic_average_safe(pred_tifs, pred_mosaic_path, compress="deflate")

    gt_mosaic_path = os.path.join(out_dir, "gt_mosaic.tif")
    print("Mosaicking ground truth →", gt_mosaic_path)
    mosaic_average_safe(sorted(list(set(gt_tifs))), gt_mosaic_path, compress="deflate")

    return pred_mosaic_path, gt_mosaic_path

def align_and_save_diff(pred_mosaic_path, gt_mosaic_path, out_dir):
    """Ensures alignment of pred to GT, computes and saves diff."""
    with rasterio.open(gt_mosaic_path) as g, rasterio.open(pred_mosaic_path) as p:
        gt_array = g.read(1).astype(np.float32)
        pred_array = p.read(1).astype(np.float32)

        # Align if shapes/transforms differ
        if (gt_array.shape != pred_array.shape) or (g.transform != p.transform):
            print("Aligning prediction mosaic to ground truth grid...")
            pred_aligned = np.zeros_like(gt_array, dtype=np.float32)
            reproject(
                source=pred_array, destination=pred_aligned,
                src_transform=p.transform, src_crs=p.crs,
                dst_transform=g.transform, dst_crs=g.crs,
                resampling=Resampling.bilinear
            )
            pred_array = pred_aligned

        diff_array = pred_array - gt_array

        diff_path = os.path.join(out_dir, "diff_pred_minus_gt.tif")
        prof = g.profile.copy()
        prof.update(dtype="float32", count=1, compress="deflate")
        with rasterio.open(diff_path, "w", **prof) as dst:
            dst.write(diff_array.astype(np.float32), 1)

        print("Wrote diff raster →", diff_path)

    return gt_array, pred_array, diff_array, diff_path

def make_all_plots(gt_array, pred_array, diff_array, out_dir, step_combined=50, step_split=4):
    """Creates combined 3-panel plot and the two split 3-panel plots."""
    # Combined
    combined_path = os.path.join(out_dir, "combined_3d_plots.png")
    plot_all_three_3d_surfaces(
        gt_array=gt_array, pred_array=pred_array, diff_array=diff_array,
        step=step_combined, out_path=combined_path,
        plot_title="LiDAR: GT vs Pred vs Error (Combined)", y_gt_x=True
    )

    # Split masks
    h, w = gt_array.shape
    y_coords, x_coords = np.indices((h, w))
    mask_y_gt_x = y_coords > x_coords
    mask_y_lt_x = y_coords < x_coords

    gt_1   = np.where(mask_y_gt_x, gt_array,   np.nan)
    pred_1 = np.where(mask_y_gt_x, pred_array, np.nan)
    diff_1 = np.where(mask_y_gt_x, diff_array, np.nan)

    gt_2   = np.where(mask_y_lt_x, gt_array,   np.nan)
    pred_2 = np.where(mask_y_lt_x, pred_array, np.nan)
    diff_2 = np.where(mask_y_lt_x, diff_array, np.nan)

    # Region 1: y > x
    p1 = os.path.join(out_dir, "3d_plots_region_y_gt_x.png")
    plot_all_three_3d_surfaces(
        gt_array=gt_1, pred_array=pred_1, diff_array=diff_1,
        step=step_split, out_path=p1,
        plot_title="LiDAR Maps for Validation Region 4.1", y_gt_x=True
    )

    # Region 2: y < x
    p2 = os.path.join(out_dir, "3d_plots_region_y_lt_x.png")
    plot_all_three_3d_surfaces(
        gt_array=gt_2, pred_array=pred_2, diff_array=diff_2,
        step=step_split, out_path=p2,
        plot_title="LiDAR Maps for Validation Region 4.2", y_gt_x=False
    )

    return {"combined": combined_path, "split_gtx": p1, "split_ltx": p2}


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Unified 3D modeling: predict, mosaic, diff, plot (combined + split).")
    ap.add_argument("--ckpt", type=str, required=False, help="Path to model checkpoint (.pth). Required unless --skip-predict.")
    ap.add_argument("--config", type=str, default=None, help="Optional YAML config. If not given, use checkpoint-embedded config.")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory.")
    ap.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim", "plms"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--region-ids", type=int, nargs="*", default=None)
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic-order", action="store_true", default=True)
    ap.add_argument("--skip-predict", action="store_true", help="Skip prediction; expect pred_mosaic.tif and gt_mosaic.tif in out-dir.")
    ap.add_argument("--combined-step", type=int, default=50, help="Subsample step for combined plot.")
    ap.add_argument("--split-step", type=int, default=4, help="Subsample step for split plots.")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.skip_predict:
        pred_mosaic_path = os.path.join(args.out_dir, "pred_mosaic.tif")
        gt_mosaic_path   = os.path.join(args.out_dir, "gt_mosaic.tif")
        if not (os.path.exists(pred_mosaic_path) and os.path.exists(gt_mosaic_path)):
            raise FileNotFoundError("With --skip-predict, expected pred_mosaic.tif and gt_mosaic.tif in out-dir.")
        print("Skipping prediction. Using existing mosaics.")
    else:
        if not args.ckpt:
            raise ValueError("ckpt is required unless --skip-predict is set.")
        pred_mosaic_path, gt_mosaic_path = run_predictions_and_mosaics(
            ckpt_path=args.ckpt, config_yaml=args.config, out_dir=args.out_dir,
            sampler_name=args.sampler, batch_size=args.batch_size, num_workers=args.num_workers,
            device=args.device, region_ids=args.region_ids, max_tiles=args.max_tiles,
            seed=args.seed, deterministic_order=args.deterministic_order
        )

    # Align + diff
    gt_array, pred_array, diff_array, diff_path = align_and_save_diff(
        pred_mosaic_path=pred_mosaic_path,
        gt_mosaic_path=gt_mosaic_path,
        out_dir=args.out_dir
    )

    # Plots (combined + split)
    plot_paths = make_all_plots(
        gt_array=gt_array, pred_array=pred_array, diff_array=diff_array,
        out_dir=args.out_dir, step_combined=args.combined_step, step_split=args.split_step
    )

    print("\nDone.")
    print("Outputs:")
    print(f"  GT mosaic:    {gt_mosaic_path}")
    print(f"  Pred mosaic:  {pred_mosaic_path}")
    print(f"  Diff raster:  {diff_path}")
    print(f"  Combined plot:{plot_paths['combined']}")
    print(f"  Split y>x:    {plot_paths['split_gtx']}")
    print(f"  Split y<x:    {plot_paths['split_ltx']}")

if __name__ == "__main__":
    main()
