import glob # type: ignore
import numpy as np
import torch
import rasterio
from tqdm import tqdm
import os

# =============================================================================
# DATA PREPOCESSING METHODS
# =============================================================================

def compute_s2_mean_std_multi(s2_root, num_times=6, num_bands=4, filenames=None):
    """
    Compute dataset-level mean/std for multi-temporal S2 sets saved as:
      s2_root/s2_patch_{id}/t{i}.tif,  i=0..num_times-1
    Returns tensors of shape [num_times*num_bands] == [24].
    """
    if filenames is None:
        filenames = [f"t{i}.tif" for i in range(num_times)]

    group_dirs = sorted([d for d in glob.glob(os.path.join(s2_root, "s2_patch_*"))
                         if os.path.isdir(d)])

    C = num_times * num_bands
    sums   = torch.zeros(C, dtype=torch.float64)
    sums2  = torch.zeros(C, dtype=torch.float64)
    counts = torch.zeros(C, dtype=torch.float64)

    for gdir in tqdm(group_dirs, desc="Computing S2 stats (6×4)"):
        for ti, fname in enumerate(filenames):
            fp = os.path.join(gdir, fname)
            if not os.path.exists(fp):
                continue
            with rasterio.open(fp) as src:
                arr = src.read()[:num_bands].astype(np.float32)  # [4,h,w]
            arr = torch.from_numpy(arr).reshape(num_bands, -1)   # [4, N]

            # robust to NaNs/Infs
            finite = torch.isfinite(arr)
            safe   = torch.where(finite, arr, torch.zeros_like(arr))

            idx0, idx1 = ti*num_bands, (ti+1)*num_bands
            sums[idx0:idx1]  += safe.sum(dim=1, dtype=torch.float64)
            sums2[idx0:idx1] += (safe**2).sum(dim=1, dtype=torch.float64)
            counts[idx0:idx1]+= finite.sum(dim=1, dtype=torch.float64)

    # avoid div-by-zero
    counts = torch.clamp(counts, min=1.0)
    mean = (sums / counts).to(torch.float32)
    var  = (sums2 / counts) - (mean.to(torch.float64)**2)
    std  = torch.sqrt(torch.clamp(var, min=1e-12)).to(torch.float32)

    return mean, std