import glob # type: ignore
import random # type: ignore
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import rasterio
from tqdm import tqdm
import os
import json
import datetime as _dt
import hashlib


# =============================================================================
# DATASET
# =============================================================================

class LidarS2Dataset(Dataset):
    """
    Returns:
      dict {
        lidar: [1, H, W],      # the data band (RANSAC residuals)
        s2:    [k×4, Hc, Wc],  # k × (R,G,B,NIR)
        attrs: [k×8],          # k × (cloud[1] + sun[3] + view[3] + age_days[1])
        mask:  [H, W],         # validity mask from LiDAR channel 1
        chosen_ids: [k],       # Indices of the S2 patches used for conditioning
      }
    """

    def __init__(self, lidar_dir, s2_dir, s2_means, s2_stds,
             context_k=1, randomize_context=True, augment=True,
             debug=False, target_s2_hw=(256, 256), ref_date="2024-04-26",
             split_pids=None, split="train"):
        super().__init__()
        self.lidar_dir = lidar_dir
        self.s2_dir = s2_dir
        self.s2_means = s2_means.float().view(-1)
        self.s2_stds = s2_stds.float().view(-1)
        self.augment = augment
        self.target_s2_hw = target_s2_hw
        self.context_k = context_k
        self.randomize_context = randomize_context
        self.max_s2 = 6
        self.ref_date = _dt.date.fromisoformat(str(ref_date)[:10])
        self.split = split

        # Load all patch paths based on the provided IDs
        if split_pids is None:
            all_lidar_paths = sorted(glob.glob(os.path.join(lidar_dir, "lidar_patch_*.tif")))
            lidar_paths_to_load = all_lidar_paths
        else:
            lidar_paths_to_load = [os.path.join(lidar_dir, f"lidar_patch_{pid}.tif") for pid in split_pids]

        if debug:
            print("DEBUG MODE: Using a subset of 100 samples.")
            lidar_paths_to_load = lidar_paths_to_load[:100]

        # Store only the file paths and pids, not the data itself
        self.samples = []
        for lidar_path in lidar_paths_to_load:
            pid = self._extract_id(lidar_path)
            s2_group_dir = os.path.join(self.s2_dir, f"s2_patch_{pid}")
            # Ensure all S2 files for the patch exist
            available_ids = [i for i in range(self.max_s2)
                 if os.path.exists(os.path.join(s2_group_dir, f"t{i}.tif"))]

            if len(available_ids) >= self.context_k:
                self.samples.append({
                    "lidar_path": lidar_path,
                    "s2_group_dir": s2_group_dir,
                    "tile_id": pid,
                    "available_ids": available_ids
                })

        self.num_samples = len(self.samples)
        print(f"Prepared {self.num_samples} matched LiDAR↔S2 groups (k={self.context_k}).")

    def _extract_id(self, path: str) -> str:
        return os.path.basename(path).split("_")[-1].split(".")[0]

    @staticmethod
    def _encode_angles_deg(az_deg, ze_deg):
        rad = math.pi / 180.0
        az = float(az_deg); ze = float(ze_deg)
        return torch.tensor([math.sin(az * rad), math.cos(az * rad), ze / 90.0], dtype=torch.float32)

    def _days_from_ref(self, date_val) -> float:
        if date_val is None:
            return 0.0
        try:
            d = _dt.date.fromisoformat(str(date_val)[:10])
            return float((d - self.ref_date).days)
        except Exception:
            return 0.0

    def _parse_attrs_json(self, json_path):
        if not os.path.exists(json_path):
            return [torch.zeros(8) for _ in range(self.max_s2)]
        with open(json_path, "r") as f:
            recs = json.load(f)
        feats = []
        for r in recs[:self.max_s2]:
            if r is None:
                feats.append(torch.zeros(8))
                continue
            cloud = torch.tensor(float(r.get("cloud_cover", 0.0)) / 100.0).clamp(0, 1)
            saz = self._encode_angles_deg(r.get("sun_azimuth_mean", 0.0), r.get("sun_zenith_mean", 0.0))
            vaz = self._encode_angles_deg(r.get("view_azimuth_mean", 0.0), r.get("view_zenith_mean", 0.0))
            age = torch.tensor(self._days_from_ref(r.get("acquisition_date")), dtype=torch.float32).view(1)
            feats.append(torch.cat([cloud.view(1), saz, vaz, age], dim=0))
        while len(feats) < self.max_s2:
            feats.append(torch.zeros(8))
        return feats

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        sample_paths = self.samples[idx]
        lidar_path   = sample_paths["lidar_path"]
        s2_group_dir = sample_paths["s2_group_dir"]
        tile_id      = sample_paths["tile_id"]
        available_ids = sample_paths.get("available_ids", list(range(self.max_s2)))

        # -------------------------
        # 1) Choose which S2 times to use (define chosen_ids FIRST)
        # -------------------------
        pool = sorted(available_ids)
        if len(pool) < self.context_k:
            raise RuntimeError(
                f"Patch {tile_id}: only {len(pool)} S2 times available, "
                f"but context_k={self.context_k}."
            )

        if self.randomize_context:
            if self.split == "train":
                chosen_ids = sorted(random.sample(pool, self.context_k))
            else:
                # deterministic selection for val/test
                seed = int(hashlib.sha1(tile_id.encode("utf-8")).hexdigest(), 16) % (2**32 - 1)
                rng = random.Random(seed)
                chosen_ids = sorted(rng.sample(pool, self.context_k))
        else:
            # take the first k available deterministically
            chosen_ids = pool[:self.context_k]

        # -------------------------
        # 2) Load LiDAR (data + mask)
        # -------------------------
        with rasterio.open(lidar_path) as src:
            lidar_full = torch.from_numpy(src.read().astype(np.float32))
        if lidar_full.shape[0] == 1:
            data = lidar_full[0:1]
            mask = torch.ones_like(data[0])
        else:
            data = lidar_full[0:1]
            mask = lidar_full[1]

        # -------------------------
        # 3) Read only the chosen S2 times
        # -------------------------
        s2_list = []
        for i in chosen_ids:
            s2_path = os.path.join(s2_group_dir, f"t{i}.tif")
            with rasterio.open(s2_path) as src:
                arr = torch.from_numpy(src.read()[:4].astype(np.float32))  # R,G,B,NIR
            if arr.shape[-2:] != self.target_s2_hw:
                arr = F.interpolate(arr.unsqueeze(0), size=self.target_s2_hw,
                                    mode="bilinear", align_corners=False).squeeze(0)
            s2_list.append(arr)
        s2 = torch.cat(s2_list, dim=0)   # [k*4, Hc, Wc]

        # -------------------------
        # 4) Attributes (only for chosen times)
        # -------------------------
        all_attrs = self._parse_attrs_json(os.path.join(s2_group_dir, "attrs.json"))
        attrs = torch.cat([all_attrs[i] for i in chosen_ids], dim=0)  # [k*8]

        # -------------------------
        # 5) Normalize S2 with training stats (per chosen time)
        # -------------------------
        chosen_means = torch.cat([self.s2_means[i*4:(i+1)*4] for i in chosen_ids], dim=0)
        chosen_stds  = torch.cat([self.s2_stds[i*4:(i+1)*4]  for i in chosen_ids], dim=0)
        s2 = (s2 - chosen_means.view(-1,1,1)) / (chosen_stds.view(-1,1,1) + 1e-6)

        # -------------------------
        # 6) Optional augmentation (train only)
        # -------------------------
        if self.augment and self.split == "train":
            if random.random() > 0.5:
                data = TF.hflip(data); s2 = TF.hflip(s2); mask = TF.hflip(mask)
            if random.random() > 0.5:
                data = TF.vflip(data); s2 = TF.vflip(s2); mask = TF.vflip(mask)
            if random.random() > 0.5:
                ang = random.choice([90, 180, 270])
                data = TF.rotate(data, ang)
                s2   = TF.rotate(s2, ang)
                mask = TF.rotate(mask.unsqueeze(0), ang).squeeze(0)

        return {
            "s2": s2.float(),
            "lidar": data.float(),
            "mask": mask.float(),
            "attrs": attrs.float(),
            "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "tile_id": tile_id
        }
