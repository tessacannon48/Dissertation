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
                 context_k=1, randomize_context=True, split="train",
                 augment=True, debug=False, target_s2_hw=(26, 26), ref_date="2024-04-26"):
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
        self.split = split

        self.ref_date = _dt.date.fromisoformat(str(ref_date)[:10])
        
        # Load all lidar paths
        self.lidar_paths = sorted(glob.glob(os.path.join(lidar_dir, "lidar_patch_*.tif")))
        self.s2_group_dirs = {
            os.path.basename(p).split(".")[0].split("_")[-1]: p
            for p in glob.glob(os.path.join(s2_dir, "s2_patch_*")) if os.path.isdir(p)
        }

        # Conditionally load a subset of the data for debugging
        if debug:
            print("DEBUG MODE: Loading only 100 samples.")
            lidar_paths_to_load = self.lidar_paths[:100]
        else:
            lidar_paths_to_load = self.lidar_paths

        # Pre-load all data into memory
        self.data_cache = []
        for lidar_path in tqdm(lidar_paths_to_load, desc="Pre-loading dataset"):
            pid = self._extract_id(lidar_path)
            s2_group_dir = os.path.join(self.s2_dir, f"s2_patch_{pid}")
            if all(os.path.exists(os.path.join(s2_group_dir, f"t{i}.tif")) for i in range(self.max_s2)):
                with rasterio.open(lidar_path) as src:
                    lidar_full = torch.from_numpy(src.read().astype(np.float32))
                data = lidar_full[0:1].clamp(-1.0, 1.0)
                mask = lidar_full[1]
                
                s2_patches = []
                for i in range(self.max_s2):
                    with rasterio.open(os.path.join(s2_group_dir, f"t{i}.tif")) as src:
                        arr = torch.from_numpy(src.read()[:4].astype(np.float32))
                    s2_patches.append(arr)
                
                all_attrs = self._parse_attrs_json(os.path.join(s2_group_dir, "attrs.json"))

                self.data_cache.append({
                    "lidar": data,
                    "mask": mask,
                    "s2_patches": s2_patches,
                    "attrs": all_attrs,
                    "tile_id": pid
                })
        
        # New: Pre-generate patch selections for the entire dataset
        self.patch_selections = {}
        for idx, sample in enumerate(self.data_cache):
            tile_id = sample['tile_id']
            # For deterministic evaluation, always use the first k patches
            if not self.randomize_context:
                chosen_ids = list(range(self.context_k))
            # For randomized evaluation, select k random patches
            else:
                chosen_ids = sorted(random.sample(range(self.max_s2), self.context_k))
            self.patch_selections[tile_id] = chosen_ids
            
        self.num_samples = len(self.data_cache)
        print(f"Loaded {self.num_samples} matched LiDAR↔6×S2 groups into memory.")

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
        sample = self.data_cache[idx]
        data = sample["lidar"]
        mask = sample["mask"]
        s2_patches_full = sample["s2_patches"]
        all_attrs = sample["attrs"]
        tile_id = sample["tile_id"]

        # Use the pre-generated selection from __init__
        chosen_ids = self.patch_selections[tile_id]
        
        s2_list = [s2_patches_full[i] for i in chosen_ids]
        attrs_list = [all_attrs[i] for i in chosen_ids]

        s2_processed = []
        for arr in s2_list:
            if arr.shape[-2:] != self.target_s2_hw:
                arr = F.interpolate(arr.unsqueeze(0), size=self.target_s2_hw, mode="bilinear", align_corners=False).squeeze(0)
            s2_processed.append(arr)
        s2 = torch.cat(s2_processed, dim=0)

        if self.s2_means.numel() == 4:
            means = self.s2_means.repeat(self.context_k).view(-1,1,1)
            stds  = self.s2_stds.repeat(self.context_k).view(-1,1,1)
        else:
            means = self.s2_means.view(-1,1,1)
            stds  = self.s2_stds.view(-1,1,1)
        s2 = (s2 - means) / (stds + 1e-6)
        s2 = torch.clamp(s2 / 4, -1, 1)

        attrs = torch.cat(attrs_list, dim=0)

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
            "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long)
        }