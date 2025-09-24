
# Mapping Arctic Fast Ice Terrain Using Diffusion-Based Super-Resolution of Satellite Imagery
![alt text](https://github.com/tessacannon48/Dissertation/blob/main/figures/3d_plots_region_y_gt_x.png)

## About The Project

### Goal

This project aims to develop and validate a conditional diffusion model capable of generating super-resolved digital elevation maps (DEMs) of Arctic fast ice from Sentinel-2 imagery. Specifically, the model learns a mapping from 10-m Sentinel-2 observations to 1-m LiDAR data, corresponding to a 10× increase in spatial resolution.

### Motivation

The motivation of this study is to enable safer navigation for Indigenous communities in the Arctic by improving the estimation of fast ice conditions. Through the generation of high-resolution DEMs derived from remote sensing data, this work seeks to provide accurate representations of ice surface roughness and topography, which are key indicators of ice safety.

## Getting Started

### Installation
To set up the environment:

```bash
git clone https://github.com/tessacannon48/Dissertation.git
cd Dissertation
pip install -r requirements.txt
```

### Repository Structure

```
Dissertation
├── config.yaml                   # Project configuration file
├── copernicus_login.py           # Copernicus data service authentication
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── figures/                      # Figures and visualizations
│   ├── 3d_lidar.png
│   ├── lidar_histogram.png
│   └── ...                       # Other figure files
├── input_data/                   # Input data for experiments (samples only)
│   ├── sample_lidar_patch/
│   └── sample_s2_patch/
├── models/                       # Trained model checkpoints (.pth)
├── notebooks/                    # Jupyter notebooks for analysis and experiments
├── scripts/                      # Python scripts for processing and modeling
├── src/                          # Source code for the project
```

---

## Data Source

### LiDAR 
![alt text](https://github.com/tessacannon48/Dissertation/blob/main/figures/map_diagram.png)

**Provider:** Private research team. 

This study uses LiDAR data collected from the Qikiqtaaluk Region of Nunavut, Canada, located on northern Baffin Island (see figure 3.1). This region is home to Pond Inlet, an Inuit community of approximately 1500 people, making it the largest community on the northern part of the island. This region was chosen for this study due to it being the location of extensive research performed by the UCL Earth Science department because of its relevance to the Inuit populations. The LiDAR data was collected during a field-based surface topography mapping mission using a RIEGL VQ-580 mounted on an Unmanned Aerial Vehicle (UAV) by a research team on April 26th, 2024. 

### Satellite Images
![alt text](https://github.com/tessacannon48/Dissertation/blob/main/figures/sentinel2_images.png)

**Provider:** [ESA Copernicus Data Space](https://dataspace.copernicus.eu/)

The multispectral satellite imagery used in this study is obtained from the European Space Agency’s (ESA) Sentinel-2 mission. Four of the 13 bands, RGB+NIR (10 m), were selected for this study due to their ability to capture fine-scale surface texture and reflectance characteristics that may be correlated with surface roughness and elevation. 
---

## Dataset Construction
![alt text](https://github.com/tessacannon48/Dissertation/blob/main/figures/sample_patch.png)

1. Preprocessing
**Script:** `/Dissertation/scripts/lidar_preprocessing.py`  
The LiDAR data were originally recorded as three-dimensional point clouds at 1m resolution (WGS84). The coordinates were reprojected into a locally optimized, meter-based coordinate system using a custom Transverse Mercator projection. To remove large-scale elevation trends and emphasize local surface roughness, the raw elevations were converted to RANSAC residuals by fitting a quadratic surface to each dataset. 

2. Geolocation
**Notebook:** `/Dissertation/notebooks/data_collocation.ipynb`  
To identify valid Sentinel-2 imagery for training, a querying pipeline was developed using the Copernicus Data Space Ecosystem (CDSE) API to match optical satellite images with the spatial extent of the airborne LiDAR data.

3. Patching
**Notebook:** `/Dissertation/notebooks/patching.ipynb` 
The dataset construction followed three main steps. First, the LiDAR GeoTIFF tiles were mosaicked into a unified, geographically aligned grid. Second, a sliding window of 256⇥256 pixels with a stride of 128 pixels (50% overlap) was applied to extract LiDAR patches. For each patch, the geographic bounds were reprojected from the LiDAR coordinate reference system (CRS) to the Sentinel-2 CRS in order to extract the corresponding 26⇥26 pixel windows from each of the six Sentinel-2 products.

4. Transformations
**Script:** `/Dissertation/scripts/main.py` 
The dataset class used to create the input dataset applies several selections and trans- formations to adequately prepare the data for modeling. First, a specified k number of Sentinel-2 patches are selected either randomly or deterministically depending on the given experiment being performed. Following selection, each Sentinel-2 patch is resized to the dimensions of the LiDAR patch (256⇥256 pixels) using bilinear interpolation. Next, Sentinel-2 data is normalized using the global mean and standard deviation calculated across the training dataset: these statistics are computed for each of the 24 channels independently, treating each of the six temporal images and their four bands as a separate channel. The LiDAR data is not transformed as the values are already centered around zero from the RANSAC calculation. The training set is then randomly augmented using horizontal flips, vertical flips, and rotations to both the LiDAR and Sentinel-2 data to increase the variety of the training samples and improve the model’s robustness.Finally, the attributes are parsed for each Sentinel-2 patch and encoded in the following manner: cloud coverage percentage is scaled to be between 0 and 1, the age of the image is calculated as a positive or negative scalar value which represents the days relative to the LiDAR acquisition date, the Zenith angles are scaled to be between 0 and 1, and the Azimuth angles are transformed into two features, the cosine and sine of the original angle, using sinusoidal encoding. 

---

## Model
![alt text](https://github.com/tessacannon48/Dissertation/blob/main/figures/model_diagram.png)

### Model
The model is a conditional U-Net diffusion architecture designed for cross-modal generation. It takes as input the noisy LiDAR patch (1 channel) and conditions on the collocated Sentinel-2 patches (4 bands per patch, with *k* selectable patches) as well as auxiliary metadata vectors. The network is trained within a denoising diffusion probabilistic model (DDPM) framework to iteratively recover high-resolution synthetic elevation maps from noisy inputs, guided by optical satellite imagery.  

Note that the modeling setup enables dynamic adjustment of the model architecture to allow for ablation studies of architectural variants and sampling methods. 

### Architecture
The baseline architecture is a U-Net with depth 4 and base channels of 128, which is then dynamically adjusted according to experimental settings.  

**Inputs**  
- LiDAR residual map: `[1, H, W]`  
- Sentinel-2 context: `[4k, H, W]` (4 bands × k patches)  
- Attributes: `[8k]` (per-patch metadata, optional)  
- Diffusion timestep: `[1]`  

**Conditioning**  
- Timestep → MLP: Linear → SiLU → Linear (output size 256)  
- Attributes → MLP: Linear → SiLU → Linear (output size 256)  
- Combined vector injected at all layers via **FiLM** modulation of GroupNorm.  

**Encoder (Down path)**  
- Input DoubleConv: `[1 + 4k] → 128`  
- Four Down blocks (MaxPool2d + DoubleConv):  
  - 128 → 256  
  - 256 → 512  
  - 512 → 1024  
  - 1024 → 1024 (capped at 8× base)  
- Skip connections saved at each stage.  

**Bottleneck**  
- DoubleConv: `1024 → 1024`  
- **Self-Attention** applied here only (1024 channels, 1×1 conv q/k/v/proj).  

**Decoder (Up path)**  
- Four Up blocks (ConvTranspose2d + Concat + DoubleConv):  
  - 1024↑ (→512) + skip(1024) → 1024  
  - 1024↑ (→512) + skip(512) → 512  
  - 512↑ (→256) + skip(256) → 256  
  - 256↑ (→128) + skip(128) → 256  

**Output**  
- Final 1×1 convolution: `256 → 1` (predicted LiDAR residuals).  

**Block details**  
- **DoubleConv**: two 3×3 conv layers, each → GroupNorm(8 groups, affine=False) → FiLM conditioning → GELU activation.  
- **Up**: ConvTranspose2d (2×2, stride 2) for upsampling.  
- **Down**: MaxPool2d (2×2) for downsampling.  
- **Attention**: Configurable via variants (`none`, `all`, `mid`, `heavy`), but default = **bottleneck only**.  

**Dynamic behavior**  
- `unet_depth`, `base_channels`, and attention placement are configurable by experiment.  
- Supports ablation studies by toggling context patches, metadata, and attention variants.


### Training Configuration

## Experiments

---

## Limitations


### Data Limitations

### Computational Constraints
---

## Future Work

## Acknowledgements

## References
<a id="1">[1]</a> 
#link
