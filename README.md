
# Mapping Arctic Fast Ice Terrain Using Diffusion-Based Super-Resolution of Satellite Imagery
![alt text](https://github.com/tessacannon48/Dissertation/blob/main/figures/3d_plots_region_y_gt_x.png)

## About The Project

### Goal

This project aims to develop and validate a conditional diffusion model capable of generating super-resolved digital elevation maps (DEMs) of Arctic fast ice from Sentinel-2 imagery. Specifically, the model learns a mapping from 10-m Sentinel-2 observations to 1-m LiDAR data, corresponding to a 10× increase in spatial resolution.

### Motivation

The motivation of this study is to enable safer navigation for Indigenous communities in the Arctic by improving the estimation of fast ice conditions. Through the generation of high-resolution DEMs derived from remote sensing data, this work seeks to provide accurate representations of ice surface roughness and topography, which are key indicators of ice safety.
  
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
- Script: `/Dissertation/scripts/lidar_preprocessing.py`  
- The LiDAR data were originally recorded as three-dimensional point clouds at 1m resolution (WGS84). The coordinates were reprojected into a locally optimized, meter-based coordinate system using a custom Transverse Mercator projection. To remove large-scale elevation trends and emphasize local surface roughness, the raw elevations were converted to RANSAC residuals by fitting a quadratic surface to each dataset. 

2. Geolocation
- Notebook:`/Dissertation/notebooks/data_collocation.ipynb`  
- To identify valid Sentinel-2 imagery for training, a querying pipeline was developed using the Copernicus Data Space Ecosystem (CDSE) API to match optical satellite images with the spatial extent of the airborne LiDAR data.

3. Patching
   
- Notebook: `/Dissertation/notebooks/patching.ipynb`
- The dataset construction followed three main steps. First, the LiDAR GeoTIFF tiles were mosaicked into a unified, geographically aligned grid. Second, a sliding window of 256⇥256 pixels with a stride of 128 pixels (50% overlap) was applied to extract LiDAR patches. For each patch, the geographic bounds were reprojected from the LiDAR coordinate reference system (CRS) to the Sentinel-2 CRS in order to extract the corresponding 26⇥26 pixel windows from each of the six Sentinel-2 products.

5. Transformations
- Script: `/Dissertation/scripts/main.py` 
- The dataset class used to create the input dataset applies several selections and transformations to adequately prepare the data for modeling. First, a specified k number of Sentinel-2 patches are selected either randomly or deterministically depending on the given experiment being performed. Each Sentinel-2 patch is then resized to the dimensions of the LiDAR patch (256x256 pixels) using bilinear interpolation. Sentinel-2 data is normalized using the global mean and standard deviation calculated across the training dataset, computed for each of the bands channels independently. The LiDAR data is not transformed as the values are already centered around zero from the RANSAC calculation. The training set is then randomly augmented to increase the variety of the training samples and improve the model’s robustness. Finally, the Sentinel-2 attributes are encoded in the following manner: cloud coverage percentage is scaled to be between 0 and 1, the age of the image is calculated as a positive or negative scalar value which represents the days relative to the LiDAR acquisition date, the Zenith angles are scaled to be between 0 and 1, and the Azimuth angles are transformed into two features, the cosine and sine of the original angle, using sinusoidal encoding. 

---

## Model
![alt text](https://github.com/tessacannon48/Dissertation/blob/main/figures/model_diagram.png)

### Model
The model is a conditional U-Net diffusion architecture designed for cross-modal generation. It takes as input the noisy LiDAR patch (1 channel) and conditions on the collocated Sentinel-2 patches (4 bands per patch, with *k* selectable patches) as well as Sentinel-2 attribute vectors. The network is trained within a denoising diffusion probabilistic model (DDPM) framework to iteratively recover high-resolution synthetic elevation maps from noisy inputs, guided by the conditioning.  

Note that the modeling setup enables dynamic adjustment of the model architecture to allow for ablation studies of architectural variants and sampling methods. 

**Inputs**  
- LiDAR residual map: `[1, H, W]`  
- Sentinel-2 context: `[4k, H, W]` (4 bands × k patches)  
- Attributes: `[8k]` (per-patch attributes)  
- Diffusion timestep: `[1]`  

**Architecture**
- **Base**: U-Net with dynamic depth (default = 4) and base channels (default = 128).  
- **Conditioning**: Sentinel-2 patches (4 bands × k), metadata vectors, and diffusion timestep. Conditioning injected at every block via FiLM-modulated GroupNorm.  
- **Encoder/Decoder**: Standard downsampling (MaxPool + DoubleConv) and upsampling (TransposeConv + DoubleConv) with skip connections.  
- **Attention**: Optional self-attention modules, default = bottleneck only.  
- **Blocks**: DoubleConv = two 3×3 convs with GroupNorm, FiLM conditioning, and GELU activation.  
- **Output**: Final 1×1 conv producing a single-channel LiDAR residual map.  
- **Dynamic behavior**: Depth, channels, number of context patches (*k*), and attention placement can all be varied for experiments.

## Training Configuration

### Default Training parameters
- **Batch Size**: 8
- **Epochs**: 200
- **Learning Rate**: 0.0001
- **Timesteps**: 1000
- **Noise Schedule**: Linear
- **Loss Function**: MSE (masked on valid LiDAR regions)
- **Context *k***: 1
- **Randomize Context**: False

### Experiments

The code in this repository is designed to allow dynamic configuration of model architecture and hyperparameters. Below are the parameters, architectural variants, and methods tested during a series of controlled ablation studies to determine the optimal model approach for this task. Alternative values of hyperparameters can be tested using command-line arguments to adjust the config.  

- **Baseline tuning**:
  - Input: 1 Sentinel-2 patch + attributes, output: 256×256 LiDAR residuals  
  - Diffusion: DDPM (1000 steps), loss: Masked MSE  
  - Sweeps:  
    - Learning rate (1e-3, 1e-4, 1e-5)  
    - Noise schedule (linear vs. cosine)  
    - Embedding dimension (128, 256)  
    - Loss variations: Masked MAE, Hybrid (MAE/MSE + gradient loss, λ = {0.1, 0.5, 1.0})  

- **Experiment 1: Architectural variations**  
  - Tested on baseline:  
    - Attention placement (bottleneck, medium attention, heavy attention)  
    - UNet depth (shallow (3) vs. deep (5))  
    - Channel width (narrow vs. wide)  

- **Experiment 2: Sampling strategies**  
  - Compared deterministic samplers: DDPM, DDIM, PLMS  
  - Measured trade-off between reconstruction quality and runtime  

- **Experiment 3: Additional context**  
  - Added multiple Sentinel-2 patches as conditioning  
    - 1 deterministically selected patch
    - 2 deterministically selected patches
    - 3 deterministically selected patches

- **Experiment 4: Randomized context**  
  - Tested robustness to randomly chosen Sentinel-2 patches  
    - k = 1, 3, 6 random patches per sample  

---

## Limitations and Constraints

- During model development, training was limited to 50 epochs in order to balance computational resource use with experimental breadth.
- A sequential search strategy was adopted with only a limited number of overlapping parameter combinations tested, rather than a full grid search, due to computational and time constraints.
- Evaluation metrics used to judge model variants not perfect indicators of reconstruction quality.
---

## Setup & Execution

Follow the steps below to setup and execute the project with your LiDAR data.

### Installation
1. Set up the environment:

```bash
git clone https://github.com/tessacannon48/Dissertation.git
cd Dissertation
pip install -r requirements.txt
```

2. Set up LiDAR data:
- Download LiDAR data to /raw_data folder.
- LiDAR data should be preprocessed as RANSAC residuals such that they are roughly normally distributed around zero.

### Data Collocation

1. Launch Jupyter and open the notebook:
```bash
jupyter notebook data_collocation.ipynb
```
   - This notebook identifies all Sentinel-2 Level-2A products that overlap LiDAR coverage area, filters for usable (cloud-free) imagery, and prepares them for training. 

2. In cell 3, set:
   - Your Copernicus Data Space (CDSE) username and password
   - The LiDAR `.tif` directory path (e.g., `raw_data/pondinlet_lidar`)
   - The date range for Sentinel-2 products to query (+/- 4 days of the LiDAR collection date)
     
3. Execute cells 3-7 to:
   - Cell 3: Query CDSE to find Sentinel-2 products that are geolocated with the LiDAR area
   - Cell 4: Visualize the results
   - Cell 5: Ensure products have 100% coverage the LiDAR area
   - Cell 6: Download the Sentinel-2 products
   - Cell 7: Visualize the Sentinel-2 products for manual inspection

### Patching

1. Launch Jupyter and open the notebook:
```bash
jupyter notebook patching.ipynb
```
   - This notebook extracts spatially aligned Sentinel-2 and LiDAR patch sets to prepare datasets for model training.

2. Set the input and output paths in cell 4:
   ```python
   sentinel_granule_dirs = [
       "/path/to/S2A_MSIL2A_20240422T173911.../GRANULE",
       "/path/to/S2B_MSIL2A_20240424T172859.../GRANULE",
       "/path/to/S2A_MSIL2A_20240426T171901.../GRANULE",
       "/path/to/S2B_MSIL2A_20240427T173859.../GRANULE",
       "/path/to/S2A_MSIL2A_20240429T172901.../GRANULE",
       "/path/to/S2B_MSIL2A_20240430T174909.../GRANULE",
   ]

   lidar_dir = "/path/to/lidar_tifs"
   out_lidar_dir = "./lidar_patches"
   out_s2_dir = "./s2_patches_multi"
   ```

- Replace these with your own Sentinel-2 and LiDAR directories.
- Output directories are where the patch sets will be stored.
     
3. Execute cells 5-7 to:
   - Cell 5: Merge LiDAR files
   - Cell 6: Divide LiDAR into 10 regions with roughly equal number of patches
   - Cell 7: Execute patching:
      - Stacks Sentinel-2 bands (RBG + NIR)
      - Slides window across LiDAR region
      - Transforms LiDAR bounds to extract corresponding Sentinel-2 patch
      - Saves each Sentinel-2 patch set + metadata and LiDAR patch set to out directories (patches are paired using tile IDs ex. 00001)
      - Visualizes sample patch sets.

### Training

1. Edit config.yaml: 
   - Specify directories to LiDAR and Sentinel-2
   - Set default training parameters
   - Set WANDB login
2. Run main.py, specifying optional config changes in terminal:
```bash
python Dissertation/scripts/main.py --context_k 1 --attention_variant mid --sampling_methods plms --lr 1e-4 --epochs 200 --unet_depth 4 --noise_schedule cosine --base_channels 64 --loss_name masked_hybrid_mse_loss --loss_alpha 1.0 --evaluate --run_name final_improved_baseline
```
- Outputs trained model, reconstruction figure, and patch-wise reconstruction statistics
## Acknowledgements

- This project was completed as the dissertation for the MSc in Artificial Intelligence for Sustainable Development at University College London.
- The project was supervised by Dr. Michel Tsamados from the UCL Earth Science Department and Petru Manescu from the UCL Computer Science Department.
- Thanks to contributors Thomas Newman, Weibin Chen, and Alex Saoulis.

## Contact

Please email me tessacannon48@gmail.com if you would like to discuss this work.
