import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Utility helpers
def subsample(arr, step):
    """Downsamples a 2D array by a given step."""
    return arr[::step, ::step]

def plot_single_3d_surface(ax, lidar_array, title="3D LiDAR Surface Plot", cmap='terrain', z_label='Elevation Deviations (m)',vmin=None, vmax=None):
    """
    Plots a single 3D surface, properly handling NaNs and zooming to valid data.\
    """
    # Replace zeros with NaNs for better visualization
    lidar_array = np.where(lidar_array == 0, np.nan, lidar_array)
    # Get dimensions of the array for meshgrid creation
    height, width = lidar_array.shape
    x_coords = np.arange(0, width)
    y_coords = np.arange(0, height)
    X, Y = np.meshgrid(x_coords, y_coords)

    # Set fixed z-axis limits for consistent plotting
    ax.set_zlim(-1.5, 1.5)

    # Create the surface plot. The `plot_surface` function automatically
    surface = ax.plot_surface(
        X, Y, lidar_array, 
        cmap=cmap,
        alpha=0.9,
        edgecolor='none',
        rstride=1, cstride=1,
        vmin=vmin, vmax=vmax
    )
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel(z_label)
    ax.set_title(title)
    ax.view_init(elev=15, azim=70)
    return surface

def plot_all_three_3d_surfaces(gt_array, pred_array, diff_array, step=4, out_path=None, plot_title="Combined 3D Plots",y_gt_x=True):
    """
    Plots all three 3D surfaces side-by-side with fixed zoom and z-axis.
    """
    gt_s = subsample(gt_array, step)
    pr_s = subsample(pred_array, step)
    diff_s = subsample(diff_array, step)

    fig = plt.figure(figsize=(24, 8))
    fig.suptitle(plot_title, fontsize=16)

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surface1 = plot_single_3d_surface(ax1, gt_s, title="Ground Truth LiDAR Elevation", cmap='terrain', z_label='Elevation (m)',vmin=-0.1, vmax=1.1)
    fig.colorbar(surface1, ax=ax1, shrink=0.5, aspect=5, label='Elevation (m)')

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    surface2 = plot_single_3d_surface(ax2, pr_s, title="Predicted LiDAR Elevation", cmap='terrain', z_label='Elevation (m)',vmin=-0.1, vmax=1.1)
    fig.colorbar(surface2, ax=ax2, shrink=0.5, aspect=5, label='Elevation (m)')

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    surface3 = plot_single_3d_surface(ax3, diff_s, title="Prediction - Ground Truth (Error)", cmap='RdBu', z_label='Difference (m)',vmin=-0.15, vmax=0.15)
    fig.colorbar(surface3, ax=ax3, shrink=0.5, aspect=5, label='Difference (m)')

    for a in [ax1, ax2, ax3]:
        if y_gt_x:
            a.set_xlim(0, 275)
            a.set_ylim(400, 1500)
            a.set_zlim(-1.5, 1.5)
        else:
            a.set_xlim(700, 1400)
            a.set_ylim(0, 800)
            a.set_zlim(-1.5, 1.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if out_path:
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        print(f"Saved 3D plots to {out_path}")

    plt.show()

# Core function for loading and plotting split mosaics

def load_and_plot_split_mosaics(out_dir):
    """
    Loads saved mosaic GeoTIFF files, splits them across y=x, and plots two sets of 3D surfaces.
    """
    pred_mosaic_path = os.path.join(out_dir, "pred_mosaic.tif")
    gt_mosaic_path   = os.path.join(out_dir, "gt_mosaic.tif")
    diff_path        = os.path.join(out_dir, "diff_pred_minus_gt.tif")
    

    print("Loading saved mosaic files...")
    with rasterio.open(gt_mosaic_path) as src:
        gt_array = src.read(1)
        
    with rasterio.open(pred_mosaic_path) as src:
        pred_array = src.read(1)

    with rasterio.open(diff_path) as src:
        diff_array = src.read(1)

    print("Files loaded. Splitting data...")
    
    h, w = gt_array.shape
    y_coords, x_coords = np.indices((h, w))

    # Create masks for the two regions
    mask_y_gt_x = y_coords > x_coords
    mask_y_lt_x = y_coords < x_coords
    
    # Create split arrays for Region 1 (y > x)
    gt_array_1 = np.where(mask_y_gt_x, gt_array, np.nan)
    pred_array_1 = np.where(mask_y_gt_x, pred_array, np.nan)
    diff_array_1 = np.where(mask_y_gt_x, diff_array, np.nan)

    # Create split arrays for Region 2 (y < x)
    gt_array_2 = np.where(mask_y_lt_x, gt_array, np.nan)
    pred_array_2 = np.where(mask_y_lt_x, pred_array, np.nan)
    diff_array_2 = np.where(mask_y_lt_x, diff_array, np.nan)

    print("Generating plots for Region 1 (y > x)...")
    plot_all_three_3d_surfaces(
        gt_array=gt_array_1,
        pred_array=pred_array_1,
        diff_array=diff_array_1,
        step=4,
        out_path=os.path.join(out_dir, "3d_plots_region_y_gt_x.png"),
        plot_title="LiDAR Maps for Validation Region 4.1",
        y_gt_x=True
    )

    print("Generating plots for Region 2 (y < x)...")
    plot_all_three_3d_surfaces(
        gt_array=gt_array_2,
        pred_array=pred_array_2,
        diff_array=diff_array_2,
        step=4,
        out_path=os.path.join(out_dir, "3d_plots_region_y_lt_x.png"),
        plot_title="LiDAR Maps for Validation Region 4.2",
        y_gt_x=False
    )

    print("\nPlotting complete.")

# Execution
output_directory = '/cs/student/projects2/aisd/2024/tcannon/dissertation/Dissertation/final/mod_final/diagrams'
load_and_plot_split_mosaics(output_directory)