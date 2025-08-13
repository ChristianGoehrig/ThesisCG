# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 16:48:08 2025

@author: goehrigc
"""

import pandas as pd
import numpy as np
import glob
import os
import seaborn as sns
import rasterio
import matplotlib.pyplot as plt
import sys
import re
from cmcrameri import cm

# Append library directory
library_dir = r"E:\manned_aircraft\christiangoehrig\python\functions\00_dev"
sys.path.append(library_dir)
import powdersearch as ps

def feature_sensitivity(
    library_dir,
    input_snowdepth,
    feature_paths,
    elevation_path,
    output_folder_name,
    feature_names,
    feature_classes_list,
    feature_class_labels_list,
    elevation_classes,
    elevation_class_labels,
    nodata_value
    ):
    
    """
    Perform terrain feature sensitivity analysis on snow depth data with elevation stratification
    
    Output:
        Plot: Snowdepth per feature class for different elevation bands and pixelcount per class in one plot

    
    !!! Elevation classes should be chosen by literture (Grünewald) or correlation results of the ROI!!!
    
    Parameters:
    -----------
    -----------
    library_dir : str
        Directory containing libraries
    input_folder : str
        Folder with snow depth raster files
    feature_path : str
        Path to terrain feature raster
    output_folder_name : str
        Output directory for results
    feature_name : str, optional
        Name of the terrain feature (default: 'slope')
    classes : list, optional
        Slope angle classes (default: [0, 30, 35, 40, 45, np.inf])
    nodata_value : float, optional
        Value representing no data in the rasters
    elevation_path : str
        Path to elevation raster
    elevation_classes : list, optional
        Elevation classification bins
    elevation_class_labels : list, optional
        Labels for elevation classes
    """
    
    # Create output folder
    os.makedirs(output_folder_name, exist_ok=True)
    
    # Load snow depth raster
    snowdepth, meta_ref, no_data_value_ref = ps.load_raster(input_snowdepth, metadata = ["meta","nodata"])
    
    # Load and match features and elevation
    features = []
    feature_classified_list = []
    
    for feature_path, feature_name, feature_classes, feature_class_labels in zip(
        feature_paths, feature_names, feature_classes_list, feature_class_labels_list
    ):
        print(f"\nProcessing feature: {feature_name}")
        print(f"Feature path: {feature_path}")
        
        # FIXED: Use proper spatial alignment instead of naive cropping
        feature, _ = ps.properly_align_rasters(feature_path, input_snowdepth)
        print(f"Used proper spatial alignment for {feature_name}")
        
        print(f"Aligned {feature_name} shape: {feature.shape}")
        print(f"Feature range: {np.nanmin(feature):.3f} to {np.nanmax(feature):.3f}")
        
        features.append(feature)
        
        # Flatten features
        feature_flat = feature.flatten()
        
        # Classify features
        feature_classes = np.array(feature_classes, dtype=np.float32)
        n_feature_labels = len(feature_classes) - 1
        feature_classified = np.digitize(feature_flat, feature_classes, right=True) - 1

        # After classification, merge 315-360° (class 8) with 0-45° (class 0)
        if feature_name.lower() == "aspect":
            print("Merging North aspect classes (315-360° with 0-45°)")
            feature_classified[feature_classified == 8] = 0  # Merge last class with first
            
        feature_classified_list.append(feature_classified)
        
    
    for feature_path, feature_name, feature_classes, feature_class_labels in zip(
        feature_paths, feature_names, feature_classes_list, feature_class_labels_list
    ):
        # Load feature raster
        feature, no_data_value_feature = ps.load_raster(feature_path, metadata=["nodata"])
        
        # Match raster shapes
        feature, _ = ps.match_array_shapes(feature, snowdepth)
    # Load and properly align elevation
    print(f"\nProcessing elevation data")
    print(f"Elevation path: {elevation_path}")
    
    elevation, _ = ps.properly_align_rasters(elevation_path, input_snowdepth)
    print("Used proper spatial alignment for elevation")
    
    # Flatten rasters
    snowdepth_flat = snowdepth.flatten()
    elevation_flat = elevation.flatten()
    
    # Classify elevation bands
    elevation_classes = np.array(elevation_classes, dtype=np.float32)
    elevation_classified = np.digitize(elevation_flat, elevation_classes, right=True) - 1
    
    # Create DataFrame
    df = pd.DataFrame.from_dict({
        'snowdepth': snowdepth_flat,
        'elevation': elevation_flat,
        **{f'{feature_name}_class': fc for feature_name, fc in zip(feature_names, feature_classified_list)},
        'elevation_class': elevation_classified
    })
    
    # No-data filtering and excluding negative values
    valid_mask = (
        (df['snowdepth'] != nodata_value) & 
        #(df['snowdepth'] > 0) &#TODO: change for HS
        (df['elevation'] != nodata_value)
    )
    for feature_name in feature_names:
        valid_mask &= (df[f'{feature_name}_class'] != -1)
    valid_mask &= (df['elevation_class'] != -1)
    
    df = df[valid_mask]
    
        # Create a figure with subplots for each elevation band
    fig, axes = plt.subplots(
        nrows=len(elevation_class_labels), 
        ncols=1, 
        figsize=(12, 4*len(elevation_class_labels)), 
        sharex=True
    )
    
    # If only one elevation band, convert to list
    if len(elevation_class_labels) == 1:
        axes = [axes]
    
    # First pass: determine global min/max for snow depth to set consistent y-axis
    global_min = float('inf')
    global_max = float('-inf')
    
    for elev_idx in range(len(elevation_class_labels)):
        elev_class_data = df[df['elevation_class'] == elev_idx]
        if not elev_class_data.empty:
            snow_depths = elev_class_data['snowdepth']
            global_min = min(global_min, snow_depths.min())
            global_max = max(global_max, snow_depths.max())
    
    # Round to nearest meter and add some padding
   # y_min = int(np.floor(global_min))#TODO:HS
    #y_max = int(np.ceil(global_max))
    
    y_min =-2 #TODO:representativeness
    y_max = 2
    
    # Iterate through elevation classes
    for elev_idx, (elev_label, ax) in enumerate(zip(elevation_class_labels, axes)):
        # Filter data for current elevation class
        elev_class_data = df[df['elevation_class'] == elev_idx]
        
        ############################ snowdeptrh ############################TODO:snowdepth
        # Prepare subplot
       # ax.set_title(f'Snow Depth Distribution - {elev_label}')
       # ax.set_ylabel('Snow Depth (m)')
        
        # Set consistent y-axis limits and ticks
       # ax.set_ylim(y_min, y_max)
       # ax.set_yticks(range(y_min, y_max + 1, 1))  # 1m ticks
        
       ################################### representativeness ##############
        # Set y-axis limits and ticks for 0-1 scale
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(-2, 2.1, 0.5))  # 0.5 increments from -2 to 2
        ax.set_title(f'Deviation from mean snowdepth of ROI - {elev_label}')
        ax.set_ylabel('Deviation (m)')
        
        
        ########################################################################
        
        # Prepare data for boxplot and pixel count
        box_data = []
        labels = []
        colors = []
        pixel_counts = []
        
        for feature_idx, (feature_name, feature_classes, feature_labels) in enumerate(
            zip(feature_names, feature_classes_list, feature_class_labels_list)
        ):
            # Choose colormap based on feature type
            if feature_name == "Aspect":
                cmap = cm.roma_r 
            elif feature_name == "Slope":
                cmap = cm.hawaii_r 
            elif feature_name == "TPI":
                cmap = cm.lipari
            elif feature_name == "Curvature":
                cmap = cm.managua
            else:
                cmap = cm.davos
            
            feature_colors = cmap(np.linspace(0, 1, len(feature_labels)))
            
            # Boxplot for each feature class in this elevation band
            for class_idx, label in enumerate(feature_labels):
                feature_class_data = elev_class_data[
                    elev_class_data[f'{feature_name}_class'] == class_idx
                ]
                
                # Snow depth data
                snow_depth_data = feature_class_data['snowdepth']
                box_data.append(snow_depth_data)
                
                # Pixel count for this feature class
                pixel_count = len(feature_class_data)
                pixel_counts.append(pixel_count)
                
                labels.append(f'{feature_name}: {label}')
                colors.append(feature_colors[class_idx])
        
        # Create boxplot with updated parameters
        bp = ax.boxplot(box_data,
                        tick_labels=labels,
                        patch_artist=True,
                        flierprops={"marker": "o", "markersize": 1, "markerfacecolor": "black", "markeredgecolor": "black"}
                        )
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Create a twin axis for pixel count bar plot
        ax2 = ax.twinx()
        
        # Plot pixel count bars in the background
        bar_width = 0.5
        ax2.bar(range(1, len(pixel_counts) + 1), pixel_counts, 
                alpha=0.3, color='gray', width=bar_width, 
                label='Pixel Count')
        
        # Configure pixel count axis
        ax2.set_ylabel('Pixel Count')
        ax2.set_xlim(0.5, len(pixel_counts) + 0.5)
        
        # Rotate x-axis labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder_name, f"SDD_HS_med_elev_per_{feature_name}_classes.png"))
    plt.show()
    plt.close()




# Example usage
feature_sensitivity(
    library_dir=r"E:\manned_aircraft\christiangoehrig\data\datasets\001_2m_no18_common_extent",
    input_snowdepth=r"E:\manned_aircraft\christiangoehrig\data\representativeness\deviation_mean_negatives.tif",
    feature_paths=[
        r"E:\manned_aircraft\christiangoehrig\data\datasets\max_common_acquisition_extent\terrain_layer/aspect_2m.tif"  ,
        #r"E:\manned_aircraft\christiangoehrig\data\terrain_layer_2m_uniform/aspect.tif",
        #r"E:\manned_aircraft\christiangoehrig\data\datasets\001_2m_no18_common_extent\terrain_layer\tpi_20m_2m_001.tif",
        #r"E:\manned_aircraft\christiangoehrig\data\datasets\001_2m_no18_common_extent\terrain_layer\curvature_20m_2m_001.tif"
    ],
    elevation_path=r"E:\manned_aircraft\christiangoehrig\data\datasets\max_common_acquisition_extent/terrain_layer\dem_2m.tif" ,
    #elevation_path= r"E:\manned_aircraft\christiangoehrig\data\dem\dem_2m_correlation_raster.tif",
    output_folder_name=r"E:\manned_aircraft\christiangoehrig\data\representativeness\negatives",
    feature_names=["Aspect"],
    feature_classes_list=[
        [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360], # aspect classes
        #[-5, -1,-0.5,0,0.5,1,5] #TPI and curvature classes
        #[1, 10, 25, 30,35,40, 45, 90],  # Slope classes
        #[0,0.11,0.33,0.45,0.76,0.97,3.64]
       
    ],
    feature_class_labels_list=[
       ["North","Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest"],    #aspect labels
       #["Strong Concave","Concave","Moderately Concave", "Flat", "Moderately Convex","Convex", "Strong Convex"],  # Curvature labels
        #['0-9°', '10-19°', '20-29°', '30-34°', '35-39°',"40-45°", ">45°"],  # Slope labels
       # ["Very Low", "Low", "Moderately Low", "Moderately High", "High", "Very High"],
        #["Very Consistent", "Consistent", "Moderately Consistent","Moderately Volatile", "Volatile","Very Volatile"]
    ],
    elevation_classes=[1800,2200,2600,2800, np.inf],
    elevation_class_labels=["1801-2200m","2201-2600","2601-2800", ">2800m"],
    nodata_value=-999
)