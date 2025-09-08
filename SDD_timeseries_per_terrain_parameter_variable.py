# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 09:04:08 2025

@author: goehrigc




IMPORTANT NOTICE:
    
    Script is dependent on feature and snowdepth size #
    
    if feature is bigger then snowdepth wrong calculations
    only correct statistics using same geometry data input
    
"""

#   function
#   feature_subgroup_sensitivity

import pandas as pd
import numpy as np
import glob
import os
import seaborn as sns
import rasterio
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
from cmcrameri import cm
import sys
sys.path.append(library_dir)
import powdersearch as ps

# Define no-data value
NO_DATA_VALUE = -999

#   define output folder
output_folder_name =  # output folder
input_folder =   # snow depth train
feature_path = # Terrain Parameter TIFF


# ====================== CONFIGURATION SECTION ======================
# Change these parameters based on your feature type

# FEATURE CONFIGURATION
feature_name = "Curvature"  # Change to "Aspect" or "Slope"

# CLASSIFICATION CONFIGURATION
if feature_name == "Aspect":
    # Aspect classification (degrees)
    classes = [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360]
    class_names = {0: 'North', 1: 'Northeast', 2: 'East', 3: 'Southeast', 4: 'South', 5: 'Southwest', 6: 'West', 7: 'Northwest', 8: 'North'}
    merge_classes = True  # Merge North classes (315-360 with 0-45)
    
    # Generate colors using roma colormap
    cmap = cm.romaO_r
    colors = cmap(np.linspace(0, 1, len(class_names)))
    #colors = ['lightblue', 'green',"brown", 'yellow', 'orange', 'red', "purple"]
    
    #colors = ['blue', 'yellow', 'red', 'orange']
    
elif feature_name == "Slope":
    # Slope classification (degrees) - adjust these ranges as needed
    classes = [1, 10, 25, 30,35,40, 45, 90]  # Example: flat, gentle, moderate, steep, very steep
    class_names = {0: '0-9°', 1: '10-19°', 2: '20-29°', 
                   3: '30-34°', 4: '35-39°', 5:"40-45°", 6:">45°"}
    merge_classes = False  # No class merging needed for slope
    
    # Generate colors using Hawaii colormap
    cmap = cm.hawaii_r
    colors = cmap(np.linspace(0, 1, len(class_names)))
    #colors = ['lightblue', 'green',"brown", 'yellow', 'orange', 'red', "purple"]
    
elif feature_name == "Geomorphons":
    # Generic configuration - customize as needed
    classes = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]  # Percentile-based or custom ranges
    class_names = {0: 'Flat', 1: 'Peak', 2: 'Ridge', 3: 'Shoulder', 4:"Spur",
                   5: "Slope", 6: "Hollow", 7:"Footslope", 8: "Valley", 9:"Pit"}
    merge_classes = False
    cmap = cm.lipari
    colors = cmap(np.linspace(0, 1, len(class_names)))
    #colors = ['blue', 'green', 'orange', 'red']
    
    
elif feature_name == "TPI_old":
    # TPI (Topographic Position Index) classification
    # Based on Weiss (2001) and Jenness (2006) standard TPI classifications
    # TPI values typically range from -3 to +3 (standardized units)
    classes = [-4, -2, -1,0,1, 2, 4]  # TPI threshold values
    class_names = {
        0: "Very Low",      # TPI < -2.5 (very negative)
        1: 'Low',   # -2.5 ≤ TPI < -1.5 (negative)
        2: 'Moderately Low',      # -1.5 ≤ TPI < -0.5 (slightly negative)
        3: 'Moderately High',      # 0.5 ≤ TPI < 1.5 (slightly positive)
        4: 'High',            # 1.5 ≤ TPI < 2.5 (positive)
        5: 'Very High'              # TPI ≥ 2.5 (very positive)
    }
    merge_classes = False
    cmap = cm.lipari  # Terrain colormap suitable for topographic features
    colors = cmap(np.linspace(0, 1, len(class_names)))
    
    
        
elif feature_name == "TPI":
    # TPI (Topographic Position Index) classification
    # Based on Weiss (2001) and Jenness (2006) standard TPI classifications
    # TPI values typically range from -3 to +3 (standardized units)
    classes = [-5,-1, -0.5, 0, 0.5, 1, 5]  # TPI threshold values
    class_names = {
        0: "Very Low",      # TPI < -2.5 (very negative)
        1: 'Low',   # -2.5 ≤ TPI < -1.5 (negative)
        2: 'Moderately Low',      # -1.5 ≤ TPI < -0.5 (slightly negative)
        3: 'Moderately High',      # 0.5 ≤ TPI < 1.5 (slightly positive)
        4: 'High',            # 1.5 ≤ TPI < 2.5 (positive)
        5: 'Very High'              # TPI ≥ 2.5 (very positive)
    }
    merge_classes = False
    cmap = cm.lipari  # Terrain colormap suitable for topographic features
    colors = cmap(np.linspace(0, 1, len(class_names)))
    
    
elif feature_name == "curvature_original":
    # Generic configuration - customize as needed
    classes = [-0.5, -0.04, -0.012, 0.012, 0.04, 0.5]# Percentile-based or custom ranges
    class_names = {0: 'Strongly concave', 1: 'Concave', 2: 'Flat', 3: 'Convex', 4:"Strongly Convex"}
    merge_classes = False
    cmap = cm.managua
    colors = cmap(np.linspace(0, 1, len(class_names)))
    #colors = ['blue', 'green', 'orange', 'red']
    
elif feature_name == "Curvature":
    # ArcGIS curvature classification (typical range -10 to +10)
    # ArcGIS scales curvature values by 100 from standard units
    # Negative values = concave (valleys, hollows)
    # Positive values = convex (ridges, peaks)
    classes = [-5,-1, -0.5, 0, 0.5, 1, 5]  # ArcGIS curvature threshold values
    class_names = {
        0: 'Strong Concave',     
        1: 'Concave',          
        2: 'Moderatley Concave',       
        3: 'Moderately Convex',   
        4: 'Convex', 
        5: 'Strong Convex',
        }
        
    merge_classes = False
    cmap = cm.managua  # Red-Yellow-Blue colormap (red=convex, blue=concave)
    colors = cmap(np.linspace(0, 1, len(class_names)))
    
elif feature_name == "Elevation":
    # Generic configuration - customize as needed
    classes = [0, 1800, 2200, 2600, 2800, 5000]  #
    class_names = {0: '<1800m', 1: '1800-2200m', 2: '2201-2600m', 3:"2601m-2800", 4:">2800"}
    merge_classes = False
    cmap = cm.lapaz
    colors = cmap(np.linspace(0, 1, len(class_names)))
    #colors = ['green', 'brown', 'grey', 'blue']
    
    
else:
    # Generic configuration - customize as needed
    classes = [0, 25, 50, 75, 100]  # Percentile-based or custom ranges
    class_names = {0: 'Low', 1: 'Medium-Low', 2: 'Medium-High', 3: 'High'}
    merge_classes = False
    colors = ['blue', 'green', 'orange', 'red']

# ================================================================

#   define snowdepth paths of both areas
filepaths = glob.glob(os.path.join(input_folder, "*.tif"))
n_timesteps = len(filepaths)

# Define years based on the number of time steps (e.g., starting from 2010)
all_years = np.arange(2010, 2010 + n_timesteps)
# Remove 2011 and 2018
years = all_years[(all_years != 2011) & (all_years != 2018)]
kwargs = None

# --------------------------raster preprocessing -----------------------------
#   load snowdepth as reference
reference_raster, meta_ref = ps.load_raster(filepaths[0], metadata="meta")
shape_ref = reference_raster.shape
n_pixels = shape_ref[0] * shape_ref[1]

#   load feature
#feature = ps.load_raster(feature_path)

# match both rasters
#feature, snowdepth = ps.match_array_shapes(feature, reference_raster)

# With this:
feature, reference_raster = ps.properly_align_rasters(feature_path, filepaths[0])

# Verify alignment by checking some statistics
print("Alignment verification:")
print(f"Feature range: {np.nanmin(feature):.3f} to {np.nanmax(feature):.3f}")
print(f"Snow range: {np.nanmin(reference_raster):.3f} to {np.nanmax(reference_raster):.3f}")
print(f"Shapes match: {feature.shape == reference_raster.shape}")

# Check for no-data alignment
feature_nodata_mask = (feature == NO_DATA_VALUE) | np.isnan(feature)
snow_nodata_mask = (reference_raster == NO_DATA_VALUE) | np.isnan(reference_raster)

print(f"Feature no-data pixels: {np.sum(feature_nodata_mask)}")
print(f"Snow no-data pixels: {np.sum(snow_nodata_mask)}")
print(f"Overlapping no-data: {np.sum(feature_nodata_mask & snow_nodata_mask)}")

# flatten features
feature_flat = feature.flatten()

years = [2010, 2012, 2013, 2014, 2015, 2016, 2017, 2019, 2020, 2021, 2022, 2023, 2024]

print("data preprocessed")
# -----------------------------------------------------------------------------

# Define feature classification using NumPy vectorized operations
classes = np.array(classes, dtype=np.float32)

# Dynamically create labels based on the number of bins
n_labels = len(classes) - 1  
labels = np.arange(n_labels, dtype=np.uint8)  
feature_classes = np.digitize(feature_flat, classes) -1 # Assign category indices

filename = 'snowdepth_timeseries17.dat'
if os.path.exists(filename):
    os.remove(filename)
    
snowdepth_timeseries = np.memmap('snowdepth_timeseries17.dat', dtype=np.float32, mode='w+', shape=(n_pixels, n_timesteps))

for i, filepath in enumerate(filepaths):
    with rasterio.open(os.path.join(input_folder, filepath)) as src:
        snowdepth = src.read(1).flatten()  # Read the first band and flatten it into 1D array
        
        # Replace no-data values with NaN for proper handling
        snowdepth[snowdepth == NO_DATA_VALUE] = np.nan
        snowdepth[snowdepth < 0] = np.nan

        
        
        snowdepth_timeseries[:, i] = snowdepth  # Assign this year's data to the correct column

# Also handle no-data values in feature data
feature_flat[feature_flat == NO_DATA_VALUE] = np.nan

# Create mask for valid pixels (pixels that have valid data in both feature and at least some timesteps)
valid_feature_mask = ~np.isnan(feature_flat)
# For each pixel, check if it has at least some valid snow depth data
valid_snow_mask = ~np.isnan(snowdepth_timeseries).all(axis=1)
# Combined mask: pixels that have valid feature data AND some valid snow data
valid_mask = valid_feature_mask & valid_snow_mask

print(f"Total pixels: {n_pixels}")
print(f"Valid pixels (excluding no-data): {np.sum(valid_mask)}")
print(f"Pixels with no-data: {n_pixels - np.sum(valid_mask)}")

# Filter data to only include valid pixels
feature_flat_valid = feature_flat[valid_mask]
snowdepth_timeseries_valid = snowdepth_timeseries[valid_mask, :]
feature_classes_valid = feature_classes[valid_mask]

# CONDITIONAL CLASS MERGING (only for aspect)
if merge_classes and feature_name == "Aspect":
    print("Merging North classes for aspect analysis...")
    # Combine class 4 (315°-360°) with class 0 (0°-45°) to make one North class
    feature_classes_clean = feature_classes_valid.copy()
    feature_classes_clean[feature_classes_valid == 8] = 0  # Merge class  into class 0
    max_valid_class = 8  # After merging, valid classes are 0-3
else:
    print(f"No class merging needed for {feature_name} analysis...")
    feature_classes_clean = feature_classes_valid.copy()
    max_valid_class = n_labels - 1  # All original classes are valid

# Remove any invalid classes (like -1 from digitize)
valid_indices = (feature_classes_clean >= 0) & (feature_classes_clean <= max_valid_class)
feature_classes_final = feature_classes_clean[valid_indices]
snowdepth_timeseries_final = snowdepth_timeseries_valid[valid_indices, :]

# Create DataFrame with clean data
df_dict = {
    'feature_class': feature_classes_final
}

# Add snow depth columns
for i in range(n_timesteps):
    df_dict[f'snowdepth_t{i+1}'] = snowdepth_timeseries_final[:, i]

df = pd.DataFrame.from_dict(df_dict)

print(f"Classes in final data: {sorted(df['feature_class'].unique())}")

# Compute median and standard deviation per terrain class
def safe_median(x):
    return np.nanmedian(x) if not np.isnan(x).all() else np.nan

def safe_std(x):
    return np.nanstd(x) if not np.isnan(x).all() else np.nan

median_per_class = df.groupby('feature_class').agg({f'snowdepth_t{i+1}': safe_median for i in range(n_timesteps)})
std_per_class = df.groupby('feature_class').agg({f'snowdepth_t{i+1}': safe_std for i in range(n_timesteps)})

# Create output filename suffix
n_classes = len(df['feature_class'].unique())
suffix = f"{n_classes}classes"

# Save results
median_per_class.to_csv(os.path.join(output_folder_name, f"{feature_name}_mean_HS_per_class.csv"))
std_per_class.to_csv(os.path.join(output_folder_name, f"{feature_name}_std_HS_per_class.csv"))

# Print data availability
print(f"\nData availability per {feature_name} class:")
for feature_class in sorted(df['feature_class'].unique()):
    class_data = df[df['feature_class'] == feature_class]
    total_pixels = len(class_data)
    class_name = class_names.get(feature_class, f'Class {feature_class}')
    print(f"{class_name}: {total_pixels} pixels")

# PLOTTING - MAXIMIZED WITH LEGEND INSIDE PLOT AREA
plt.figure(figsize=(16, 12))  # Larger figure size for maximum visibility
timesteps = np.arange(1, n_timesteps + 1)
unique_classes = sorted(median_per_class.index)

# Ensure we have enough colors
if len(colors) < len(unique_classes):
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))

for i, feature_class in enumerate(unique_classes):
    mean_values = median_per_class.loc[feature_class].values
    std_values = std_per_class.loc[feature_class].values
    
    # Only plot if we have valid data
    if not np.isnan(mean_values).all():
        class_name = class_names.get(feature_class, f'Class {feature_class}')
        plt.plot(timesteps, mean_values, label=class_name, color=colors[i], 
                marker='o', linewidth=2, markersize=6)
        
        # Add error bars where we have valid std data
        # Add error bars where we have valid std data - FIXED VERSION
        valid_std_mask = ~np.isnan(std_values)
        if np.any(valid_std_mask):
            # Calculate bounds and clip lower bound to 0 (no negative snow depth)
            lower_bound = np.maximum(mean_values - std_values, 0)  # Don't go below 0
            upper_bound = mean_values + std_values
            
            plt.fill_between(timesteps[valid_std_mask], 
                           lower_bound[valid_std_mask], 
                           upper_bound[valid_std_mask], 
                           color=colors[i], alpha=0.2)

# Calculate global min/max for consistent Y-axis
global_min = 0  # Snow depth minimum
#global_max = np.nanmax(median_per_class.values + std_per_class.values)
global_max = 5
# Round max up to nearest 0.5m
global_max_rounded = np.ceil(global_max * 2) / 2

# Create Y-ticks every 0.5m
y_ticks = np.arange(0, global_max_rounded + 0.5, 0.5)

# Customize plot
plt.xlabel('Year', fontsize=24)
plt.ylabel('HS (m)', fontsize=24)
plt.title(f'Median Snow Depth Over Time by {feature_name.capitalize()} Class', 
          fontsize=24)
plt.xticks(timesteps, years, rotation=45, fontsize=24)
plt.yticks(y_ticks, fontsize=24)
plt.ylim(0, global_max_rounded)



# Legend positioned inside plot area at bottom with multiple columns
n_cols = min(len(unique_classes), 3)  # Max 3 columns to avoid overcrowding
plt.legend(loc='upper center', fontsize=24, ncol=n_cols, 
          frameon=True, fancybox=True, shadow=True, framealpha=0.9,
          columnspacing=1.0, handlelength=2.0)

plt.grid(True, alpha=0.3)

plt.text(0.5, 0.05, 'Missing Years: 2011, 2018', 
         transform=plt.gca().transAxes,  # Use axis coordinate system
         ha='center', va='top', fontsize=24, color='black',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)

# Create class names string for filename
class_names_str = "_".join([name.replace(' ', '').replace('(', '').replace(')', '').replace('°', 'deg')                            for name in [class_names.get(c, f'Class{c}') for c in unique_classes]])

plt.savefig(os.path.join(output_folder_name, f"SDD_per_{feature_name}_classes.png"), 
           dpi=300, bbox_inches='tight')
plt.show()

# Clean up
del snowdepth_timeseries
snowdepth_timeseries = None
print(f"Analysis completed with {n_classes} {feature_name} classes: {list(class_names.values())}!")

# Clean up
del snowdepth_timeseries
snowdepth_timeseries = None


print(f"Analysis completed with {n_classes} {feature_name} classes: {list(class_names.values())}!")
