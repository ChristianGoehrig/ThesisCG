# -*- coding: utf-8 -*-
"""
Created on Wed May 28 15:40:58 2025

@author: goehrigc
"""


# -*- coding: utf-8 -*-
"""
Snow Distribution Terrain Analysis - Mixed Approach

This script analyzes the relationship between terrain features and snow depth,
and generates predictions of snow depth based on terrain characteristics.

Key features:
- Calculates terrain parameters on bounding box extent
- Trains and predicts models using exact outline geometry


UPDATE:
    
    Buffered terrain parameter calculation for having no artefacts in fina lprocessing extent
    
    
Author: Christian Goehrig
Modified: Mixed approach implementation
"""

# special case with geopandas
# must install before first console run individually 
# 1. conda install geopandas
# 2. restart kernel



library_dir = r"E:\manned_aircraft\christiangoehrig\python\powdersearch_application"
import sys
sys.path.append(library_dir)
import powdersearch as ps

import os
import sys
from pathlib import Path
import glob
import re
import yaml
from typing import Dict, List, Optional, Tuple, Union, Any
from shapely.geometry import box,mapping
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from sklearn.linear_model import LinearRegression
import geopandas as gpd
from rasterio.mask import mask
from shapely.geometry import mapping
from cmcrameri import cm as cmc


# Global data storage
TERRAIN_LAYERS = {
    "train": {},
    "pred": {}
}

# Store geometries for later use in training/prediction
OUTLINE_GEOMETRIES = {
    "train": {},
    "pred": {}
}

def debug_calibration_setup(config, area_name="75129"):
    """
    Debug function to check if calibration setup is working correctly.
    Run this to see what's happening with your reference point.
    """
    import rasterio
    import numpy as np
    from rasterio.transform import rowcol
    import matplotlib.pyplot as plt
    
    print(f"=== DEBUGGING CALIBRATION FOR AREA {area_name} ===")
    
    # Get calibration config
    if 'calibration' not in config or not config['calibration'].get('enabled', False):
        print(" Calibration not enabled")
        return
    
    calibration_config = config['calibration']
    if area_name not in calibration_config.get('areas', {}):
        print(f" Area {area_name} not in calibration config")
        return
    
    area_config = calibration_config['areas'][area_name]
    reference_coord = area_config['reference_coordinate']
    ground_truth_depth = area_config['ground_truth_depth']
    
    print(f"✓ Reference coordinate: {reference_coord}")
    print(f"✓ Ground truth depth: {ground_truth_depth} m")
    
    # Check DEM file
    year_pred = config['analysis']['year_pred']
    case_folder = config['paths']['output_folder'] / "data" / "your_case_name"  # You'll need to update this
    dem_path = case_folder / f"{year_pred}__{area_name}" / f"dem_{year_pred}_{area_name}.tif"
    
    print(f"\nChecking DEM file: {dem_path}")
    if not dem_path.exists():
        print(f" DEM file not found: {dem_path}")
        return
    
    # Load and examine DEM
    with rasterio.open(dem_path) as src:
        print(f"✓ DEM CRS: {src.crs}")
        print(f"✓ DEM bounds: {src.bounds}")
        print(f"✓ DEM shape: {src.width} x {src.height}")
        print(f"✓ DEM transform: {src.transform}")
        
        # Convert reference coordinate to pixel
        try:
            row, col = rowcol(src.transform, reference_coord[0], reference_coord[1])
            print(f"✓ Reference pixel: ({row}, {col})")
            
            # Check if pixel is within bounds
            if 0 <= row < src.height and 0 <= col < src.width:
                print(f"✓ Reference pixel is within DEM bounds")
                
                # Read elevation at reference point
                elevation = src.read(1, window=((row, row+1), (col, col+1)))
                if elevation.size > 0:
                    elev_value = elevation[0]
                    print(f"✓ Elevation at reference: {elev_value:.1f} m")
                    
                    if elev_value == src.nodata:
                        print(f" Reference point has nodata value!")
                    else:
                        print(f"Valid elevation data at reference point")
                else:
                    print(f" Could not read elevation at reference point")
            else:
                print(f" Reference pixel ({row}, {col}) is outside DEM bounds ({src.height}, {src.width})")
                
        except Exception as e:
            print(f" Error converting coordinates: {e}")
    
    # Check if prediction files exist
    result_folder = config['paths']['output_folder'] / "results" / "your_case_name"  # Update this
    outline_folder = result_folder / f"{year_pred}__{area_name}"
    
    print(f"\nChecking prediction files in: {outline_folder}")
    if outline_folder.exists():
        pred_files = list(outline_folder.glob("predicted_snowdepth_*.tif"))
        print(f"✓ Found {len(pred_files)} prediction files")
        for pf in pred_files:
            print(f"  - {pf.name}")
    else:
        print(f"❌ Prediction folder not found: {outline_folder}")


def visualize_reference_point_context(config, area_name="75129", feature_name="tpi_20_elevation_enhanced"):
    """
    Create a detailed visualization around your reference point to see what's happening.
    """
    import rasterio
    import numpy as np
    import matplotlib.pyplot as plt
    from rasterio.transform import rowcol
    
    print(f"=== VISUALIZING REFERENCE POINT CONTEXT ===")
    
    # Get paths (you'll need to update these with your actual paths)
    year_pred = config['analysis']['year_pred']
    output_folder = config['paths']['output_folder']
    
    # Build paths - you may need to adjust these
    case_name = "your_case_name"  # Update this
    case_folder = output_folder / "data" / case_name
    result_folder = output_folder / "results" / case_name
    
    dem_path = case_folder / f"{year_pred}__{area_name}" / f"dem_{year_pred}_{area_name}.tif"
    outline_folder = result_folder / f"{year_pred}__{area_name}"
    pred_file = outline_folder / f"predicted_snowdepth_{year_pred}_{area_name}_{feature_name}.tif"
    
    # Get reference point
    area_config = config['calibration']['areas'][area_name]
    reference_coord = area_config['reference_coordinate']
    ground_truth_depth = area_config['ground_truth_depth']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Load DEM
    if dem_path.exists():
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)
            transform = src.transform
            
            # Convert reference to pixel coordinates
            ref_row, ref_col = rowcol(transform, reference_coord[0], reference_coord[1])
            
            # Plot DEM
            masked_dem = np.ma.masked_where(dem_data == src.nodata, dem_data)
            im1 = axes[0, 0].imshow(masked_dem, cmap='terrain')
            axes[0, 0].plot(ref_col, ref_row, 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)
            axes[0, 0].set_title(f'DEM - Reference Point\nElevation: {dem_data[ref_row, ref_col]:.1f}m')
            plt.colorbar(im1, ax=axes[0, 0], label='Elevation (m)')
            
            print(f"Reference pixel: ({ref_row}, {ref_col})")
            print(f"DEM value at reference: {dem_data[ref_row, ref_col]:.1f} m")
    
    # Load prediction
    if pred_file.exists():
        with rasterio.open(pred_file) as src:
            pred_data = src.read(1)
            
            # Plot prediction
            masked_pred = np.ma.masked_where(pred_data == -999, pred_data)
            im2 = axes[0, 1].imshow(masked_pred, cmap='viridis')
            axes[0, 1].plot(ref_col, ref_row, 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)
            
            pred_value_at_ref = pred_data[ref_row, ref_col]
            axes[0, 1].set_title(f'Predicted Snow Depth\nValue at ref: {pred_value_at_ref:.2f}m')
            plt.colorbar(im2, ax=axes[0, 1], label='Snow Depth (m)')
            
            print(f"Predicted value at reference: {pred_value_at_ref:.2f} m")
            print(f"Ground truth: {ground_truth_depth:.2f} m")
            print(f"Required offset: {ground_truth_depth - pred_value_at_ref:.2f} m")
    
    # Load test/validation data if available
    test_file = case_folder / f"{year_pred}__{area_name}" / f"snowdepth_{year_pred}_{area_name}.tif"
    if test_file.exists():
        with rasterio.open(test_file) as src:
            test_data = src.read(1)
            
            # Plot test data
            masked_test = np.ma.masked_where(test_data == -999, test_data)
            im3 = axes[1, 0].imshow(masked_test, cmap='viridis')
            axes[1, 0].plot(ref_col, ref_row, 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)
            
            test_value_at_ref = test_data[ref_row, ref_col]
            axes[1, 0].set_title(f'Test Snow Depth\nValue at ref: {test_value_at_ref:.2f}m')
            plt.colorbar(im3, ax=axes[1, 0], label='Snow Depth (m)')
            
            print(f"Test/validation value at reference: {test_value_at_ref:.2f} m")
            
            # Calculate and plot difference
            if pred_file.exists():
                diff_data = test_data - pred_data
                masked_diff = np.ma.masked_where((test_data == -999) | (pred_data == -999), diff_data)
                
                im4 = axes[1, 1].imshow(masked_diff, cmap='RdBu_r', vmin=-2, vmax=2)
                axes[1, 1].plot(ref_col, ref_row, 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)
                
                diff_at_ref = diff_data[ref_row, ref_col]
                axes[1, 1].set_title(f'Test - Predicted\nDiff at ref: {diff_at_ref:.2f}m')
                plt.colorbar(im4, ax=axes[1, 1], label='Difference (m)')
                
                print(f"Test - Predicted at reference: {diff_at_ref:.2f} m")
    
    plt.tight_layout()
    plt.savefig(outline_folder / "reference_point_debug.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Reference coordinate: {reference_coord}")
    print(f"Reference pixel: ({ref_row}, {ref_col})")
    print(f"Ground truth depth: {ground_truth_depth:.2f} m")
    if pred_file.exists():
        print(f"Model prediction: {pred_value_at_ref:.2f} m")
        print(f"Required calibration offset: {ground_truth_depth - pred_value_at_ref:.2f} m")


def check_prediction_variation(config, area_name="75129", feature_name="tpi_20_elevation_enhanced"):
    """
    Check if your prediction actually has spatial variation or if it's somehow constant.
    """
    import rasterio
    import numpy as np
    
    print(f"=== CHECKING PREDICTION SPATIAL VARIATION ===")
    
    # Get prediction file path (update with your actual path structure)
    year_pred = config['analysis']['year_pred']
    output_folder = config['paths']['output_folder']
    case_name = "your_case_name"  # Update this
    result_folder = output_folder / "results" / case_name
    outline_folder = result_folder / f"{year_pred}__{area_name}"
    pred_file = outline_folder / f"predicted_snowdepth_{year_pred}_{area_name}_{feature_name}.tif"
    
    if not pred_file.exists():
        print(f"❌ Prediction file not found: {pred_file}")
        # List available files
        if outline_folder.exists():
            available = list(outline_folder.glob("*.tif"))
            print(f"Available files: {[f.name for f in available]}")
        return
    
    print(f"✓ Analyzing: {pred_file}")
    
    with rasterio.open(pred_file) as src:
        pred_data = src.read(1)
        
        # Get valid data
        valid_mask = (pred_data != -999) & (~np.isnan(pred_data))
        valid_data = pred_data[valid_mask]
        
        if len(valid_data) == 0:
            print("❌ No valid prediction data found!")
            return
        
        print(f"✓ Valid pixels: {len(valid_data):,}")
        print(f"✓ Prediction range: {np.min(valid_data):.3f} to {np.max(valid_data):.3f} m")
        print(f"✓ Prediction mean: {np.mean(valid_data):.3f} m")
        print(f"✓ Prediction std: {np.std(valid_data):.3f} m")
        
        # Check if prediction is suspiciously uniform
        if np.std(valid_data) < 0.001:
            print("❌ WARNING: Prediction has almost no variation! (std < 0.001)")
            print("   This suggests your model isn't working properly")
        elif np.std(valid_data) < 0.01:
            print("⚠️  WARNING: Very low prediction variation (std < 0.01)")
            print("   Your model might have issues")
        else:
            print(f"✓ Good: Prediction has reasonable variation (std = {np.std(valid_data):.3f})")
        
        # Check for constant values
        unique_values = np.unique(valid_data)
        print(f"✓ Number of unique values: {len(unique_values)}")
        
        if len(unique_values) == 1:
            print("❌ CRITICAL: All predicted values are identical!")
            print(f"   All values = {unique_values[0]:.6f}")
        elif len(unique_values) < 10:
            print("⚠️  WARNING: Very few unique values in prediction")
            print(f"   Unique values: {unique_values}")


# Quick fix function to run all diagnostics
def run_full_calibration_diagnostics(config):
    """
    Run all calibration diagnostics to identify the issue.
    """
    print("🔍 Running full calibration diagnostics...")
    print("=" * 50)
    
    # 1. Check basic setup
    debug_calibration_setup(config)
    print("\n" + "=" * 50)
    
    # 2. Check prediction variation
    check_prediction_variation(config)
    print("\n" + "=" * 50)
    
    # 3. Visualize reference point
    # visualize_reference_point_context(config)  # Uncomment to run this
    
    print("\n🔍 Diagnostics complete!")
    print("\nNext steps:")
    print("1. Fix any ❌ issues identified above")
    print("2. If prediction has no variation, check your terrain model training")
    print("3. If reference point is outside bounds, update coordinates")
    print("4. Re-run calibration after fixes")

def create_valid_mask(array, nodata_value=-999):
    """
    Create a mask that filters out both NaN values and nodata values,
    handling different array types safely.
    
    Parameters:
    -----------
    array : numpy.ndarray
        Input array to create mask for
    nodata_value : float or int, optional
        The nodata value to exclude, defaults to -999
        
    Returns:
    --------
    numpy.ndarray
        Boolean mask where True indicates valid data
    """
    # Check array type first - integer arrays don't support NaN
    if np.issubdtype(array.dtype, np.integer):
        # For integer arrays, just check for nodata value
        return array != nodata_value
    else:
        # For float arrays, check both NaN and nodata value
        return ~np.isnan(array) & (array != nodata_value)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Convert string paths to Path objects with robust handling
    for key, value in config['paths'].items():
        if value and isinstance(value, str):
            # Remove 'r"' prefix if present
            if value.startswith('r"') or value.startswith("r'"):
                value = value[2:-1]
            
            # Remove surrounding quotes if present
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            
            # Normalize path: replace backslashes, convert to absolute path
            normalized_path = value.replace('\\', '/')
            path_obj = Path(normalized_path).resolve()
            
            config['paths'][key] = path_obj
            
            # If path doesn't exist, print more diagnostic information
            if not path_obj.exists():
                print(f"Warning: Path for {key} does not exist: {path_obj}")
    
    return config


def validate_config(config):
    """Validate configuration parameters"""
    # Skip validation if disabled
    if not config.get('validate_paths', True):
        print("Path validation skipped")
        return
        
    # Check if neighbourhood is a multiple of pixel size
    pixel_size = config['analysis']['pixel_size']
    neighbourhood_meter = config['analysis']['neighbourhood_meter']
    if neighbourhood_meter % pixel_size != 0:
        raise ValueError(f"Neighbourhood ({neighbourhood_meter}) must be a multiple of pixel size ({pixel_size})")
    
    # Check if required paths exist
    library_dir = config['paths']['library_dir']
    if not library_dir.exists():
        raise FileNotFoundError(f"Library directory not found: {library_dir}")
    
    # List of paths to check
    paths_to_check = [
        'snow_depth_train',
        'dem_path',
        'avy_outlines',
    ]
    
    for path_key in paths_to_check:
        file_path = config['paths'].get(path_key)
        if file_path and not file_path.exists():
            print(f"Checking path for {path_key}: {file_path}")
            raise FileNotFoundError(f"Required file not found: {file_path}")
            
    print("Configuration validated successfully")


def setup_directories(config):
    """Set up directory structure for outputs"""
    # Create case name for output organization
    name_train_area = config['analysis']['name_train_area']
    name_test_area = config['analysis']['name_test_area']
    year_train = config['analysis']['year_train']
    year_pred = config['analysis']['year_pred']
    pixel_size = config['analysis']['pixel_size']
    neighbourhood_meter = config['analysis']['neighbourhood_meter']
    data_type = config['analysis']['data_type']
    
    case_name = f"{name_train_area}_{year_train}_{name_test_area}__{year_pred}_res{pixel_size}_tpi{neighbourhood_meter}"
    
    # Create main output folders
    output_folder = config['paths']['output_folder']
    result_folder = output_folder / "results"
    data_folder = output_folder / "data"
    case_folder = data_folder / case_name
    case_result_folder = result_folder / case_name
    
    # Create directories
    for folder in [output_folder, result_folder, data_folder, case_folder, case_result_folder]:
        folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Directories created")
    
    return result_folder, data_folder, case_folder, case_result_folder, case_name


def get_feature_geometry(shapefile_path, feature_id):
    """
    Extract geometry from shapefile using feature ID.
    
    Args:
        shapefile_path: Path to the shapefile (.shp)
        feature_id: ID of the feature to extract
    
    Returns:
        Shapely geometry object
    """
    
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Find ID column
    id_columns = ['ID', 'id', 'FID', 'fid', 'OBJECTID', 'objectid']
    id_column = None
    
    for col in id_columns:
        if col in gdf.columns:
            id_column = col
            break
    
    if id_column is None:
        print(f"Available columns: {list(gdf.columns)}")
        return None
    
    # Find feature by ID
    feature = gdf[gdf[id_column] == feature_id]
    
    # Try string/int conversion if not found
    if feature.empty and isinstance(feature_id, str) and feature_id.isdigit():
        feature = gdf[gdf[id_column] == int(feature_id)]
    elif feature.empty and isinstance(feature_id, (int, float)):
        feature = gdf[gdf[id_column] == str(feature_id)]
    
    if feature.empty:
        print(f"Feature ID '{feature_id}' not found")
        return None
    
    return feature.geometry.iloc[0]


def clip_raster_by_geometry(raster_path, geometry, output_path=None):
    """
    Clip a raster using exact geometry (not just extent).
    
    Args:
        raster_path: Path to input raster
        geometry: Shapely geometry object
        output_path: Optional path to save clipped raster
    
    Returns:
        Clipped array and metadata
    """
    with rasterio.open(raster_path) as src:
        # Convert geometry to GeoJSON-like dict
        geom = [mapping(geometry)]
        
        # Clip the raster with the geometry
        out_image, out_transform = mask(src, geom, crop=True, nodata=-999)
        
        # Update metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": -999
        })
        
        # Save if output path provided
        if output_path:
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
        
        return out_image, out_meta


def process_outline_terrain_2(config, job, outline, year, snowdepth_path, dem_path, case_folder, ps):
    """
    Process terrain features for a single outline using MIXED approach:
    - Calculate terrain parameters on bounding box
    - Store exact geometry for later use in training/prediction
    """
    # Create folder for this outline
    outline_folder = case_folder / f"{year}__{outline}"
    outline_folder.mkdir(parents=True, exist_ok=True)
    
    # Get and store the exact geometry for later use
    outline_geometry = get_feature_geometry(str(config['paths']['avy_outlines_shp']), outline)
    if outline_geometry is None:
        print(f"Could not find geometry for outline {outline}")
        return {}
    
    # Store geometry for later use
    OUTLINE_GEOMETRIES[job][outline] = outline_geometry
    
    # Get bounding box extent directly from geometry
    minx, miny, maxx, maxy = outline_geometry.bounds
    outline_extent = box(minx, miny, maxx, maxy)  # Create bbox geometry
    
    # Get bounding box extent for terrain parameter calculation
    #outline_extent = ps.get_feature_extent(str(config['paths']['avy_outlines_csv']), outline, str(dem_path)) #old version
    
    # Define output paths
    output_dem = outline_folder / f"dem_{year}_{outline}.tif"
    output_snowdepth = outline_folder / f"snowdepth_{year}_{outline}.tif"
    
    # Process using BOUNDING BOX for terrain calculations
    layers = {}
    
    if snowdepth_path is not None:
        
        # Clip DEM and snowdepth using bounding box
        layers["elevation"] = np.squeeze(ps.clip_raster(
            str(dem_path), str(output_dem), outline_extent), axis=0)
        layers["snowdepth"] = np.squeeze(ps.clip_raster(
            str(snowdepth_path), str(output_snowdepth), outline_extent), axis=0)
        
        # Match array shapes
        layers["snowdepth"], layers["elevation"] = ps.match_array_shapes(
            layers["snowdepth"], layers["elevation"])
        
        # Crop rasters to remove edge artifacts
        layers["snowdepth"] = layers["snowdepth"][2:-2, 2:-2]
        layers["elevation"] = layers["elevation"][2:-2, 2:-2]
        
        # Save processed layers back to files
        _, meta_export, _ = ps.load_raster(str(output_dem), metadata=["meta", "nodata"])
        
        # Update metadata for new dimensions
        meta_updated = meta_export.copy()
        meta_updated.update({
            "height": layers["elevation"].shape[0],
            "width": layers["elevation"].shape[1],
        })
        
        # Export updated layers
        with rasterio.open(str(output_dem), "w", **meta_updated) as dest:
            dest.write(layers["elevation"].astype(rasterio.float32), 1)
        
        with rasterio.open(str(output_snowdepth), "w", **meta_updated) as dest:
            dest.write(layers["snowdepth"].astype(rasterio.float32), 1)
    else:
        # Only clip DEM if no snowdepth data available
        layers["elevation"] = np.squeeze(ps.clip_raster(
            str(dem_path), str(output_dem), outline_extent), axis=0)
    
    # Calculate terrain parameters on the BOUNDING BOX
    pixel_size = config['analysis']['pixel_size']
    
    # Calculate slope
    layers["slope"] = ps.calculate_slope(str(output_dem), pixel_size, str(outline_folder))
    
    # Calculate aspect
    aspect_result = ps.calculate_aspect(str(output_dem), pixel_size, str(outline_folder))
    layers["aspect"] = aspect_result["aspect_deviation"]
    
    # Calculate TPI
    neighbourhood_meter = config['analysis']['neighbourhood_meter']
    neighbourhood_pixels = neighbourhood_meter // pixel_size
    layers[f"tpi_{neighbourhood_meter}"] = ps.calculate_tpi(
        str(output_dem), neighbourhood_pixels, str(outline_folder))
    
    # Calculate curvature
    layers[f"curvature_{neighbourhood_meter}"] = ps.calculate_curvature(
        str(output_dem), neighbourhood_pixels, str(outline_folder))
    
    # Calculate laplace
    layers["laplace"] = ps.calculate_laplace(
        str(output_dem), pixel_size, str(outline_folder))
    
    return layers

def process_outline_terrain_buuferattempt(config, job, outline, year, snowdepth_path, dem_path, case_folder, ps):
    """
    Process terrain features using buffered bounding box to avoid edge artifacts
    """
    # Create folder for this outline
    outline_folder = case_folder / f"{year}__{outline}"
    outline_folder.mkdir(parents=True, exist_ok=True)
    
    # Get and store the exact geometry for later use
    outline_geometry = get_feature_geometry(str(config['paths']['avy_outlines_shp']), outline)
    if outline_geometry is None:
        print(f"Could not find geometry for outline {outline}")
        return {}
    
    # Store geometry for later use
    OUTLINE_GEOMETRIES[job][outline] = outline_geometry
    
    # Calculate buffer size based on neighborhood parameters
    neighbourhood_meter = config['analysis']['neighbourhood_meter']
    pixel_size = config['analysis']['pixel_size']
    
    # Buffer the bounding box by neighborhood size (+ small safety margin)
    buffer_meters = neighbourhood_meter + (pixel_size * 2)  # Extra pixels for safety
    
    print(f"  Using {buffer_meters}m buffer for outline {outline}")
    
    # Get original bounding box and buffer it
    minx, miny, maxx, maxy = outline_geometry.bounds
    outline_extent = box(minx, miny, maxx, maxy)
    
    # Create buffered bounding box
    buffered_minx = minx - buffer_meters
    buffered_miny = miny - buffer_meters  
    buffered_maxx = maxx + buffer_meters
    buffered_maxy = maxy + buffer_meters
    
    buffered_extent = box(buffered_minx, buffered_miny, buffered_maxx, buffered_maxy)
    
    print(f"  Original bounds: ({minx:.0f}, {miny:.0f}, {maxx:.0f}, {maxy:.0f})")
    print(f"  Buffered bounds: ({buffered_minx:.0f}, {buffered_miny:.0f}, {buffered_maxx:.0f}, {buffered_maxy:.0f})")
    
    # Define output paths
    output_dem = outline_folder / f"dem_{year}_{outline}.tif"
    output_snowdepth = outline_folder / f"snowdepth_{year}_{outline}.tif"
    
    
    # Process using BUFFERED BOUNDING BOX for terrain calculations
    layers = {}
    
    # Clip DEM and snowdepth using bounding box
    ps.clip_raster(str(dem_path), str(output_dem), buffered_extent)
    
    
    # Calculate terrain parameters on the BUFFERED BOUNDING BOX
    # (Now all edge pixels have complete neighborhoods!)
    
    # Calculate slope
    slope = ps.calculate_slope(str(output_dem), pixel_size, str(outline_folder))
    
    # Calculate aspect
    aspect = ps.calculate_aspect(str(output_dem), pixel_size, str(outline_folder))
    aspect = aspect["aspect_deviation"]
    
    # Calculate TPI (now with complete neighborhoods at all locations)
    neighbourhood_pixels = neighbourhood_meter // pixel_size
    tpi  = ps.calculate_tpi(
        str(output_dem), neighbourhood_pixels, str(outline_folder))
    
    # Calculate curvature
    curvature = ps.calculate_curvature(
        str(output_dem), neighbourhood_pixels, str(outline_folder))
    
    # Calculate laplace
    laplace = ps.calculate_laplace(
        str(output_dem), pixel_size, str(outline_folder))
    
    print(f"All terrain parameter sucessfully calculated on buffered bounds")

    
    #   reshaping
    layers["elevation"] = np.squeeze(ps.clip_raster(
        str(dem_path), str(output_dem), outline_extent), axis=0)
    
    layers["snowdepth"] = np.squeeze(ps.clip_raster(
       str(snowdepth_path), str(output_snowdepth), outline_extent), axis=0)
   
    target_shape = layers["elevation"].shape
    
    
    # Reshape terrain features to match aligned DEM/snow depth
    layers["slope"] = slope[:target_shape[0], :target_shape[1]]
    layers["aspect"] = aspect[:target_shape[0], :target_shape[1]]
    layers[f"tpi_{neighbourhood_meter}"] = tpi[:target_shape[0], :target_shape[1]]
    layers[f"curvature_{neighbourhood_meter}"] = curvature[:target_shape[0], :target_shape[1]]
    layers["laplace"] = laplace[:target_shape[0], :target_shape[1]]
      
    
    return layers

def process_outline_terrain(config, job, outline, year, snowdepth_path, dem_path, case_folder, ps):
    """
    Process terrain features using buffered bounding box to avoid edge artifacts
    """
    # Create folder for this outline
    outline_folder = case_folder / f"{year}__{outline}"
    outline_folder.mkdir(parents=True, exist_ok=True)
    
    # Get and store the exact geometry for later use
    outline_geometry = get_feature_geometry(str(config['paths']['avy_outlines_shp']), outline)
    if outline_geometry is None:
        print(f"Could not find geometry for outline {outline}")
        return {}
    
    # Store geometry for later use
    OUTLINE_GEOMETRIES[job][outline] = outline_geometry
    
    # Calculate buffer size based on neighborhood parameters
    neighbourhood_meter = config['analysis']['neighbourhood_meter']
    pixel_size = config['analysis']['pixel_size']
    
    # Buffer the bounding box by neighborhood size (+ small safety margin)
    buffer_meters = neighbourhood_meter + (pixel_size * 2)  # Extra pixels for safety
    
    print(f"  Using {buffer_meters}m buffer for outline {outline}")
    
    # Get original bounding box and buffer it
    minx, miny, maxx, maxy = outline_geometry.bounds
    
    # Create buffered bounding box
    buffered_minx = minx - buffer_meters
    buffered_miny = miny - buffer_meters  
    buffered_maxx = maxx + buffer_meters
    buffered_maxy = maxy + buffer_meters
    
    buffered_extent = box(buffered_minx, buffered_miny, buffered_maxx, buffered_maxy)
    
    print(f"  Original bounds: ({minx:.0f}, {miny:.0f}, {maxx:.0f}, {maxy:.0f})")
    print(f"  Buffered bounds: ({buffered_minx:.0f}, {buffered_miny:.0f}, {buffered_maxx:.0f}, {buffered_maxy:.0f})")
    
    # Define output paths
    output_dem = outline_folder / f"dem_{year}_{outline}.tif"
    output_snowdepth = outline_folder / f"snowdepth_{year}_{outline}.tif"
    
    # Process using BUFFERED BOUNDING BOX for terrain calculations
    layers = {}
    
    if snowdepth_path is not None:
        # Clip DEM and snowdepth using buffered bounding box
        layers["elevation"] = np.squeeze(ps.clip_raster(
            str(dem_path), str(output_dem), buffered_extent), axis=0)
        layers["snowdepth"] = np.squeeze(ps.clip_raster(
            str(snowdepth_path), str(output_snowdepth), buffered_extent), axis=0)
        
        # Match array shapes (no resampling - data is already aligned)
        layers["snowdepth"], layers["elevation"] = ps.match_array_shapes(
            layers["snowdepth"], layers["elevation"])
        
    else:
        # Only clip DEM if no snowdepth data available
        layers["elevation"] = np.squeeze(ps.clip_raster(
            str(dem_path), str(output_dem), buffered_extent), axis=0)
    
    # Calculate terrain parameters on the BUFFERED BOUNDING BOX
    # (Now all edge pixels have complete neighborhoods!)
    
    # Calculate slope
    layers["slope"] = ps.calculate_slope(str(output_dem), pixel_size, str(outline_folder))
    
    # Calculate aspect
    aspect_result = ps.calculate_aspect(str(output_dem), pixel_size, str(outline_folder))
    layers["aspect"] = aspect_result["aspect_deviation"]
    
    # Calculate TPI (now with complete neighborhoods at all locations)
    neighbourhood_pixels = neighbourhood_meter // pixel_size
    layers[f"tpi_{neighbourhood_meter}"] = ps.calculate_tpi(
        str(output_dem), neighbourhood_pixels, str(outline_folder))
    
    # Calculate curvature
    layers[f"curvature_{neighbourhood_meter}"] = ps.calculate_curvature(
        str(output_dem), neighbourhood_pixels, str(outline_folder))
    
    # Calculate laplace
    layers["laplace"] = ps.calculate_laplace(
        str(output_dem), pixel_size, str(outline_folder))
    
    # NO MORE MANUAL CROPPING! 
    # The geometry masking in extract_data_within_geometry() will handle the final clipping
    
    return layers

def extract_data_within_geometry(terrain_data, snowdepth_data, geometry, raster_path):
    """
    Extract data from rasters within exact geometry bounds.
    
    Args:
        terrain_data: 2D array of terrain feature data
        snowdepth_data: 2D array of snow depth data (can be None)
        geometry: Shapely geometry object
        raster_path: Path to reference raster for transform info
    
    Returns:
        terrain_data_masked, snowdepth_data_masked (both masked to geometry)
    """
    # Open the raster to get transform information
    with rasterio.open(raster_path) as src:
        # Create a mask for the geometry
        geom = [mapping(geometry)]
        mask_array, mask_transform = mask(src, geom, crop=False, nodata=-999)
        
        # Get the mask (True where data is valid)
        geometry_mask = mask_array[0] != -999
        
        # Apply mask to terrain data
        terrain_masked = np.where(geometry_mask, terrain_data, -999)
        
        # Apply mask to snowdepth data if available
        if snowdepth_data is not None:
            snowdepth_masked = np.where(geometry_mask, snowdepth_data, -999)
        else:
            snowdepth_masked = None
    
    return terrain_masked, snowdepth_masked, geometry_mask


def process_outlines(config, case_folder, ps):
    """Process all training and prediction outlines"""
    # Setup outline lists and parameters
    outline_sets = {
        "train": config['areas']['avy_outline_train'],
        "pred": config['areas']['avy_outline_pred']
    }
    
    years = {
        "train": config['analysis']['year_train'],
        "pred": config['analysis']['year_pred']
    }
    
    snowdepths = {
        "train": config['paths']['snow_depth_train'],
        "pred": config['paths']['snow_depth_test']
    }
    
    snowdepth_paths = []
    
    # Get list of terrain features
    terrain_features = []
    for feature in config['terrain_features']:
        # Replace any placeholders in the feature names
        if '{neighbourhood_meter}' in feature:
            feature = feature.format(neighbourhood_meter=config['analysis']['neighbourhood_meter'])
        terrain_features.append(feature)
    
    # Initialize TERRAIN_LAYERS structure
    for job in ["train", "pred"]:
        if job not in TERRAIN_LAYERS:
            TERRAIN_LAYERS[job] = {}
        for feature in terrain_features:
            if feature not in TERRAIN_LAYERS[job]:
                TERRAIN_LAYERS[job][feature] = {}
    
    # Process each job type (train/pred)
    for job in ["train", "pred"]:
        # Use corresponding DEM
        dem_path = config['paths']['dem_path_pred'] if job == "pred" else config['paths']['dem_path']
        
        # Check if we have valid snowdepth data
        has_snowdepth = snowdepths[job] is not None and Path(snowdepths[job]).exists()
        snowdepth_path = snowdepths[job] if has_snowdepth else None
        
        # Process each outline
        for outline in outline_sets[job]:
            print(f"Processing {job} outline: {outline}")
            
            # Process terrain on bounding box
            layers = process_outline_terrain(
                config, job, outline, years[job], snowdepth_path, dem_path, case_folder, ps)
            
            # Store layers in TERRAIN_LAYERS dictionary (still bounding box data)
            for feature in terrain_features:
                if feature in layers:
                    TERRAIN_LAYERS[job][feature][outline] = layers[feature]
            
            # Add snowdepth path to list if available
            if snowdepth_path is not None:
                output_snowdepth = case_folder / f"{years[job]}__{outline}" / f"snowdepth_{years[job]}_{outline}.tif"
                snowdepth_paths.append(str(output_snowdepth))
    
    return snowdepth_paths


    
def plot_correlation(config, x_data, y_data, feature_name, area_name, correlation, 
                    pixel_count, output_folder):
    """Create correlation plot between terrain feature and snow depth"""
    plt.figure(figsize=(8, 6))
    hexbin_gridsize = config['visualization'].get('hexbin_gridsize', 200)
    fontsize = config["visualization"]["fontsize_corr_plots"]
    
    hb = plt.hexbin(x_data, y_data, gridsize=hexbin_gridsize, cmap='magma_r', bins='log')
    plt.xlabel(feature_name,fontsize=fontsize)
    plt.ylabel("HS (m)", fontsize=fontsize)
    
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    
    data_type = config['analysis']['data_type']
    year_train = config['analysis']['year_train']
    plt.title(f" HS ({year_train}) vs. {feature_name}",fontsize=fontsize)
    
    #plt.colorbar(hb, label="Log Count Density")
    cbar = plt.colorbar(hb)
    cbar.set_label("Log Count Density", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize-2)
    
    plt.grid(True)
    
    # Add statistics text
    stats_text = f"r = {correlation:.2f}\nn = {pixel_count:,}"
    plt.text(
        0.05, 0.95, stats_text,
        transform=plt.gca().transAxes, fontsize=fontsize,
        verticalalignment='top', 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )
    
    # Save and display
    plt.savefig(output_folder / f"{feature_name}__truesnow_corr_{area_name}_fontsize{fontsize}.png", format="png", bbox_inches='tight', dpi=300)
    plt.close()


def save_prediction(predicted_snowdepth, terrain_feature_name, area_name, output_folder, case_folder, config):
    """Save prediction to file"""
    # Define output file path
    year_pred = config['analysis']['year_pred']
    output_snowdepth_raster = output_folder / f"predicted_snowdepth_{year_pred}_{area_name}_{terrain_feature_name}.tif"
    
    # Get metadata from DEM or another existing file
    output_dem = case_folder / f"{year_pred}__{area_name}" / f"dem_{year_pred}_{area_name}.tif"
    
    if output_dem.exists():
        _, meta_export, _ = ps.load_raster(str(output_dem), metadata=["meta", "nodata"])
    else:
        # Fallback to original DEM if specific output doesn't exist
        _, meta_export, _ = ps.load_raster(str(config['paths']['dem_path']), metadata=["meta", "nodata"])
    
    # Update metadata for new dimensions
    meta_updated = meta_export.copy()
    meta_updated.update({
        "height": predicted_snowdepth.shape[0],
        "width": predicted_snowdepth.shape[1],
    })
    
    # Export prediction
    with rasterio.open(str(output_snowdepth_raster), "w", **meta_updated) as dest:
        dest.write(predicted_snowdepth.astype(rasterio.float32), 1)
    
    #print(f"Saved snow depth prediction to {output_snowdepth_raster}")


def plot_prediction_map(predicted_snowdepth, feature_name, area_name, output_folder, config):
    """Create visualization of predicted snow depth"""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Get visualization settings
    colormap = config['visualization'].get('colormap', 'cmc.navia_r')
    snow_depth_range = config['visualization'].get('snow_depth_range', [0, 5])
    
    # Ensure colormap exists
    try:
        if colormap.startswith('cmc.'):
            # Import cmcrameri if needed
            from cmcrameri import cm as cmc
            cmap = getattr(cmc, colormap[4:])
        else:
            cmap = colormap
    except (ImportError, AttributeError):
        print(f"Colormap {colormap} not found, using default")
        cmap = 'viridis'
    
    # Plot predicted snow depth
    cax = ax.imshow(predicted_snowdepth, cmap=cmap, 
                   interpolation='nearest', vmin=snow_depth_range[0], vmax=snow_depth_range[1])
    
    data_type = config['analysis']['data_type']
    ax.set_title(f'Predicted {data_type} Snowdepth ({feature_name}) ({area_name})')
    
    # Add colorbar
    fig.colorbar(cax, ax=ax, orientation='vertical', label='HS (m)')
    
    # Save and display
    plt.savefig(output_folder / f"snow_map_predicted_{feature_name}_{area_name}.png", format="png")
    plt.close()


def validate_and_visualize_prediction(predicted_snowdepth, test_snowdepth, 
                                     feature_name, area_name, pixel_count, output_folder, config):
    """Validate and visualize prediction against test data"""
    
    # Mask invalid data
    predicted_mask = create_valid_mask(predicted_snowdepth, -999)
    test_mask = create_valid_mask(test_snowdepth, -999)
    valid_mask = predicted_mask & test_mask

    snow_predicted = predicted_snowdepth[valid_mask]
    snow_pred_flat = snow_predicted.flatten()
    snow_validate = test_snowdepth[valid_mask]
    snow_validate_flat = snow_validate.flatten()
    
    # Calculate metrics
    correlation = np.corrcoef(snow_pred_flat, snow_validate_flat)[0, 1]
    print(f"    Validation: r={correlation:.2f}")
    
    # Calculate RMSE
    diff_pred_true = test_snowdepth - predicted_snowdepth
    diff_1d = snow_validate - snow_predicted
    rmse = np.sqrt(np.mean(diff_1d ** 2))
    print(f"    RMSE: {rmse:.2f}")
    
    # Create correlation plot
    plt.figure(figsize=(8, 6))
    hexbin_gridsize = config['visualization'].get('hexbin_gridsize', 200)
    fontsize = config["visualization"]["fontsize_corr_plots"]
    
    hb = plt.hexbin(snow_pred_flat, snow_validate_flat, gridsize=hexbin_gridsize, cmap='magma_r', bins='log')
    plt.xlabel("HS modeled", fontsize=fontsize)
    plt.ylabel("HS observed ", fontsize=fontsize)
    
    plt.tick_params(axis='both', which='major', labelsize=fontsize)

    
    data_type = config['analysis']['data_type']
    plt.title(f"HS modeled vs. HS observed", fontsize=fontsize)
    
    #plt.colorbar(hb, label="Log Count Density")
    cbar = plt.colorbar(hb)
    cbar.set_label("Log Count Density", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize-2)
    
    plt.grid(True)
    
    # Add statistics text
    stats_text = f"r = {correlation:.2f}\nn = {pixel_count:,}\nRMSE = {rmse:.2f}"
    plt.text(
        0.05, 0.95, stats_text,
        transform=plt.gca().transAxes, fontsize=fontsize,
        verticalalignment='top', 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )
    
    # Save correlation plot
    plt.savefig(output_folder / f"predicted_snow_{feature_name}_{area_name}_font{fontsize}.png", format="png",bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get visualization settings
    colormap = config['visualization'].get('colormap', 'cmc.navia_r')
    diff_colormap = config['visualization'].get('diff_colormap', 'cmc.vik_r')
    
    # Ensure colormaps exist
    try:
        if colormap.startswith('cmc.'):
            from cmcrameri import cm as cmc
            cmap = getattr(cmc, colormap[4:])
        else:
            cmap = colormap
    except (ImportError, AttributeError):
        print(f"Colormap {colormap} not found, using default")
        cmap = 'viridis'
        
    try:
        if diff_colormap.startswith('cmc.'):
            from cmcrameri import cm as cmc
            diff_cmap = getattr(cmc, diff_colormap[4:])
        else:
            diff_cmap = diff_colormap
    except (ImportError, AttributeError):
        print(f"Diff colormap {diff_colormap} not found, using default")
        diff_cmap = 'RdBu_r'
    
    # -----------------------------------------------------------------------
                #NEW
    masked_test = np.ma.masked_where(test_snowdepth == -999, test_snowdepth)
    masked_pred = np.ma.masked_where(predicted_snowdepth == -999, predicted_snowdepth)
    masked_diff = np.ma.masked_where(~valid_mask, diff_pred_true)

    # --------------------------------------------------------------------
    
    # Plot true snow depth (min max scale)
    vmin_true = np.percentile(test_snowdepth[test_snowdepth != -999], 10)
    vmax_true = np.percentile(test_snowdepth[test_snowdepth != -999], 90)
    year_pred = config['analysis']['year_pred']
    cax1 = axes[0].imshow(masked_test, cmap=cmap, interpolation='nearest', vmin=vmin_true, vmax=vmax_true)
    #cax1 = axes[0].imshow(test_snowdepth, cmap=cmap, 
     #                     interpolation='nearest', vmin=vmin_true, vmax=vmax_true)
    axes[0].set_title(f'Observed Snow Depth ({year_pred})')
    fig.colorbar(cax1, ax=axes[0], orientation='vertical', label='Snow Depth (m)')
    
    # Plot predicted snow depth (min max scale)
    vmin_pred = np.percentile(predicted_snowdepth[predicted_snowdepth != -999], 10)
    vmax_pred = np.percentile(predicted_snowdepth[predicted_snowdepth != -999], 90)
    cax2 = axes[1].imshow(masked_pred, cmap=cmap, interpolation='nearest', vmin=vmin_pred, vmax=vmax_pred)
    #cax2 = axes[1].imshow(predicted_snowdepth, cmap=cmap, 
              #           interpolation='nearest', vmin=vmin_pred, vmax=vmax_pred)
    axes[1].set_title(f'Relative Model')
    fig.colorbar(cax2, ax=axes[1], orientation='vertical', label='Snow Depth (m)')
    
    # Plot difference (min max scale)
    valid_diff = diff_pred_true[valid_mask]
    
    if len(valid_diff) > 0:
        # Use percentile-based range for better visualization
        vmin_diff = np.percentile(valid_diff, 10)
        vmax_diff = np.percentile(valid_diff, 90)
        # Make range symmetric around zero
        vmax_abs = max(abs(vmin_diff), abs(vmax_diff))
        vmin_diff, vmax_diff = -vmax_abs, vmax_abs
    else:
        vmin_diff, vmax_diff = -1, 1
        
    
    cax3 = axes[2].imshow(masked_diff, cmap=diff_cmap, interpolation='nearest', vmin=vmin_diff, vmax=vmax_diff)
    #cax3 = axes[2].imshow(diff_pred_true, cmap=diff_cmap, 
     #                    interpolation='nearest', vmin=vmin_diff, vmax=vmax_diff)
    axes[2].set_title(f'Model Error\n(Observed - Modeled relative)')
    fig.colorbar(cax3, ax=axes[2], orientation='vertical', label='Snow Depth (m)')
    
    # Add title
    fig.suptitle(f'Comparison: True vs. Predicted {data_type} Snow Depths ({feature_name})', 
                fontsize=16)
    
    # Save comparison
    plt.tight_layout()
    plt.savefig(output_folder / f"snow_maps_{feature_name}_{area_name}.png", format="png")
    plt.close()

def create_tpi_elevation_combined(config, case_folder):
    """
    
    NOTE: Not checked and optimized after AI
    
    Create combined TPI + elevation residual feature and save it to the case folder.
    This function combines TPI with elevation residuals (deviation from regional trend)
    to enhance terrain characterization without letting elevation dominate.
    
    The residual approach:
    1. Calculates regional elevation trend using polynomial surface fitting
    2. Computes elevation residuals (observed - trend)
    3. Combines normalized TPI with normalized elevation residuals
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary containing analysis parameters
    case_folder : Path
        Path to the case folder where data is stored
    """
    from scipy import ndimage
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    print("Creating enhanced TPI + elevation residual feature...")
    
    # Get the TPI feature name with neighbourhood parameter
    neighbourhood_meter = config['analysis']['neighbourhood_meter']
    tpi_feature_name = f"tpi_{neighbourhood_meter}"
    elevation_feature_name = "elevation"
    combined_feature_name = f"tpi_{neighbourhood_meter}_elevation_enhanced"
    
    # Configuration for enhancement method
    enhancement_config = config.get('tpi_enhancement', {})
    method = enhancement_config.get('method', 'residual')  # 'residual', 'weighted', 'adaptive'
    tpi_weight = enhancement_config.get('tpi_weight', 0.7)
    elevation_weight = enhancement_config.get('elevation_weight', 0.3)
    polynomial_degree = enhancement_config.get('polynomial_degree', 2)
    smoothing_sigma = enhancement_config.get('smoothing_sigma', 5.0)
    
    # Initialize combined feature in TERRAIN_LAYERS if not exists
    for job in ["train", "pred"]:
        if combined_feature_name not in TERRAIN_LAYERS[job]:
            TERRAIN_LAYERS[job][combined_feature_name] = {}
    
    def calculate_elevation_residuals(elevation_array, valid_mask, method='polynomial'):
        """Calculate elevation residuals by removing regional trend"""
        if method == 'polynomial':
            # Create coordinate grids
            rows, cols = elevation_array.shape
            x, y = np.meshgrid(np.arange(cols), np.arange(rows))
            
            # Get valid coordinates and elevation values
            valid_coords = np.column_stack([
                x[valid_mask].flatten(),
                y[valid_mask].flatten()
            ])
            valid_elevations = elevation_array[valid_mask].flatten()
            
            # Fit polynomial surface (degree 2 by default)
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            
            # Create polynomial features
            poly_features = PolynomialFeatures(degree=polynomial_degree)
            X_poly = poly_features.fit_transform(valid_coords)
            
            # Fit regression model
            model = LinearRegression()
            model.fit(X_poly, valid_elevations)
            
            # Predict trend surface for all pixels
            all_coords = np.column_stack([x.flatten(), y.flatten()])
            all_coords_poly = poly_features.transform(all_coords)
            trend_surface = model.predict(all_coords_poly).reshape(elevation_array.shape)
            
            # Calculate residuals
            residuals = np.full_like(elevation_array, -999, dtype=np.float32)
            residuals[valid_mask] = elevation_array[valid_mask] - trend_surface[valid_mask]
            
        elif method == 'gaussian':
            # Use Gaussian smoothing to estimate regional trend
            smoothed = ndimage.gaussian_filter(
                np.where(valid_mask, elevation_array, np.nan), 
                sigma=smoothing_sigma
            )
            residuals = np.full_like(elevation_array, -999, dtype=np.float32)
            residuals[valid_mask] = elevation_array[valid_mask] - smoothed[valid_mask]
        
        return residuals
    
    def normalize_array(array, valid_mask, method='zscore'):
        """Normalize array values"""
        normalized = np.full_like(array, -999, dtype=np.float32)
        
        if method == 'zscore':
            # Z-score normalization
            valid_values = array[valid_mask]
            if len(valid_values) > 1:
                mean_val = np.mean(valid_values)
                std_val = np.std(valid_values)
                if std_val > 0:
                    normalized[valid_mask] = (valid_values - mean_val) / std_val
                else:
                    normalized[valid_mask] = valid_values - mean_val
        
        elif method == 'minmax':
            # Min-max normalization to [0, 1]
            valid_values = array[valid_mask]
            if len(valid_values) > 1:
                min_val = np.min(valid_values)
                max_val = np.max(valid_values)
                if max_val > min_val:
                    normalized[valid_mask] = (valid_values - min_val) / (max_val - min_val)
        
        return normalized
    
    def combine_features(tpi_norm, elevation_feature, valid_mask, method='weighted'):
        """Combine normalized TPI with elevation feature"""
        combined = np.full_like(tpi_norm, -999, dtype=np.float32)
        
        if method == 'weighted':
            # Simple weighted combination
            combined[valid_mask] = (tpi_weight * tpi_norm[valid_mask] + 
                                  elevation_weight * elevation_feature[valid_mask])
        
        elif method == 'adaptive':
            # Adaptive weighting based on TPI magnitude
            tpi_abs = np.abs(tpi_norm[valid_mask])
            # Higher TPI magnitude = more weight on TPI, less on elevation
            adaptive_tpi_weight = 0.5 + 0.4 * (tpi_abs / (np.max(tpi_abs) + 1e-6))
            adaptive_elev_weight = 1.0 - adaptive_tpi_weight
            
            combined[valid_mask] = (adaptive_tpi_weight * tpi_norm[valid_mask] + 
                                  adaptive_elev_weight * elevation_feature[valid_mask])
        
        elif method == 'multiplicative':
            # Multiplicative enhancement (for preserving TPI structure)
            elevation_factor = 1.0 + elevation_weight * elevation_feature[valid_mask]
            combined[valid_mask] = tpi_norm[valid_mask] * elevation_factor
        
        return combined
    
    # Process both training and prediction data
    for job in ["train", "pred"]:
        print(f"Processing {job} data...")
        
        # Check if both TPI and elevation data exist
        if (tpi_feature_name not in TERRAIN_LAYERS[job] or 
            elevation_feature_name not in TERRAIN_LAYERS[job]):
            print(f"Warning: Missing TPI or elevation data for {job}, skipping")
            continue
        
        # Get all outlines for this job
        tpi_outlines = set(TERRAIN_LAYERS[job][tpi_feature_name].keys())
        elevation_outlines = set(TERRAIN_LAYERS[job][elevation_feature_name].keys())
        common_outlines = tpi_outlines.intersection(elevation_outlines)
        
        if not common_outlines:
            print(f"Warning: No common outlines found for {job} data")
            continue
        
        # Process each outline
        for outline in common_outlines:
            print(f"  Processing outline: {outline}")
            
            # Get TPI and elevation arrays
            tpi_array = TERRAIN_LAYERS[job][tpi_feature_name][outline]
            elevation_array = TERRAIN_LAYERS[job][elevation_feature_name][outline]
     
            # ------------------------------ change for flipping coorelation------------
    #       Get TPI and elevation arrays
            tpi_array = TERRAIN_LAYERS[job][tpi_feature_name][outline]
            elevation_array = TERRAIN_LAYERS[job][elevation_feature_name][outline]
            
            # Flip TPI sign since it typically has negative correlation with target
            tpi_array = -tpi_array
                    #    ----------------------------------------------------------
                        
            # Check array shapes match
            if tpi_array.shape != elevation_array.shape:
                print(f"Warning: Shape mismatch for outline {outline} in {job} data")
                print(f"  TPI shape: {tpi_array.shape}, Elevation shape: {elevation_array.shape}")
                continue
            
            # Create valid masks for both arrays
            tpi_valid = create_valid_mask(tpi_array, -999)
            elevation_valid = create_valid_mask(elevation_array, -999)
            valid_mask = tpi_valid & elevation_valid
            
            if np.sum(valid_mask) < 100:  # Need minimum pixels for meaningful analysis
                print(f"Warning: Too few valid pixels ({np.sum(valid_mask)}) for outline {outline}")
                continue
            
            # Calculate elevation feature based on method
            if method == 'residual':
                print(f"    Calculating elevation residuals...")
                elevation_feature = calculate_elevation_residuals(
                    elevation_array, valid_mask, method='polynomial'
                )
            elif method == 'raw':
                elevation_feature = elevation_array.copy()
            else:
                # Default to residual
                elevation_feature = calculate_elevation_residuals(
                    elevation_array, valid_mask, method='polynomial'
                )
            
            # Normalize both TPI and elevation feature
            print(f"    Normalizing features...")
            tpi_normalized = normalize_array(tpi_array, valid_mask, method='zscore')
            elevation_normalized = normalize_array(elevation_feature, valid_mask, method='zscore')
            
            # Combine features
            print(f"    Combining features with method: {method}")
            combined_array = combine_features(
                tpi_normalized, elevation_normalized, valid_mask, 
                method=enhancement_config.get('combination_method', 'weighted')
            )
            
            # Store combined array in TERRAIN_LAYERS
            TERRAIN_LAYERS[job][combined_feature_name][outline] = combined_array
            
            # Save combined array to file for reference
            year = config['analysis']['year_train'] if job == "train" else config['analysis']['year_pred']
            outline_folder = case_folder / f"{year}__{outline}"
            output_file = outline_folder / f"{combined_feature_name}_{year}_{outline}.tif"
            
            # Get metadata from elevation file for consistency
            elevation_file = outline_folder / f"dem_{year}_{outline}.tif"
            if elevation_file.exists():
                try:
                    _, meta_export, _ = ps.load_raster(str(elevation_file), metadata=["meta", "nodata"])
                    
                    # Update metadata for the combined array
                    meta_updated = meta_export.copy()
                    meta_updated.update({
                        "height": combined_array.shape[0],
                        "width": combined_array.shape[1],
                        "nodata": -999
                    })
                    
                    # Save combined array to file
                    with rasterio.open(str(output_file), "w", **meta_updated) as dest:
                        dest.write(combined_array.astype(rasterio.float32), 1)
                    
                    print(f"    Saved enhanced feature to: {output_file}")
                    
                except Exception as e:
                    print(f"    Warning: Could not save enhanced feature for {outline}: {e}")
            else:
                print(f"    Warning: Reference elevation file not found for {outline}")
            
            # Print comprehensive statistics
            valid_pixels = np.sum(valid_mask)
            if valid_pixels > 0:
                # Original TPI stats
                tpi_min, tpi_max, tpi_mean = (np.min(tpi_array[valid_mask]), 
                                             np.max(tpi_array[valid_mask]), 
                                             np.mean(tpi_array[valid_mask]))
                
                # Elevation residual stats (if using residuals)
                if method == 'residual':
                    elev_res_min, elev_res_max, elev_res_mean = (np.min(elevation_feature[valid_mask]),
                                                               np.max(elevation_feature[valid_mask]),
                                                               np.mean(elevation_feature[valid_mask]))
                    print(f"    Elevation residuals - Min: {elev_res_min:.2f}, Max: {elev_res_max:.2f}, Mean: {elev_res_mean:.2f}")
                
                # Combined feature stats
                combined_min, combined_max, combined_mean = (np.min(combined_array[valid_mask]),
                                                          np.max(combined_array[valid_mask]),
                                                          np.mean(combined_array[valid_mask]))
                
                print(f"    Original TPI - Min: {tpi_min:.2f}, Max: {tpi_max:.2f}, Mean: {tpi_mean:.2f}")
                print(f"    Enhanced TPI - Min: {combined_min:.2f}, Max: {combined_max:.2f}, Mean: {combined_mean:.2f}")
                print(f"    Valid pixels: {valid_pixels}")
                
                # Calculate enhancement effectiveness
                tpi_range = tpi_max - tpi_min
                combined_range = combined_max - combined_min
                if tpi_range > 0:
                    range_ratio = combined_range / tpi_range
                    print(f"    Range enhancement factor: {range_ratio:.2f}")
            else:
                print(f"    Warning: No valid pixels found for outline {outline}")
    
    # Add combined feature to terrain_features list in config if not already present
    if 'terrain_features' in config:
        if combined_feature_name not in config['terrain_features']:
            config['terrain_features'].append(combined_feature_name)
            print(f"Added {combined_feature_name} to terrain features list")
    
    # Log configuration used
    print(f"\nEnhancement configuration used:")
    print(f"  Method: {method}")
    print(f"  TPI weight: {tpi_weight}")
    print(f"  Elevation weight: {elevation_weight}")
    if method == 'residual':
        print(f"  Polynomial degree: {polynomial_degree}")
    
    print(f"Enhanced TPI + elevation feature creation completed: {combined_feature_name}")
    return combined_feature_name

def train_and_predict(case_folder, case_result_folder, config, ps):
    """
    Train models on terrain features and predict snow depth
    MIXED APPROACH: Extract training data using exact geometry, predict using exact geometry
    """
    # Get terrain features list with placeholders replaced
    terrain_features = []
    for feature in config['terrain_features']:
        # Replace any placeholders in the feature names
        if '{neighbourhood_meter}' in feature:
            feature = feature.format(neighbourhood_meter=config['analysis']['neighbourhood_meter'])
        terrain_features.append(feature)
    
    # Process each terrain feature
    for terrain_feature_name in terrain_features:
        # Skip snowdepth as it's the target variable
        if terrain_feature_name == "snowdepth":
            continue
        
        # Skip if feature is not in TERRAIN_LAYERS
        if terrain_feature_name not in TERRAIN_LAYERS["train"]:
            print(f"Terrain feature {terrain_feature_name} not found in training data, skipping")
            continue
            
        print(f"Processing terrain feature: {terrain_feature_name}")
        
        # Create combined arrays for training
        training_features_combined = np.array([]).reshape(-1, 1)
        training_snowdepth_combined = np.array([])
        
        # Process each training area using EXACT GEOMETRY
        year_train = config['analysis']['year_train']
        for single_outline_train in TERRAIN_LAYERS["train"][terrain_feature_name].keys():
            # Create result folder for this terrain feature
            terrain_folder_result = case_result_folder / f"{year_train}_{terrain_feature_name}_train"
            terrain_folder_result.mkdir(parents=True, exist_ok=True)
            
            # Get the stored geometry for this outline
            if single_outline_train not in OUTLINE_GEOMETRIES["train"]:
                print(f"Geometry not found for {single_outline_train}, skipping")
                continue
                
            geometry = OUTLINE_GEOMETRIES["train"][single_outline_train]
            
            # Extract data from dictionary (this is bounding box data)
            training_terrain_bbox = TERRAIN_LAYERS["train"][terrain_feature_name][single_outline_train]
            
            # Skip if snowdepth data is not available
            if ("snowdepth" not in TERRAIN_LAYERS["train"] or 
                single_outline_train not in TERRAIN_LAYERS["train"]["snowdepth"]):
                print(f"Snow depth data missing for {single_outline_train}, skipping")
                continue
                
            training_snowdepth_bbox = TERRAIN_LAYERS["train"]["snowdepth"][single_outline_train]
            
            # Get path to DEM for reference
            dem_path = case_folder / f"{year_train}__{single_outline_train}" / f"dem_{year_train}_{single_outline_train}.tif"
            
            #   NEW
            training_terrain_bbox, training_snowdepth_bbox = ps.match_array_shapes(training_terrain_bbox, training_snowdepth_bbox)#TODO:NEW

        
            # Extract data within exact geometry
            training_terrain_geom, training_snowdepth_geom, geometry_mask = extract_data_within_geometry(
                training_terrain_bbox, training_snowdepth_bbox, geometry, str(dem_path))
            
            # Get nodata value from config or use default
            nodata_value = config.get('analysis', {}).get('nodata_value', -999)
            
            # Create valid mask combining geometry mask and data validity
            terrain_mask = create_valid_mask(training_terrain_geom, nodata_value)
            snowdepth_mask = create_valid_mask(training_snowdepth_geom, nodata_value)
            valid_mask = terrain_mask & snowdepth_mask & geometry_mask

            training_snowdepth_flat = training_snowdepth_geom[valid_mask].flatten()
            training_terrain_reshaped = training_terrain_geom[valid_mask].reshape(-1, 1)
            
            # Append to combined arrays
            training_features_combined = np.append(
                training_features_combined, training_terrain_reshaped, axis=0)
            training_snowdepth_combined = np.append(
                training_snowdepth_combined, training_snowdepth_flat, axis=0)
        
        # Check if we have enough training data
        pixelcount_training_terrain = training_features_combined.size
        if pixelcount_training_terrain == 0:
            print(f"No valid training data found for {terrain_feature_name}, skipping")
            continue
            
        print(f"    n={pixelcount_training_terrain}")
        
        # Calculate correlation
        correlation = np.corrcoef(training_features_combined.flatten(), 
                                 training_snowdepth_combined)[0, 1]
        print(f"    r={correlation:.2f}")
        
        # Create correlation plot
        plot_correlation(
            config,
            training_features_combined.flatten(), 
            training_snowdepth_combined,
            terrain_feature_name,
            single_outline_train,
            correlation,
            pixelcount_training_terrain,
            terrain_folder_result
        )
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(training_features_combined, training_snowdepth_combined)
        print(f"    Model: Snow Depth = {model.coef_[0]:.2f} * {terrain_feature_name} + {model.intercept_:.2f}")
        
        # Apply predictions based on prediction mode
        prediction_mode = config['analysis']['prediction']
        if prediction_mode == "outlines":
            predict_for_outlines(model, terrain_feature_name, case_folder, case_result_folder, config, ps)
        elif prediction_mode == "extent":
            predict_for_extent(model, terrain_feature_name, case_folder, case_result_folder, config, ps)
        else:
            print(f"Warning: Unrecognized prediction mode '{prediction_mode}', skipping prediction")


def predict_for_outlines(model, terrain_feature_name, case_folder, case_result_folder, config, ps):
    """Generate predictions for specified outlines using EXACT GEOMETRY"""
    # Process each prediction outline
    year_pred = config['analysis']['year_pred']
    for single_outline_pred in TERRAIN_LAYERS["pred"][terrain_feature_name].keys():
        # Create result folder for this outline
        outline_folder_result = case_result_folder / f"{year_pred}__{single_outline_pred}"
        outline_folder_result.mkdir(parents=True, exist_ok=True)
        
        # Get the stored geometry for this outline
        if single_outline_pred not in OUTLINE_GEOMETRIES["pred"]:
            print(f"Geometry not found for {single_outline_pred}, skipping prediction")
            continue
            
        geometry = OUTLINE_GEOMETRIES["pred"][single_outline_pred]
        
        # Extract terrain data for prediction (bounding box data)
        prediction_terrain_bbox = TERRAIN_LAYERS["pred"][terrain_feature_name][single_outline_pred]
        
        # Check if we have test data (for validation)
        snow_depth_test = config['paths']['snow_depth_test']
        has_test_data = (snow_depth_test is not None and
                        "snowdepth" in TERRAIN_LAYERS["pred"] and
                        single_outline_pred in TERRAIN_LAYERS["pred"]["snowdepth"])
        
        # Get path to DEM for reference
        dem_path = case_folder / f"{year_pred}__{single_outline_pred}" / f"dem_{year_pred}_{single_outline_pred}.tif"
        
        # Different processing depending on whether we have test data
        if has_test_data:
            test_snowdepth_bbox = TERRAIN_LAYERS["pred"]["snowdepth"][single_outline_pred]
            
            prediction_terrain_bbox, test_snowdepth_bbox = ps.match_array_shapes(prediction_terrain_bbox, test_snowdepth_bbox) #TODO:NEW

            
            # Extract data within exact geometry
            prediction_terrain_geom, test_snowdepth_geom, geometry_mask = extract_data_within_geometry(
                prediction_terrain_bbox, test_snowdepth_bbox, geometry, str(dem_path))
            
            test_mask = create_valid_mask(test_snowdepth_geom, -999)
            pred_mask = create_valid_mask(prediction_terrain_geom, -999)
            valid_mask = test_mask & pred_mask & geometry_mask
            
            test_snowdepth_valid = test_snowdepth_geom[valid_mask]
            prediction_terrain_valid = prediction_terrain_geom[valid_mask].reshape(-1, 1)
            
            # Pixel count for reporting
            pixelcount_prediction = test_snowdepth_valid.size
            
            # Predict snow depth
            predicted_snowdepth_flat = model.predict(prediction_terrain_valid)
            
            # Reshape to original dimensions
            predicted_snowdepth = ps.reshape_after_nandrop(
                predicted_snowdepth_flat, valid_mask, test_snowdepth_geom, nodata_value=-999)
            
            # Save prediction
            save_prediction(
                predicted_snowdepth,
                terrain_feature_name,
                single_outline_pred,
                outline_folder_result,
                case_folder,
                config
            )
            
            # Validate and visualize
            validate_and_visualize_prediction(
                predicted_snowdepth, 
                test_snowdepth_geom,
                terrain_feature_name,
                single_outline_pred,
                pixelcount_prediction,
                outline_folder_result,
                config
            )
        else:
            # No test data - just predict
            # Extract data within exact geometry
            prediction_terrain_geom, _, geometry_mask = extract_data_within_geometry(
                prediction_terrain_bbox, None, geometry, str(dem_path))
            
            valid_mask = create_valid_mask(prediction_terrain_geom, -999) & geometry_mask
            prediction_terrain_valid = prediction_terrain_geom[valid_mask].reshape(-1, 1)
            
            # Predict snow depth
            predicted_snowdepth_flat = model.predict(prediction_terrain_valid)
            
            # Reshape to original dimensions
            predicted_snowdepth = ps.reshape_after_nandrop(
                predicted_snowdepth_flat, valid_mask, prediction_terrain_geom, nodata_value=-999)
            
            # Save prediction
            save_prediction(
                predicted_snowdepth,
                terrain_feature_name,
                single_outline_pred,
                outline_folder_result,
                case_folder,
                config
            )
            
            # Create visualization
            plot_prediction_map(
                predicted_snowdepth,
                terrain_feature_name,
                single_outline_pred,
                outline_folder_result,
                config
            )


def predict_for_extent(model, terrain_feature_name, case_folder, case_result_folder, config, ps):
    """Generate predictions for specified extent"""
    # Create result folder
    year_pred = config['analysis']['year_pred']
    name_test_area = config['analysis']['name_test_area']
    extent_folder_result = case_result_folder / f"{year_pred}__{name_test_area}"
    extent_folder_result.mkdir(parents=True, exist_ok=True)
    
    # Extract terrain data for the extent
    prediction_terrain = TERRAIN_LAYERS["pred"][terrain_feature_name][name_test_area]
    
    # Mask invalid data
    valid_mask = ~np.isnan(prediction_terrain)
    prediction_terrain_valid = prediction_terrain[valid_mask].reshape(-1, 1)
    
    # Pixel count for reporting
    pixelcount_prediction = prediction_terrain_valid.size
    print(f"    n={pixelcount_prediction}")
    
    # Predict snow depth
    predicted_snowdepth_flat = model.predict(prediction_terrain_valid)
    
    # Reshape to original dimensions
    predicted_snowdepth = ps.reshape_after_nandrop(
        predicted_snowdepth_flat, valid_mask, prediction_terrain)
    
    # Save prediction
    save_prediction(
        predicted_snowdepth,
        terrain_feature_name,
        name_test_area,
        extent_folder_result,
        case_folder,
        config
    )
    
    # Create visualization
    plot_prediction_map(
        predicted_snowdepth,
        terrain_feature_name,
        name_test_area,
        extent_folder_result,
        config
    )

# Integration example - Add this to your existing workflow

def calibrate_snow_depth_model(predicted_snowdepth, reference_coord, ground_truth_depth, 
                              raster_path, output_folder, feature_name, area_name, config):
    """
    Calibrate relative snow depth model to absolute values using a single ground truth measurement.
    
    Parameters:
    -----------
    predicted_snowdepth : numpy.ndarray
        2D array of predicted snow depths (relative model)
    reference_coord : tuple
        (x, y) coordinate of the ground truth measurement in the coordinate system of the raster
    ground_truth_depth : float
        Actual snow depth measurement at the reference coordinate (in meters)
    raster_path : str or Path
        Path to reference raster file for coordinate transformation
    output_folder : Path
        Folder to save calibrated results and plots
    feature_name : str
        Name of the terrain feature used for modeling
    area_name : str
        Name of the prediction area
    config : dict
        Configuration dictionary
    
    Returns:
    --------
    calibrated_snowdepth : numpy.ndarray
        Calibrated (absolute) snow depth array
    calibration_offset : float
        The offset applied to calibrate the model
    reference_pixel : tuple
        (row, col) pixel coordinates of the reference point
    """
    import rasterio
    import numpy as np
    import matplotlib.pyplot as plt
    from rasterio.transform import rowcol
    
    print(f"Calibrating snow depth model using reference point...")
    print(f"  Reference coordinate: {reference_coord}")
    print(f"  Ground truth depth: {ground_truth_depth:.2f} m")
    
    # Open raster to get transform information
    with rasterio.open(raster_path) as src:
        # Convert geographic coordinates to pixel coordinates
        try:
            row, col = rowcol(src.transform, reference_coord[0], reference_coord[1])
            reference_pixel = (row, col)
            print(f"  Reference pixel: ({row}, {col})")
        except Exception as e:
            print(f"Error converting coordinates: {e}")
            return None, None, None
    
    # Check if pixel coordinates are within the array bounds
    if (row < 0 or row >= predicted_snowdepth.shape[0] or 
        col < 0 or col >= predicted_snowdepth.shape[1]):
        print(f"Error: Reference pixel ({row}, {col}) is outside array bounds {predicted_snowdepth.shape}")
        return None, None, None
    
    # Get the modeled value at the reference pixel
    modeled_value_at_reference = predicted_snowdepth[row, col]
    
    # Check if the reference pixel has valid data
    if modeled_value_at_reference == -999 or np.isnan(modeled_value_at_reference):
        print(f"Error: No valid modeled data at reference pixel ({row}, {col})")
        return None, None, None
    
    # Calculate calibration offset
    calibration_offset = ground_truth_depth - modeled_value_at_reference
    print(f"  Modeled value at reference: {modeled_value_at_reference:.2f} m")
    print(f"  Calibration offset: {calibration_offset:.2f} m")
    
    # Apply calibration to the entire array
    calibrated_snowdepth = predicted_snowdepth.copy()
    valid_mask = (predicted_snowdepth != -999) & (~np.isnan(predicted_snowdepth))
    calibrated_snowdepth[valid_mask] = predicted_snowdepth[valid_mask] + calibration_offset
    
    # Calculate statistics
    valid_calibrated = calibrated_snowdepth[valid_mask]
    print(f"  Calibrated snow depth range: {np.min(valid_calibrated):.2f} to {np.max(valid_calibrated):.2f} m")
    print(f"  Calibrated snow depth mean: {np.mean(valid_calibrated):.2f} m")
    
    # Save calibrated snow depth
    save_calibrated_prediction(calibrated_snowdepth, feature_name, area_name, 
                              output_folder, raster_path, config, calibration_offset)
    
    return calibrated_snowdepth, calibration_offset, reference_pixel


def save_calibrated_prediction(calibrated_snowdepth, terrain_feature_name, area_name, 
                              output_folder, reference_raster_path, config, calibration_offset):
    """Save calibrated prediction to file"""
    year_pred = config['analysis']['year_pred']
    output_snowdepth_raster = output_folder / f"calibrated_snowdepth_{year_pred}_{area_name}_{terrain_feature_name}.tif"
    
    # Get metadata from reference raster
    _, meta_export, _ = ps.load_raster(str(reference_raster_path), metadata=["meta", "nodata"])
    
    # Update metadata for new dimensions
    meta_updated = meta_export.copy()
    meta_updated.update({
        "height": calibrated_snowdepth.shape[0],
        "width": calibrated_snowdepth.shape[1],
        "nodata": -999
    })
    
    # Add calibration info to metadata
    meta_updated['descriptions'] = [f'Calibrated snow depth (offset: {calibration_offset:.2f}m)']
    
    # Export calibrated prediction
    with rasterio.open(str(output_snowdepth_raster), "w", **meta_updated) as dest:
        dest.write(calibrated_snowdepth.astype(rasterio.float32), 1)
    
    #print(f"  Saved calibrated snow depth to: {output_snowdepth_raster}")


def plot_calibrated_comparison(ground_truth_depth, calibrated_snowdepth, predicted_snowdepth,
                                    test_snowdepth, reference_pixel, feature_name, area_name, 
                                    output_folder, config):
    """
    FIXED VERSION: Create comprehensive visualization comparing ground truth reference, 
    calibrated model, and TEST DATA (not original prediction).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # gezt fontsize from config file    
    fontsize = config["visualization"]["fontsize_maps"]

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Get visualization settings
    colormap = config['visualization'].get('colormap', 'cmc.navia_r')
    diff_colormap = config['visualization'].get('diff_colormap', 'cmc.vik_r')
    
    # Handle colormaps
    try:
        if colormap.startswith('cmc.'):
            from cmcrameri import cm as cmc
            cmap = getattr(cmc, colormap[4:])
        else:
            cmap = colormap
    except (ImportError, AttributeError):
        print(f"Colormap {colormap} not found, using default")
        cmap = 'viridis'
        
    try:
        if diff_colormap.startswith('cmc.'):
            from cmcrameri import cm as cmc
            diff_cmap = getattr(cmc, diff_colormap[4:])
        else:
            diff_cmap = diff_colormap
    except (ImportError, AttributeError):
        print(f"Diff colormap {diff_colormap} not found, using default")
        diff_cmap = 'RdBu_r'
    
    # Create masks for valid data
    calibrated_mask = np.ma.masked_where(calibrated_snowdepth == -999, calibrated_snowdepth)
    test_mask = np.ma.masked_where(test_snowdepth == -999, test_snowdepth)
    
    # Calculate difference: TEST - CALIBRATED (this should show real model errors)
    difference = test_snowdepth - calibrated_snowdepth  
    valid_diff_mask = (calibrated_snowdepth != -999) & (test_snowdepth != -999)
    difference_mask = np.ma.masked_where(~valid_diff_mask, difference)
    
    # Plot 1: Test/Observed Snow Depth
    valid_test = test_snowdepth[test_snowdepth != -999]
    valid_cal = calibrated_snowdepth[calibrated_snowdepth != -999]
    
    # Calculate common scale using both datasets
    all_valid = np.concatenate([valid_test, valid_cal]) if len(valid_test) > 0 and len(valid_cal) > 0 else (valid_test if len(valid_test) > 0 else valid_cal)
    
    if len(all_valid) > 0:
        vmin_common = np.percentile(all_valid, 10)
        vmax_common = np.percentile(all_valid, 90)
    else:
        vmin_common, vmax_common = 0, 1
    
    cax1 = axes[0].imshow(test_mask, cmap=cmap, interpolation='nearest', 
                         vmin=vmin_common, vmax=vmax_common)
    year_pred = config['analysis']['year_pred']
    data_type = config['analysis']['data_type']
    axes[0].set_title(f'Observed ({year_pred})', fontsize=fontsize)
    axes[0].plot(reference_pixel[1], reference_pixel[0], 'r*', markersize=15, 
                markeredgecolor='white', markeredgewidth=2, label='Reference Point')
    axes[0].legend(loc='upper right')
    #fig.colorbar(cax1, ax=axes[0], orientation='vertical', label='HS (m)',fontsize=fontsize)
    cbar1 = fig.colorbar(cax1, ax=axes[0], orientation='vertical', label='HS (m)')
    cbar1.ax.tick_params(labelsize=fontsize)  # for tick labels
    cbar1.set_label('HS (m)', fontsize=fontsize)  # for the label
    
    # Plot 2: Calibrated Model (using same scale)
    cax2 = axes[1].imshow(calibrated_mask, cmap=cmap, interpolation='nearest',
                         vmin=vmin_common, vmax=vmax_common)
    axes[1].set_title(f'Modeled \n(Reference: {ground_truth_depth:.2f}m)', fontsize=fontsize)
    axes[1].plot(reference_pixel[1], reference_pixel[0], 'r*', markersize=fontsize,
                markeredgecolor='white', markeredgewidth=2, label='Reference Point')
    axes[1].legend(loc='upper right')
    #fig.colorbar(cax2, ax=axes[1], orientation='vertical', label='HS (m)',fontsize=fontsize)

    cbar2 = fig.colorbar(cax2, ax=axes[1], orientation='vertical')
    cbar2.set_label('HS (m)', fontsize=fontsize)
    cbar2.ax.tick_params(labelsize=fontsize)
    
    # Plot 3: Model Error (Observed - Calibrated)
    valid_diff = difference[valid_diff_mask]
    if len(valid_diff) > 0:
        # Use percentile-based range for better visualization
        vmin_diff = np.percentile(valid_diff, 10)
        vmax_diff = np.percentile(valid_diff, 90)
        # Make range symmetric around zero
        vmax_abs = max(abs(vmin_diff), abs(vmax_diff))
        vmin_diff, vmax_diff = -vmax_abs, vmax_abs
    else:
        vmin_diff, vmax_diff = -1, 1
    
    cax3 = axes[2].imshow(difference_mask, cmap=diff_cmap, interpolation='nearest',
                         vmin=vmin_diff, vmax=vmax_diff)
    axes[2].set_title(f'Model Error\n(Observed - Calibrated)',fontsize=fontsize)
    axes[2].plot(reference_pixel[1], reference_pixel[0], 'r*', markersize=fontsize,
                markeredgecolor='white', markeredgewidth=2, label='Reference Point')
    axes[2].legend(loc='upper right')
    #fig.colorbar(cax3, ax=axes[2], orientation='vertical', label='Error (m)', fontsize=fontsize)
    cbar3 = fig.colorbar(cax3, ax=axes[2], orientation='vertical')
    cbar3.set_label('Error (m)', fontsize=fontsize)
    cbar3.ax.tick_params(labelsize=fontsize)
    
    # Add overall title
    fig.suptitle(f'Calibrated Model Validation', fontsize=fontsize)
    
    # Calculate and display error statistics
    if len(valid_diff) > 0:
        rmse = np.sqrt(np.mean(valid_diff**2))
        mae = np.mean(np.abs(valid_diff))
        bias = np.mean(valid_diff)
        
        # Add text box with statistics
    if fontsize < 16:
        stats_text = f'RMSE: {rmse:.2f}m\nMAE: {mae:.2f}m\nBias: {bias:.2f}m\nn: {len(valid_diff):,}'
        axes[2].text(0.02, 0.98, stats_text, transform=axes[2].transAxes,
                    verticalalignment='top', fontsize=fontsize,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    else:
        # Export statistics to CSV
        import pandas as pd
        stats_df = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'Bias', 'Valid_Pixels'],
            'Value': [rmse, mae, bias, len(valid_diff)],
            'Unit': ['m', 'm', 'm', 'count']
        })
        stats_filename = output_folder / f"validation_stats_{feature_name}_{area_name}.csv"
        stats_df.to_csv(stats_filename, index=False)
        print(f"Validation statistics exported to: {stats_filename}")
            
        print(f"Model validation statistics:")
        print(f"  RMSE: {rmse:.2f} m")
        print(f"  MAE: {mae:.2f} m") 
        print(f"  Bias: {bias:.2f} m")
        print(f"  Valid pixels: {len(valid_diff):,}")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_folder / f"calibrated_model_validation_{feature_name}_{area_name}.png", 
                format="png", bbox_inches='tight', dpi=300)
    plt.close()

def create_calibration_statistics_plot(calibrated_snowdepth, predicted_snowdepth, 
                                     ground_truth_depth, reference_pixel, 
                                     feature_name, area_name, output_folder, config,
                                     test_snowdepth=None):
    """Create detailed statistics and histogram comparison with observed vs modeled correlation plot"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Get valid data
    valid_mask = (calibrated_snowdepth != -999) & (predicted_snowdepth != -999)
    calibrated_valid = calibrated_snowdepth[valid_mask]
    predicted_valid = predicted_snowdepth[valid_mask]
    
    # Plot 1: Histogram comparison
    axes[0, 0].hist(predicted_valid, bins=50, alpha=0.7, label='Original Relative', color='blue')
    axes[0, 0].hist(calibrated_valid, bins=50, alpha=0.7, label='Calibrated Absolute', color='red')
    axes[0, 0].axvline(ground_truth_depth, color='green', linestyle='--', linewidth=2, 
                      label=f'Ground Truth ({ground_truth_depth:.2f}m)')
    axes[0, 0].set_xlabel('HS (m)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: REPLACED - Use EXACT correlation plot from validate_and_visualize_calibrated_prediction
    if test_snowdepth is not None:
        # Mask invalid data (EXACTLY as in your existing function)
        calibrated_mask = create_valid_mask(calibrated_snowdepth, -999)
        test_mask = create_valid_mask(test_snowdepth, -999)
        valid_mask = calibrated_mask & test_mask

        snow_calibrated = calibrated_snowdepth[valid_mask]
        snow_cal_flat = snow_calibrated.flatten()
        snow_test = test_snowdepth[valid_mask]  
        snow_test_flat = snow_test.flatten()
        
        # Calculate metrics: Calibrated vs Test (EXACTLY as in your existing function)
        correlation = np.corrcoef(snow_cal_flat, snow_test_flat)[0, 1]
        diff_1d = snow_test - snow_calibrated
        rmse = np.sqrt(np.mean(diff_1d ** 2))
        r2 = correlation ** 2
        bias = np.mean(diff_1d)
        pixel_count = len(snow_cal_flat)
        
        # Create correlation plot (EXACTLY as in your existing function)
        hexbin_gridsize = config['visualization'].get('hexbin_gridsize', 200)
        fontsize = config["visualization"]["fontsize_corr_plots_cal_analysis"]
        
        hb = axes[0, 1].hexbin(snow_cal_flat, snow_test_flat, gridsize=hexbin_gridsize, cmap='magma_r', bins='log', 
                        extent=[0, 5, 0, 5])
        axes[0, 1].set_xlabel("HS modeled (m)", fontsize=fontsize)
        axes[0, 1].set_ylabel("HS observed (m)", fontsize=fontsize)  
        
        axes[0, 1].tick_params(axis='both', which='major', labelsize=fontsize)
        
        axes[0, 1].set_title(f"Modeled vs Observed", fontsize=fontsize)
        
        # Add colorbar
        cbar = plt.colorbar(hb, ax=axes[0, 1])
        cbar.set_label("Log Count Density", fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize-2)
        
        # Set axis limits to 0-5 range (EXACTLY as in your existing function)
        axes[0, 1].set_xlim(0, 5)
        axes[0, 1].set_ylim(0, 5)
        
        # Make axes equal and square
        axes[0, 1].set_aspect('equal', adjustable='box')
        
        # Set custom grid with matching ticks
        axes[0, 1].set_xticks(np.arange(0, 5.1, 1))
        axes[0, 1].set_yticks(np.arange(0, 5.1, 1))
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add statistics text (EXACTLY as in your existing function)
        stats_text = f"R2 = {r2:.2f}\nn = {pixel_count:,}\nRMSE = {rmse:.2f}\nBias = {bias:.2f}"
        axes[0, 1].text(
            0.05, 0.95, stats_text,
            transform=axes[0, 1].transAxes, fontsize=fontsize,
            verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )
        
        # Add 1:1 line (EXACTLY as in your existing function)
        axes[0, 1].plot([0, 5], [0, 5], 'r--', alpha=0.8, linewidth=2, label='1:1 line')
        axes[0, 1].legend()
        
    else:
        axes[0, 1].text(0.5, 0.5, 'No observed data\navailable for validation', 
                       transform=axes[0, 1].transAxes, ha='center', va='center',
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[0, 1].set_title('Observed vs Modeled')
    
    # Plot 3: Reference point context (zoomed view around reference) - UNCHANGED
    zoom_size = 20  # pixels around reference point
    row, col = reference_pixel
    row_start = max(0, row - zoom_size)
    row_end = min(calibrated_snowdepth.shape[0], row + zoom_size + 1)
    col_start = max(0, col - zoom_size)
    col_end = min(calibrated_snowdepth.shape[1], col + zoom_size + 1)
    
    zoomed_calibrated = calibrated_snowdepth[row_start:row_end, col_start:col_end]
    zoomed_mask = np.ma.masked_where(zoomed_calibrated == -999, zoomed_calibrated)
    
    # Get colormap from config
    colormap = config['visualization'].get('colormap', 'cmc.navia_r')
    try:
        if colormap.startswith('cmc.'):
            from cmcrameri import cm as cmc
            cmap = getattr(cmc, colormap[4:])
        else:
            cmap = colormap
    except (ImportError, AttributeError):
        cmap = 'viridis'
    
    im = axes[1, 0].imshow(zoomed_mask, cmap=cmap, interpolation='nearest')
    # Adjust reference point coordinates for zoomed view
    ref_row_zoom = row - row_start
    ref_col_zoom = col - col_start
    axes[1, 0].plot(ref_col_zoom, ref_row_zoom, 'r*', markersize=20, 
                   markeredgecolor='white', markeredgewidth=2)
    axes[1, 0].set_title(f'Reference Point Context\n(Value: {ground_truth_depth:.2f}m)')
    plt.colorbar(im, ax=axes[1, 0], label='HS (m)')
    
    # Plot 4: Statistics summary - ENHANCED
    axes[1, 1].axis('off')
    
    # Calculate statistics
    offset = calibrated_valid[0] - predicted_valid[0] if len(calibrated_valid) > 0 else 0
    
    # Enhanced stats including validation metrics if available
    stats_text = f"""Model Summary

Reference Point:
• Coordinate: {reference_pixel}
• Ground Truth: {ground_truth_depth:.2f} m
• Calibration Offset: {offset:.2f} m

Original Model Stats:
• Mean: {np.mean(predicted_valid):.2f} m
• Std: {np.std(predicted_valid):.2f} m
• Range: {np.min(predicted_valid):.2f} to {np.max(predicted_valid):.2f} m

Calibrated Model Stats:
• Mean: {np.mean(calibrated_valid):.2f} m
• Std: {np.std(calibrated_valid):.2f} m
• Range: {np.min(calibrated_valid):.2f} to {np.max(calibrated_valid):.2f} m

• Valid Pixels: {len(calibrated_valid):,}

"""

#stats_text += f"""
    
    # Add validation stats if test data is available
    #if test_snowdepth is not None:
     #   test_mask = create_valid_mask(test_snowdepth, -999)
      #  cal_mask = create_valid_mask(calibrated_snowdepth, -999)
       # validation_mask = test_mask & cal_mask
        
        #if np.sum(validation_mask) > 0:
         #   test_valid = test_snowdepth[validation_mask]
          #  cal_valid = calibrated_snowdepth[validation_mask]
            
           # correlation = np.corrcoef(cal_valid, test_valid)[0, 1]
            #rmse = np.sqrt(np.mean((test_valid - cal_valid) ** 2))
            #bias = np.mean(test_valid - cal_valid)
            
            #stats_text += f"""

#Validation vs Observed:
#• Correlation (r): {correlation:.3f}
#• RMSE: {rmse:.2f} m
#• Bias: {bias:.2f} m
#• Valid pixels: {len(test_valid):,}"""
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                   fontsize=fontsize, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Add overall title
    fig.suptitle(f'Model Analysis', fontsize=fontsize+2)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_folder / f"model_analysis_{feature_name}_{area_name}.png", 
                format="png", bbox_inches='tight', dpi=300)
    plt.close()



def apply_model_calibration(predicted_snowdepth, reference_coord, ground_truth_depth,
                                 test_snowdepth, raster_path, feature_name, area_name, 
                                 output_folder, config):
    """
    UPDATED VERSION: Main function to apply calibration and validate against test data.
    """
    print(f"\n=== CALIBRATING SNOW DEPTH MODEL ===")
    print(f"Feature: {feature_name}")
    print(f"Area: {area_name}")
    
    # Perform calibration (this part is correct)
    calibrated_snowdepth, calibration_offset, reference_pixel = calibrate_snow_depth_model(
        predicted_snowdepth, reference_coord, ground_truth_depth, 
        raster_path, output_folder, feature_name, area_name, config
    )
    
    if calibrated_snowdepth is None:
        print("Calibration failed!")
        return None
    
    # Create 4-panel calibration analysis WITH test data for correlation plot
    create_calibration_statistics_plot(
        calibrated_snowdepth, predicted_snowdepth, 
        ground_truth_depth, reference_pixel, 
        feature_name, area_name, output_folder, config,
        test_snowdepth=test_snowdepth  # ✅ PASS TEST DATA HERE
    )
    
    # Create comprehensive visualizations comparing against TEST DATA
    if test_snowdepth is not None:
        plot_calibrated_comparison(
            ground_truth_depth, calibrated_snowdepth, predicted_snowdepth,
            test_snowdepth,  # ✅ PASS TEST DATA for comparison
            reference_pixel, feature_name, area_name, output_folder, config
        )
        
        # Validate against test data
        correlation, rmse, bias = validate_and_visualize_calibrated_prediction(
            calibrated_snowdepth, test_snowdepth,  # ✅ Compare against test data
            feature_name, area_name, 
            np.sum((calibrated_snowdepth != -999) & (test_snowdepth != -999)),
            output_folder, config
        )
        
        print(f"Calibration completed successfully!")
        print(f"Offset applied: {calibration_offset:.2f} m")
        print(f"Final model performance vs observed:")
        print(f"  Correlation: {correlation:.3f}")
        print(f"  RMSE: {rmse:.2f} m")
        print(f"  Bias: {bias:.2f} m")
    else:
        print("No test data available for validation")
    
    return calibrated_snowdepth

def predict_for_outlines_with_calibration(model, terrain_feature_name, case_folder, case_result_folder, config, ps):
    """
    Enhanced prediction function that includes optional calibration step.
    This replaces the original predict_for_outlines function when calibration is enabled.
    """
    
    # Process each prediction outline
    year_pred = config['analysis']['year_pred']
    for single_outline_pred in TERRAIN_LAYERS["pred"][terrain_feature_name].keys():
        # Create result folder for this outline
        outline_folder_result = case_result_folder / f"{year_pred}__{single_outline_pred}"
        outline_folder_result.mkdir(parents=True, exist_ok=True)
        
        # Get the stored geometry for this outline
        if single_outline_pred not in OUTLINE_GEOMETRIES["pred"]:
            print(f"Geometry not found for {single_outline_pred}, skipping prediction")
            continue
            
        geometry = OUTLINE_GEOMETRIES["pred"][single_outline_pred]
        
        # Extract terrain data for prediction (bounding box data)
        prediction_terrain_bbox = TERRAIN_LAYERS["pred"][terrain_feature_name][single_outline_pred]
        
        # Check if we have test data (for validation)
        snow_depth_test = config['paths']['snow_depth_test']
        has_test_data = (snow_depth_test is not None and
                        "snowdepth" in TERRAIN_LAYERS["pred"] and
                        single_outline_pred in TERRAIN_LAYERS["pred"]["snowdepth"])
        
        # Get path to DEM for reference
        dem_path = case_folder / f"{year_pred}__{single_outline_pred}" / f"dem_{year_pred}_{single_outline_pred}.tif"
        
        # Different processing depending on whether we have test data
        if has_test_data:
            test_snowdepth_bbox = TERRAIN_LAYERS["pred"]["snowdepth"][single_outline_pred]
            
            #new ENSURE ARRAYS ARE ALIGNED BEFORE GEOMETRY EXTRACTION
            print(f"Before alignment - Terrain: {prediction_terrain_bbox.shape}, Snow: {test_snowdepth_bbox.shape}")
           
            # Align them to same grid (either DEM or snow as reference)
            #prediction_terrain_bbox, test_snowdepth_bbox = ps.properly_align_rasters(config['paths']['dem_path'], config['paths']['snow_depth_test']
            #)
           
            print(f"After alignment - Terrain: {prediction_terrain_bbox.shape}, Snow: {test_snowdepth_bbox.shape}")
   
            # end of new
            
            # new again
            prediction_terrain_bbox, test_snowdepth_bbox = ps.match_array_shapes(prediction_terrain_bbox, test_snowdepth_bbox)
            
            # Extract data within exact geometry
            prediction_terrain_geom, test_snowdepth_geom, geometry_mask = extract_data_within_geometry(
                prediction_terrain_bbox, test_snowdepth_bbox, geometry, str(dem_path))
            
            test_mask = create_valid_mask(test_snowdepth_geom, -999)
            pred_mask = create_valid_mask(prediction_terrain_geom, -999)
            valid_mask = test_mask & pred_mask & geometry_mask
            
            test_snowdepth_valid = test_snowdepth_geom[valid_mask]
            prediction_terrain_valid = prediction_terrain_geom[valid_mask].reshape(-1, 1)
            
            # Pixel count for reporting
            pixelcount_prediction = test_snowdepth_valid.size
            
            # Predict snow depth
            predicted_snowdepth_flat = model.predict(prediction_terrain_valid)
            
            # Reshape to original dimensions
            predicted_snowdepth = ps.reshape_after_nandrop(
                predicted_snowdepth_flat, valid_mask, test_snowdepth_geom, nodata_value=-999)
            
            # Save prediction
            save_prediction(
                predicted_snowdepth,
                terrain_feature_name,
                single_outline_pred,
                outline_folder_result,
                case_folder,
                config
            )
            
            # Validate and visualize (original prediction)
            validate_and_visualize_prediction(
                predicted_snowdepth, 
                test_snowdepth_geom,
                terrain_feature_name,
                single_outline_pred,
                pixelcount_prediction,
                outline_folder_result,
                config
            )
        else:
            # No test data - just predict
            # Extract data within exact geometry
            prediction_terrain_geom, _, geometry_mask = extract_data_within_geometry(
                prediction_terrain_bbox, None, geometry, str(dem_path))
            
            valid_mask = create_valid_mask(prediction_terrain_geom, -999) & geometry_mask
            prediction_terrain_valid = prediction_terrain_geom[valid_mask].reshape(-1, 1)
            
            # Predict snow depth
            predicted_snowdepth_flat = model.predict(prediction_terrain_valid)
            
            # Reshape to original dimensions
            predicted_snowdepth = ps.reshape_after_nandrop(
                predicted_snowdepth_flat, valid_mask, prediction_terrain_geom, nodata_value=-999)
            
            # Save prediction
            save_prediction(
                predicted_snowdepth,
                terrain_feature_name,
                single_outline_pred,
                outline_folder_result,
                case_folder,
                config
            )
            
            # Create visualization (original prediction)
            plot_prediction_map(
                predicted_snowdepth,
                terrain_feature_name,
                single_outline_pred,
                outline_folder_result,
                config
            )
        
        # === CALIBRATION SECTION ===
        # After prediction is complete, check for calibration config
        if 'calibration' in config and config['calibration'].get('enabled', False):
            calibration_config = config['calibration']
            
            # Check if this outline should be calibrated
            if single_outline_pred in calibration_config.get('areas', {}):
                area_calibration = calibration_config['areas'][single_outline_pred]
                
                reference_coord = area_calibration['reference_coordinate']  # (x, y)
                ground_truth_depth = area_calibration['ground_truth_depth']  # meters
                
                print(f"\n--- APPLYING CALIBRATION TO {single_outline_pred} ---")
                
                # Apply calibration to the predicted snow depth
                calibrated_snowdepth = apply_model_calibration(
                    predicted_snowdepth=predicted_snowdepth,  # From prediction above
                    reference_coord=reference_coord,
                    ground_truth_depth=ground_truth_depth,
                    test_snowdepth = test_snowdepth_geom,
                    raster_path=str(dem_path),
                    feature_name=terrain_feature_name,
                    area_name=single_outline_pred,
                    output_folder=outline_folder_result,
                    config=config
                )
                
                if calibrated_snowdepth is not None:
                    print(f"Calibration successful for {single_outline_pred}")
                    
                    # If we have test data, also validate the calibrated model
                    if has_test_data:
                        print(f"Validating calibrated model against test data...")
                        validate_and_visualize_calibrated_prediction(
                            calibrated_snowdepth, 
                            test_snowdepth_geom,
                            terrain_feature_name,
                            single_outline_pred,
                            pixelcount_prediction,
                            outline_folder_result,
                            config
                        )
                else:
                    print(f"Calibration failed for {single_outline_pred}")


def validate_and_visualize_calibrated_prediction(calibrated_snowdepth, test_snowdepth, 
                                                      feature_name, area_name, pixel_count, 
                                                      output_folder, config):
    """
    FIXED VERSION: Validate and visualize calibrated prediction against TEST DATA.
    """
    
    # Mask invalid data
    calibrated_mask = create_valid_mask(calibrated_snowdepth, -999)
    test_mask = create_valid_mask(test_snowdepth, -999)
    valid_mask = calibrated_mask & test_mask

    snow_calibrated = calibrated_snowdepth[valid_mask]
    snow_cal_flat = snow_calibrated.flatten()
    snow_test = test_snowdepth[valid_mask]  
    snow_test_flat = snow_test.flatten()
    
    # Calculate metrics: Calibrated vs Test (not vs Original prediction!)
    correlation = np.corrcoef(snow_cal_flat, snow_test_flat)[0, 1]
    print(f"    Calibrated Model vs Test Data: r={correlation:.2f}")
    
    # Calculate RMSE: Test - Calibrated
    diff_1d = snow_test - snow_calibrated  # ✅ FIXED: Test - Calibrated
    rmse = np.sqrt(np.mean(diff_1d ** 2))
    print(f"    Calibrated Model RMSE: {rmse:.2f}")
    
    r2 = correlation ** 2
    print(f"    Calibrated Model R²: {r2:.2f}")
    
    # Calculate bias
    bias = np.mean(diff_1d)
    print(f"    Calibrated Model Bias: {bias:.2f}")
    
        # Create correlation plot for calibrated model vs test data
    plt.figure(figsize=(8, 8))
    hexbin_gridsize = config['visualization'].get('hexbin_gridsize', 200)
    fontsize = config["visualization"]["fontsize_corr_plots"]
    
    hb = plt.hexbin(snow_cal_flat, snow_test_flat, gridsize=hexbin_gridsize, cmap='magma_r', bins='log', 
                    extent=[0, 5, 0, 5])
    plt.xlabel("HS modeled", fontsize=fontsize)
    plt.ylabel("HS observed", fontsize=fontsize)  
    
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    
    data_type = config['analysis']['data_type']
    plt.title(f"Modeled vs Observed ", fontsize=fontsize)
    
    cbar = plt.colorbar(hb)
    cbar.set_label("Log Count Density", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize-2)
    
    # Set axis limits to 0-5 range
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    
    # Make axes equal and square
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Set custom grid with matching ticks
    plt.xticks(np.arange(0, 5.1, 1))  # Ticks every 0.5 units
    plt.yticks(np.arange(0, 5.1, 1))  # Ticks every 0.5 units
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"R2 = {r2:.2f}\nn = {pixel_count:,}\nRMSE = {rmse:.2f}\nBias = {bias:.2f}"
    plt.text(
        0.05, 0.95, stats_text,
        transform=plt.gca().transAxes, fontsize=fontsize,
        verticalalignment='top', 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )
    
    # Add 1:1 line (now from 0 to 5)
    plt.plot([0, 5], [0, 5], 'r--', alpha=0.8, linewidth=2, label='1:1 line')
    plt.legend()
    
    # Save correlation plot
    plt.savefig(output_folder / f"calibrated_vs_observed_{feature_name}_{area_name}_font{fontsize}.png", 
                format="png", bbox_inches='tight', dpi=300)
    plt.close()
    
    return correlation, rmse, bias


def train_and_predict_with_calibration(case_folder, case_result_folder, config, ps):
    """
    Enhanced version of your train_and_predict function that includes calibration.
    """
    # Get terrain features list with placeholders replaced
    terrain_features = []
    for feature in config['terrain_features']:
        # Replace any placeholders in the feature names
        if '{neighbourhood_meter}' in feature:
            feature = feature.format(neighbourhood_meter=config['analysis']['neighbourhood_meter'])
        terrain_features.append(feature)
    
    # Process each terrain feature
    for terrain_feature_name in terrain_features:
        # Skip snowdepth as it's the target variable
        if terrain_feature_name == "snowdepth":
            continue
        
        # Skip if feature is not in TERRAIN_LAYERS
        if terrain_feature_name not in TERRAIN_LAYERS["train"]:
            print(f"Terrain feature {terrain_feature_name} not found in training data, skipping")
            continue
            
        print(f"Processing terrain feature: {terrain_feature_name}")
        
        # Create combined arrays for training
        training_features_combined = np.array([]).reshape(-1, 1)
        training_snowdepth_combined = np.array([])
        
        print(f"Training outlines: {list(TERRAIN_LAYERS['train'][terrain_feature_name].keys())}")
        print(f"Prediction outlines: {list(TERRAIN_LAYERS['pred'][terrain_feature_name].keys())}")
    
        # Process each training area using EXACT GEOMETRY (same as original)
        year_train = config['analysis']['year_train']
        for single_outline_train in TERRAIN_LAYERS["train"][terrain_feature_name].keys():
            # Create result folder for this terrain feature
            terrain_folder_result = case_result_folder / f"{year_train}_{terrain_feature_name}_train"
            terrain_folder_result.mkdir(parents=True, exist_ok=True)
            
            # Get the stored geometry for this outline
            if single_outline_train not in OUTLINE_GEOMETRIES["train"]:
                print(f"Geometry not found for {single_outline_train}, skipping")
                continue
                
            geometry = OUTLINE_GEOMETRIES["train"][single_outline_train]
            
            # Extract data from dictionary (this is bounding box data)
            training_terrain_bbox = TERRAIN_LAYERS["train"][terrain_feature_name][single_outline_train]
            
            # Skip if snowdepth data is not available
            if ("snowdepth" not in TERRAIN_LAYERS["train"] or 
                single_outline_train not in TERRAIN_LAYERS["train"]["snowdepth"]):
                print(f"Snow depth data missing for {single_outline_train}, skipping")
                continue
                
            training_snowdepth_bbox = TERRAIN_LAYERS["train"]["snowdepth"][single_outline_train]
            
            # Get path to DEM for reference
            dem_path = case_folder / f"{year_train}__{single_outline_train}" / f"dem_{year_train}_{single_outline_train}.tif"
            
            # Extract data within exact geometry
            training_terrain_geom, training_snowdepth_geom, geometry_mask = extract_data_within_geometry(
                training_terrain_bbox, training_snowdepth_bbox, geometry, str(dem_path))
            
            # Get nodata value from config or use default
            nodata_value = config.get('analysis', {}).get('nodata_value', -999)
            
            # Create valid mask combining geometry mask and data validity
            terrain_mask = create_valid_mask(training_terrain_geom, nodata_value)
            snowdepth_mask = create_valid_mask(training_snowdepth_geom, nodata_value)
            valid_mask = terrain_mask & snowdepth_mask & geometry_mask

            training_snowdepth_flat = training_snowdepth_geom[valid_mask].flatten()
            training_terrain_reshaped = training_terrain_geom[valid_mask].reshape(-1, 1)
            
            # Append to combined arrays
            training_features_combined = np.append(
                training_features_combined, training_terrain_reshaped, axis=0)
            training_snowdepth_combined = np.append(
                training_snowdepth_combined, training_snowdepth_flat, axis=0)
            
            print(f"  Outline {single_outline_train}: {np.sum(valid_mask)} valid pixels")
            
        
        
        # Check if we have enough training data
        pixelcount_training_terrain = training_features_combined.size
        if pixelcount_training_terrain == 0:
            print(f"No valid training data found for {terrain_feature_name}, skipping")
            continue
            
        print(f"    n={pixelcount_training_terrain}")
  
        
        # Calculate correlation
        correlation = np.corrcoef(training_features_combined.flatten(), 
                                 training_snowdepth_combined)[0, 1]
        print(f"    r={correlation:.2f}")
        
        # Create correlation plot
        plot_correlation(
            config,
            training_features_combined.flatten(), 
            training_snowdepth_combined,
            terrain_feature_name,
            single_outline_train,
            correlation,
            pixelcount_training_terrain,
            terrain_folder_result
        )
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(training_features_combined, training_snowdepth_combined)
        print(f"    Model: Snow Depth = {model.coef_[0]:.2f} * {terrain_feature_name} + {model.intercept_:.2f}")
        
        # Apply predictions with calibration
        prediction_mode = config['analysis']['prediction']
        if prediction_mode == "outlines":
            predict_for_outlines_with_calibration(model, terrain_feature_name, case_folder, case_result_folder, config, ps)
        elif prediction_mode == "extent":
            # For extent prediction, you could create a similar function with calibration
            predict_for_extent(model, terrain_feature_name, case_folder, case_result_folder, config, ps)
        else:
            print(f"Warning: Unrecognized prediction mode '{prediction_mode}', skipping prediction")
    
def main(config_path="config.yaml"):
    """Main function to run the terrain analysis with optional calibration"""
    try:
        # Load configuration
        config = load_config(config_path)
        
        # After loading your config, run this:
        #run_full_calibration_diagnostics(config)
        
        # Validate configuration
        validate_config(config)
        
        # Setup output directories
        result_folder, data_folder, case_folder, case_result_folder, case_name = setup_directories(config)
        
        # Process terrain for all outlines (calculates on bounding box, stores geometry)
        process_outlines(config, case_folder, ps)
        
        # CREATE COMBINED TPI + ELEVATION FEATURE AND SAVE TO CASE FOLDER
        combined_feature_name = create_tpi_elevation_combined(config, case_folder)        
        
        # Check if calibration is enabled and choose appropriate function
        if 'calibration' in config and config['calibration'].get('enabled', False):
            print("=== CALIBRATION ENABLED ===")
            print("Using calibration-enhanced prediction workflow")
            train_and_predict_with_calibration(case_folder, case_result_folder, config, ps)
        else:
            print("=== STANDARD WORKFLOW ===")
            print("Using standard prediction workflow (no calibration)")
            train_and_predict(case_folder, case_result_folder, config, ps)
        
        print("Terrain analysis completed successfully")
     
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import argparse
    
    # Setup command line argument parser
    parser = argparse.ArgumentParser(
        description="Snow Distribution Terrain Analysis - Mixed Approach",
        epilog="This script calculates terrain parameters on bounding boxes but trains/predicts using exact geometries."
    )
    parser.add_argument("--config", "-c", default="config_snow_modelling.yaml",
                        help="Path to YAML configuration file (default: config.yaml)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run main function with specified config file
    sys.exit(main(args.config))#!/usr/bin/env python