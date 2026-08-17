#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Snow Distribution Analysis Configuration-Driven Script

This script provides a configurable approach to analyzing snow distribution data across
multiple years, leveraging terrain features and statistical methods for comparison.

Author: Christian Goehrig
Created: April 2025
"""

import os
import sys
import glob
import re
import datetime
import warnings
import yaml
import argparse
from pathlib import Path
import numpy as np
from rasterio.enums import Resampling

# Handle YAML configuration loading
def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            print(f"Configuration loaded from {config_path}")
            return config
        except yaml.YAMLError as e:
            print(f"Error parsing YAML configuration: {e}")
            sys.exit(1)

# Validate paths in configuration
def validate_paths(config):
    """Validate that paths in configuration exist."""
    required_paths = [
        config['paths']['library_dir'],
    ]

    # Check for required paths
    for path in required_paths:
        if not os.path.exists(path):
            print(f"ERROR: Required path does not exist: {path}")
            return False

    # Create output directory if it doesn't exist
    os.makedirs(config['paths']['output_folder'], exist_ok=True)

    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Snow Distribution Analysis Tool')
    parser.add_argument('--config', type=str, default='config_preprocess.yaml',
                        help='Path to YAML configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Record start time
    start = datetime.datetime.now()
    print(f"Script started at {start}")

    # Validate paths if enabled in config
    if config.get('validate_paths', True):
        if not validate_paths(config):
            print("Path validation failed. Exiting.")
            return

    # Import powdersearch module from library directory
    sys.path.append(config['paths']['library_dir'])
    import powdersearch as ps

    # Ignore specific warnings
    warnings.filterwarnings("ignore", message="angle from rectified to skew grid parameter lost in conversion to CF")

    # Create case folder if it doesn't exist
    case_folder = config['paths']['output_folder']
    os.makedirs(case_folder, exist_ok=True)

    # Select years to process from config
    # Here we assume years are defined within a config['analysis']['years'] list
    years = config.get('analysis', {}).get('years', [])
    if not years:
        # If years not defined in config, extract them from the file names in dataset dir
        dataset_files = glob.glob(os.path.join(config['paths']['dataset_dir'], "*.tif"))
        years = set()
        for file in dataset_files:
            try:
                match = re.search(r'\d{4}', os.path.basename(file))
                if match:
                    years.add(int(match.group()))
            except:
                pass
        years = sorted(list(years))

    print(f"Processing years: {years}")

    # Select desired years from datapool
    input_data = ps.select_files_by_year(config['paths'].get('dataset_dir', case_folder), years)

    if not input_data:
        print("No matching data files found. Exiting.")
        return

    # Process based on config settings

    # 1. Reproject and align rasters
    reference_raster = config['paths'].get('reference_raster')
    resampling_method_str = config.get('analysis', {}).get('resampling_method', 'nearest')#default nearest neighbor unless defined in config

    # Map string to Resampling enum
    resampling_methods = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic,
        'cubic_spline': Resampling.cubic_spline,
        'lanczos': Resampling.lanczos,
        'average': Resampling.average,
        'mode': Resampling.mode
    }
    resampling_method = resampling_methods.get(resampling_method_str, Resampling.nearest)

    print("Reprojecting and aligning rasters...")
    uniformed_rasters = ps.reproject_and_align_rasters(
        input_data,
        case_folder,
        target_crs=config.get('analysis', {}).get('crs', 'EPSG:2056'),
        reference_raster=reference_raster,
        resolution=config.get('analysis', {}).get('resolution'),
        resampling_method=resampling_method,
        apply_ref_mask=config.get('analysis', {}).get('apply_ref_mask', True),
        set_negative_to_nodata=config.get('analysis', {}).get('set_negative_to_nodata', True)
    )

    # 2. Create visualization if enabled
    if config.get('analysis', {}).get('create_violin', True):
        print("Creating violin plots...")
        violin_plot_folder = ps.violin(
            uniformed_rasters,
            combined_plot=config.get('visualization', {}).get('combined_plot', True),
            subsample_factor=config.get('visualization', {}).get('subsample_factor', 10),
            max_depth=config.get('visualization', {}).get('max_depth', 6),
            remove_outliers=config.get('visualization', {}).get('remove_outliers', True)
        )

    # 3. Calculate statistical parameters for whole timeseries
    if config.get('analysis', {}).get('calc_timeseries_stats', True):
        print("Calculating timeseries statistics...")
        timeseries_statistic_rasters = ps.calculate_timeseries_statistics(
            input_folder=uniformed_rasters,
            output_folder_name='global_statistics',
            use_parallel=config.get('processing', {}).get('parallel', False),
            max_workers=config.get('processing', {}).get('max_workers')
        )

    # 4. Compare each year to summarized global statistics
    if config.get('analysis', {}).get('calc_yearly_diffs', True):
        print("Calculating yearly difference maps...")
        yearly_difference_rasters = ps.calculate_difference_maps(
            uniformed_rasters,
            timeseries_statistic_rasters,
            config.get('analysis', {}).get('diff_mode', 'absolute')
        )

    # 5. Normalize data if enabled
    if config.get('analysis', {}).get('normalize_data', True):
        norm_method = config.get('analysis', {}).get('normalization_method', 'single_year_relative')
        print(f"Normalizing data using {norm_method} method...")

        if norm_method == 'single_year_relative':
            normalized_rasters, output_folder = ps.normalize(
                uniformed_rasters,
                norm_method,
                f"normalized_{norm_method}"
            )
        elif norm_method == 'single_year_minmax':
            yearly_max, yearly_min, output_folder = ps.normalize(
                uniformed_rasters,
                norm_method,
                f"normalized_{norm_method}"
            )

        # 6. Calculate statistics on normalized data
        if config.get('analysis', {}).get('calc_norm_stats', True):
            print("Calculating statistics on normalized data...")
            timeseries_statistic_rasters_normalized = ps.calculate_timeseries_statistics(
                input_folder=output_folder,
                output_folder_name=f'global_statistics_{norm_method}',
                use_parallel=config.get('processing', {}).get('parallel', False),
                max_workers=config.get('processing', {}).get('max_workers')
            )

            # 7. Calculate difference maps for normalized data
            yearly_difference_rasters_normalized = ps.calculate_difference_maps(
                output_folder,
                timeseries_statistic_rasters_normalized,
                f"{norm_method}"
            )

    # 8. Calculate terrain features if enabled
    if config.get('analysis', {}).get('calc_terrain_features', True):
        dem_path = config['paths'].get('dem_path')
        if dem_path and os.path.exists(dem_path):
            print("Calculating terrain features...")

            # Create terrain features folder
            terrain_folder = os.path.join(case_folder, "terrain_features")
            os.makedirs(terrain_folder, exist_ok=True)

            # Calculate pixel size
            pixel_size = config.get('analysis', {}).get('pixel_size', 2)

            # Get terrain features to calculate from config
            terrain_features = config.get('terrain_features', ['slope', 'aspect'])

            # Calculate each feature
            for feature in terrain_features:
                if feature == 'slope':
                    ps.calculate_slope(dem_path, pixel_size, terrain_folder)
                elif feature == 'aspect':
                    ps.calculate_aspect(dem_path, pixel_size, terrain_folder)
                elif feature.startswith('curvature'):
                    # Extract window size from feature name pattern (curvature_{size})
                    window_size = int(re.search(r'curvature_(\d+)', feature).group(1))
                    ps.calculate_curvature(dem_path, window_size, terrain_folder)
                elif feature.startswith('tpi'):
                    # Extract window size from feature name pattern (tpi_{size})
                    window_size = int(re.search(r'tpi_(\d+)', feature).group(1))
                    ps.calculate_tpi(dem_path, window_size, terrain_folder)

    # 9. Additional analysis based on config
    if config.get('analysis', {}).get('calc_pearson', False):
        print("Calculating Pearson correlation...")
        pearson_folder = ps.pearson_analysis(uniformed_rasters)

    if config.get('analysis', {}).get('calc_clusters', False):
        print("Calculating pixel clusters...")
        line_reduction_factor = config.get('analysis', {}).get('line_reduction_factor', 100)
        number_of_clusters = config.get('analysis', {}).get('number_of_clusters', 4)
        cluster_folder = ps.plot_raster_timeseries(
            uniformed_rasters,
            line_reduction_factor,
            number_of_clusters
        )

    # Record end time and show elapsed time
    end = datetime.datetime.now()
    print(f"Script ended at {end}")
    time_diff = end - start
    print(f"Total computation time: {time_diff}")

if __name__ == "__main__":
    main()
