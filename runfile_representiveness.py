# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 09:26:01 2025

@author: goehrigc
"""

"""
Run Script for Snow Site Representativity Analysis
-------------------------------------------------
This script runs the snow site representativity analysis with custom parameters.

Modify the parameters below to match your data and analysis needs.
"""

import os
import sys
import numpy as np
from datetime import datetime

# Add the directory containing the snow_site_analysis module to the Python path if needed
# Uncomment and modify this if snow_site_analysis.py is in a different directory
# sys.path.append("/path/to/directory/containing/snow_site_analysis")

# Import the analysis function
from snow_site_analysis import run_site_representativity_analysis

# =============================================================================
# PARAMETERS - MODIFY THESE FOR YOUR ANALYSIS
# =============================================================================

# Input and output paths
INPUT_FOLDER =   # Directory containing snow depth rasters
STATION_LOCATIONS =   # CSV file with station locations (optional)
DEM_PATH =  # DEM file for elevation analysis (optional)
OUTPUT_NAME = f"snow_analysis_{datetime.now().strftime('%Y%m%d')}"  # Output folder name

# Analysis parameters
SELECTED_YEARS = [2019, 2020]  # Years to analyze (set to None for all available years)

# Elevation analysis parameters (only used if elevation_layers=True)
ELEVATION_LAYERS = False  # Set to False to skip elevation-stratified analysis
ELEVATION_CLASSES = [0, 1800, 2300, np.inf]  # Elevation band boundaries
ELEVATION_CLASS_LABELS = ['<1800m', '1800-2300m', '>2300m']  # Labels for elevation bands

# Station buffer analysis parameters (only used if buffer_analysis=True)
BUFFER_ANALYSIS = True  # Set to False to skip buffer analysis
BUFFER_RADII = [100, 250, 500, 1000]  # Buffer radii in meters

# =============================================================================
# RUN ANALYSIS
# =============================================================================

if __name__ == "__main__":
    print(f"Starting snow site representativity analysis at {datetime.now()}")
    
    # Validate inputs
    if not os.path.exists(INPUT_FOLDER):
        raise FileNotFoundError(f"Input folder does not exist: {INPUT_FOLDER}")
    
    if STATION_LOCATIONS and not os.path.exists(STATION_LOCATIONS):
        print(f"Warning: Station locations file not found: {STATION_LOCATIONS}")
        STATION_LOCATIONS = None
    
    if ELEVATION_LAYERS and (not DEM_PATH or not os.path.exists(DEM_PATH)):
        print(f"Warning: DEM file not found: {DEM_PATH}. Elevation analysis will be skipped.")
        ELEVATION_LAYERS = False
    
    # Run the analysis
    results = run_site_representativity_analysis(
        input_folder=INPUT_FOLDER,
        output_name=OUTPUT_NAME,
        station_locations=STATION_LOCATIONS,
        dem_path=DEM_PATH,
        elevation_layers=ELEVATION_LAYERS,
        selected_years=SELECTED_YEARS,
        elevation_classes=ELEVATION_CLASSES,
        elevation_class_labels=ELEVATION_CLASS_LABELS,
        buffer_analysis=BUFFER_ANALYSIS,
        buffer_radii=BUFFER_RADII
    )
    
    # Print summary
    print("\nANALYSIS SUMMARY")
    print("================")
    print(f"Output folder: {results['output_folder']}")
    print(f"Analyzed years: {SELECTED_YEARS}")
    
    # Print yearly means
    print("\nYearly Snow Depth Means:")
    for year, mean in sorted(results['yearly_means'].items()):
        print(f"  {year}: {mean:.2f}m")
    
    # Print station results if available
    if 'station_scores' in results and results['station_scores'] is not None:
        print("\nStation Representativity (Top 5):")
        top_stations = results['station_scores'].sort_values('representativity_score').head(5)
        for _, row in top_stations.iterrows():
            optimal = "Optimal" if row['is_optimal'] else "Not optimal"
            print(f"  {row['station']}: Score={row['representativity_score']:.2f} ({optimal})")
    

    print("\nAnalysis complete!")
