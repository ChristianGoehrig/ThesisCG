
"""
Created on Tue Nov 26 17:53:33 2024

@author: chris
"""

# LIBRARY
# Summary of all functions used for the processing of the UltraCam snowdepth data


    #------------------- load modules---------------------------------------------
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import sys
from pathlib import Path
import matplotlib.colors as mcolors
import re
from scipy.ndimage import generic_filter, correlate
import csv
from shapely.geometry import box,mapping
import rioxarray as rio
import glob
import datetime
from rasterio.enums import Resampling, Compression
from rasterio.warp import reproject, calculate_default_transform
import xarray as xr
import seaborn as sns
from sklearn.cluster import KMeans
import pandas as pd
from cmcrameri import cm
import warnings
import multiprocessing
from rasterio.windows import from_bounds
import yaml
import geopandas as gpd
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore", message="angle from rectified to skew grid parameter lost in conversion to CF")
 
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
            
            # Verbose path verification
            #print(f"Path for {key}: {path_obj}")
            #print(f"Path exists: {path_obj.exists()}")
            
            # If path doesn't exist, print more diagnostic information
            if not path_obj.exists():
                print(f"Checking parent directory: {path_obj.parent}")
                print(f"Parent directory exists: {path_obj.parent.exists()}")
                print("Contents of parent directory:")
                try:
                    for item in path_obj.parent.iterdir():
                        print(f"  - {item.name}")
                except Exception as e:
                    print(f"Error listing directory contents: {e}")
    
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
        # Add other critical paths here
    ]
    
    for path_key in paths_to_check:
        file_path = config['paths'].get(path_key)
        if file_path and not file_path.exists():
            print(f"Checking path for {path_key}: {file_path}")
            raise FileNotFoundError(f"Required file not found: {file_path}")
            
    print("Configuration validated successfully")
    
def create_folder_in_parent_dir(input_folder, folder_name):
    """
    Create a folder in the parent directory of the input folder.
    
    Parameters
    ----------
    input_folder : str
        Path to input folder whose parent directory will be used
    folder_name : str
        Name of the new folder to create
        
    Returns
    -------
    str
        Path to the newly created folder
        
    Notes
    -----
    Uses exist_ok=True, so returns existing path if folder already exists
    
    Examples
    --------
    >>> create_folder_in_parent_dir('/home/user/data', 'results')
    '/home/user/results'
    """
    parent_dir = os.path.dirname(input_folder)
    new_folder = os.path.join(parent_dir, folder_name)
    os.makedirs(new_folder, exist_ok=True)
    return new_folder

# ------------------ select desired files from a folder ----------------------

def select_files_by_year(folder_path, years):
    """
    Select files from a folder that contain specified years in their filenames.
    
    Parameters
    ----------
    folder_path : str
        Path to the folder containing files to search
    years : list
        List of years (int or str) to search for in filenames
        
    Returns
    -------
    list
        List of full file paths for files containing any of the specified years
        
    Notes
    -----
    The function performs a simple string match, checking if the string 
    representation of each year appears anywhere in the filename
    
    Examples
    --------
    >>> select_files_by_year('/data/rainfall', [2020, 2021])
    ['/data/rainfall/data_2020.csv', '/data/rainfall/rainfall_2021.txt']
    """
    selected_files = []
    
    # Iterate over files in the folder using glob
    for file_path in glob.glob(os.path.join(folder_path, "*")):
        file_name = os.path.basename(file_path)
        if any(str(year) in file_name for year in years):
            selected_files.append(file_path)  # Append the full path
    
    print("Years of Data selected for processing")
    
    return selected_files

# ---------- extract year stamp  from filenames -------------------------------

def extract_years_from_paths(paths):
    """
    Extract years from a list of pathnames.
    
    Parameters
    ----------
    paths : list
        List of pathname strings containing year information
        
    Returns
    -------
    list
        Sorted list of extracted years as integers
        
    Notes
    -----
    Extracts 4-digit years or years from 8-digit dates (YYYYMMDD)
    using regex pattern \d{8}|\d{4}
    
    Examples
    --------
    >>> extract_years_from_paths(['data_2020.txt', 'rainfall_20211105.csv'])
    [2020, 2021]
    """
    
    years = []
    for path in paths:
        
        # extract dates in format year or yearmonthday from path
        date_matches = re.search(r'\d{8}|\d{4}', path).group()
        
        # shorten long dates to only year and write to sorted list
        years.append(int(date_matches[0:4]))
        years.sort()
    
    return years
# ---------------------load raster -------------------------------------------

def load_raster(raster_path, metadata=None, band=1):
    """
    Load a raster file with array always returned and optional metadata.
    
    Parameters
    ----------
    raster_path : str
        Path to the raster file.
    metadata : str, list, or None, optional
        Additional metadata item(s) to return after the array.
        Can be a single string or a list of strings.
        Options include: 'profile', 'crs', 'transform', 'bounds', 
        'nodata', 'pixel_size', "meta"
        If None, returns only the raster array (default).
        Use 'all' to return the array plus all metadata items.
    band : int, optional
        Band number to read from the raster, by default 1.
        
    Returns
    -------
    numpy.ndarray or multiple values
        Always returns the raster array as the first return value.
        If metadata is requested, returns those as additional values
        in the order specified.
        
    Examples
    --------
    >>> # Get just the raster data (default)
    >>> data = load_raster("dem.tif")
    >>> 
    >>> # Get raster data and a single metadata item
    >>> data, nodata = load_raster("dem.tif", metadata="nodata")
    >>> 
    >>> # Get array and multiple metadata items
    >>> data, crs, pixel_size = load_raster("dem.tif", metadata=["crs", "pixel_size"])
    >>> 
    >>> # Get array and all metadata components
    >>> data, profile, crs, transform, bounds, nodata, pixel_size = load_raster("dem.tif", metadata="all")
    """
    # Handle string metadata parameter (convert to list)
    if isinstance(metadata, str) and metadata != "all":
        metadata = [metadata]
    
    # Handle 'all' option
    all_metadata = ["profile", "crs", "transform", "bounds", "nodata", "pixel_size"]
    if metadata == "all":
        metadata = all_metadata
    
    with rasterio.open(raster_path) as src:
        # Always read the array
        raster_data = src.read(band)
        
        # Get nodata value from the metadata
        nodata_value = src.nodata
        
        # Mask the array if a nodata value is defined
        if nodata_value is not None:
            # Replace nodata values with NaN
            raster_data = np.where(raster_data == nodata_value, np.nan, raster_data)
        
        # Also mask -999 values, which are often used as nodata in the dataset
        raster_data = np.where(raster_data == -999, np.nan, raster_data)
        
        # If no additional metadata requested, return just the array
        if not metadata:
            return raster_data
        
        # Prepare return values with array as first item
        returns = [raster_data]
        
        # Process each requested metadata item
        for item in metadata:
            if item == "profile":
                returns.append(src.profile)
            elif item == "meta":
                returns.append(src.meta)
            elif item == "crs":
                returns.append(src.crs)
            elif item == "transform":
                returns.append(src.transform)
            elif item == "bounds":
                returns.append(src.bounds)
            elif item == "nodata":
                returns.append(src.nodata)
            elif item == "pixel_size":
                pixel_size_x = src.transform[0]
                pixel_size_y = abs(src.transform[4])
                returns.append((pixel_size_x, pixel_size_y))
        
        # Return array only if no additional metadata was found
        if len(returns) == 1:
            return raster_data
        # Otherwise return array and all requested metadata
        else:
            return tuple(returns)

# ------------------ reference a raster based on a reference----------------

def georeference_and_save(reference_raster, unreferenced_array, output_path):
    """
    Georeference an array using metadata from a reference raster and save to disk.
    
    Parameters
    ----------
    reference_raster : str
        Path to reference raster file with desired georeferencing information
    unreferenced_array : numpy.ndarray
        Array with data to be georeferenced
    output_path : str
        Path where the new georeferenced raster will be saved
        
    Returns
    -------
    None
        Function saves output to disk and prints confirmation message
        
    Notes
    -----
    The unreferenced array must match the dimensions of the reference raster
    
    Examples
    --------
    >>> import numpy as np
    >>> array = np.random.rand(100, 100)
    >>> georeference_and_save("reference.tif", array, "output.tif")
    """
    # Open the referenced raster to get metadata
    with rasterio.open(reference_raster) as src:
        ref_meta = src.meta.copy()
    # Ensure the unreferenced array has the same shape as the reference
    assert unreferenced_array.shape == (ref_meta['height'], ref_meta['width']), "Shape mismatch"
    # Update metadata (set dtype and ensure it's writable)
    ref_meta.update(dtype=unreferenced_array.dtype, count=1)
    # Save the new georeferenced raster
    with rasterio.open(output_path, 'w', **ref_meta) as dst:
        dst.write(unreferenced_array, 1)
    print(f"Saved georeferenced raster to {output_path}")


# ----------------------------------------------------------------------------
def match_array_shapes(arr1, arr2):
    """Ensure two arrays have the same shape by cropping the larger one."""
    min_rows = min(arr1.shape[0], arr2.shape[0])
    min_cols = min(arr1.shape[1], arr2.shape[1])
    
    return arr1[:min_rows, :min_cols], arr2[:min_rows, :min_cols]


def round_up_to_nearest_2m(value):
    """Round up the value to the nearest multiple of 2 meters."""
    return math.ceil(value / 2.0) * 2.0

def round_down_to_nearest_2m(value):
    return value - (value % 2)

def align_to_raster_grid_x(value, raster_transform):
    """Aligns an x-coordinate to the raster grid."""
    return raster_transform[2] + round((value - raster_transform[2]) / raster_transform[0]) * raster_transform[0]

def align_to_raster_grid_y(value, raster_transform, height):
    """Aligns a y-coordinate to the raster grid (accounting for top-left origin)."""
    return raster_transform[5] + round((value - raster_transform[5]) / raster_transform[4]) * raster_transform[4]

def transform_aspect_360_to_180(x):
    if 0 <= x <= 180:
        return x  # No change
    elif 180 < x < 360:
        return x - 360  # Convert to negative
    else:
        raise ValueError("x should be between 0 and 360")
        

def properly_align_rasters(terrain_path, snow_reference_path):
    """
    Properly align terrain data to snow depth reference using spatial coordinates.
    Returns terrain data that exactly matches the snow depth spatial extent and resolution.
    """
    
    # Load snow depth reference with full spatial metadata
    with rasterio.open(snow_reference_path) as snow_src:
        snow_data = snow_src.read(1)
        snow_meta = snow_src.meta.copy()
        snow_bounds = snow_src.bounds
        snow_transform = snow_src.transform
        snow_crs = snow_src.crs
        
    print(f"Snow depth bounds: {snow_bounds}")
    print(f"Snow depth shape: {snow_data.shape}")
    print(f"Snow depth resolution: {snow_transform[0]:.2f}m")
    
    # Load terrain data with full spatial metadata
    with rasterio.open(terrain_path) as terrain_src:
        terrain_crs = terrain_src.crs
        terrain_transform = terrain_src.transform
        terrain_bounds = terrain_src.bounds
        
        print(f"Terrain bounds: {terrain_bounds}")
        print(f"Terrain resolution: {terrain_transform[0]:.2f}m")
        
        # Check if the snow extent is within terrain bounds
        if not (terrain_bounds.left <= snow_bounds.left and 
                terrain_bounds.right >= snow_bounds.right and
                terrain_bounds.bottom <= snow_bounds.bottom and
                terrain_bounds.top >= snow_bounds.top):
            print("WARNING: Snow depth extent extends beyond terrain data!")
            print("Consider using terrain data that covers the full snow depth area.")
        
        # Method 1: If same CRS and resolution, use window reading
        if (terrain_crs == snow_crs and 
            abs(terrain_transform[0] - snow_transform[0]) < 1e-6):
            
            print("Same CRS and resolution - using window extraction")
            
            # Calculate window in terrain raster that corresponds to snow bounds
            window = from_bounds(*snow_bounds, terrain_transform)
            
            # Read only the required window
            terrain_aligned = terrain_src.read(1, window=window)
            
            # Ensure exact shape match (handle any rounding differences)
            if terrain_aligned.shape != snow_data.shape:
                min_rows = min(terrain_aligned.shape[0], snow_data.shape[0])
                min_cols = min(terrain_aligned.shape[1], snow_data.shape[1])
                terrain_aligned = terrain_aligned[:min_rows, :min_cols]
                snow_data = snow_data[:min_rows, :min_cols]
                
        else:
            print("Different CRS or resolution - using reprojection")
            
            # Method 2: Reproject terrain to exactly match snow depth grid
            terrain_aligned = np.empty_like(snow_data, dtype=np.float32)
            
            reproject(
                source=rasterio.band(terrain_src, 1),
                destination=terrain_aligned,
                src_transform=terrain_transform,
                src_crs=terrain_crs,
                dst_transform=snow_transform,
                dst_crs=snow_crs,
                resampling=Resampling.bilinear  # or Resampling.cubic_spline for smoother terrain
            )
    
    print(f"Final aligned shapes - Terrain: {terrain_aligned.shape}, Snow: {snow_data.shape}")
    
    return terrain_aligned, snow_data

# ------------------ resample and aligne--------------------------------------

def reproject_and_align_rasters(src_paths, case_folder, target_crs=None, reference_raster=None, 
                               resolution=None, resampling_method=Resampling.nearest,
                               apply_ref_mask=True, set_negative_to_nodata=True):
    """
    Reproject and align multiple rasters to a target CRS and/or match the extent and resolution of a reference raster.
    
    Parameters:
    -----------
    src_paths : list of str
        Paths to the source raster files to be processed
    dst_paths : list of str, optional
        Paths where the reprojected and aligned rasters will be saved
        If None, will generate output paths based on source paths
    target_crs : dict or str, optional
        Target coordinate reference system as a PROJ4 string, EPSG code, or WKT string
        If None and reference_raster is provided, the CRS from reference_raster will be used
    reference_raster : str, optional
        Path to a reference raster to match the extent and resolution
        If provided, the output will align with this raster
    resolution : tuple (float, float), optional
        (x_res, y_res) target resolution in target CRS units
        If None and reference_raster is provided, the resolution from reference_raster will be used
    resampling_method : rasterio.warp.Resampling, optional
        Resampling algorithm to use for reprojection (default: Resampling.nearest)
    apply_ref_mask : bool, optional
        If True and reference_raster is provided, will set output pixels to NoData where reference has NaN
        
    Returns:
    --------
    list of bool
        List of boolean values indicating success for each processed raster
    """
    
    # Ensure src_paths is a list
    if isinstance(src_paths, str):
        src_paths = [src_paths]
    
    #   define output folder
    output_folder = os.path.join(case_folder,"uniform")
    
    # create output folder 
    os.makedirs(output_folder, exist_ok=True)
    
    # Prepare reference raster mask (read only once)
    ref_mask = None
    ref_transform = None
    ref_width = None
    ref_height = None
    ref_crs = None
    common_nodata_values = [-999, -99, -9999, -3.40282e+38,]

    # Process reference raster if provided
    if reference_raster:
        with rasterio.open(reference_raster) as ref:
            target_crs = target_crs or ref.crs
            ref_transform = ref.transform
            ref_width = ref.width
            ref_height = ref.height
            ref_crs = ref.crs
            
            # Read reference mask (NaN values) if apply_ref_mask is True
            if apply_ref_mask:
                ref_data = ref.read(1)  # Read first band, assuming mask applies to all bands
                ref_nodata = ref.nodata
                
                # Create initial mask from NaN values
                ref_mask = np.isnan(ref_data)
                
                # Add NoData values to mask if defined
                if ref_nodata is not None:
                    ref_mask = np.logical_or(ref_mask, ref_data == ref_nodata)
                
                # Try common NoData values if no mask was created
                if not np.any(ref_mask):
                    for value in common_nodata_values:
                        potential_mask = (ref_data == value)
                        if np.any(potential_mask):
                            ref_mask = np.logical_or(ref_mask, potential_mask)
                            print(f"Using {value} as NoData in reference raster")
    elif target_crs is None:
        raise ValueError("Either target_crs or reference_raster must be provided")
    
    # List to store processing results
    processing_results = []
    
    # Process each raster
    for src_path in src_paths:
        
        #   Extract name for identification and prints
        date = re.search(r'\d{8}|\d{4}', src_path).group()
        
        dst_path = os.path.join(output_folder, f"{date}_uniform.tif")

        try:
            with rasterio.open(src_path) as src:
                src_crs = src.crs
                src_transform = src.transform
            
                # Read first band to change no data values
                src_data = src.read(1)  
                
                # Replace common NoData values with -999
                for value in common_nodata_values:
                    src_data = np.where(src_data == value, -999, src_data)
                
                # Determine transform and dimensions
                if reference_raster:
                    dst_transform = ref_transform
                    dst_width = ref_width
                    dst_height = ref_height
                    dst_crs = ref_crs
                else:
                    # Calculate the optimal transform for the new CRS
                    dst_transform, dst_width, dst_height = calculate_default_transform(
                        src_crs, target_crs, src.width, src.height, 
                        left=src.bounds.left, bottom=src.bounds.bottom, 
                        right=src.bounds.right, top=src.bounds.top,
                        resolution=resolution
                    )
                    dst_crs = target_crs
                 
                # Create destination dataset
                dst_kwargs = src.meta.copy()
                dst_kwargs.update({
                    'crs': dst_crs,
                    'transform': dst_transform,
                    'width': dst_width,
                    'height': dst_height,
                    'nodata': -999,
                    "compress": "lzw",
                    "predictor": 2
                })
                
            
                with rasterio.open(dst_path, 'w', **dst_kwargs) as dst:
                    # Initialize destination arrays for each band
                    #dst_data = np.zeros((src.count, dst_height, dst_width), dtype=dst_kwargs['dtype'])
                    dst_data = np.full((src.count, dst_height, dst_width), -999, dtype=dst_kwargs['dtype'])
                    
                    # Reproject each band
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=dst_data[i-1],
                            src_transform=src_transform,
                            src_crs=src_crs,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            resampling=resampling_method,
                            src_nodata=src.nodata or -999,
                            dst_nodata=-999, 
                            dst_width = ref_width,
                            dst_height = ref_height,
                            compress=Compression.lzw
                        )
                        
                        # Apply negative to NoData conversion if enabled
                        if set_negative_to_nodata:
                           dst_data[i-1][dst_data[i-1] < 0] = -999
                        
                        # Apply reference mask if available
                        if ref_mask is not None and apply_ref_mask:
                            dst_data[i-1][ref_mask] = -999
                    
                    # Write all bands at once
                    dst.write(dst_data)
                    dst.nodata = -999
                
                processing_results.append(True)
                print(f"Successfully processed {os.path.basename(src_path)}")
                
        except Exception as e:
            print(f"Error processing {os.path.basename(src_path)}: {e}")
            processing_results.append(False)
    
    return output_folder

#------------------------------------------------------------------------------



#   function to get extent from shape in csv file
def get_feature_extent(csv_file, feature_id, raster_file):
    with rasterio.open(raster_file) as src:
        height = src.height  # Get raster height to handle y-alignment correctly
        transform = src.transform  # Get raster transform for alignment

    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['ID'] == feature_id:
                min_x, min_y, max_x, max_y = map(float, [row['min_x'], row['min_y'], row['max_x'], row['max_y']])
                
                ## Round and align extents to the raster grid
                #min_x = align_to_raster_grid_x(round_down_to_nearest_2m(min_x), transform)
                #max_x = align_to_raster_grid_x(round_up_to_nearest_2m(max_x), transform)
                #min_y = align_to_raster_grid_y(round_down_to_nearest_2m(min_y), transform, height)
                #max_y = align_to_raster_grid_y(round_up_to_nearest_2m(max_y), transform, height)
             
                print("Shape extracted for clipping")
                
                return box(min_x, min_y, max_x, max_y)  # Create a bounding box

    return sys.exit("Error: Not able to retrieve extent")



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
    metdata
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

# ----------------------------------------------------------------------------

def convert_extent(extent, raster_file, adjust_extent):
    with rasterio.open(raster_file) as src:
        height = src.height  # Get raster height to handle y-alignment correctly
        transform = src.transform  # Get raster transform for alignment
        
        #   extract coordinates from given extent
        min_x, min_y, max_x, max_y = extent[0], extent[1], extent[2], extent[3]
        
        if adjust_extent == True:
            
            # Round and align extents to the raster grid
            min_x = align_to_raster_grid_x(round_down_to_nearest_2m(min_x), transform)
            max_x = align_to_raster_grid_x(round_up_to_nearest_2m(max_x), transform)
            min_y = align_to_raster_grid_y(round_down_to_nearest_2m(min_y), transform, height)
            max_y = align_to_raster_grid_y(round_up_to_nearest_2m(max_y), transform, height)
        
            print("Shape extracted for clipping")
            
            return box(min_x, min_y, max_x, max_y)  # Create a bounding box
        
# -----------------------------------------------------------------------------

#   function to clip raster with specified shape
def clip_raster(input_raster, output_raster, extent):

     with rasterio.open(input_raster) as src:
         extent_geojson = [mapping(extent)]  # Convert to GeoJSON-like format
         out_image, out_transform = rasterio.mask.mask(src, extent_geojson,crop=True, pad=False)
         out_meta = src.meta.copy()

         # Update metadata to match the clipped raster
         out_meta.update({
             "driver": "GTiff",
             "height": out_image.shape[1],
             "width": out_image.shape[2],
             "transform": out_transform,
             "crs": src.crs
         })

         with rasterio.open(output_raster, "w", **out_meta) as clipped_raster:
             clipped_raster.write(out_image)

     print("Raster clipped")
     
     return(out_image)

#   ---------------- ----------------------------------------------------------

# Function to calculate slope and aspect
def calculate_terrain_parameter(dem_path, pixel_size, output_folder, parameters):

    #   load data and extract metadata
    dem, meta_dem,nodata_value = load_raster(dem_path,metadata=["meta","nodata"])

    results={}
    
    #   calculate base for terrain feature calculation
    gradient_x, gradient_y = np.gradient(dem, pixel_size, pixel_size)
    
    if "slope" in parameters:
        #   calculate slope
        slope_radians = np.arctan(np.sqrt(gradient_x**2 + gradient_y**2))
        slope_degrees = np.degrees(slope_radians)
        
        # add metadata for export
        meta_dem.update(dtype=rasterio.float32, count=1)
        
        #   define export path
        output_raster = os.path.join(output_folder, "slope.tif")
    
        #   export
        with rasterio.open(output_raster, "w", **meta_dem) as dest:
            dest.write(slope_degrees.astype(rasterio.float32), 1)
        
        #   add variable to results for function output
        results["slope"] = slope_degrees
        
        print("Slope calculated")
    
    if "aspect" in parameters:
        #   calculate aspect
        aspect = np.arctan2(-gradient_y, gradient_x) * 180 / np.pi
        
        # Convert to absolute deviation from north
        aspect_deviation = np.abs(180 - np.abs(aspect))
        
        # add metadata for export
        meta_dem.update(dtype=rasterio.float32, count=1)

        #   define export path
        output_raster_180 = os.path.join(output_folder, "aspect.tif")
        output_raster_360 = os.path.join(output_folder, "aspect_360.tif")

        #   export
        with rasterio.open(output_raster_180, "w", **meta_dem) as dest:
            dest.write(aspect_deviation.astype(rasterio.float32), 1)
        with rasterio.open(output_raster_360, "w", **meta_dem) as dest:
            dest.write(aspect.astype(rasterio.float32), 1)
        
        results["aspect"] = aspect_deviation
        
        print("Aspect calculated")
      
    return results

# ---------------- calculate slope ------------------------------------------

def calculate_slope(dem_path, pixel_size, output_folder=None):
    """
    Calculate slope from a digital elevation model.
    
    This function computes slope (in degrees) from a DEM raster.
    Results can be saved as a GeoTIFF file and are returned as numpy arrays.
    
    Parameters
    ----------
    dem_path : str
        File path to the digital elevation model raster.
    pixel_size : float
        Size of pixels in map units (typically meters). if output should have different resolution
    output_folder : str, optional
        Directory path where output raster will be saved.
        If None, no file will be exported.
        
    Returns
    -------
    numpy.ndarray
        Slope in degrees (0-90°)
        
    Notes
    -----
    The slope calculation uses the gradient method.
    
    Examples
    --------
    >>> slope_array = calculate_slope("dem.tif", 30.0, "./output")
    """
    try:
        # Load data and extract metadata
        dem, meta_dem, nodata_value = load_raster(dem_path, metadata=["meta", "nodata"])
        
        # Calculate base for terrain feature calculation
        gradient_x, gradient_y = np.gradient(dem, pixel_size, pixel_size)
        
        # Calculate slope
        slope_radians = np.arctan(np.sqrt(gradient_x**2 + gradient_y**2))
        slope_degrees = np.degrees(slope_radians)
        
        # Apply nodata mask if available
        if nodata_value is not None:
            mask = dem == nodata_value
            slope_degrees[mask] = nodata_value
        
        # Export if output folder is provided
        if output_folder:
            # Create output directory if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            # Add metadata for export
            meta_dem.update(dtype=rasterio.float32, count=1, nodata= -999)
            
            # Define export path
            output_slope = os.path.join(output_folder, "slope.tif")
        
            # Export
            with rasterio.open(output_slope, "w", **meta_dem) as dest:
                dest.write(slope_degrees.astype(rasterio.float32), 1)
            
            print("Slope calculated sucesfully")
        
        return slope_degrees
    
    except (IOError, rasterio.errors.RasterioError) as e:
        print(f"Error processing DEM for slope calculation: {str(e)}")
        raise

def calculate_aspect(dem_path, pixel_size, output_folder=None):
    """
    Calculate aspect from a digital elevation model.
    
    This function computes aspect from a DEM raster in two formats:
    1. Full 360° aspect where 0°/360° is north, 90° is east, 180° is south, 270° is west
    2. Deviation from north (0-180°) where 0° is north and 180° is south
    
    Results can be saved as GeoTIFF files and are returned as a dictionary.
    
    Parameters
    ----------
    dem_path : str
        File path to the digital elevation model raster.
    pixel_size : float
        Size of pixels in map units (typically meters).
    output_folder : str, optional
        Directory path where output rasters will be saved.
        If None, no files will be exported.
        
    Returns
    -------
    dict
        Dictionary containing:
        - "aspect": Full 360° aspect (0-360°)
        - "aspect_deviation": Aspect as deviation from north (0-180°)
        
    Examples
    --------
    >>> aspect_result = calculate_aspect("dem.tif", 30.0, "./output")
    >>> aspect_360 = aspect_result["aspect"]
    >>> aspect_dev = aspect_result["aspect_deviation"]
    """
    try:
        # Load data and extract metadata
        dem, meta_dem, nodata_value = load_raster(dem_path, metadata=["meta", "nodata"])
        results = {}
        
        # Calculate base for terrain feature calculation
        gradient_x, gradient_y = np.gradient(dem, pixel_size, pixel_size)
        
        # Calculate aspect (0° is east, 90° is north, etc.)
        aspect = np.arctan2(-gradient_y, gradient_x) * 180 / np.pi
        
        # Convert to 0-360° format where 0°/360° is north, 90° is east, etc.
        aspect = 90 - aspect  # Convert so 0° is north
        aspect = np.mod(aspect, 360)  # Ensure values are in 0-360 range
        
        # Calculate deviation from north (0-180°)
        aspect_deviation = np.minimum(aspect, 360 - aspect)
        
        # Apply nodata mask if available
        if nodata_value is not None:
            mask = dem == nodata_value
            aspect[mask] = nodata_value
            aspect_deviation[mask] = nodata_value
        
        # Export if output folder is provided
        if output_folder:
            # Create output directory if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            # Add metadata for export
            meta_dem.update(dtype=rasterio.float32, count=1)
            
            # Define export paths
            output_aspect_deviation = os.path.join(output_folder, "aspect_deviation.tif")
            output_aspect_360 = os.path.join(output_folder, "aspect_360.tif")
            
            # Export aspect deviation from north
            with rasterio.open(output_aspect_deviation, "w", **meta_dem) as dest:
                dest.write(aspect_deviation.astype(rasterio.float32), 1)
                
            # Export full 360° aspect
            with rasterio.open(output_aspect_360, "w", **meta_dem) as dest:
                dest.write(aspect.astype(rasterio.float32), 1)
            
            print("Aspect calculated sucesfully")
        
        results["aspect"] = aspect
        results["aspect_deviation"] = aspect_deviation
        
        return results
    
    except (IOError, rasterio.errors.RasterioError) as e:
        print(f"Error processing DEM for aspect calculation: {str(e)}")
        raise
#---------------------- curvature function --------------------------------------
def calculate_curvature(dem_path, window_size, output_folder, **kwargs):
    """
    Wrapper function for backward compatibility with existing code.
    Calls calculate_profile_curvature regardless of window_size parameter.
    
    Parameters:
        dem_path (str): Path to the input DEM raster.
        window_size (int): Ignored - profile curvature always uses 3x3 window.
        output_folder (str): Directory to save the output raster.
        **kwargs: Additional parameters (ignored for profile curvature).
    
    Returns:
        arr: profile curvature raster as np.array().
    """
    # Call the new profile curvature function
    # window_size is ignored since profile curvature requires 3x3
    return calculate_profile_curvature(dem_path, output_folder)


def calculate_profile_curvature(dem_path, output_folder):
    """
    Calculate profile curvature for slope transition analysis
    
    Profile curvature measures concavity and convexity along the vertical slope profile,
    giving information about slope transitions into steeper or more gentle characteristics.
    Provides insights about potential avalanche starting areas due to transitions from 
    less steep to more steep slopes.
    
    Uses standard 3x3 window for precise geomorphometric calculations.
    
    Parameters:
        dem_path (str): Path to the input DEM raster.
        output_folder (str): Directory to save the output profile curvature raster.
    
    Returns:
        arr: profile curvature raster as np.array().
        
    Interpretation:
        - Positive values: Convex slopes (slope gets gentler downhill) - potential avalanche release zones
        - Negative values: Concave slopes (slope gets steeper downhill) - accumulation zones
        - Near zero: Uniform slope gradient
    """
    
    def profile_curvature_function(window):
        """
        Calculate profile curvature using 3x3 window
        Profile curvature measures curvature along the slope direction (maximum gradient direction)
        """
        # generic_filter passes a 1D array, reshape to 3x3
        window = window.reshape(3, 3)
        
        # Standard 3x3 window notation
        # z1 z2 z3
        # z4 z5 z6  
        # z7 z8 z9
        z1, z2, z3 = window[0, 0], window[0, 1], window[0, 2]
        z4, z5, z6 = window[1, 0], window[1, 1], window[1, 2]
        z7, z8, z9 = window[2, 0], window[2, 1], window[2, 2]
        
        # Calculate first derivatives (slope components)
        # Using Zevenbergen & Thorne (1987) central difference method
        dz_dx = (z3 + 2*z6 + z9 - z1 - 2*z4 - z7) / 8.0
        dz_dy = (z1 + 2*z2 + z3 - z7 - 2*z8 - z9) / 8.0
        
        # Calculate second derivatives
        d2z_dx2 = (z3 + z1 + z6 + z4 - 2*z2 - 2*z5) / 4.0
        d2z_dy2 = (z1 + z7 + z2 + z8 - 2*z4 - 2*z6) / 4.0
        d2z_dxdy = (z3 + z7 - z1 - z9) / 4.0
        
        # Calculate gradient magnitude squared
        grad_mag_sq = dz_dx**2 + dz_dy**2
        
        if grad_mag_sq < 1e-10:  # Avoid division by zero for flat areas
            return 0.0
        
        # Profile curvature formula (curvature along slope direction)
        # Measures how slope steepness changes along the fall line
        # Positive values: convex slopes (getting gentler downhill)
        # Negative values: concave slopes (getting steeper downhill)
        numerator = (dz_dx**2 * d2z_dx2 + 2*dz_dx*dz_dy*d2z_dxdy + dz_dy**2 * d2z_dy2)
        denominator = grad_mag_sq**(3/2)
        
        profile_curv = numerator / denominator
        
        profile_curv_ranged = np.where((profile_curv < -10) | (profile_curv > 10), np.nan, profile_curv)
        
        # Apply scaling factor to make values more visible for slope analysis
        # Typical curvature values are very small, scaling helps with avalanche analysis
        return profile_curv_ranged
    
    # Load dem and metadata
    dem, meta_dem, nodata_value = load_raster(dem_path, metadata=["meta", "nodata"])
    
    # Handle no data values
    mean = np.nanmean(dem)
    dem_array = np.where(np.isnan(dem), mean, dem)
    
    # Apply profile curvature calculation using standard 3x3 window
    profile_curvature = generic_filter(dem_array, profile_curvature_function, 
                                      size=3, mode="nearest")
    
    #NEW
    # Add z-score standardization
    mean_curv = np.nanmean(profile_curvature)
    std_curv = np.nanstd(profile_curvature)
    
    # Z-score standardization
    profile_curvature_zscore = (profile_curvature - mean_curv) / std_curv
    
    # Optional: clip extreme z-scores (e.g., beyond ±3 standard deviations)
    profile_curvature_clipped = np.clip(profile_curvature_zscore, -5, 5)
    
    #   end NEW
    # Define output file path
    output_raster = os.path.join(output_folder, "profile_curvature_3x3.tif")
    
    # Write output raster
    with rasterio.open(output_raster, "w", **meta_dem) as dest:
        dest.write(profile_curvature.astype(rasterio.float32), 1)
    
    print(f"Profile curvature calculated for slope transition analysis using standard 3x3 window")
    print(f"Profile curvature range: {np.nanmin(profile_curvature):.3f} to {np.nanmax(profile_curvature):.3f}")
    print(f"Profile curvature std: {np.nanstd(profile_curvature):.3f}")
    
    return profile_curvature_clipped


# Example usage:
"""
# Calculate profile curvature (backward compatible)
result = calculate_curvature("dem.tif", 5, "output/")  # window_size ignored, uses 3x3

# Direct profile curvature calculation
result = calculate_profile_curvature("dem.tif", "output/")

# Both produce profile curvature optimized for slope transition and avalanche analysis
"""

def calculate_curvature_deprecated(dem_path, window_size, output_folder,**kwargs):
    """
    Calculate curvature

    Parameters:
        dem_path (str): Path to the input DEM raster.
        window_size (int): Size of the moving window (must be an odd number).
        output_folder (str): Directory to save the output TPI raster.

    Returns:
        arr: curvature raster as np.array().
    """
    
    def curvature_function(window):
        center = window[len(window) // 2]  # Center pixel
        neighbors = window[window != center]  # Exclude center for some calculations
    
        # Compute second derivative approximation (curvature)
        mean_neighbors = np.nanmean(neighbors)  # Mean of surrounding pixels
        curvature = center - mean_neighbors  # Difference from center
    
        return curvature
    
    #   load dem and metadata
    dem, meta_dem, nodata_value = load_raster(dem_path,metadata=["meta", "nodata"])
    
    #   handle no data values
    mean = np.nanmean(dem)
    dem_array = np.where(np.isnan(dem),mean,dem)


    # Apply curvature calculation using a moving window
    curvature = generic_filter(dem_array, curvature_function, size=window_size, mode="nearest")
    
    #   if kwargs is not calles instead of None write nothing to the path
    terrain_type = kwargs.get("terrain_type")
    
    #   if kwargs is not calles instead of None write nothing to the path
    terrain_type = f"_{terrain_type}" if isinstance(terrain_type, str) and terrain_type else ""
 
    # Define output file path
    output_raster = os.path.join(output_folder, f"curvature_{window_size}{terrain_type}.tif")

    # Write output raster
    with rasterio.open(output_raster, "w", **meta_dem) as dest:
        dest.write(curvature.astype(rasterio.float32), 1)

    print(f"Curvature calculated with neigbourhood of {window_size}")
    
    return curvature

    
#   ---------------------------------------------------------------------------
# Function to calculate Topographic Position Index (TPI)
def calculate_tpi_raw(dem_path, window_size, output_folder, **kwargs):
    """
    Calculate Topographic Position Index
    
    Parameters:
        dem_path (str): Path to the input DEM raster.
        window_size (int): Size of the moving window (must be an odd number).
        output_folder (str): Directory to save the output TPI raster.
    
    Returns:
        arr: TPI raster as np.array().
    """

    try:
        #   load array and metadata
        dem, meta_dem,nodata_value = load_raster(dem_path, metadata=["meta", "nodata"])
        
        # Replace NoData values with NaN for processing
        #dem_array = dem.filled(np.nan)
    
    #TODO: problem because of nan. what to do?
        mean = np.nanmean(dem)
        dem_array = np.where(np.isnan(dem),mean,dem)
    
        # Compute mean elevation of neighborhood using a uniform filter
        mean_elevation = uniform_filter(dem_array, size=window_size, mode='nearest')
    
        # Compute TPI: elevation - mean(neighbors)
        tpi = dem_array - mean_elevation
    
        # Handle NoData values properly
        #tpi[np.isnan(dem_array)] = nodata_value  # Restore NoData values
    
        # Update metadata for output
        meta_dem.update(dtype=rasterio.float32, count=1, nodata=nodata_value, compression="lzw") #TODO_compression try
        
        #   extract additional path name variable from kwargs
        terrain_type = kwargs.get("terrain_type")
        
        #   if kwargs is not calles instead of None write nothing to the path
        terrain_type = f"_{terrain_type}" if isinstance(terrain_type, str) and terrain_type else ""
    
        # Define output file path
        output_raster = os.path.join(output_folder, f"tpi_{window_size}{terrain_type}.tif")
    
        # Write output raster
        with rasterio.open(output_raster, "w", **meta_dem) as dest:
            dest.write(tpi.astype(rasterio.float32), 1)
    
        print(f"TPI calculated with neigbourhood of {window_size}")
        
        return tpi
    
    except Exception as e:
        print(f"Error calculating TPI: {e}")
        return None
    
    
# Function to calculate Topographic Position Index (TPI)
def calculate_tpi(dem_path, window_size, output_folder, **kwargs):
    """
    Calculate Topographic Position Index

    Parameters:
        dem_path (str): Path to the input DEM raster.
        window_size (int): Size of the moving window (must be an odd number).
        output_folder (str): Directory to save the output TPI raster.

    Returns:
        arr: Standardized TPI raster as np.array().
    """
    
    try:
        #   load array and metadata
        dem, meta_dem,nodata_value = load_raster(dem_path, metadata=["meta", "nodata"])
        
        # Replace NoData values with NaN for processing
        #dem_array = dem.filled(np.nan)
    
    #TODO: problem because of nan. what to do?
        mean = np.nanmean(dem)
        dem_array = np.where(np.isnan(dem),mean,dem)
    
        # Compute mean elevation of neighborhood using a uniform filter
        mean_elevation = uniform_filter(dem_array, size=window_size, mode='nearest')
    
        # Compute TPI: elevation - mean(neighbors)
        tpi = dem_array - mean_elevation
        
        valid_mask = ~np.isnan(dem)  # Original valid data mask

        if np.sum(valid_mask) > 0:  # Ensure we have valid data
            tpi_mean = np.nanmean(tpi[valid_mask])
            tpi_std = np.nanstd(tpi[valid_mask])
            
            # Apply z-score transformation: (x - mean) / std
            if tpi_std > 0:  # Avoid division by zero
                tpi = (tpi - tpi_mean) / tpi_std
            else:
                print("Warning: TPI standard deviation is 0. Using raw TPI values.")
        else:
            print("Warning: No valid data found. Using raw TPI values.")
                
        tpi[~np.isnan(dem)] = tpi[~np.isnan(dem)]  # Keep calculated values for valid areas
        tpi[np.isnan(dem)] = np.nan  # Set original NoData areas back to NaN
        
        # Handle NoData values properly
        #tpi[np.isnan(dem_array)] = nodata_value  # Restore NoData values
    
        # Update metadata for output
        meta_dem.update(dtype=rasterio.float32, count=1, nodata=nodata_value, compression="lzw") #TODO_compression try
        
        #   extract additional path name variable from kwargs
        terrain_type = kwargs.get("terrain_type")
        
        #   if kwargs is not calles instead of None write nothing to the path
        terrain_type = f"_{terrain_type}" if isinstance(terrain_type, str) and terrain_type else ""
    
        # Define output file path
        output_raster = os.path.join(output_folder, f"tpi_{window_size}{terrain_type}.tif")
    
        # Write output raster
        with rasterio.open(output_raster, "w", **meta_dem) as dest:
            dest.write(tpi.astype(rasterio.float32), 1)
    
        print(f"TPI calculated with neigbourhood of {window_size}")
        
        return tpi
    
    except Exception as e:
        print(f"Error calculating TPI: {e}")
        return None

# -----------------------------------------------------------------------------

# Function to calculate Laplace filter (Laplacian)
def calculate_laplace(dem_path, pixel_size, output_folder):
    """
   Calculate Topographic Position Index (TPI)

   Parameters:
       dem_path (str): Path to the input DEM raster.
       window_size (int): Size of the moving window (must be an odd number).
       output_folder (str): Directory to save the output TPI raster.

   Returns:
       array: TPI raster as np.array().
   """
    #   load array and metadata
    dem, meta_dem, nodata_value = load_raster(dem_path, metadata=["meta", "nodata"])
    
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / pixel_size**2
    laplace = correlate(dem, kernel)
    
    # add metadata for export
    meta_dem.update(dtype=rasterio.float32, count=1)

    #   define export path
    output_raster = os.path.join(output_folder, "laplace.tif")

    #   export
    with rasterio.open(output_raster, "w", **meta_dem) as dest:
        dest.write(laplace.astype(rasterio.float32), 1)
    
    print("laplace calculated")
    return laplace

# ------------------ direct raster correlation ------------------------------

def correlate_two_raster(raster1_path, raster2_path, **kwargs):
    
    raster1, raster1_meta, raster1_nodata_value = load_raster(raster1_path,metadata=["meta", "nodata"])
    raster2, raster2_meta,raster2_nodata_value = load_raster(raster2_path, metadata=["meta", "nodata"])

    
    # Ensure same shape
    if raster1.shape != raster2.shape:
        raise ValueError("Rasters must have the same shape!")
    
    # Flatten arrays and remove NaN values
    valid_mask = ~np.isnan(raster1) & ~np.isnan(raster2)
    raster1_flat = raster1[valid_mask].flatten()
    raster2_flat = raster2[valid_mask].flatten()
    
    # Calculate Pearson correlation
    correlation, _ = pearsonr(raster1_flat, raster2_flat)
    
    #   define plot axis names
    if "raster1_name" in kwargs:
       raster1_name = kwargs["raster1_name"]
    else:
       raster1_name = os.path.basename(raster1_path)
           
    if "raster2_name" in kwargs:
       raster2_name = kwargs["raster2_name"]
    else:
       raster2_name = os.path.basename(raster2_path)

    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(raster1_flat, raster2_flat, gridsize=200, cmap='magma_r', bins='log')
    plt.xlabel(raster1_name)
    plt.ylabel(raster2_name)
    plt.title(f"Density Plot of {raster1_name} vs. {raster2_name}")
    plt.colorbar(hb, label="Log Count Density")
    plt.grid(True)
    # Add correlation text
    plt.text(
        0.05, 0.95, f"Correlation: {correlation:.2f}", 
        transform=plt.gca().transAxes, fontsize=12, 
        verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
    )
    plt.show()
    plt.close()
    
    print(f"Pearson Correlation: {correlation:.4f}")
    
    return (correlation)
    
# --------------------- terrain subclass assessment -------------------------

def feature_subgroup_sensitivity(
    library_dir,
    input_folder,
    feature_path,
    output_folder_name,
    feature_name='slope',
    classes=[0, 30, 35, 40, 45, np.inf],
    nodata_value=-999
):
    """
    Perform terrain feature sensitivity analysis on snow depth data
    
    Parameters:
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
    """
    # Preprocessing
    filepaths = glob.glob(os.path.join(input_folder, "*.tif"))
    n_timesteps = len(filepaths)
    
    # Load reference raster and feature
    reference_raster, meta_ref = load_raster(filepaths[0], metadata="meta")
    feature = load_raster(feature_path)
    
    # Match raster shapes
    feature, _ = match_array_shapes(feature, reference_raster)
    
    # Flatten features
    feature_flat = feature.flatten()
    
    # Define years
    years = extract_years_from_paths(filepaths)
    
    # Classification
    classes = np.array(classes, dtype=np.float32)
    n_labels = len(classes) - 1
    labels = np.arange(n_labels, dtype=np.uint8)
    feature_classes = np.digitize(feature_flat, classes) - 1
    
    # Prepare memory-mapped snow depth time series
    filename = 'snowdepth_timeseries.dat'
    if os.path.exists(filename):
        os.remove(filename)
    
    snowdepth_timeseries = np.memmap(
        filename, 
        dtype=np.float32, 
        mode='w+', 
        shape=(feature_flat.size, n_timesteps)
    )
    
    # Load snow depth data
    for i, filepath in enumerate(filepaths):
        with rasterio.open(filepath) as src:
            snowdepth = src.read(1).flatten()
            snowdepth_timeseries[:, i] = snowdepth
    
    # Create DataFrame
    df = pd.DataFrame.from_dict({
        'feature': feature_flat,
        **{f'snowdepth_t{i+1}': snowdepth_timeseries[:, i] for i in range(n_timesteps)},
        'feature_class': feature_classes
    })
    
    # Robust no-data filtering
    if nodata_value is not None:
        # Filter based on both feature and snow depth columns
        valid_mask = (
            (df['feature'] != nodata_value) & 
            (df['feature'] != np.nan) &
            (df['feature_class'] != -1)  # Exclude values outside of classes
        )
        for col in [f'snowdepth_t{i+1}' for i in range(n_timesteps)]:
            valid_mask &= (
                (df[col] != nodata_value) & 
                (df[col] != np.nan) &
                (df[col] > 0)  # Optional: ensure positive snow depth
            )
        
        df = df[valid_mask]
    
    # Statistical Analysis
    median_per_class = df.groupby('feature_class').agg(
        {f'snowdepth_t{i+1}': 'median' for i in range(n_timesteps)}
    )
    std_per_class = df.groupby('feature_class').agg(
        {f'snowdepth_t{i+1}': 'std' for i in range(n_timesteps)}
    )
    
    # Save statistical results
    median_per_class.to_csv(
        os.path.join(output_folder_name, f"{feature_name}_median_per_class_and_year.csv")
    )
    std_per_class.to_csv(
        os.path.join(output_folder_name, f"{feature_name}_std_per_class_and_year.csv")
    )
    
    # Visualization
    plt.figure(figsize=(10, 6))
    
    # Line plot for median snow depth
    colors = ['blue', 'yellow', 'orange', 'red', 'purple']
    feature_classes = median_per_class.index
    
    for i, feature_class in enumerate(feature_classes):
        mean_values = median_per_class.loc[feature_class].values
        std_values = std_per_class.loc[feature_class].values
        timesteps = np.arange(1, n_timesteps + 1)
        
        plt.plot(timesteps, mean_values, label=f'Class {feature_class}', 
                 color=colors[i], marker='o')
        plt.fill_between(timesteps, mean_values - std_values, 
                         mean_values + std_values, 
                         color=colors[i], alpha=0.2)
    
    plt.xlabel('Year')
    plt.ylabel('Median Snow Depth (m)')
    plt.title('Median Snow Depth Over Time by Terrain Class')
    plt.xticks(timesteps, years)
    plt.legend(title='Terrain Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder_name, f"{feature_name}_analysis.png"))
    plt.show()
    plt.close()
    
    snowdepth_timeseries = None
# -------------------- linear regression prediction---------------------------

def predict_by_linear_regression(training_raster_target_path, training_raster_feature_path, test_raster_target_path, test_raster_feature_path, prediction_raster_path):
    
    #   load rasters
    training_target= load_raster(training_raster_target_path)
    training_feature= load_raster(training_raster_feature_path)
    test_target, meta_test = load_raster(test_raster_target_path, metadata="meta")
    test_feature = load_raster(test_raster_feature_path)
    
    #   match shapes
    training_target, training_feature = match_array_shapes(training_target, training_feature)
    test_target, test_feature = match_array_shapes(test_target, test_feature)
    
    #   mask training data
    valid_mask = ~np.isnan(training_target) & ~np.isnan(training_feature)
    training_target_valid_flat = training_target[valid_mask].flatten()
    training_feature_valid_flat = training_feature[valid_mask].reshape(-1, 1)

    #   mask test data
    valid_mask = ~np.isnan(test_target) & ~np.isnan(test_feature)
    test_target_valid = test_target[valid_mask]
    test_feature_valid = test_feature[valid_mask].reshape(-1,1)
    
    correlation_train = correlate_two_raster(training_raster_target_path, training_raster_feature_path)
    
    # define and train model
    model = LinearRegression()
    model.fit(training_feature_valid_flat, training_target_valid_flat)
    
    #   model prediction
    predicted_target_flat  = model.predict(test_feature_valid.reshape(-1,1))
    
    #   reshape to original shape wincluding nan
    predicted_target = reshape_after_nandrop(predicted_target_flat, valid_mask, test_target)

    # Save predicted target raster
    meta_test.update(dtype=rasterio.float32, count=1)
    
    with rasterio.open(prediction_raster_path, "w", **meta_test) as dest:
        dest.write(predicted_target.astype(rasterio.float32), 1)

    print("prediction done")
    
    correlation_test = correlate_two_raster(test_raster_target_path, prediction_raster_path)
    
    return(predicted_target)

# --------------------------------------------------------------------------------

def mask(input_folder, resampling_method, mask_path):
    
    start_function = datetime.datetime.now()
    print("Start masking data at " + str(start_function))
    
    #   create list of tiffs to be processed
    filepaths = glob.glob(os.path.join(input_folder, "*.tif*"))
    
    #   create output folder for method 
    output_folder_method = create_folder_in_parent_dir(input_folder, "uniform_masked")
    
    # check if reference data is already opened dataarray or open tiff path
    if not isinstance(mask_path, xr.DataArray):    
        # open reference raster
        mask = rio.open_rasterio(mask_path)
        
    for filepath in filepaths:
        
        #   read raster
        raster = rio.open_rasterio(filepath)
        
        #   Extract name for identification and prints
        date = re.search(r'\d{8}|\d{4}', filepath).group()
        
        #   reproject/resample mask to fit data
        mask = mask.rio.reproject_match(raster,
                                        resampling=resampling_method)
        
        #   control sucess of resampling/reprojection TODO: adapted to each file probably not so good
        if (raster.rio.crs == mask.rio.crs and
              raster.rio.resolution() == mask.rio.resolution() and
              raster.rio.bounds() == mask.rio.bounds()
              ):
              print("Mask reprojected/resampled to fit data")
              
              # create output folder for year
              #output_folder_year = create_folder_in_parent_dir(input_folder, date)
     
        else:
              raise Exception ("Mask is not compatible and reprojection or alignment of " + date + " was unsucessfull")
    
        #   Mask all pixels that are rated valid
        raster_masked = raster.where(mask != 0, np.nan)
        
        
        # ---------------------save raster to folders ----------------------
        
        # method folder
        output_file_method =(f"{output_folder_method}/{date}_uniform_masked.tif")
        raster_masked.rio.to_raster(output_file_method)
    
        # year folder
        #output_file_year =(f"{output_folder_year}/{date}_uniform_masked.tif")
        #raster_masked.rio.to_raster(output_file_year)
        
        print(date + " is masked sucessfully")
        
    return output_folder_method

#   ------------ calculate buffer around center coordinates -------------------

def create_circular_filter(raster_path, center_x, center_y, radius_meters):
    """
    Create a circular filter for the given raster centered on the pixel containing the point.
    
    Parameters
    ----------
    raster_path : str
        Path to the raster file
    center_x : float
        X-coordinate of the center point
    center_y : float
        Y-coordinate of the center point
    radius_meters : float
        Radius of the circular filter in meters
        
    Returns
    -------
    tuple
        (masked_data, mask) where masked_data is the raster with values outside 
        the circle set to NaN, and mask is a boolean array
    """
    # Use your load_raster function to get data and metadata
    raster_data, meta, transform = load_raster(raster_path, metadata=["meta", "transform"])

    # Find the pixel that contains the center point
    center_row, center_col = rasterio.transform.rowcol(transform, center_x, center_y)
    
    # Get pixel size in meters
    pixel_size = transform[0]  # Assuming square pixels
    
    # Convert radius from meters to pixels
    radius_pixels = radius_meters / pixel_size
    
    # Create coordinates for each pixel
    rows, cols = np.indices(raster_data.shape)
    
    # Calculate distance from each pixel to the center pixel
    distances = np.sqrt((rows - center_row)**2 + (cols - center_col)**2)
    
    # Create circular mask
    mask = distances <= radius_pixels
    
    # Apply mask to data
    masked_data = np.copy(raster_data)
    masked_data[~mask] = np.nan
    
    return masked_data, mask

#   ------------ create new buffer based ROi around points -------------------

def analyze_stations_with_circular_filter(csv_path, input_filepaths, buffer_radius, output_folder=None, meta=None):
    """
    Analyze raster data using circular filters centered on weather stations.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing weather station data with coordinates
    input_filepaths : list
        list of filepaths
    buffer_radius : float or int
        Radius of the circular filter in the same units as the raster
        (typically meters for projected data)
    analysis_type : str
        Type of analysis to perform: 'statistics', 'timeseries', or 'difference'
    output_folder : str, optional
        Path to save results. If None, uses a default location.
        
    Returns
    -------
    dict
        Dictionary with station IDs as keys and analysis results as values
    """
    import os
    import glob
    import pandas as pd
    import numpy as np
    import rasterio
    from scipy.ndimage import generic_filter
    import matplotlib.pyplot as plt
    import re
    import warnings
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Read station locations from CSV
    try:
        stations_df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        try:
            stations_df = pd.read_csv(csv_path, encoding='utf-8-sig')
        except:
            stations_df = pd.read_csv(csv_path, encoding='latin1')
    
    # Look for coordinate columns in the dataframe
    potential_x_cols = ['x', 'X', 'lon', 'longitude', 'easting', 'east', 'POINT_X']
    potential_y_cols = ['y', 'Y', 'lat', 'latitude', 'northing', 'north', 'POINT_Y']
    potential_id_cols = ['id', 'ID', 'name', 'NAME', 'station', 'STATION', 'code', 'CODE']
    
    x_col = None
    y_col = None
    id_col = None
    
    # Find the coordinate columns
    for col in stations_df.columns:
        if col in potential_x_cols:
            x_col = col
        elif col in potential_y_cols:
            y_col = col
        elif col in potential_id_cols:
            id_col = col
    
    # Validate that we have the necessary columns
    if x_col is None or y_col is None or id_col is None:
        raise ValueError("Could not identify coordinate or ID columns in the CSV file.")
    
    print(f"Using columns: ID={id_col}, X={x_col}, Y={y_col}")
    
        
    # Extract dates from filenames
    dates = []
    for file in input_filepaths:
        date_match = re.search(r'\d{8}|\d{4}', os.path.basename(file))
        if date_match:
            dates.append(date_match.group())
        else:
            dates.append(os.path.basename(file))
    
    # Initialize dictionary to store all clipped data for timeseries statistics
    station_timeseries = {}
    all_year_stats ={}
    
    # Process each station
    for _, station_row in stations_df.iterrows():
        station_id = str(station_row[id_col])
        center_x = float(station_row[x_col])
        center_y = float(station_row[y_col])
        
        print(f"Processing station {station_id} at ({center_x}, {center_y})...")
        
        # Create station-specific output folder
        station_folder = os.path.join(output_folder, f"station_{station_id}")
        os.makedirs(station_folder, exist_ok=True)
        
        # Storage for yearly clip of raster based on buffer
        station_areas = {}
        all_year_stats_buffer = {  
            'mean': {},
            'std': {},
            'median': {},
            'variance': {},
            'min': {},
            'max': {}
        }
        all_year_stats_buffer_path = os.path.join(station_folder, f"all_year_stats_{station_id}_{buffer_radius}_meter")
      
        # Process each raster file
        for raster_file, date in zip(input_filepaths, dates):
            
                # Apply circular filter
                masked_data, mask = create_circular_filter(raster_file, center_x, center_y, buffer_radius)
                
                # Mask raster data
               #valid_data = masked_data[~np.isnan(masked_data)]
                
                # Store yearly cxlipping area and date  in the station_areas dictionary
                station_areas[date] = {
                    "data": masked_data,
                    "date": date
                }
                
                #   add yearly data and date to the station dict
                station_timeseries[station_id] = station_areas
                
                #   define output csv path
                #yearly_stats_output_csv_buffer = os.path.join(station_folder, f"{station_id}_{date}_{buffer_radius}_meter")
        
                #   calculate yearly statistics
                #TODO: original                yearly_stats_buffer = calculate_yearly_statistics(valid_data, yearly_stats_output_csv_buffer, date)
                #yearly_stats_buffer = calculate_yearly_statistics(masked_data, yearly_stats_output_csv_buffer, date)
                yearly_stats_buffer = calculate_yearly_statistics(masked_data, date=date)

                # Merge the new stats into the existing dictionary
                for stat_type, stat_values in yearly_stats_buffer.items():
                    all_year_stats_buffer[stat_type].update(stat_values)
      
                #   ouput path for deviation results
                deviations_output_path = os.path.join(station_folder, f"{station_id}_{date}_{buffer_radius}_meter_deviations")
        
        #   save statistics from buffer area
        # Save to CSV file if path is provided
        if all_year_stats_buffer_path:
            # Get all unique dates across all statistic types
            all_dates = set()
            for stat_values in all_year_stats_buffer.values():
                all_dates.update(stat_values.keys())
            all_dates = sorted(all_dates)
            
            with open(all_year_stats_buffer_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header row with Date and all stat types
                header = ['Date'] + list(all_year_stats_buffer.keys())
                writer.writerow(header)
                
                # Write data rows
                for date in all_dates:
                    row = [date]
                    for stat_type in all_year_stats_buffer.keys():
                        value = all_year_stats_buffer[stat_type].get(date, '')
                        row.append(value)
                    writer.writerow(row)
        
        #   calculate deviations 
        deviations = calculate_yearly_deviations_dictionary_test(station_areas, all_year_stats_buffer, deviations_output_path,meta)


        
        #   calculate timeseries statistics #TODO: Necessary?
        #timeseries_statistics_folder, timeseries_statistics_dict = calculate_timeseries_statistics()
                
 
# -------------------- calculate yearly statistics ----------------------------

def calculate_yearly_statistics(input_data, output_csv_path=None, date=None):  
    """
    Calculate basic statistics for each snow depth raster.
    
    Parameters
    ----------
    input_data : list or numpy.ndarray
        Either a list of file paths to snow depth rasters, or a 3D numpy array with shape (n_years, height, width)
    output_csv_path : str, optional
        Path where the statistics will be saved as a CSV file
    dates : list, optional
        List of date strings corresponding to each layer in the numpy array.
        Required if input_data is a numpy array.
        
    Returns
    -------
    dict
        Dictionary of yearly statistics with keys 'mean', 'std', 'median', 'variance', 'min', 'max'
        Each contains a sub-dictionary mapping dates to their respective statistic values
    """
    # Initialize the restructured dictionary to collect stats by type across years
    yearly_stats = {  
        'mean': {},
        'std': {},
        'median': {},
        'variance': {},
        'min': {},
        'max': {}
    }
    
    # Handle case where input is a list of file paths
    if isinstance(input_data, list):
        filepaths = input_data
        for filepath in filepaths:
            # Extract date from filename
            date_match = re.search(r'\d{8}|\d{4}', filepath)
            if not date_match:
                print(f"Warning: Could not extract date from {filepath}, skipping")
                continue
            
            date = date_match.group()
            print(f"Processing {date} for basic statistics...")
            
            # Read data
            data = load_raster(filepath)
            
            # Calculate statistics
            calculate_and_store_stats(data, date, yearly_stats)
    
    # Handle case where input is a numpy array
    elif isinstance(input_data, np.ndarray):
        if date is None:
           raise ValueError("When providing a numpy array, 'dates' parameter is required")
        
       # if len(dates) != input_data.shape[0]:
        #    raise ValueError(f"Length of dates ({len(dates)}) must match first dimension of array ({input_data.shape[0]})")
        
        #for i, date in enumerate(dates):
        #print(f"Processing layer {i+1} (date: {date}) for basic statistics...")
          #  data = input_data[i]
        #date =dates
        
        print(f"Processing {date} for basic statistics...")
        # Calculate statistics
        calculate_and_store_stats(input_data, date, yearly_stats)
    
    else:
        raise TypeError("input_data must be either a list of file paths or a numpy array")
        
    # Save to CSV file if path is provided
    if output_csv_path:
        # Get all unique dates across all statistic types
        all_dates = set()
        for stat_values in yearly_stats.values():
            all_dates.update(stat_values.keys())
        all_dates = sorted(all_dates)
        
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header row with Date and all stat types
            header = ['Date'] + list(yearly_stats.keys())
            writer.writerow(header)
            
            # Write data rows
            for date in all_dates:
                row = [date]
                for stat_type in yearly_stats.keys():
                    value = yearly_stats[stat_type].get(date, '')
                    row.append(value)
                writer.writerow(row)
                
    return yearly_stats

def calculate_and_store_stats(data, date, yearly_stats):
    """Helper function to calculate statistics and store them in the yearly_stats dictionary"""
    # Calculate statistics
    one_year_mean = np.nanmean(data)
    one_year_std = np.nanstd(data)
    one_year_median = np.nanmedian(data)
    one_year_variance = np.nanvar(data)
    one_year_min = np.nanmin(data)
    one_year_max = np.nanmax(data)
    
    # Store statistics by type for this year
    yearly_stats['mean'][date] = one_year_mean
    yearly_stats['std'][date] = one_year_std
    yearly_stats['median'][date] = one_year_median
    yearly_stats['variance'][date] = one_year_variance
    yearly_stats['min'][date] = one_year_min
    yearly_stats['max'][date] = one_year_max
    
    print(f"Year {date}: Mean={one_year_mean:.2f}m, StdDev={one_year_std:.2f}m")
# ------------------------- calculate timeseries statistics ------------------

def calculate_timeseries_statistics(input_filepaths, output_folder=None, output_folder_name='global_statistics', 
                    use_parallel=True, max_workers=None):
    """
    Calculate pixel-wise statistical parameters for a time series of raster files.
    
    Args:
        input_filepaths (list): List of paths to input raster files
        output_folder (str, optional): Path to the output folder. If None, will create folder in parent dir of first file.
        output_folder_name (str, optional): Name of the output folder if output_folder not specified. Defaults to 'global_statistics'.
        use_parallel (bool, optional): Use parallel processing for loading rasters. Defaults to True.
        max_workers (int, optional): Maximum number of parallel workers. Defaults to None (auto).
    
    Returns:
        tuple: (str, dict) Path to the output folder and dictionary with statistical parameter arrays
    """
    start_function = datetime.datetime.now()
    print(f"Calculating timeseries global statistics started at {start_function}")
    
    # Validate input
    if not input_filepaths:
        raise ValueError("No input filepaths provided")
    
    files = sorted(input_filepaths)  # Sort to ensure consistent order
    print(f"Processing {len(files)} specified files")
    
    # Create output folder
    if output_folder is None:
        # Get parent directory of the first file
        first_file_dir = os.path.dirname(files[0])
        output_folder = create_folder_in_parent_dir(first_file_dir, output_folder_name)
    else:
        os.makedirs(output_folder, exist_ok=True)
        
    # Extract dates from filenames
    rasters_dates = []
    for file in files:
        try:
            date = re.search(r'\d{8}|\d{4}', os.path.basename(file)).group()
            rasters_dates.append(date)
            print(f"{date} added for stacking")
        except AttributeError:
            print(f"Warning: Could not extract date from {os.path.basename(file)}")
    
    # Get reference metadata from first file
    with rasterio.open(files[0]) as src:
        profile = src.profile.copy()
        shape = (src.height, src.width)
    
    # Load all rasters
    all_rasters = []
    
    # First try parallel loading
    if use_parallel:
        try:
            # Determine number of workers
            if max_workers is None:
                max_workers = min(multiprocessing.cpu_count() - 1 or 1, 4)  # Limit to 4 workers
            
            # Parallel loading
            print(f"Using parallel processing with {max_workers} workers")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(load_raster, file): file for file in files}
                
                for future in as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        data = future.result()
                        if data is not None:
                            all_rasters.append(data)
                    except Exception as e:
                        print(f"Error processing {os.path.basename(file)}: {e}")
        except Exception as e:
            print(f"Parallel processing failed: {e}. Falling back to sequential processing.")
            use_parallel = False
    
    # Fall back to sequential processing if parallel fails or is disabled
    if not use_parallel or not all_rasters:
        print("Using sequential processing")
        all_rasters = []  # Reset in case partial results were collected
        for file in files:
            data = load_raster(file)
            if data is not None:
                all_rasters.append(data)
    
    if not all_rasters:
        raise ValueError("No valid raster data could be loaded")
    
    print(f"Successfully loaded {len(all_rasters)} out of {len(files)} rasters")
    
    # Stack all arrays along a new axis (time dimension)
    try:
        stacked_data = np.ma.stack(all_rasters, axis=0)
        print(f"Stacked data shape: {stacked_data.shape}")
    except Exception as e:
        print(f"Error stacking data: {e}")
        # Try to identify inconsistent shapes
        shapes = [arr.shape for arr in all_rasters]
        unique_shapes = set(shapes)
        if len(unique_shapes) > 1:
            print(f"Found inconsistent shapes: {unique_shapes}")
            raise ValueError("Rasters have inconsistent shapes. Cannot stack.")
        raise
    
    # Calculate pixel-wise statistics along the time axis
    print("Calculating statistics...")
    
    # Define a helper function to safely calculate statistics
    def safe_calculation(func, data, axis=0, fill_value=None):
        try:
            result = func(data, axis=axis)
            if fill_value is not None:
                return result.filled(fill_value)
            return result
        except Exception as e:
            print(f"Error calculating {func.__name__}: {e}")
            return np.zeros(data.shape[1:], dtype=np.float32)
    
    # Get nodata value from profile
    nodata_value = profile.get('nodata')
    if nodata_value is None:
        nodata_value = -999    
        
    # Calculate statistics
    mean_array = safe_calculation(np.ma.mean, stacked_data, fill_value=nodata_value)
    median_array = safe_calculation(np.ma.median, stacked_data, fill_value=nodata_value)
    std_array = safe_calculation(np.ma.std, stacked_data, fill_value=nodata_value)
    min_array = safe_calculation(np.ma.min, stacked_data, fill_value=nodata_value)
    max_array = safe_calculation(np.ma.max, stacked_data, fill_value=nodata_value)
    variance_array = safe_calculation(np.ma.var, stacked_data, fill_value=nodata_value)
    
    # Extract years from filenames for output naming
    years = []
    for file in files:
        try:
            # Try to get 4-digit year from filename
            year_match = re.search(r'(?<!\d)(\d{4})(?!\d)', os.path.basename(file))
            if year_match:
                years.append(year_match.group(1))
        except Exception:
            pass
    
    # Create years suffix for filenames
    if years and len(set(years)) < len(files):  # If we have years and they're not all unique
        years_suffix = "_" + "_".join(sorted(set(years)))
    else:
        years_suffix = "_custom_selection"  # Generic suffix for custom selection
    
    # Save results as GeoTIFFs
    statistical_parameters = {
        "mean": mean_array,
        "median": median_array,
        "std": std_array,
        "min": min_array,
        "max": max_array,
        "variance": variance_array
    }
    
    for parameter, result in statistical_parameters.items():
        output_file = os.path.join(output_folder, f"{parameter}{years_suffix}.tif")
        
        # Update profile for output with LZW compression
        out_profile = profile.copy()
        out_profile.update({
            'dtype': rasterio.float32,
            'compress': 'lzw',  # Add LZW compression
            'predictor': 2,
            "nodata": -999# Horizontal differencing predictor improves compression for continuous data
        })
        
        # Save the result
        try:
            with rasterio.open(output_file, 'w', **out_profile) as dst:
                dst.write(result.astype(np.float32), 1)
            print(f"{parameter.capitalize()} raster saved")
        except Exception as e:
            print(f"Error saving {parameter} raster: {e}")
    
    end_function = datetime.datetime.now()
    time_diff_function = str(end_function - start_function)
    print(f"All rasters processed and statistical parameters calculated in {time_diff_function}")
    
    return output_folder, statistical_parameters

# --------- calculate yearly deviations from global statistrics ------------

def calculate_yearly_deviations(filepaths, yearly_means, output_folder):
    """
    Calculate deviation metrics for each year's snow depth.
    
    Parameters
    ----------
    filepaths : list
        List of paths to snow depth rasters
    yearly_means : dict
        Dictionary of yearly mean values
    output_folder : str
        Path to output folder
        
    Returns
    -------
    dict
        Dictionary containing:
        1. Per-year rasters and metrics:
           - 'yearly_abs_deviations': List of absolute deviation rasters
           - 'yearly_norm_deviations': List of normalized deviation rasters
           - 'yearly_global_metrics': Dict of global metrics per year (min, max, median, std)
           - 'dates': List of dates corresponding to each year
        
        2. Per-pixel timeseries metrics for absolute deviations:
           - 'pixel_abs_mean_deviation': Raster of mean absolute deviation per pixel across all years
           - 'pixel_abs_median_deviation': Raster of median absolute deviation per pixel across all years
           - 'pixel_abs_std_deviation': Raster of standard deviation of the absolute deviations per pixel
           
        3. Per-pixel timeseries metrics for normalized deviations:
           - 'pixel_norm_mean_deviation': Raster of mean normalized deviation per pixel across all years
           - 'pixel_norm_median_deviation': Raster of median normalized deviation per pixel across all years
           - 'pixel_norm_std_deviation': Raster of standard deviation of the normalized deviations per pixel
    """
    # Initialize result structure
    results = {
        'yearly_abs_deviations': [],
        'yearly_norm_deviations': [],
        'yearly_global_metrics': {},
        'dates': [],
        # These will be populated after processing all years - absolute deviations
        'pixel_abs_mean_deviation': None,
        'pixel_abs_median_deviation': None,
        'pixel_abs_std_deviation': None,
        # These will be populated after processing all years - normalized deviations
        'pixel_norm_mean_deviation': None,
        'pixel_norm_median_deviation': None,
        'pixel_norm_std_deviation': None
    }
    
    # Initialize yearly_global_metrics structure
    results['yearly_global_metrics'] = {
        'min': [],
        'max': [],
        'median': [],
        'std': []
    }
    
    # First pass: Process each year's data
    all_abs_deviations = []
    all_norm_deviations = []
    sample_meta = None
    
    for filepath in filepaths:
        # Extract date from filename
        date_match = re.search(r'\d{8}|\d{4}', filepath)
        if not date_match:
            continue
        
        date = date_match.group()
        results['dates'].append(date)
        print(f"Calculating deviations for {date}...")
        
        # Read data
        data, meta = load_raster(filepath, metadata=["meta"])
        if sample_meta is None:
            sample_meta = meta
        
        # Get the mean value for this year
        one_year_mean = yearly_means[date]
        
        # 1. Calculate per-pixel deviations for this year
        abs_deviation = data - one_year_mean
        norm_deviation = abs_deviation / one_year_mean * 100
        
        # Store the deviations
        results['yearly_abs_deviations'].append(abs_deviation)
        results['yearly_norm_deviations'].append(norm_deviation)
        all_abs_deviations.append(abs_deviation)
        all_norm_deviations.append(norm_deviation)
        
        # 2. Calculate global metrics for this year (using absolute deviations)
        results['yearly_global_metrics']['min'].append(np.nanmin(abs_deviation))
        results['yearly_global_metrics']['max'].append(np.nanmax(abs_deviation))
        results['yearly_global_metrics']['median'].append(np.nanmedian(abs_deviation))
        results['yearly_global_metrics']['std'].append(np.nanstd(abs_deviation))
        
        # Create output folder for this year
        year_folder = os.path.join(output_folder, date)
        os.makedirs(year_folder, exist_ok=True)
        
        # Save yearly deviation maps
        abs_output = os.path.join(year_folder, f"{date}_abs_deviation.tif")
        with rasterio.open(abs_output, "w", **meta) as dest:
            dest.write(abs_deviation.astype(rasterio.float32), 1)
    
        norm_output = os.path.join(year_folder, f"{date}_norm_deviation.tif")
        with rasterio.open(norm_output, "w", **meta) as dest:
            dest.write(norm_deviation.astype(rasterio.float32), 1)
    
    # Second pass: Calculate per-pixel timeseries metrics across all years
    if all_abs_deviations and all_norm_deviations:
        # Stack all deviations along a new axis (time)
        stacked_abs_deviations = np.stack(all_abs_deviations, axis=0)
        stacked_norm_deviations = np.stack(all_norm_deviations, axis=0)
        
        # 3a. Calculate per-pixel timeseries metrics for absolute deviations
        pixel_abs_mean_deviation = np.nanmean(stacked_abs_deviations, axis=0)
        pixel_abs_median_deviation = np.nanmedian(stacked_abs_deviations, axis=0)
        pixel_abs_std_deviation = np.nanstd(stacked_abs_deviations, axis=0)
        
        # 3b. Calculate per-pixel timeseries metrics for normalized deviations
        pixel_norm_mean_deviation = np.nanmean(stacked_norm_deviations, axis=0)
        pixel_norm_median_deviation = np.nanmedian(stacked_norm_deviations, axis=0)
        pixel_norm_std_deviation = np.nanstd(stacked_norm_deviations, axis=0)
        
        # Add to results - absolute deviations
        results['pixel_abs_mean_deviation'] = pixel_abs_mean_deviation
        results['pixel_abs_median_deviation'] = pixel_abs_median_deviation
        results['pixel_abs_std_deviation'] = pixel_abs_std_deviation
        
        # Add to results - normalized deviations
        results['pixel_norm_mean_deviation'] = pixel_norm_mean_deviation
        results['pixel_norm_median_deviation'] = pixel_norm_median_deviation
        results['pixel_norm_std_deviation'] = pixel_norm_std_deviation
        
        # Save the per-pixel timeseries metrics - absolute deviations
        abs_mean_output = os.path.join(output_folder, "pixel_abs_mean_deviation.tif")
        with rasterio.open(abs_mean_output, "w", **sample_meta) as dest:
            dest.write(pixel_abs_mean_deviation.astype(rasterio.float32), 1)
        
        abs_median_output = os.path.join(output_folder, "pixel_abs_median_deviation.tif")
        with rasterio.open(abs_median_output, "w", **sample_meta) as dest:
            dest.write(pixel_abs_median_deviation.astype(rasterio.float32), 1)
        
        abs_std_output = os.path.join(output_folder, "pixel_abs_std_deviation.tif")
        with rasterio.open(abs_std_output, "w", **sample_meta) as dest:
            dest.write(pixel_abs_std_deviation.astype(rasterio.float32), 1)
        
        # Save the per-pixel timeseries metrics - normalized deviations
        norm_mean_output = os.path.join(output_folder, "pixel_norm_mean_deviation.tif")
        with rasterio.open(norm_mean_output, "w", **sample_meta) as dest:
            dest.write(pixel_norm_mean_deviation.astype(rasterio.float32), 1)
        
        norm_median_output = os.path.join(output_folder, "pixel_norm_median_deviation.tif")
        with rasterio.open(norm_median_output, "w", **sample_meta) as dest:
            dest.write(pixel_norm_median_deviation.astype(rasterio.float32), 1)
        
        norm_std_output = os.path.join(output_folder, "pixel_norm_std_deviation.tif")
        with rasterio.open(norm_std_output, "w", **sample_meta) as dest:
            dest.write(pixel_norm_std_deviation.astype(rasterio.float32), 1)
    
    return results

# ------------------- calculate yearly deviations from dictionary ----------

def calculate_yearly_deviations_dictionary_test(input_data, yearly_stats, output_folder=None, meta = None, date=None):
    """
    Calculate deviation metrics for each year's snow depth.
    
    Parameters
    ----------
    input_data : dict or list
        Either a dictionary mapping dates to numpy arrays,
        or a list of file paths to snow depth rasters
    yearly_means : dict
        Dictionary of yearly mean values
    output_folder : str, optional
        Path to output folder for saving results. If None, results won't be saved to disk.
        
    Returns
    -------
    dict
        Dictionary containing deviation metrics
    """
    # Initialize result structure
    results = {
        'yearly_abs_deviations': {},
        'yearly_norm_deviations': {},
        'yearly_global_metrics': {
            'min_deviation': {},
            'max_deviation': {},
            'median_deviation': {},
            'std_of_deviation': {},
            'var_of_deviation': {}
        }
    }
    
  
    # Process dictionary input
    if isinstance(input_data, dict):
        for date, data_dict in input_data.items():
            
            if meta is None:
                raise Exception(" metadata is missing")
                
            print(f"Calculating deviations for {date}...")
            
            # Get the mean value for this year
            one_year_mean = yearly_stats["mean"][date]
            
            # Access the actual numeric data from your dictionary
            data = data_dict["data"]
            
            
            # Calculate deviations
            abs_deviation = data - one_year_mean
            abs_positive_deviation = abs(data- one_year_mean)
            norm_deviation = abs_positive_deviation / one_year_mean * 100
            
            # Store the deviations
            results['yearly_abs_deviations'][date] = abs_deviation
            results['yearly_norm_deviations'][date] = norm_deviation
            
            # Calculate global metrics for this year
            results['yearly_global_metrics']['min_deviation'][date] = np.nanmin(abs_deviation)
            results['yearly_global_metrics']['max_deviation'][date] = np.nanmax(abs_deviation)
            results['yearly_global_metrics']['median_deviation'][date] = np.nanmedian(abs_deviation)
            results['yearly_global_metrics']['std_of_deviation'][date] = np.nanstd(abs_deviation)
            results['yearly_global_metrics']['var_of_deviation'][date] = np.nanvar(abs_deviation)

            
            # Save yearly deviation maps if output folder provided
            if output_folder:
                save_deviation_maps(date, abs_deviation, norm_deviation, output_folder, meta)
    
       
    # Process file paths input
    elif isinstance(input_data, list):
        for filepath in input_data:
            # Extract date from filename
            date_match = re.search(r'\d{8}|\d{4}', filepath)
            if not date_match:
                continue
            
            date = date_match.group()
            print(f"Calculating deviations for {date}...")
            
            # Read data
            data, meta = load_raster(filepath, metadata=["meta"])
            
            # Get the mean value for this year
            one_year_mean = yearly_stats["mean"][date]
            
            # Calculate deviations
            abs_deviation = data - one_year_mean
            abs_positive_deviation = abs(data- one_year_mean)
            norm_deviation = abs_positive_deviation / one_year_mean * 100
            
            # Store the deviations
            results['yearly_abs_deviations'][date] = abs_deviation
            results['yearly_norm_deviations'][date] = norm_deviation
            
            # Calculate global metrics for this year
            results['yearly_global_metrics']['min_deviation'][date] = np.nanmin(abs_deviation)
            results['yearly_global_metrics']['max_deviation'][date] = np.nanmax(abs_deviation)
            results['yearly_global_metrics']['median_deviation'][date] = np.nanmedian(abs_deviation)
            results['yearly_global_metrics']['std_of_deviation'][date] = np.nanstd(abs_deviation)
            results['yearly_global_metrics']['var_of_deviation'][date] = np.nanvar(abs_deviation)

            # Save yearly deviation maps if output folder provided
            if output_folder:
                save_deviation_maps(date, abs_deviation, norm_deviation, output_folder, meta)
    
    else:
        raise TypeError("input_data must be either a dictionary of arrays or a list of file paths")
        
    # Calculate pixel-level timeseries metrics
    if output_folder and results['yearly_abs_deviations']:
        calculate_and_save_pixel_metrics(results, output_folder, meta)
    
    return results

def save_deviation_maps(date, abs_deviation, norm_deviation, output_folder, meta=None):
    """Helper function to save deviation maps to disk"""
    if meta is None:
        # Create basic metadata if not provided
        height, width = abs_deviation.shape
        meta = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': abs_deviation.dtype,
        }
    
    # Create output folder for this year
    year_folder = os.path.join(output_folder, date)
    os.makedirs(year_folder, exist_ok=True)
    
    # Save absolute deviation map
    abs_output = os.path.join(year_folder, f"{date}_abs_deviation.tif")
    with rasterio.open(abs_output, "w", **meta) as dest:
        dest.write(abs_deviation.astype(rasterio.float32), 1)

    # Save normalized deviation map
    norm_output = os.path.join(year_folder, f"{date}_norm_deviation.tif")
    with rasterio.open(norm_output, "w", **meta) as dest:
        dest.write(norm_deviation.astype(rasterio.float32), 1)

def calculate_and_save_pixel_metrics(results, output_folder, meta):
    """
    Calculate pixel-level statistical metrics from yearly deviation arrays and save as GeoTIFF files.
    
    This function processes yearly absolute and normalized deviation arrays to compute
    per-pixel statistical metrics (mean, median, standard deviation, variance) across
    all years. The calculated metrics are then saved as individual GeoTIFF files.
    
    Parameters
    ----------
    results : dict
        Dictionary containing yearly deviation data with the following required keys:
        - 'yearly_abs_deviations': dict with date keys and 2D numpy arrays as values
        - 'yearly_norm_deviations': dict with date keys and 2D numpy arrays as values
        The arrays for each date should have the same spatial dimensions.
    
    output_folder : str or Path
        Directory path where the output GeoTIFF files will be saved. Directory must exist.
    
    meta : dict
        Rasterio metadata dictionary containing geospatial information for output files.
        Should include keys like 'driver', 'height', 'width', 'count', 'dtype', 'crs', 
        'transform', etc. This metadata will be used for all output GeoTIFF files.
    
    Returns
    -------
    None
        Function modifies the input `results` dictionary in-place by adding pixel 
        metrics and saves GeoTIFF files to disk.
    
    Notes
    -----
    The function calculates the following metrics for both absolute and normalized deviations:
    - Mean (pixel_abs_mean, pixel_norm_mean)
    - Median (pixel_abs_median, pixel_norm_median) 
    - Standard deviation (pixel_abs_std, pixel_norm_std)
    - Variance (pixel_abs_var, pixel_norm_var)
    
    All calculations use numpy's nan-aware functions to handle missing data properly.
    Output arrays are converted to float32 dtype before saving to optimize file size.
    
    """
    # Get all dates
    dates = list(results['yearly_abs_deviations'].keys())
    
    # Stack arrays for calculation
    abs_arrays = [results['yearly_abs_deviations'][date] for date in dates]
    norm_arrays = [results['yearly_norm_deviations'][date] for date in dates]
    
    stacked_abs = np.stack(abs_arrays, axis=0)
    stacked_norm = np.stack(norm_arrays, axis=0)
    
       # Calculate means and stds first
    abs_mean = np.nanmean(stacked_abs, axis=0)
    abs_std = np.nanstd(stacked_abs, axis=0, ddof=1)
    norm_mean = np.nanmean(stacked_norm, axis=0)
    norm_std = np.nanstd(stacked_norm, axis=0, ddof=1)
    
    # Calculate per-pixel metrics including CV
    pixel_metrics = {
        'pixel_abs_mean': abs_mean,
        'pixel_abs_median': np.nanmedian(stacked_abs, axis=0),
        'pixel_abs_std': abs_std,
        'pixel_abs_var': np.nanvar(stacked_abs, axis=0, ddof=1),
        'pixel_abs_cv': np.divide(abs_std, abs_mean, out=np.full_like(abs_std, np.nan), where=(abs_mean != 0)),
        'pixel_norm_mean': norm_mean,
        'pixel_norm_median': np.nanmedian(stacked_norm, axis=0),
        'pixel_norm_std': norm_std,
        'pixel_norm_var': np.nanvar(stacked_norm, axis=0, ddof=1),
        'pixel_norm_cv': np.divide(norm_std, norm_mean, out=np.full_like(norm_std, np.nan), where=(norm_mean != 0))
    }
    
    # Add to results
    results.update(pixel_metrics)
    
    # Get sample metadata from the first year #TODO: get all meta 
    #sample_date = dates[0]
    #sample_array = results['yearly_abs_deviations'][sample_date]
    #height, width = sample_array.shape
    #basic_meta = {
     #   'driver': 'GTiff',
      #  'height': height,
       # 'width': width,
        #'count': 1,
        #'dtype': rasterio.float32,
    #}
    
    # Save pixel metrics to files
    for metric_name, metric_array in pixel_metrics.items():
        output_file = os.path.join(output_folder, f"{metric_name}.tif")
        with rasterio.open(output_file, "w", **meta) as dest:
            dest.write(metric_array.astype(rasterio.float32), 1)
            
            
            


# ------------ calculate yearly deviations from filepaths --------------------

def calculate_yearly_deviations_deprecated(filepaths, yearly_means, output_folder):
    """
    Calculate deviation metrics for each year's snow depth.
    
    Parameters
    ----------
    filepaths : list
        List of paths to snow depth rasters
    yearly_means : dict
        Dictionary of yearly mean values
    output_folder : str
        Path to output folder
        
    Returns
    -------
    tuple
        (deviation_mean_timeseries, normalized_deviation_timeseries, std_timeseries) - 
        Lists of deviation, normalized deviation, and standard deviation rasters
    """
    deviation_mean_timeseries = []
    normalized_deviation_timeseries = []
    std_timeseries = []
    
    for filepath in filepaths:
        # Extract date from filename
        date_match = re.search(r'\d{8}|\d{4}', filepath)
        if not date_match:
            continue
        
        date = date_match.group()
        print(f"Calculating deviations for {date}...")
        
        # Read data as xarray for easier calculations
        data, meta = load_raster(filepath, metadata= ["meta"])
        
        # Get the mean value for this year
        one_year_mean = yearly_means[date]
        
        # Calculate per-pixel deviation from mean
        abs_deviation = data - one_year_mean
        # Calculate normalized deviation (% of mean)
        norm_deviation = abs_deviation / one_year_mean * 100

        # Calculate per-pixel standard deviation
        #data = np.expand_dims(data, axis=0)
        #pixel_std = np.nanstd(data)
        
        # Store for timeseries analysis
        deviation_mean_timeseries.append(abs_deviation)
        normalized_deviation_timeseries.append(norm_deviation)
        #std_timeseries.append(pixel_std)
        
        # Create output folder for this year
        year_folder = os.path.join(output_folder, date)
        os.makedirs(year_folder, exist_ok=True)
        
        # Save yearly deviation and std maps
        abs_output = os.path.join(year_folder, f"{date}_abs_deviation.tif")
        with rasterio.open(abs_output, "w", **meta) as dest:
            dest.write(abs_deviation.astype(rasterio.float32),1)
    
        norm_output = os.path.join(year_folder, f"{date}_norm_deviation.tif")
        with rasterio.open(norm_output, "w", **meta) as dest:
           dest.write(norm_deviation.astype(rasterio.float32),1)
        
        #std_output = os.path.join(year_folder, f"{date}_pixel_std.tif")
        #with rasterio.open(std_output, "w", **meta) as dest:
         #  dest.write(pixel_std.astype(rasterio.float32))
        
    
    return deviation_mean_timeseries, normalized_deviation_timeseries, std_timeseries

#------------------ calculate difference maps to timeseries statistics -

def calculate_difference_maps(input_folder, timeseries_reference_folder, data_type):
    
        """
        Calculate difference maps between yearly raster data and statistical trend references.
        
        This function compares each raster file in the input folder against trend statistics
        (mean, median, standard deviation) from reference files. It generates and saves
        difference maps showing deviations from these trends.
        
        Parameters
        ----------
        input_folder : str
            Path to the folder containing yearly raster files (*.tif) to be compared
            against trend statistics.
        timeseries_reference_folder : str
            Path to the folder containing trend statistic files. This folder must contain
            files with 'mean', 'median', and 'std' in their filenames.
        data_type: str
            Appendix possibility for different datasets like absolute or normalized
        Returns
        -------
        str
            Path to the output folder containing all generated difference maps organized
            by trend parameter (mean, median, std).
        
        Raises
        ------
        FileNotFoundError
            If input_folder or timeseries_reference_folder does not exist.
        ValueError
            If no reference files or input files are found in 
        """
            
        # Record start time for performance tracking
        start_function = datetime.datetime.now()
        print("Calculating difference maps started at " + str(start_function))
        
        # Validate input directories
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
        if not os.path.exists(timeseries_reference_folder):
            raise FileNotFoundError(f"Timeseries reference folder does not exist: {timeseries_reference_folder}")
        
        # Define trend parameters to compare with
        try:
            mean_files = glob.glob(os.path.join(timeseries_reference_folder, "*mean*.tif"))
            median_files = glob.glob(os.path.join(timeseries_reference_folder, "*median*.tif"))
            std_files = glob.glob(os.path.join(timeseries_reference_folder, "*std*.tif"))
            
            if not (mean_files and median_files and std_files):
                raise ValueError("One or more trend files not found. Files must contain 'mean', 'median', or 'std' in their names.")
            
            mean = mean_files[0]
            median = median_files[0]
            std = std_files[0]
        
        except Exception as e:
            raise ValueError(f"Error finding timeseries statistics files: {e}")
        
        # Create list of tiffs to be processed
        files = glob.glob(os.path.join(input_folder, "*.tif"))
        if not files:
            raise ValueError(f"No .tif files found in {input_folder}")
        
        trends = [mean, median, std]
        trends_names = ["mean", "median", "std"]
        
        # Create folder for export
        output_folder_method = create_folder_in_parent_dir(input_folder, f"yearly_differences_{data_type}")
         
        # --------------- Processing Part --------------------------------------------
        
        # Define a list that stores all statistic parameters for each raster
        all_statistics = [] 
        
        for trend, trend_name in zip(trends, trends_names):
            # Create folder for this trend parameter
            trend_folder = os.path.join(output_folder_method, trend_name)
            os.makedirs(trend_folder, exist_ok=True)
            
            # Read trend raster
            trend_raster, meta = load_raster(trend, metadata=["meta"])
                        
            for file in files:
                # Extract date from filename
                date_match = re.search(r'\d{8}|\d{4}', os.path.basename(file))
                if not date_match:
                    print(f"Warning: Could not extract date from {file}, skipping")
                    continue
                    
                date = date_match.group()
                
                # Read single year raster
                single_year_raster = load_raster(file)
                  
                # Calculate difference
                difference_raster = single_year_raster - trend_raster
                
                # Calculate statistics for this year's data
                # Fix: Use NumPy functions directly without .values attribute
                year_stats = {
                    'date': date,
                    'filename': os.path.basename(file),
                    'max': float(np.nanmax(single_year_raster)),
                    'min': float(np.nanmin(single_year_raster)),
                    'mean': float(np.nanmean(single_year_raster)),
                    'std': float(np.nanstd(single_year_raster))
                }
                
                # Add to statistics collection
                all_statistics.append(year_stats)
        
                # ------------------ Save Difference Raster -----------------------
                  
                # Extract name for output
                basename, ext = os.path.splitext(os.path.basename(file))
                  
                # Save raster to trend-specific folder
                output_file_method = f"{trend_folder}/{date}_{trend_name}_trend_diff.tif"
                
                with rasterio.open(output_file_method, 'w', **meta) as dst:
                    dst.write(difference_raster, 1)  # Added the band index 1
            
                print(f"Difference Raster between {trend_name} and year {date} successfully created and saved")
        
        # Save table of yearly statistics
        stats_df = pd.DataFrame(all_statistics)
        stats_csv_path = os.path.join(output_folder_method, "yearly_statistics.csv")
        stats_df.to_csv(stats_csv_path, index=False)
        print(f"Yearly statistics saved to {stats_csv_path}")
                 
        # Calculate processing time for function
        end_function = datetime.datetime.now()
        time_diff_function = str(end_function - start_function)
        
        print(f"All difference Rasters created and saved in {time_diff_function}")
        
        return output_folder_method


#--------------compare to trend is deprecatetd and replaced by timeseries difference

def compare_year_to_trend(input_folder, trend_reference_folder):

    start_function = datetime.datetime.now()
    print("Calculating difference maps started at " + str(start_function))
    
    # define trend parameter to compare with
    try:
        mean = glob.glob(os.path.join(trend_reference_folder, "*mean*.tif"))[0]
        median = glob.glob(os.path.join(trend_reference_folder, "*median*.tif"))[0]
        std = glob.glob(os.path.join(trend_reference_folder, "*std*.tif"))[0]
    
    except Exception as e:
        print("Filename probably wrong.  Must contain parameters name (mean/median/std)") 
        print(f"Error: {e}")      
    
    #   create list of tiffs to be processed
    files = glob.glob(os.path.join(input_folder, "*.tif"))
    trends = glob.glob(mean) + glob.glob(median) + glob.glob(std)
    trends_names =["mean", "median", "std"]
    
    #   create folder for export
    output_folder_method = create_folder_in_parent_dir(input_folder, "yearly_difference_maps")
     
    # --------------- Processing Part --------------------------------------------
    
    # define a list that stores all statistic parameters for each raster
    all_statistics = [] 
    
    for trend, trend_name in zip(trends, trends_names):
              
          trend_folder = os.path.join(output_folder_method, trend_name)
        
        
          os.makedirs(trend_folder, exist_ok=True)
          # read single year raster
          trend_raster = rio.open_rasterio(trend)
                    
          for file in files:
          
              # read single year raster
              single_year_raster = load_raster(file)
              
              # for comparison of global std with yearly stf first calculate yearly std otherwise use absolute values of year
              if trend_name == "std":
                  #   define parameters to calculate
                  single_year_raster = single_year_raster.std(dim="band", skipna=True),
              
              # calculate difference
              difference_raster = single_year_raster - trend_raster
              
              # extract date from filename
              date = re.search(r'\d{8}|\d{4}', file).group()  # extract date from pathname
              
              # create output folder for year if not already exists
              output_folder_year = create_folder_in_parent_dir(input_folder, date)
              
              #------------------- Save Difference Raster -----------------------
              
              # extract name for output
              basename, ext = os.path.splitext(os.path.basename(file))
              
              # save raster to folders
              # method folder
              output_file_method =(f"{trend_folder}/{date}_{trend_name}_trend_diff.tif")
              difference_raster.rio.to_raster(output_file_method, crs= trend_raster.rio.crs, transform = trend_raster.rio.transform)
            
              print("Difference Raster between " + trend_name + " and year " + date + " succesfully created and  saved")
                  
    # calculate processing time for function
    end_function = datetime.datetime.now()
    time_diff_function = str(end_function-start_function)
    
    print("All difference Rasters created and saved in " + time_diff_function)
    
    return(output_file_method)


def normalize(input_folder, mode, output_folder_name):
    """
    Normalize snow depth raster files using different methods.
    
    Parameters:
    - input_folder: Path to the folder containing input .tif files
    - mode: Normalization method ('single_year_relative' or 'single_year_minmax')
    - output_folder_name: Optional custom name for the output folder. 
                           If None, a default name will be used based on the mode.
    
    Returns:
    - Depends on the mode:
      - 'single_year_relative': (yearly_average, output_folder)
      - 'single_year_minmax': (yearly_max, yearly_min, output_folder)
    """

    start_function = datetime.datetime.now()
    print("Calculating normalized maps started at " + str(start_function))

    
    # ------------------- PREPROCESSING PART -------------------------------------
    
    #   create list of tiffs in folder
    files = glob.glob(os.path.join(input_folder, "*.tif"))
    
    #---------- PROCESSING PART -------------------------------------------------
    
    if mode == "single_year_relative":
        
        #   create output folder
        output_folder = create_folder_in_parent_dir(input_folder, "normalized_relative")
        
        yearly_average = {}
        for file in files:
            # Use load_raster to get data and metadata including nodata value
            data, metadata, nodata_value = load_raster(file, metadata=["meta", "nodata"])
            
            # Calculate mean excluding nodata values
            valid_data = data[data != nodata_value]
            one_year_mean = np.nanmean(valid_data)
            
            #   save yearly average to list of all years averages
            date = re.search(r'\d{8}|\d{4}', file).group()  # extract date from pathname
            yearly_average[date] = str(one_year_mean)
            print(f"Average snow depth in {date} is {one_year_mean} meters (excluding NoData values)")
            
            #   normalize (keeping NoData as NoData)
            data_normalized = np.copy(data)
            valid_mask = (data != nodata_value) & ~np.isnan(data)
            data_normalized[valid_mask] = data[valid_mask] / one_year_mean
            
            # Save the result to a new raster file
            basename, ext = os.path.splitext(os.path.basename(file))
            output_path = os.path.join(output_folder, f"{basename}_relnorm.tif")
            
            # Update metadata for output
            metadata.update(dtype=data_normalized.dtype)
            
            # Write output
            with rasterio.open(output_path, 'w', **metadata) as dst:
                dst.write(data_normalized, 1)
                dst.nodata = nodata_value
            
            print(f"Normalized raster saved to {output_path}")
            
        #   return processing time
        end_function = datetime.datetime.now()
        time_diff_function = str(end_function - start_function)
        print(f"All rasters normalized in {time_diff_function}")
            
        return output_folder
        
    elif mode == "single_year_minmax":
        
        #   create output folder
        output_folder = create_folder_in_parent_dir(input_folder, "normalized_minmax")

        yearly_min = {}
        yearly_max = {}
        for file in files:
            # Use load_raster to get data and metadata including nodata value
            data, metadata, nodata_value = load_raster(file, metadata=["meta", "nodata"])
            
            # Create mask for valid data
            valid_mask = (data != nodata_value) & ~np.isnan(data)
            valid_data = data[valid_mask]
            
            # Calculate min and max excluding NoData values
            one_year_min = np.min(valid_data)
            one_year_max = np.max(valid_data)
            
            #   save yearly min/max to dictionaries
            date = re.search(r'\d{8}|\d{4}', file).group()  # extract date from pathname
            yearly_min[date] = str(one_year_min)
            yearly_max[date] = str(one_year_max)
    
            print(f"Minimum snow depth in {date} is {one_year_min} meters (excluding NoData values)")
            print(f"Maximum snow depth in {date} is {one_year_max} meters (excluding NoData values)")
    
            #   normalize (keeping NoData as NoData)
            data_normalized = np.copy(data)
            data_normalized[valid_mask] = (data[valid_mask] - one_year_min) / (one_year_max - one_year_min)
            
            # Save the result to a new raster file
            basename, ext = os.path.splitext(os.path.basename(file))
            output_path = os.path.join(output_folder, f"{basename}_minmaxnorm.tif")
            
            # Update metadata for output
            metadata.update(dtype=data_normalized.dtype)
            
            # Write output
            with rasterio.open(output_path, 'w', **metadata) as dst:
                dst.write(data_normalized, 1)
                dst.nodata = nodata_value
                
            print(f"Normalized raster saved to {output_path}")
            
        #   return processing time
        end_function = datetime.datetime.now()
        time_diff_function = str(end_function - start_function)
        print(f"All rasters normalized in {time_diff_function}")
    
        return output_folder
    
    else:
       raise ValueError("Invalid mode. Please use either 'single_year_relative' or 'single_year_minmax'")
        

def pearson_analysis(input_folder):
    
    start_function = datetime.datetime.now()
    print("Calculating temporal and Pearson correlation started at " + str(start_function))
    
    
    # Create folder for export
    output_folder_method = create_folder_in_parent_dir(input_folder, "Pearson")
    
    # --------------------- Main Processing ----------------------------------------
    
    # List of rasters to be processed
    filepaths = glob.glob(os.path.join(input_folder, "*.tif"))
    
    # List to store Pearson correlation results
    all_years = []
    
    #   identify number of rasters
    n_years = len(filepaths)
    
    # Initialize correlation matrix for pairwise correlations
    correlation_matrix = np.zeros((n_years, n_years))
    
    # Prepare list to store spatial correlation values
    temporal_correlations = []
    
    #   list to store dates
    dates=[]
    
    # Iterate through all raster files
    for filepath in filepaths:
        
        #   read raster
        raster = rio.open_rasterio(filepath)  # Open the raster
        
        #   Extract name for identification and prints
        date = re.search(r'\d{8}|\d{4}', filepath).group()
        
        # Flatten raster values
        raster_flat = raster.values.flatten()
        
        # Store flattened raster for temporal correlation later
        temporal_correlations.append(raster_flat)
        
        print( date + " flattened and added to list for timeseries calculation")
        
#   ------------------- Matric correlation -----------------------------------
    
    # Calculate pairwise correlation after all rasters are processed
    for year_one in range(n_years):
        for year_two in range(year_one + 1, n_years): 
            
            # Mask NaN values in both rasters
            raster_1_flat = temporal_correlations[year_one]
            raster_2_flat = temporal_correlations[year_two]
            valid_mask = ~np.isnan(raster_1_flat) & ~np.isnan(raster_2_flat)
            valid_raster_1_flat = raster_1_flat[valid_mask]
            valid_raster_2_flat = raster_2_flat[valid_mask]
    
            if len(valid_raster_1_flat) > 0:
                correlation = np.corrcoef(valid_raster_1_flat, valid_raster_2_flat)[0, 1]
                correlation_matrix[year_one, year_two] = correlation
                correlation_matrix[year_two, year_one] = correlation  # Symmetric matrix
    
    
    # Save Pearson results to CSV
    with open(f"{output_folder_method}/pearson.csv", "w", newline="") as file:
        csv.writer(file).writerows([[item] for item in all_years])
    
    # Visualize pairwise correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix,
                annot=True,
                cmap='coolwarm',
                xticklabels=[f"Year {i}" for i in range(1, n_years+1)],
                yticklabels=[f"Year {i}" for i in range(1, n_years+1)])
    plt.title("Pairwise Pearson Correlation of Snow Depth Distribution")
    # Save the plot as a PNG file
    plt.savefig(os.path.join(output_folder_method, "Pearson_matrix.png"), format="png")
    plt.show()
    
    
#   ------------------- Pixel correlation ---------------------------------------
    
    # Convert temporal_correlations to a 2D array
    temporal_correlations_array = np.vstack(temporal_correlations)
    
    print(" Full timeseries stacked as virtual raster")
    
    # Calculate temporal correlation for each pixel if at least two values for the timeseries are present, ignoring NaN values
    temporal_correlation = np.apply_along_axis(
        lambda x: np.corrcoef(x[~np.isnan(x)], np.arange(n_years)[~np.isnan(x)])[0, 1]
        if np.sum(~np.isnan(x)) > 1 else np.nan,
        axis=0,
        arr=temporal_correlations_array
    )
    
    # Reshape the correlation map back to the raster's spatial dimensions
    temporal_map = temporal_correlation.reshape(raster.shape[1:])
    
    output_file = os.path.join(output_folder_method, "Pearson_Map.tif")
    
    #   georeference and save numpy array 
    georeference_and_save(temporal_map, filepath, output_file)
    
    # Plot the temporal correlatiomap
    #plt.imshow(temporal_map, cmap="RdYlGn")
    #plt.colorbar(label='Temporal Correlation')
    #plt.title("Spatial Distribution of Temporal Correlation")
    #plt.show()
    
    # Calculate processing time for the function
    end_function = datetime.datetime.now()
    time_diff_function = str(end_function - start_function)
    print("Pearson and temporal correlations calculated and saved. Processing time: " + time_diff_function)
    
    
    return output_folder_method

# ---------------------- timeseries plot with std ----------------------------

def plot_snow_depth_timeseries(yearly_stats, output_folder, feature_name='snowdepth', plot_classes=False, 
                               class_labels=None, colors=None):
    """
    Plot snow depth time series with standard deviation shown as shaded bands.
    
    This function can create either:
    1. A single line showing mean snow depth with standard deviation bands
    2. Multiple lines for different terrain classes, each with its own standard deviation band
    
    The function properly handles missing years in the time series for both modes.
    
    Parameters:
    -----------
    yearly_stats : dict or pd.DataFrame
        - If plot_classes=False: Dictionary with years as keys and nested dictionaries as values
          containing 'mean', 'std', and other statistics
        - If plot_classes=True: DataFrame with classes as index and years as columns
    
    output_folder : str
        Path to the folder where the output plot will be saved.
    
    feature_name : str, optional
        Name of the feature being plotted, used in the output filename.
        Default is 'snowdepth'.
    
    plot_classes : bool, optional
        Whether to plot multiple classes (True) or a single line (False).
        Default is False.
    
    class_labels : list or dict, optional
        Labels for each class in the legend. If None, uses class indices.
        For plot_classes=True: either a list in the same order as DataFrame indices
        or a dictionary mapping class indices to labels.
    
    colors : list, optional
        Colors for each line and fill. If None, uses default colors.
        
    Returns:
    --------
    None
        Function saves the plot to the specified output folder.
        
    Notes:
    ------
    - Years are assumed to be sortable (strings or integers).
    - Missing years are handled properly by checking for gaps in the sequence.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import pandas as pd
    
    if plot_classes:
        # For class plots, we expect DataFrames
        median_per_class = yearly_stats
        
        # Get years from column names and convert to integers
        years = [int(col) for col in median_per_class.columns]
        
        # Check for missing years
        min_year = min(years)
        max_year = max(years)
        full_range = list(range(min_year, max_year + 1))
        missing_years = [y for y in full_range if y not in years]
        
        # Set default colors if not provided
        if colors is None:
            colors = ['blue', 'yellow', 'orange', 'red', 'purple']
            
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Get feature classes from the index
        feature_classes = median_per_class.index
        
        # Set up class labels
        if class_labels is None:
            # Use index values as labels
            class_labels = {cls: f"Class {cls}" for cls in feature_classes}
        elif isinstance(class_labels, list):
            # Convert list to dict
            class_labels = {cls: label for cls, label in zip(feature_classes, class_labels)}
            
        # Plot each class
        for i, feature_class in enumerate(feature_classes):
            color_idx = i % len(colors)  # Handle case with more classes than colors
            
            # Get values for this class - need to reshape to plot with missing years
            class_data = {}
            for j, year in enumerate(years):
                class_data[year] = {
                    'mean': median_per_class.loc[feature_class, str(year)]['mean'],
                    'std': median_per_class.loc[feature_class, str(year)]['std']
                }
            
            # Create lists of x, y values in proper order
            x_values = []
            means = []
            stds = []
            
            for year in sorted(class_data.keys()):
                x_values.append(year)
                means.append(class_data[year]['mean'])
                stds.append(class_data[year]['std'])
            
            # Plot disconnected lines - only connect consecutive years
            for j in range(len(x_values) - 1):
                if x_values[j + 1] == x_values[j] + 1:  # Only connect if consecutive
                    plt.plot([x_values[j], x_values[j + 1]], 
                             [means[j], means[j + 1]], 
                             color=colors[color_idx], linewidth=2)
            
            # Plot markers for all data points
            plt.plot(x_values, means, marker='o', linestyle='', 
                     color=colors[color_idx], markersize=6, 
                     label=class_labels[feature_class])
            
            # Add standard deviation bands
            plt.fill_between(x_values, 
                            [m - s for m, s in zip(means, stds)], 
                            [m + s for m, s in zip(means, stds)], 
                            color=colors[color_idx], alpha=0.2)
        
        # Add vertical lines at missing years
        for year in missing_years:
            plt.axvline(x=year, color='lightgray', linestyle='--', alpha=0.5)
        
        # Add annotation about missing years
        if missing_years:
            missing_str = ', '.join(str(y) for y in missing_years)
            plt.figtext(0.5, 0.01, f"Missing years: {missing_str}", 
                        ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        # Customize plot
        plt.xlabel('Year')
        plt.ylabel('Snow Depth (m)')
        plt.title('Median Snow Depth Over Time by Terrain Class')
        plt.xticks(full_range, rotation=45)
        plt.legend(title='Terrain Class', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        # Save the plot with adjusted layout
        if missing_years:
            plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for annotation
        else:
            plt.tight_layout()
            
        plt.savefig(os.path.join(output_folder, f"{feature_name}_by_class.png"), dpi=300)
        plt.close()
        
    else:
        # Extract years, means and std devs from the yearly_stats dictionary
        years = sorted(yearly_stats.keys())
        snow_depths = [yearly_stats[date]['mean'] for date in years]
        std_devs = [yearly_stats[date]['std'] for date in years]
        
        # Create years as integers for plotting
        year_ints = [int(y) for y in years]
        
        # Get full range to show gaps
        min_year = min(year_ints)
        max_year = max(year_ints)
        full_range = list(range(min_year, max_year + 1))
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot disconnected lines - only connect consecutive years
        for i in range(len(year_ints) - 1):
            if year_ints[i + 1] == year_ints[i] + 1:  # Only connect if consecutive
                plt.plot([year_ints[i], year_ints[i + 1]], 
                         [snow_depths[i], snow_depths[i + 1]], 
                         'b-', linewidth=2)
        
        # Plot markers for data points
        plt.plot(year_ints, snow_depths, 'o', color='blue', 
                 markersize=8, label="ROI Mean Snow Depth")
        
        # Add standard deviation as filled area
        plt.fill_between(year_ints, 
                        [y - std for y, std in zip(snow_depths, std_devs)],
                        [y + std for y, std in zip(snow_depths, std_devs)],
                        color='blue', alpha=0.2)
        
        # Identify missing years
        missing_years = [y for y in range(min_year, max_year + 1) if y not in year_ints]
        
        # Add vertical lines at missing years
        for year in missing_years:
            plt.axvline(x=year, color='lightgray', linestyle='--', alpha=0.5)
        
        # Add annotation about missing years
        if missing_years:
            missing_str = ', '.join(str(y) for y in missing_years)
            plt.figtext(0.5, 0.01, f"Missing years: {missing_str}", 
                        ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        # Customize plot
        plt.xlabel("Year")
        plt.ylabel("Snow Depth (m)")
        plt.title("Annual Snow Depth with Standard Deviation")
        plt.xticks(full_range, rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Adjust layout
        if missing_years:
            plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for annotation
        else:
            plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_folder, f"{feature_name}_annual.png"), dpi=300)
        plt.close()

def plot_raster_timeseries(input_folder, line_reduction_factor, number_of_clusters):
    
    start_function = datetime.datetime.now()
    print("Analyzing and clustering pixel timeseries started at:  " + str(start_function))
    
    # Create folder for export
    output_folder_method = create_folder_in_parent_dir(input_folder, "Global_statistics")
    
    # List of rasters to be processed
    filepaths = glob.glob(os.path.join(input_folder, "*.tif"))


    # Prepare list to store spatial correlation values
    timeseries = []
   
    for filepath in filepaths:
        
        with rasterio.open(filepath) as src:
            data = src.read()  
            
        #   Extract name for identification and prints
        date = re.search(r'\d{8}|\d{4}', filepath).group()
       
        # Store flattened raster for temporal correlation later
        timeseries.append(data)
       
        print( date + " added vor virtual stacking")

    # Convert temporal_correlations to a 2D array
    timeseries_stack = np.vstack(timeseries)
    print("Virtual Raster created")
    
    #   get shape
    num_bands, height, width = timeseries_stack.shape

    # Reshape to (num_bands, num_pixels)
    timeseries_reshaped = timeseries_stack.reshape(num_bands, -1)  # Shape: (time, pixels)
    print("Virtual Raster flattened")
    
    # drop NaN
    valid_pixels = ~np.isnan(timeseries_reshaped).any(axis=0)
    timeseries_reshaped_clean = timeseries_reshaped[:, valid_pixels]
    print("NaN values of Virtual Raster dropped") 
  
    #    --------------- Process clustered lines------------------------------
    
    # Reduce number of lines by factor 100 while preserving variance
    sampled_indices = np.random.choice(timeseries_reshaped_clean.shape[1], timeseries_reshaped_clean.shape[1] // line_reduction_factor, replace=False)
    sampled_data = timeseries_reshaped_clean[:, sampled_indices]
    print("Number of pixel timeseries produced by predefined factor: " + str(line_reduction_factor) )

    # Cluster pixels into 4 groups based on their time series similarity
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(sampled_data.T)
    print("Sample of reduced data clustered into predefined amount of clusters (" + str(number_of_clusters) + ")")
    
    # ----------------------- plot clustered lines ---------------------------
    
    # Define colors for clusters
    colors = ['blue', 'green', 'red', 'purple']
    
    # Plot time series for selected pixels, colored by cluster
    plt.figure(figsize=(10, 6))
    for pixel in range(sampled_data.shape[1]):
        plt.plot(range(num_bands), sampled_data[:, pixel], alpha=0.1, color=colors[cluster_labels[pixel]])
    
    plt.xlabel("Time Step")
    plt.ylabel("Pixel Value")
    plt.title("Time Series of Pixel Values in 3D Raster (Clustered)")
    #   save plot
    plt.savefig(os.path.join(output_folder_method, "pixel_timelines_all.png"), format="png")
    plt.show()
        
    plt.close()
    
    #   -------------------- plot individual cluster lines--------------------
    
    # Create separate plots for each cluster
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for cluster in range(4):
        ax = axes[cluster]
        cluster_pixels = sampled_data[:, cluster_labels == cluster]
        for pixel in range(cluster_pixels.shape[1]):
            ax.plot(range(num_bands), cluster_pixels[:, pixel], alpha=0.1, color=colors[cluster])
        
        ax.set_title(f"Cluster {cluster + 1}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Pixel Value")

    plt.suptitle("Time Series of Pixel Values in 3D Raster (Clustered)")
    plt.tight_layout()
    #   save plot
    plt.savefig(os.path.join(output_folder_method, "pixel_timelines_seperate.png"), format="png")
    plt.show()
        
    plt.close()
    
    #   ------------------ cluster on map -----------------------------------
    
    #   Mask all pixels that have a single nan in whole timeseries
    mask = ~np.isnan(timeseries_stack).any(axis=0)
    mask_3d = np.repeat(mask[np.newaxis,:,:], str(num_bands), axis=0)
    
    #   extract valid values
    valid_values = timeseries_stack[mask_3d]
    valid_values = valid_values.reshape(num_bands, -1)
    
    #   statistics for lost pixels
    #   get shape
    num_bands, height, width = timeseries_stack.shape
    all_pixels = height * width
    remaining_pixels = np.sum(mask == True)
    lost_pixels = all_pixels - remaining_pixels
    print("From " + str(all_pixels) + " input_pixels " + str(remaining_pixels) + " have values for the whole timeseries and are used for clustering")
    
    # create array with original shape for later reshaping
    empty_original = np.full_like(timeseries_stack[0,:,:], np.nan)
    
    #   cluster pixels with kmean
    cluster_labels_map = kmeans.fit_predict(valid_values.T)

    # Place the valid values back at the positions of the original non-NaN values
    empty_original[mask] = cluster_labels_map

    # Plot the clustered map
    plt.figure(figsize=(10, 6))
    plt.imshow(empty_original, cmap='tab10')
    plt.colorbar(label='Cluster Label')
    plt.title("Clustered Snow Depth Map")
    plt.axis("off")
    plt.show()
    
    #   define output name
    output_file = os.path.join(output_folder_method, "KMean_cluster.tif")
    
    #   georeference and save numpy array 
    georeference_and_save(filepath, empty_original, output_file)
    
#   --------------------------------------------------------------------------------------
    # Calculate processing time for the function
    end_function = datetime.datetime.now()
    time_diff_function = str(end_function - start_function)
    print("Pixel timeseries plotted and clustered in " + time_diff_function)
    
    return output_folder_method

def reshape_after_nandrop(cleaned_1d_raster, mask, original_2d_raster, nodata_value=-999):
    """
    Reshape a 1D array back to its original 2D shape after filtering NaN and no data values.
    
    Parameters
    ----------
    cleaned_1d_raster : numpy.ndarray
        1D array of valid values after processing
    mask : numpy.ndarray
        Boolean mask of valid values in the original array
    original_2d_raster : numpy.ndarray
        Original 2D array with the desired output shape
    nodata_value : float, optional
        Value to use for invalid or missing data (default: -999)
        
    Returns
    -------
    numpy.ndarray
        2D array with the same shape as original_2d_raster, with processed values
        inserted at valid positions and nodata_value elsewhere
    """
    # Create array with original shape filled with nodata_value instead of np.nan
    # This ensures consistent handling of missing data
    filled_original = np.full_like(original_2d_raster, nodata_value)
    
    # Place the valid values back at the positions of the original valid values
    filled_original[mask] = cleaned_1d_raster
    
    return filled_original

def create_valid_mask(array1, array2=None, nodata_value=-999):
    """
    Create a mask for valid values, filtering out NaN and nodata values.
    
    Parameters:
    -----------
    array1 : numpy.ndarray
        First array to check
    array2 : numpy.ndarray, optional
        Second array to check (if provided)
    nodata_value : float, optional
        Value representing no data (default: -999)
        
    Returns:
    --------
    numpy.ndarray
        Boolean mask where True indicates valid values
    """
    # Check for NaN and nodata in first array
    mask1 = ~np.isnan(array1) & (array1 != nodata_value)
    
    # If second array provided, also check it
    if array2 is not None:
        mask2 = ~np.isnan(array2) & (array2 != nodata_value)
        return mask1 & mask2
    
    return mask1
    
    # ----------------------------------------testing -------------------------
    
def violin(input_folder, combined_plot=True, subsample_factor=None, max_depth=10, remove_outliers=True):
    """
    Create violin plots for raster data in the input folder.
    
    Parameters:
    -----------
    input_folder : str
        Path to folder containing tiff files
    combined_plot : bool, default=True
        If True, create a combined plot with all dates
    subsample_factor : int, optional
        If provided, use only every Nth pixel to reduce memory usage
    max_depth : float, default=10
        Maximum snow depth to display in meters
    remove_outliers : bool, default=True
        If True, remove outliers beyond 1.5*IQR
    
    Returns:
    --------
    str
        Path to the output folder containing the plots
    """
    import os
    import glob
    import re
    import datetime
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    start_function = datetime.datetime.now()
    print(f"Calculating violin diagrams started at {start_function}")
     
    # Create list of tiffs to be processed
    files = glob.glob(os.path.join(input_folder, "*.tif"))
    
    if not files:
        print(f"No .tif files found in {input_folder}")
        return None
    
    # Create folder for export using powdersearch function
    output_folder = create_folder_in_parent_dir(input_folder, "violin")
    
    # Store results for combined plot
    all_results = []
    
    # Process each file sequentially
    for file in files:
        try:
            # Extract date from filename
            date_match = re.search(r'\d{8}|\d{4}', os.path.basename(file))
            if date_match:
                date_str = date_match.group()
                # Extract just the year if it's a longer date format
                if len(date_str) > 4:
                    date = date_str[:4]
                else:
                    date = date_str
            else:
                date = os.path.splitext(os.path.basename(file))[0]
                        
            # Load raster using powdersearch function
            raster_data, meta = load_raster(file, metadata="meta")
            
            # Subsample data if requested
            if subsample_factor and subsample_factor > 1:
                raster_data = raster_data[::subsample_factor, ::subsample_factor]
            
            # Remove NaN and NoData values
            valid_mask = ~np.isnan(raster_data)
            if meta.get('nodata'):
                valid_mask &= (raster_data != meta.get('nodata'))
            
            # Extract valid data
            valid_data = raster_data[valid_mask].flatten()
            
            if len(valid_data) == 0:
                print(f"Warning: No valid data in {file}, skipping")
                continue
            
            # Remove outliers if requested
            if remove_outliers:
                q1 = np.percentile(valid_data, 25)
                q3 = np.percentile(valid_data, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                valid_data = valid_data[(valid_data >= lower_bound) & (valid_data <= upper_bound)]
                print(f"Removed {len(raster_data[valid_mask].flatten()) - len(valid_data)} outliers")
            
            # Cap max depth
            valid_data = valid_data[valid_data <= max_depth]
            
            # Calculate statistics
            median = float(np.median(valid_data))
            std = float(np.std(valid_data))
            min_val = float(np.min(valid_data))
            max_val = float(np.min([np.max(valid_data), max_depth]))
            pixel_count = len(valid_data)
            
            # Store results for combined plot
            all_results.append({
                'date': date,
                'median': median,
                'std': std,
                'min': min_val,
                'max': max_val,
                'pixel_count': pixel_count,
                'data': valid_data if combined_plot else None  # Store data only if needed for combined plot
            })
            
            # Create dataframe for plotting
            df = pd.DataFrame({'value': valid_data})
            
            # Create individual violin plot
            plt.figure(figsize=(10, 6))
            ax = sns.violinplot(y='value', data=df, inner='quartile', color='black')
            
            # Add horizontal line for median
            plt.axhline(y=median, color='red', linestyle='--', linewidth=1.5)
            
            # Add statistics text box
            stats_text = (
                f"Median: {median:.2f} m\n"
                f"Std: {std:.2f} m\n"
                f"Min: {min_val:.2f} m\n"
                f"Max: {max_val:.2f} m\n"
                f"Pixels: {pixel_count:,}"
            )
            plt.annotate(
                stats_text,
                xy=(0.02, 0.96),
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                verticalalignment='top',
                fontsize=10
            )
            
            # Set labels and title
            plt.title(f"Snow Depth Distribution - {date}")
            plt.ylabel("Snow Depth (m)")
            plt.ylim(0, max_depth)  # Set max to specified max_depth
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            
            # Save plot to both folders
            basename = os.path.splitext(os.path.basename(file))[0]
            plt.savefig(os.path.join(output_folder, f"{basename}_violin.png"), dpi=300)
            plt.close()
            
            print(f"Violin plot for {date} created successfully")
            
        except Exception as e:
            print(f"Error processing {os.path.basename(file)}: {e}")
    
    # Create combined plot if requested and we have results
    if combined_plot and all_results:
        try:
            create_combined_violin_plot(all_results, output_folder, max_depth)
        except Exception as e:
            print(f"Error creating combined plot: {e}")
    
    end_function = datetime.datetime.now()
    time_diff = str(end_function - start_function)
    print(f"All violin diagrams created and saved in {time_diff}")
    
    return output_folder


def create_combined_violin_plot(results, output_folder, max_depth=10):
    """
    Create a combined violin plot from all individual results.
    
    Parameters:
    -----------
    results : list
        List of dictionaries containing plot data and statistics
    output_folder : str
        Folder to save the combined plot
    max_depth : float, default=10
        Maximum snow depth to display in meters
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    # Sort results by date
    results = sorted(results, key=lambda x: x['date'])
    
    # Create dataframe for combined plot
    combined_data = []
    
    # Add data for each date
    for result in results:
        # Sample data to a reasonable size if needed
        data = result['data']
        if len(data) > 10000:
            # Randomly sample 10,000 points to keep size manageable
            indices = np.random.choice(len(data), 10000, replace=False)
            data = data[indices]
        
        # Add to combined dataframe
        for value in data:
            combined_data.append({
                'Date': result['date'],
                'Snow Depth (m)': value
            })
    
    combined_df = pd.DataFrame(combined_data)
    
    # Create combined violin plot
    plt.figure(figsize=(12, 8))
    
    # Create the violin plot with black color for all violins
    ax = sns.violinplot(
        x='Date', 
        y='Snow Depth (m)', 
        data=combined_df,
        color='black',   # All violins black
        inner='quartile',
        scale='width'
    )
    
    # Extract median values and positions for the line plot
    x_positions = np.arange(len(results))
    median_values = [r['median'] for r in results]
    
    # Add a line connecting all medians
    plt.plot(x_positions, median_values, 'ro-', linewidth=2, markersize=6, label='Median Trend')
    
    # Add median annotations
    for i, result in enumerate(results):
        ax.text(
            i, result['median'] + 0.2, 
            f"{result['median']:.2f}m",
            ha='center', va='bottom',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7)
        )
    
    # Set title and labels
    plt.title("Snow Depth Distribution by Year", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Set y limits to max_depth
    plt.ylim(0, max_depth)
    
    plt.tight_layout()
    
    # Save combined plot
    plt.savefig(f"{output_folder}/combined_violin.png", dpi=300)
    
    # Save statistics as CSV
    stats_df = pd.DataFrame([{
        'year': r['date'],
        'median': r['median'],
        'std': r['std'],
        'min': r['min'],
        'max': r['max'],
        'pixel_count': r['pixel_count']
    } for r in results])
    
    stats_df.to_csv(f"{output_folder}/snow_depth_statistics.csv", index=False)
    
    print(f"Combined violin plot and statistics saved to {output_folder}")
    plt.close()