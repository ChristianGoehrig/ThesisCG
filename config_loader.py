# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 14:27:43 2025

@author: goehrigc
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration loader for Snow Distribution Terrain Analysis

This module handles loading and validation of configuration settings from
a YAML file to support the terrain analysis scripts.

Author: Christian Goehrig
"""

import os
import sys
import yaml
from pathlib import Path
import string
import argparse
from typing import Dict, Any, Optional, List, Union


class ConfigLoader:
    """Loads and validates configuration from a YAML file for terrain analysis."""
    
    def __init__(self, config_path='config.yaml'):
        """
        Initialize the configuration loader.
        
        Parameters:
        -----------
        config_path : str or Path
            Path to the configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = None
        
    def load(self):
        """
        Load configuration from YAML file.
        
        Returns:
        --------
        dict
            Dictionary containing all configuration settings
        
        Raises:
        -------
        FileNotFoundError
            If the configuration file is not found
        ValueError
            If the configuration file is invalid
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Process dynamic terrain feature names
            self._process_template_strings()
            
            # Generate case name
            self._generate_case_name()
            
            # Validate the configuration
            self._validate()
            
            return self.config
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in config file: {e}")
    
    def _process_template_strings(self):
        """Replace template variables in terrain feature names."""
        if self.config and 'terrain_features' in self.config and 'analysis' in self.config:
            template_vars = self.config['analysis']
            processed_features = []
            
            for feature in self.config['terrain_features']:
                # Check if feature name contains a template variable
                if '{' in feature and '}' in feature:
                    template = string.Template(feature)
                    processed_feature = template.safe_substitute(template_vars)
                    processed_features.append(processed_feature)
                else:
                    processed_features.append(feature)
            
            self.config['terrain_features'] = processed_features
    
    def _generate_case_name(self):
        """Generate a case name based on analysis settings."""
        if 'analysis' in self.config:
            analysis = self.config['analysis']
            if all(key in analysis for key in ['year_train', 'name_train_area', 'year_pred', 'name_test_area', 'pixel_size', 'neighbourhood_meter', 'data_type']):
                case_name = (
                    f"{analysis['year_train']}_{analysis['name_train_area']}_train_vs_"
                    f"{analysis['year_pred']}_{analysis['name_test_area']}_pred_"
                    f"res{analysis['pixel_size']}_tpi{analysis['neighbourhood_meter']}_"
                    f"{analysis['data_type']}"
                )
                self.config['case_name'] = case_name
    
    def _validate(self):
        """
        Validate the loaded configuration.
        
        Raises:
        -------
        ValueError
            If the configuration is invalid
        """
        if not self.config:
            raise ValueError("Configuration not loaded")
        
        # Check required sections
        required_sections = ['paths', 'analysis', 'areas', 'terrain_features']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate path existence if specified in the config
        if self.config.get('validate_paths', True):
            self._validate_paths()
        
        # Validate analysis settings
        self._validate_analysis_settings()
    
    def _validate_paths(self):
        """Validate that configured paths exist."""
        critical_paths = ['library_dir', 'avy_outlines']
        optional_paths = ['snow_depth_train', 'snow_depth_test', 'dem_path', 'dem_path_pred']
        
        for key in critical_paths:
            path = self.config['paths'].get(key)
            if path:
                path_obj = Path(path)
                if not path_obj.exists():
                    raise ValueError(f"Critical path does not exist: {path} (key: {key})")
        
        for key in optional_paths:
            path = self.config['paths'].get(key)
            if path:
                path_obj = Path(path)
                if not path_obj.exists():
                    print(f"Warning: Optional path does not exist: {path} (key: {key})")
    
    def _validate_analysis_settings(self):
        """Validate analysis settings."""
        analysis = self.config['analysis']
        
        # Check if neighbourhood is a multiple of pixel size
        if 'neighbourhood_meter' in analysis and 'pixel_size' in analysis:
            if analysis['neighbourhood_meter'] % analysis['pixel_size'] != 0:
                raise ValueError(
                    f"Neighbourhood ({analysis['neighbourhood_meter']}) "
                    f"must be a multiple of pixel size ({analysis['pixel_size']})"
                )
        
        # Validate training and prediction modes
        valid_modes = ['outlines', 'extent']
        if analysis.get('training') not in valid_modes:
            raise ValueError(f"Invalid training mode: {analysis.get('training')}")
        if analysis.get('prediction') not in valid_modes:
            raise ValueError(f"Invalid prediction mode: {analysis.get('prediction')}")
        
        # Validate extent configuration if using 'extent' mode
        if analysis.get('training') == 'extent' and not self.config['areas'].get('area_train_extent'):
            raise ValueError("Training mode is 'extent' but no 'area_train_extent' is defined")
        if analysis.get('prediction') == 'extent' and not self.config['areas'].get('area_predict_extent'):
            raise ValueError("Prediction mode is 'extent' but no 'area_predict_extent' is defined")


def load_config(config_path='config.yaml'):
    """
    Load and validate configuration from a YAML file.
    
    Parameters:
    -----------
    config_path : str or Path
        Path to the configuration YAML file
    
    Returns:
    --------
    dict
        Dictionary containing all configuration settings
    """
    loader = ConfigLoader(config_path)
    return loader.load()


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
    --------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Snow Distribution Terrain Analysis with configuration file'
    )
    
    parser.add_argument(
        '-c', '--config',
        default='config.yaml',
        help='Path to the configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '-v', '--validate-only',
        action='store_true',
        help='Validate configuration file and exit'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip validation of paths in configuration'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Load and validate configuration
    try:
        config = load_config(args.config)
        
        if args.validate_only:
            print(f"Configuration file '{args.config}' is valid")
            sys.exit(0)
            
        print(f"Loaded configuration from: {args.config}")
        print(f"  Case name: {config.get('case_name', 'Not computed')}")
        print(f"  Training: {config['analysis']['year_train']} ({config['analysis']['name_train_area']})")
        print(f"  Prediction: {config['analysis']['year_pred']} ({config['analysis']['name_test_area']})")
        print(f"  Terrain features: {', '.join(config['terrain_features'])}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)