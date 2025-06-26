#!/usr/bin/env python3
"""
Test script to verify that the radial heating correction is working properly.
This script runs the 1D simulation with and without radial correction and compares the results.
"""

import yaml
import os
import numpy as np
import pandas as pd
from run_no_diamond_1d import run_1d

def test_radial_correction_effect():
    """Test the effect of radial correction on simulation results."""
    
    # Load configuration
    config_file = "cfgs/geballe_no_diamond_read_flux.yaml"
    
    if not os.path.exists(config_file):
        print(f"Configuration file {config_file} not found.")
        return
    
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Define mesh folder
    mesh_folder_2d = "mesh_outputs/geballe_no_diamond_read_flux"
    
    if not os.path.exists(mesh_folder_2d):
        print(f"Mesh folder {mesh_folder_2d} not found.")
        return
    
    print("=== Testing Radial Heating Correction Effect ===\n")
    
    # Define watcher points to monitor temperature changes
    watcher_points = {
        'sample_center': (0.0, 0.0),  # Center of sample
        'sample_edge': (0.5, 0.0),    # Edge of sample
        'insulator': (1.0, 0.0),      # In insulator
    }
    
    # Test 1: Without radial correction
    print("Running simulation WITHOUT radial correction...")
    try:
        run_1d(
            cfg=cfg,
            mesh_folder_2d=mesh_folder_2d,
            output_folder="test_outputs/without_radial",
            watcher_points=watcher_points,
            use_radial_correction=False,
            suppress_print=False
        )
        print("✓ Simulation without radial correction completed\n")
    except Exception as e:
        print(f"✗ Simulation without radial correction failed: {e}\n")
        return
    
    # Test 2: With radial correction
    print("Running simulation WITH radial correction...")
    try:
        run_1d(
            cfg=cfg,
            mesh_folder_2d=mesh_folder_2d,
            output_folder="test_outputs/with_radial",
            watcher_points=watcher_points,
            use_radial_correction=True,
            suppress_print=False
        )
        print("✓ Simulation with radial correction completed\n")
    except Exception as e:
        print(f"✗ Simulation with radial correction failed: {e}\n")
        return
    
    # Compare results
    print("=== Comparing Results ===\n")
    
    # Load watcher data
    without_radial_file = "test_outputs/without_radial/watcher_points.csv"
    with_radial_file = "test_outputs/with_radial/watcher_points.csv"
    
    if not os.path.exists(without_radial_file) or not os.path.exists(with_radial_file):
        print("Watcher data files not found. Cannot compare results.")
        return
    
    df_without = pd.read_csv(without_radial_file)
    df_with = pd.read_csv(with_radial_file)
    
    print("Temperature differences (With - Without radial correction):")
    print("-" * 60)
    
    for point_name in watcher_points.keys():
        if point_name in df_without.columns and point_name in df_with.columns:
            # Calculate differences
            temp_diff = df_with[point_name] - df_without[point_name]
            max_diff = temp_diff.max()
            min_diff = temp_diff.min()
            mean_diff = temp_diff.mean()
            std_diff = temp_diff.std()
            
            print(f"{point_name}:")
            print(f"  Max difference: {max_diff:.6f} K")
            print(f"  Min difference: {min_diff:.6f} K")
            print(f"  Mean difference: {mean_diff:.6f} K")
            print(f"  Std difference: {std_diff:.6f} K")
            print()
    
    # Check if there are any significant differences
    all_diffs = []
    for point_name in watcher_points.keys():
        if point_name in df_without.columns and point_name in df_with.columns:
            temp_diff = df_with[point_name] - df_without[point_name]
            all_diffs.extend(temp_diff.values)
    
    max_abs_diff = max(abs(d) for d in all_diffs)
    
    print("=== Summary ===")
    if max_abs_diff > 1e-6:
        print(f"✓ Radial correction IS having an effect!")
        print(f"  Maximum absolute temperature difference: {max_abs_diff:.6f} K")
    else:
        print("⚠ Radial correction is NOT having a significant effect.")
        print("  Maximum absolute temperature difference: {max_abs_diff:.6e} K")
        print("  This could indicate:")
        print("  1. The gradient data is very small or zero")
        print("  2. The source term is not being applied correctly")
        print("  3. The correction is too small to be noticeable")
    
    print("\n=== Recommendations ===")
    print("1. Check the debug output during simulation for source term values")
    print("2. Verify that the gradient data file exists and contains non-zero values")
    print("3. Check the timing and spatial ranges of the gradient data")
    print("4. Consider increasing the gradient magnitude for testing")

if __name__ == "__main__":
    test_radial_correction_effect() 