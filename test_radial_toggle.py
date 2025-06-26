#!/usr/bin/env python3
"""
Test script to demonstrate the radial heating correction toggle in run_1d function.
"""

import yaml
import os
from run_no_diamond_1d import run_1d

def test_radial_correction_toggle():
    """Test the radial correction toggle functionality."""
    
    # Load a sample configuration
    config_file = "cfgs/geballe_no_diamond_read_flux.yaml"
    
    if not os.path.exists(config_file):
        print(f"Configuration file {config_file} not found. Please ensure it exists.")
        return
    
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Define mesh folder (adjust path as needed)
    mesh_folder_2d = "mesh_outputs/geballe_no_diamond_read_flux"
    
    if not os.path.exists(mesh_folder_2d):
        print(f"Mesh folder {mesh_folder_2d} not found. Please ensure the 2D mesh exists.")
        return
    
    print("=== Testing Radial Heating Correction Toggle ===\n")
    
    # Test 1: With radial correction enabled
    print("Test 1: Running with radial correction ENABLED")
    print("-" * 50)
    try:
        run_1d(
            cfg=cfg,
            mesh_folder_2d=mesh_folder_2d,
            output_folder="test_outputs/with_radial_correction",
            use_radial_correction=True,
            suppress_print=False
        )
        print("✓ Test 1 completed successfully\n")
    except Exception as e:
        print(f"✗ Test 1 failed: {e}\n")
    
    # Test 2: With radial correction disabled
    print("Test 2: Running with radial correction DISABLED")
    print("-" * 50)
    try:
        run_1d(
            cfg=cfg,
            mesh_folder_2d=mesh_folder_2d,
            output_folder="test_outputs/without_radial_correction",
            use_radial_correction=False,
            suppress_print=False
        )
        print("✓ Test 2 completed successfully\n")
    except Exception as e:
        print(f"✗ Test 2 failed: {e}\n")
    
    print("=== Test Summary ===")
    print("The radial heating correction can now be toggled using the")
    print("use_radial_correction parameter in the run_1d function.")
    print("\nUsage examples:")
    print("  # Enable radial correction (default)")
    print("  run_1d(cfg, mesh_folder, use_radial_correction=True)")
    print("\n  # Disable radial correction")
    print("  run_1d(cfg, mesh_folder, use_radial_correction=False)")

if __name__ == "__main__":
    test_radial_correction_toggle() 