#!/usr/bin/env python3
"""
Test script to verify that the radial source term function is working correctly.
"""

import numpy as np
import pandas as pd
import os
from scipy.interpolate import RegularGridInterpolator

def test_radial_source_term():
    """Test the radial source term function with known values."""
    
    print("=== Testing Radial Source Term Function ===\n")
    
    # First, check if gradient data exists
    gradient_file = None
    potential_locations = [
        "outputs/geballe_no_diamond_read_flux/radial_gradient_raw.csv",
        "sim_outputs/geballe_no_diamond_read_flux/radial_gradient_raw.csv",
    ]
    
    for location in potential_locations:
        if os.path.exists(location):
            gradient_file = location
            break
    
    if gradient_file is None:
        print("✗ No gradient file found. Cannot test radial source term.")
        return
    
    # Load gradient data
    df = pd.read_csv(gradient_file, index_col=0)
    times = df.index.values.astype(float)
    z_positions = df.columns.values.astype(float)
    values = df.values
    
    print(f"Loaded gradient data: {values.shape[0]} timesteps, {values.shape[1]} z-positions")
    print(f"Time range: [{times.min():.6e}, {times.max():.6e}]")
    print(f"Z range: [{z_positions.min():.6e}, {z_positions.max():.6e}]")
    
    # Create interpolation function
    grad_interp = RegularGridInterpolator((times, z_positions), values, method='linear')
    
    # Test parameters (typical values for the simulation)
    test_kappa = 100.0  # W/m/K (typical thermal conductivity)
    
    # Define a simple radial source term function (similar to the one in run_1d)
    def radial_source_term(z_coord, t):
        """Compute radial source term: 2 * gradient * kappa"""
        try:
            # Interpolate gradient at this z-coordinate and time
            grad_val = grad_interp([t, z_coord])[0]
            
            # Apply correction: 2 * gradient * kappa
            source_val = 2.0 * grad_val * test_kappa
            return source_val
        except Exception as e:
            print(f"Error in radial_source_term: {e}")
            return 0.0
    
    # Test the function
    print(f"\n--- Testing Radial Source Term Function ---")
    
    # Test at different times and positions
    test_cases = [
        (times[0], z_positions[0]),
        (times[len(times)//2], z_positions[len(z_positions)//2]),
        (times[-1], z_positions[-1]),
    ]
    
    for t, z in test_cases:
        source_val = radial_source_term(z, t)
        print(f"t={t:.6e}, z={z:.6e}: source_term = {source_val:.6e}")
    
    # Test with values that might be outside the interpolation range
    print(f"\n--- Testing Edge Cases ---")
    
    # Test at time 0
    source_val_0 = radial_source_term(z_positions[0], 0.0)
    print(f"t=0.0, z={z_positions[0]:.6e}: source_term = {source_val_0:.6e}")
    
    # Test at a time in the middle
    mid_time = times[len(times)//2]
    source_val_mid = radial_source_term(z_positions[0], mid_time)
    print(f"t={mid_time:.6e}, z={z_positions[0]:.6e}: source_term = {source_val_mid:.6e}")
    
    # Check if we get any non-zero values
    all_test_values = []
    for t in np.linspace(times[0], times[-1], 10):
        for z in np.linspace(z_positions[0], z_positions[-1], 10):
            val = radial_source_term(z, t)
            all_test_values.append(val)
    
    max_abs_val = max(abs(v) for v in all_test_values)
    print(f"\n--- Summary ---")
    print(f"Maximum absolute source term value: {max_abs_val:.6e}")
    
    if max_abs_val > 1e-10:
        print("✓ Radial source term function is producing non-zero values")
    else:
        print("⚠ Radial source term function is producing only very small or zero values")
        print("This could explain why the radial correction isn't affecting the simulation.")
    
    # Check the gradient values directly
    max_grad = np.max(np.abs(values))
    print(f"Maximum absolute gradient value: {max_grad:.6e}")
    print(f"Expected maximum source term: {2.0 * max_grad * test_kappa:.6e}")

if __name__ == "__main__":
    test_radial_source_term() 