#!/usr/bin/env python3
"""
Diagnostic script to analyze radial gradient data and verify source term calculation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_gradient_data(gradient_file_path):
    """Analyze the radial gradient data to understand its scale and characteristics."""
    
    if not os.path.exists(gradient_file_path):
        print(f"Error: Gradient file not found at {gradient_file_path}")
        return None
    
    print(f"Loading gradient data from: {gradient_file_path}")
    
    # Load the gradient data
    df = pd.read_csv(gradient_file_path, index_col=0)
    
    # Convert column names to float and ensure ascending order
    col_float = df.columns.astype(float)
    sort_idx = np.argsort(col_float)
    z_positions = col_float[sort_idx]
    data = df.to_numpy()[:, sort_idx]  # reorder columns accordingly
    
    times = df.index.values.astype(float)
    
    print(f"Gradient data shape: {data.shape}")
    print(f"Time range: [{times.min():.2e}, {times.max():.2e}] seconds")
    print(f"Z range: [{z_positions.min():.2e}, {z_positions.max():.2e}] meters")
    
    # Analyze the data
    print(f"\n--- Gradient Data Analysis ---")
    print(f"Min gradient value: {np.nanmin(data):.2e} K/m")
    print(f"Max gradient value: {np.nanmax(data):.2e} K/m")
    print(f"Mean gradient value: {np.nanmean(data):.2e} K/m")
    print(f"Std gradient value: {np.nanstd(data):.2e} K/m")
    
    # Check for non-zero values
    non_zero_mask = np.abs(data) > 1e-12
    non_zero_count = np.sum(non_zero_mask)
    total_count = data.size
    print(f"Non-zero gradient values: {non_zero_count}/{total_count} ({100*non_zero_count/total_count:.1f}%)")
    
    # Find maximum gradient at each timestep
    max_gradients = np.nanmax(np.abs(data), axis=1)
    print(f"Max gradient range: [{max_gradients.min():.2e}, {max_gradients.max():.2e}] K/m")
    
    return {
        'times': times,
        'z_positions': z_positions,
        'data': data,
        'max_gradients': max_gradients
    }

def test_source_term_calculation(gradient_data, kappa_values, delta_r=1e-6):
    """Test the source term calculation with sample data."""
    
    if gradient_data is None:
        return
    
    print(f"\n--- Source Term Calculation Test ---")
    print(f"Using characteristic length scale Δr = {delta_r:.2e} m")
    
    # Test with a few sample points
    test_indices = [0, len(gradient_data['times'])//4, len(gradient_data['times'])//2, -1]
    
    for i, time_idx in enumerate(test_indices):
        t = gradient_data['times'][time_idx]
        max_grad = gradient_data['max_gradients'][time_idx]
        
        print(f"\nTest {i+1}: t = {t:.2e} s")
        print(f"  Max gradient at this time: {max_grad:.2e} K/m")
        
        # Test with different kappa values
        for kappa in kappa_values:
            source_term = 3.0 * kappa * max_grad / delta_r
            print(f"  Source term (κ={kappa}): {source_term:.2e} W/m³")
    
    # Calculate typical source term magnitudes
    print(f"\n--- Typical Source Term Magnitudes ---")
    typical_gradient = np.nanmean(np.abs(gradient_data['data']))
    print(f"Typical gradient magnitude: {typical_gradient:.2e} K/m")
    
    for kappa in kappa_values:
        typical_source = 3.0 * kappa * typical_gradient / delta_r
        print(f"Typical source term (κ={kappa}): {typical_source:.2e} W/m³")

def plot_gradient_evolution(gradient_data):
    """Plot the evolution of maximum gradient over time."""
    
    if gradient_data is None:
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(gradient_data['times'] * 1e6, gradient_data['max_gradients'], 'b-', linewidth=2)
    plt.xlabel('Time (μs)')
    plt.ylabel('Max |∂T/∂r| (K/m)')
    plt.title('Evolution of Maximum Radial Temperature Gradient')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the diagnostic analysis."""
    
    # Look for gradient data file
    potential_paths = [
        'outputs/geballe_no_diamond_read_flux/radial_gradient.csv',
        'outputs/geballe_no_diamond_read_flux/radial_gradient_raw.csv',
        'sim_outputs/geballe_no_diamond_read_flux/radial_gradient.csv',
        'sim_outputs/geballe_no_diamond_read_flux/radial_gradient_raw.csv',
        'meshes/geballe_no_diamond_read_flux/radial_gradient.csv',
        'meshes/geballe_no_diamond_read_flux/radial_gradient_raw.csv'
    ]
    
    gradient_file = None
    for path in potential_paths:
        if os.path.exists(path):
            gradient_file = path
            break
    
    if gradient_file is None:
        print("Error: Could not find radial gradient file.")
        print("Please ensure the 2D simulation has been run and the gradient data saved.")
        return
    
    # Determine if we're using smoothed or raw data
    using_smoothed_data = 'radial_gradient.csv' in gradient_file
    if using_smoothed_data:
        print("Found smoothed gradient data (recommended)")
        delta_r = 0.25e-6  # Smoothing window width
    else:
        print("Found raw gradient data (fallback)")
        delta_r = 0.07e-6  # Typical mesh size
    
    # Analyze the gradient data
    gradient_data = analyze_gradient_data(gradient_file)
    
    # Test source term calculation with typical kappa values
    kappa_values = [3.8, 10.0, 352.0]  # Sample, insulator, coupler
    test_source_term_calculation(gradient_data, kappa_values, delta_r)
    
    # Plot the evolution
    plot_gradient_evolution(gradient_data)
    
    print(f"\n--- Summary ---")
    print("The source term calculation uses the physically motivated formula:")
    print("Source = 3 * κ * (∂T/∂r) / Δr")
    print("where:")
    print("  κ = thermal conductivity")
    print("  ∂T/∂r = radial temperature gradient")
    if using_smoothed_data:
        print("  Δr = smoothing window width (0.25 μm) - recommended")
        print("\nUsing smoothed gradient data is recommended because:")
        print("  1. It represents an average over a radial band (0-0.25 μm)")
        print("  2. The smoothing window width provides a natural characteristic length scale")
        print("  3. It reduces noise and provides more stable source terms")
    else:
        print("  Δr = typical mesh size (0.07 μm) - fallback")
        print("\nUsing raw gradient data as fallback:")
        print("  1. Raw gradients at r=0 may be noisy")
        print("  2. Using typical mesh size as characteristic length scale")
        print("  3. Consider running 2D simulation to get smoothed data")
    
    print("\nThis accounts for the missing radial heat flow in the 1D approximation.")

if __name__ == "__main__":
    main() 