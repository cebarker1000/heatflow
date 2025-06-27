#!/usr/bin/env python3
"""
Example script demonstrating how to use the RadialGradientPlotter class
for analyzing radial gradient data from parameter sweep results.
"""

import matplotlib.pyplot as plt
from plot_radial_gradient import RadialGradientPlotter
import numpy as np


def main():
    """Example analysis of radial gradient data."""
    
    # Path to the radial gradient data
    data_path = "outputs/sweep_test/fwhm_1.30e-5_k_3.68_width_1.90e-6/radial_gradient.csv"
    
    # Create the plotter
    plotter = RadialGradientPlotter(data_path)
    
    # Get data summary
    summary = plotter.get_data_summary()
    print("Data Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Example 1: Plot evolution at specific time points
    print("\nExample 1: Plotting evolution at specific time points...")
    
    # Select time indices for plotting (e.g., every 8th time point)
    time_indices = list(range(0, len(plotter.time_values), 8))
    
    fig1, ax1 = plotter.plot_gradient_evolution(
        time_indices=time_indices,
        figsize=(14, 8),
        save_path="radial_gradient_evolution_example.png",
        show_plot=False  # Don't show in interactive mode
    )
    
    # Example 2: Create heatmap
    print("\nExample 2: Creating heatmap...")
    fig2, ax2 = plotter.plot_heatmap(
        figsize=(12, 8),
        save_path="radial_gradient_heatmap_example.png",
        show_plot=False
    )
    
    # Example 3: Custom analysis - find peak gradient at each time
    print("\nExample 3: Analyzing peak gradients...")
    
    gradient_data = plotter.data.iloc[:, 1:].values
    peak_gradients = np.max(gradient_data, axis=1)
    peak_positions = [plotter.radial_positions[np.argmax(gradient_data[i, :])] 
                     for i in range(len(plotter.time_values))]
    
    # Plot peak gradient evolution
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Peak gradient magnitude
    ax3a.plot(plotter.time_values, peak_gradients, 'b-', linewidth=2)
    ax3a.set_xlabel('Time (s)')
    ax3a.set_ylabel('Peak Radial Gradient (K/m)')
    ax3a.set_title('Peak Radial Gradient Evolution')
    ax3a.grid(True, alpha=0.3)
    ax3a.set_yscale('log')
    
    # Peak gradient position
    ax3b.plot(plotter.time_values, peak_positions, 'r-', linewidth=2)
    ax3b.set_xlabel('Time (s)')
    ax3b.set_ylabel('Radial Position (m)')
    ax3b.set_title('Peak Gradient Position Evolution')
    ax3b.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("peak_gradient_analysis.png", dpi=300, bbox_inches='tight')
    print("Peak gradient analysis saved to: peak_gradient_analysis.png")
    
    # Example 4: Analyze gradient profiles at specific times
    print("\nExample 4: Analyzing specific time points...")
    
    # Find interesting time points (e.g., early, middle, late)
    early_idx = 5   # Early time
    middle_idx = len(plotter.time_values) // 2  # Middle time
    late_idx = len(plotter.time_values) - 5  # Late time
    
    interesting_times = [early_idx, middle_idx, late_idx]
    
    fig4, ax4 = plotter.plot_gradient_evolution(
        time_indices=interesting_times,
        figsize=(12, 8),
        save_path="specific_time_analysis.png",
        show_plot=False
    )
    
    # Add some analysis annotations
    for i, time_idx in enumerate(interesting_times):
        time = plotter.time_values[time_idx]
        gradients = gradient_data[time_idx, :]
        max_grad = np.max(gradients)
        max_pos = plotter.radial_positions[np.argmax(gradients)]
        
        print(f"  Time {i+1} (t={time:.2e}s):")
        print(f"    Max gradient: {max_grad:.2e} K/m at r={max_pos:.2e} m")
    
    print("\nAll plots saved successfully!")
    print("Files created:")
    print("  - radial_gradient_evolution_example.png")
    print("  - radial_gradient_heatmap_example.png") 
    print("  - peak_gradient_analysis.png")
    print("  - specific_time_analysis.png")


if __name__ == '__main__':
    main() 