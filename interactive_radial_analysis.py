#!/usr/bin/env python3
"""
Interactive script for exploring radial gradient data.

This script provides an interactive interface to explore radial gradient data
from parameter sweep results with various plotting options.
"""

import matplotlib.pyplot as plt
from plot_radial_gradient import RadialGradientPlotter
import numpy as np


def interactive_analysis():
    """Interactive analysis of radial gradient data."""
    
    # Path to the radial gradient data
    data_path = "outputs/sweep_test/fwhm_1.30e-5_k_3.68_width_1.90e-6/radial_gradient.csv"
    
    print("Radial Gradient Data Explorer")
    print("=" * 40)
    
    # Create the plotter
    plotter = RadialGradientPlotter(data_path)
    
    # Get data summary
    summary = plotter.get_data_summary()
    print("\nData Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    while True:
        print("\n" + "=" * 40)
        print("Available options:")
        print("1. Plot gradient evolution at all time points")
        print("2. Plot gradient evolution at specific time points")
        print("3. Create heatmap")
        print("4. Analyze peak gradients over time")
        print("5. Custom time range analysis")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            print("\nPlotting gradient evolution at all time points...")
            plotter.plot_gradient_evolution()
            
        elif choice == '2':
            print("\nPlotting gradient evolution at specific time points...")
            print(f"Available time indices: 0 to {len(plotter.time_values)-1}")
            print("Enter time indices separated by spaces (e.g., 0 10 20 30):")
            
            try:
                indices_input = input("Time indices: ").strip()
                time_indices = [int(x) for x in indices_input.split()]
                plotter.plot_gradient_evolution(time_indices=time_indices)
            except ValueError:
                print("Invalid input. Please enter valid integers.")
                
        elif choice == '3':
            print("\nCreating heatmap...")
            plotter.plot_heatmap()
            
        elif choice == '4':
            print("\nAnalyzing peak gradients over time...")
            
            gradient_data = plotter.data.iloc[:, 1:].values
            peak_gradients = np.max(gradient_data, axis=1)
            peak_positions = [plotter.radial_positions[np.argmax(gradient_data[i, :])] 
                             for i in range(len(plotter.time_values))]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Peak gradient magnitude
            ax1.plot(plotter.time_values, peak_gradients, 'b-', linewidth=2)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Peak Radial Gradient (K/m)')
            ax1.set_title('Peak Radial Gradient Evolution')
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
            
            # Peak gradient position
            ax2.plot(plotter.time_values, peak_positions, 'r-', linewidth=2)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Radial Position (m)')
            ax2.set_title('Peak Gradient Position Evolution')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        elif choice == '5':
            print("\nCustom time range analysis...")
            print(f"Time range: {plotter.time_values[0]:.2e} to {plotter.time_values[-1]:.2e} s")
            
            try:
                start_time = float(input("Enter start time (s): "))
                end_time = float(input("Enter end time (s): "))
                
                # Find indices for the time range
                start_idx = np.argmin(np.abs(plotter.time_values - start_time))
                end_idx = np.argmin(np.abs(plotter.time_values - end_time))
                
                time_indices = list(range(start_idx, end_idx + 1))
                print(f"Using time indices {start_idx} to {end_idx}")
                
                plotter.plot_gradient_evolution(time_indices=time_indices)
                
            except ValueError:
                print("Invalid input. Please enter valid numbers.")
                
        elif choice == '6':
            print("\nExiting...")
            break
            
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")


if __name__ == '__main__':
    interactive_analysis() 