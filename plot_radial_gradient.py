#!/usr/bin/env python3
"""
Radial gradient plotting and analysis script.

This script reads radial gradient data from parameter sweep results and provides
flexible plotting capabilities for analyzing the time evolution of radial gradients.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path
import argparse
import sys
from typing import Optional, List, Tuple, Dict, Any


class RadialGradientPlotter:
    """Class for plotting and analyzing radial gradient data."""
    
    def __init__(self, data_path: str):
        """
        Initialize the plotter with data from a CSV file.
        
        Parameters:
        -----------
        data_path : str
            Path to the radial gradient CSV file
        """
        self.data_path = Path(data_path)
        self.data: Optional[pd.DataFrame] = None
        self.time_values: Optional[np.ndarray] = None
        self.radial_positions: Optional[List[float]] = None
        self.max_gradient: Optional[float] = None
        self.min_gradient: Optional[float] = None
        
        self.load_data()
    
    def load_data(self) -> None:
        """Load and process the radial gradient data."""
        try:
            # Read the CSV file
            self.data = pd.read_csv(self.data_path)
            
            if self.data is None or self.data.empty:
                raise ValueError("Data file is empty or could not be read")
            
            # Extract time values (first column)
            self.time_values = self.data.iloc[:, 0].values
            
            # Extract radial positions (column headers, excluding 'time')
            self.radial_positions = [float(col) for col in self.data.columns[1:]]
            
            # Extract gradient values (all columns except time)
            gradient_data = self.data.iloc[:, 1:].values
            
            # Calculate global min/max for consistent axis scaling
            self.max_gradient = float(np.max(gradient_data))
            self.min_gradient = float(np.min(gradient_data))
            
            print(f"Data loaded successfully:")
            print(f"  Time range: {self.time_values[0]:.2e} to {self.time_values[-1]:.2e} s")
            print(f"  Radial range: {self.radial_positions[0]:.2e} to {self.radial_positions[-1]:.2e} m")
            print(f"  Gradient range: {self.min_gradient:.2e} to {self.max_gradient:.2e} K/m")
            print(f"  Number of time points: {len(self.time_values)}")
            print(f"  Number of radial points: {len(self.radial_positions)}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
    
    def plot_gradient_evolution(self, time_indices: Optional[List[int]] = None, 
                               figsize: Tuple[float, float] = (12, 8), 
                               save_path: Optional[str] = None, 
                               show_plot: bool = True) -> Tuple[Figure, Axes]:
        """
        Plot the radial gradient evolution at specified time points.
        
        Parameters:
        -----------
        time_indices : list, optional
            List of time indices to plot. If None, plots all time points.
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            Path to save the plot
        show_plot : bool
            Whether to display the plot
        """
        if self.data is None or self.time_values is None or self.radial_positions is None:
            raise ValueError("Data not loaded properly")
            
        if time_indices is None:
            time_indices = list(range(len(self.time_values)))
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get gradient data
        gradient_data = self.data.iloc[:, 1:].values
        
        # Plot each time point
        for i in time_indices:
            if i < len(self.time_values):
                time = self.time_values[i]
                gradients = gradient_data[i, :]
                
                # Create label with time in scientific notation
                time_label = f"t = {time:.2e} s"
                
                ax.plot(self.radial_positions, gradients, 
                       label=time_label, linewidth=1.5, alpha=0.8)
        
        # Set axis labels and title
        ax.set_xlabel('Radial Position (m)', fontsize=12)
        ax.set_ylabel('Radial Temperature Gradient (K/m)', fontsize=12)
        ax.set_title('Radial Temperature Gradient Evolution', fontsize=14, fontweight='bold')
        
        # Set consistent axis limits based on global min/max
        if self.min_gradient is not None and self.max_gradient is not None:
            ax.set_ylim(self.min_gradient, self.max_gradient)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend if multiple time points
        if len(time_indices) > 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        
        return fig, ax
    
    def plot_heatmap(self, figsize: Tuple[float, float] = (12, 8), 
                     save_path: Optional[str] = None, 
                     show_plot: bool = True) -> Tuple[Figure, Axes]:
        """
        Create a heatmap showing the full time evolution of radial gradients.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            Path to save the plot
        show_plot : bool
            Whether to display the plot
        """
        if self.data is None or self.time_values is None or self.radial_positions is None:
            raise ValueError("Data not loaded properly")
            
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get gradient data
        gradient_data = self.data.iloc[:, 1:].values
        
        # Create meshgrid for plotting
        R, T = np.meshgrid(self.radial_positions, self.time_values)
        
        # Create heatmap
        im = ax.pcolormesh(R, T, gradient_data, 
                          cmap='RdBu_r', 
                          vmin=self.min_gradient, 
                          vmax=self.max_gradient,
                          shading='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Radial Temperature Gradient (K/m)', fontsize=12)
        
        # Set axis labels and title
        ax.set_xlabel('Radial Position (m)', fontsize=12)
        ax.set_ylabel('Time (s)', fontsize=12)
        ax.set_title('Radial Temperature Gradient Heatmap', fontsize=14, fontweight='bold')
        
        # Use scientific notation for axes
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        
        return fig, ax
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Return a summary of the loaded data."""
        if self.data is None or self.time_values is None or self.radial_positions is None:
            raise ValueError("Data not loaded properly")
            
        gradient_data = self.data.iloc[:, 1:].values
        
        summary = {
            'file_path': str(self.data_path),
            'time_range': (self.time_values[0], self.time_values[-1]),
            'radial_range': (self.radial_positions[0], self.radial_positions[-1]),
            'gradient_range': (self.min_gradient, self.max_gradient),
            'num_time_points': len(self.time_values),
            'num_radial_points': len(self.radial_positions),
            'max_gradient_time': self.time_values[np.argmax(np.max(gradient_data, axis=1))],
            'max_gradient_position': self.radial_positions[np.argmax(np.max(gradient_data, axis=0))]
        }
        
        return summary


def main() -> None:
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Plot radial gradient data from parameter sweep')
    
    # Required arguments
    parser.add_argument('data_path', type=str,
                       help='Path to radial gradient CSV file')
    
    # Optional arguments
    parser.add_argument('--plot-type', type=str, choices=['evolution', 'heatmap', 'both'], 
                       default='evolution',
                       help='Type of plot to generate')
    parser.add_argument('--time-indices', type=int, nargs='+',
                       help='Specific time indices to plot (for evolution plot)')
    parser.add_argument('--save-evolution', type=str,
                       help='Path to save evolution plot')
    parser.add_argument('--save-heatmap', type=str,
                       help='Path to save heatmap plot')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots (only save if paths provided)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[12, 8],
                       help='Figure size (width height)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.data_path).exists():
        print(f"Error: File {args.data_path} does not exist")
        sys.exit(1)
    
    # Create plotter
    plotter = RadialGradientPlotter(args.data_path)
    
    # Print data summary
    summary = plotter.get_data_summary()
    print("\nData Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Generate plots based on type
    if args.plot_type in ['evolution', 'both']:
        print("\nGenerating evolution plot...")
        plotter.plot_gradient_evolution(
            time_indices=args.time_indices,
            figsize=tuple(args.figsize),
            save_path=args.save_evolution,
            show_plot=not args.no_show
        )
    
    if args.plot_type in ['heatmap', 'both']:
        print("\nGenerating heatmap...")
        plotter.plot_heatmap(
            figsize=tuple(args.figsize),
            save_path=args.save_heatmap,
            show_plot=not args.no_show
        )


if __name__ == '__main__':
    main() 