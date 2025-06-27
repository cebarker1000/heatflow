# Radial Gradient Analysis Tools

This directory contains tools for analyzing radial gradient data from heatflow parameter sweep simulations.

## Files

- `plot_radial_gradient.py` - Main plotting class and command-line interface
- `example_radial_analysis.py` - Example script demonstrating various analysis techniques
- `interactive_radial_analysis.py` - Interactive script for exploring data
- `README_radial_analysis.md` - This documentation file

## Quick Start

### Command Line Usage

Plot the evolution of radial gradients at all time points:
```bash
python plot_radial_gradient.py outputs/sweep_test/fwhm_1.30e-5_k_3.68_width_1.90e-6/radial_gradient.csv
```

Create a heatmap showing the full time evolution:
```bash
python plot_radial_gradient.py outputs/sweep_test/fwhm_1.30e-5_k_3.68_width_1.90e-6/radial_gradient.csv --plot-type heatmap
```

Plot both evolution and heatmap:
```bash
python plot_radial_gradient.py outputs/sweep_test/fwhm_1.30e-5_k_3.68_width_1.90e-6/radial_gradient.csv --plot-type both
```

Plot specific time points:
```bash
python plot_radial_gradient.py outputs/sweep_test/fwhm_1.30e-5_k_3.68_width_1.90e-6/radial_gradient.csv --time-indices 0 10 20 30
```

### Programmatic Usage

```python
from plot_radial_gradient import RadialGradientPlotter

# Load data
plotter = RadialGradientPlotter("path/to/radial_gradient.csv")

# Get data summary
summary = plotter.get_data_summary()
print(summary)

# Plot evolution at specific time points
plotter.plot_gradient_evolution(time_indices=[0, 10, 20, 30])

# Create heatmap
plotter.plot_heatmap()
```

## Data Format

The radial gradient CSV files should have the following format:
- First column: Time values (in seconds)
- Subsequent columns: Radial gradient values at different radial positions
- Column headers: Radial positions (in meters)

Example:
```
time,-4.112e-06,-3.912e-06,-3.712e-06,...
1.875e-07,1.39e-06,2.24e-06,2.07e-06,...
3.75e-07,7.09e+00,1.95e+01,3.64e+01,...
...
```

## Features

### RadialGradientPlotter Class

- **Automatic data loading and validation**
- **Consistent axis scaling** - All plots use the same min/max gradient values
- **Flexible time point selection** - Plot any combination of time points
- **Multiple plot types** - Evolution plots and heatmaps
- **Data summary** - Get key statistics about the data

### Plot Types

1. **Evolution Plot** - Shows radial gradient profiles at selected time points
   - X-axis: Radial position (m)
   - Y-axis: Radial temperature gradient (K/m)
   - Each line represents a different time point

2. **Heatmap** - Shows the full time evolution as a color-coded plot
   - X-axis: Radial position (m)
   - Y-axis: Time (s)
   - Color: Radial temperature gradient (K/m)

### Analysis Capabilities

- **Peak gradient tracking** - Find maximum gradient at each time point
- **Position tracking** - Track where the peak gradient occurs
- **Time range analysis** - Focus on specific time periods
- **Custom visualization** - Create specialized plots for analysis

## Example Analysis

Run the example script to see various analysis techniques:
```bash
python example_radial_analysis.py
```

This will create several plots:
- `radial_gradient_evolution_example.png` - Evolution at selected time points
- `radial_gradient_heatmap_example.png` - Full heatmap
- `peak_gradient_analysis.png` - Peak gradient magnitude and position over time
- `specific_time_analysis.png` - Analysis at early, middle, and late times

## Interactive Exploration

For interactive data exploration:
```bash
python interactive_radial_analysis.py
```

This provides a menu-driven interface to:
- Plot different time points
- Create heatmaps
- Analyze peak gradients
- Custom time range analysis

## Data Insights

From the example data (`fwhm_1.30e-5_k_3.68_width_1.90e-6`):

- **Time range**: 1.88e-07 to 7.50e-06 seconds
- **Radial range**: -4.11e-06 to 7.29e-06 meters
- **Gradient range**: -2.16e+06 to 9.86e+03 K/m
- **Peak gradient time**: 5.625e-07 seconds
- **Peak gradient position**: -9.12e-07 meters

The data shows strong temporal evolution of the radial gradient, with the peak gradient occurring early in the simulation and then decaying over time.

## Future Enhancements

The plotting framework is designed to be extensible. You can easily add:
- Curve fitting capabilities
- Statistical analysis
- Comparison between different parameter sets
- Export functionality for further analysis
- Custom color schemes and styling 