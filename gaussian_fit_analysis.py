#!/usr/bin/env python3
"""
Gaussian fitting analysis for radial gradient data.

This script fits Gaussian functions to radial gradient profiles at each time point
and tracks the evolution of fitting parameters and RMSE over time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.stats import pearsonr
from plot_radial_gradient import RadialGradientPlotter
import argparse
import sys
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.cm as cm


def split_normal_function(r: np.ndarray, amplitude: float, center: float, sigma_left: float, sigma_right: float, offset: float = 0.0) -> np.ndarray:
    """
    Split normal (double-sided Gaussian) function for fitting radial gradient data.
    
    Parameters:
    -----------
    r : np.ndarray
        Radial positions
    amplitude : float
        Amplitude of the curve
    center : float
        Center position
    sigma_left : float
        Width parameter for r < center
    sigma_right : float
        Width parameter for r >= center
    offset : float
        Vertical offset
    
    Returns:
    --------
    np.ndarray : Split normal values
    """
    result = np.empty_like(r)
    mask_left = r < center
    mask_right = ~mask_left
    result[mask_left] = amplitude * np.exp(-0.5 * ((r[mask_left] - center) / sigma_left) ** 2) + offset
    result[mask_right] = amplitude * np.exp(-0.5 * ((r[mask_right] - center) / sigma_right) ** 2) + offset
    return result


def fit_split_normal_to_profile(radial_positions: np.ndarray, gradient_values: np.ndarray, initial_guess: Optional[List[float]] = None, fit_method: str = 'rmse') -> Tuple[List[float], float]:
    """
    Fit a split normal function to a radial gradient profile.
    Tries both positive and negative amplitude initial guesses, returns the best fit (lowest error).
    fit_method: 'rmse' (default) or 'maxerr' (minimize maximum absolute error)
    """
    valid_mask = ~(np.isnan(gradient_values) | np.isnan(radial_positions))
    r_valid = radial_positions[valid_mask]
    grad_valid = gradient_values[valid_mask]
    if len(r_valid) < 4:
        return [0.0, 0.0, 1.0, 1.0, 0.0], np.inf
    max_grad = np.max(grad_valid)
    min_grad = np.min(grad_valid)
    amplitude_guess_abs = np.abs(max_grad - min_grad)
    center_guess = r_valid[np.argmax(np.abs(grad_valid))]
    sigma_guess = np.std(r_valid) / 4 if np.std(r_valid) > 0 else 1e-6
    offset_guess = min_grad
    r_range = np.max(r_valid) - np.min(r_valid)
    bounds = (
        [-np.inf, np.min(r_valid), 1e-12, 1e-12, -np.inf],
        [np.inf, np.max(r_valid), r_range, r_range, np.inf]
    )
    guesses = [
        [ amplitude_guess_abs, center_guess, sigma_guess, sigma_guess, offset_guess],
        [-amplitude_guess_abs, center_guess, sigma_guess, sigma_guess, offset_guess],
    ]
    best_err = np.inf
    best_params = [0.0, 0.0, 1.0, 1.0, 0.0]
    for guess in guesses:
        try:
            if fit_method == 'rmse':
                popt, _ = curve_fit(split_normal_function, r_valid, grad_valid, p0=guess, bounds=bounds, maxfev=20000)
                fitted_values = split_normal_function(r_valid, *popt)
                err = np.sqrt(np.mean((grad_valid - fitted_values)**2))
            elif fit_method == 'maxerr':
                def max_abs_error(params):
                    return np.max(np.abs(grad_valid - split_normal_function(r_valid, *params)))
                res = minimize(max_abs_error, guess, method='Powell')
                popt = res.x
                fitted_values = split_normal_function(r_valid, *popt)
                err = np.max(np.abs(grad_valid - fitted_values))
            else:
                raise ValueError(f"Unknown fit_method: {fit_method}")
            if err < best_err:
                best_err = err
                best_params = list(popt)
        except Exception:
            continue
    return best_params, best_err


def fit_split_normal_amplitude_only(radial_positions: np.ndarray, gradient_values: np.ndarray, fixed_params: List[float]) -> Tuple[float, float]:
    """
    Fit only the amplitude of a split normal, with center, sigma_left, sigma_right, and offset fixed.
    Returns (amplitude, rmse).
    """
    def fixed_split_normal(r, amplitude):
        center, sigma_left, sigma_right, offset = fixed_params
        return split_normal_function(r, amplitude, center, sigma_left, sigma_right, offset)
    valid_mask = ~(np.isnan(gradient_values) | np.isnan(radial_positions))
    r_valid = radial_positions[valid_mask]
    grad_valid = gradient_values[valid_mask]
    if len(r_valid) < 4:
        return 0.0, np.inf
    amplitude_guess = grad_valid[np.argmax(np.abs(grad_valid))]
    try:
        popt, _ = curve_fit(fixed_split_normal, r_valid, grad_valid, p0=[amplitude_guess], maxfev=10000)
        fitted_values = fixed_split_normal(r_valid, *popt)
        rmse = np.sqrt(np.mean((grad_valid - fitted_values) ** 2))
        return popt[0], rmse
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return 0.0, np.inf


def analyze_split_normal_fits(plotter: RadialGradientPlotter, fit_method: str = 'rmse') -> Dict[str, Any]:
    """
    Fit split normal functions to all time points and analyze the results.
    """
    print(f"Fitting split normal functions to all time points (method: {fit_method})...")
    time_values = plotter.time_values
    radial_positions = np.array(plotter.radial_positions)
    gradient_data = plotter.data.iloc[:, 1:].values
    amplitudes = []
    centers = []
    sigma_lefts = []
    sigma_rights = []
    offsets = []
    rmse_values = []
    r_squared_values = []
    for i, time in enumerate(time_values):
        if i % 10 == 0:
            print(f"  Processing time point {i+1}/{len(time_values)} (t={time:.2e}s)")
        gradients = gradient_data[i, :]
        params, err = fit_split_normal_to_profile(radial_positions, gradients, fit_method=fit_method)
        amplitude, center, sigma_left, sigma_right, offset = params
        valid_mask = ~(np.isnan(gradients) | np.isnan(radial_positions))
        if np.sum(valid_mask) > 4:
            fitted_values = split_normal_function(radial_positions[valid_mask], amplitude, center, sigma_left, sigma_right, offset)
            actual_values = gradients[valid_mask]
            ss_res = np.sum((actual_values - fitted_values)**2)
            ss_tot = np.sum((actual_values - np.mean(actual_values))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            r_squared = 0
        amplitudes.append(amplitude)
        centers.append(center)
        sigma_lefts.append(sigma_left)
        sigma_rights.append(sigma_right)
        offsets.append(offset)
        rmse_values.append(err)
        r_squared_values.append(r_squared)
    print("Split normal fitting completed!")
    return {
        'time_values': time_values,
        'amplitudes': np.array(amplitudes),
        'centers': np.array(centers),
        'sigma_lefts': np.array(sigma_lefts),
        'sigma_rights': np.array(sigma_rights),
        'offsets': np.array(offsets),
        'rmse_values': np.array(rmse_values),
        'r_squared_values': np.array(r_squared_values)
    }


def analyze_split_normal_fits_amplitude_only(plotter: RadialGradientPlotter, avg_center: float, avg_sigma_left: float, avg_sigma_right: float, avg_offset: float) -> Dict[str, Any]:
    """
    Fit only amplitude at each time step, with center, sigma_left, sigma_right, and offset fixed to averages.
    """
    print("Fitting amplitude only (split normal, fixed center, sigma_left, sigma_right, offset)...")
    time_values = plotter.time_values
    radial_positions = np.array(plotter.radial_positions)
    gradient_data = plotter.data.iloc[:, 1:].values
    amplitudes = []
    rmse_values = []
    for i, gradients in enumerate(gradient_data):
        amp, rmse = fit_split_normal_amplitude_only(radial_positions, gradients, [avg_center, avg_sigma_left, avg_sigma_right, avg_offset])
        amplitudes.append(amp)
        rmse_values.append(rmse)
    return {
        'time_values': time_values,
        'amplitudes': np.array(amplitudes),
        'center': avg_center,
        'sigma_left': avg_sigma_left,
        'sigma_right': avg_sigma_right,
        'offset': avg_offset,
        'rmse_values': np.array(rmse_values)
    }


def plot_split_normal_analysis(results: Dict[str, Any], save_path: Optional[str] = None, 
                          show_plot: bool = True) -> Tuple[Figure, List[Axes]]:
    """
    Create comprehensive plots of split normal fitting results.
    
    Parameters:
    -----------
    results : dict
        Results from analyze_split_normal_fits
    save_path : str, optional
        Path to save the plot
    show_plot : bool
        Whether to display the plot
        
    Returns:
    --------
    tuple : (figure, list of axes)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    time_values = results['time_values']
    
    # Plot 1: RMSE over time
    axes[0].plot(time_values, results['rmse_values'], 'b-', linewidth=2)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('RMSE (K/m)')
    axes[0].set_title('Split Normal Fit RMSE Over Time')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Plot 2: R-squared over time
    axes[1].plot(time_values, results['r_squared_values'], 'g-', linewidth=2)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('R²')
    axes[1].set_title('Split Normal Fit R² Over Time')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    # Plot 3: Amplitude over time
    axes[2].plot(time_values, results['amplitudes'], 'r-', linewidth=2)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude (K/m)')
    axes[2].set_title('Split Normal Amplitude Over Time')
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Center position over time
    axes[3].plot(time_values, results['centers'], 'm-', linewidth=2)
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Center Position (m)')
    axes[3].set_title('Split Normal Center Position Over Time')
    axes[3].grid(True, alpha=0.3)
    
    # Plot 5: Sigma (width) over time
    axes[4].plot(time_values, results['sigma_lefts'], 'c-', linewidth=2)
    axes[4].set_xlabel('Time (s)')
    axes[4].set_ylabel('Sigma (m) for r < center')
    axes[4].set_title('Split Normal Width (Sigma) Over Time for r < center')
    axes[4].grid(True, alpha=0.3)
    
    axes[5].plot(time_values, results['sigma_rights'], 'orange', linewidth=2)
    axes[5].set_xlabel('Time (s)')
    axes[5].set_ylabel('Sigma (m) for r >= center')
    axes[5].set_title('Split Normal Width (Sigma) Over Time for r >= center')
    axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Split Normal analysis plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig, axes


def plot_fit_comparison(plotter: RadialGradientPlotter, results: Dict[str, Any], 
                       time_indices: List[int], save_path: Optional[str] = None, 
                       show_plot: bool = True) -> Tuple[Figure, Axes]:
    """
    Plot comparison between actual data and split normal fits at specific time points.
    
    Parameters:
    -----------
    plotter : RadialGradientPlotter
        Plotter object with loaded data
    results : dict
        Results from analyze_split_normal_fits
    time_indices : list
        List of time indices to plot
    save_path : str, optional
        Path to save the plot
    show_plot : bool
        Whether to display the plot
        
    Returns:
    --------
    tuple : (figure, axis)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    radial_positions = np.array(plotter.radial_positions)
    gradient_data = plotter.data.iloc[:, 1:].values
    
    colors = cm.get_cmap('viridis')(np.linspace(0, 1, len(time_indices)))
    
    for i, time_idx in enumerate(time_indices):
        if time_idx >= len(results['time_values']):
            continue
            
        time = results['time_values'][time_idx]
        gradients = gradient_data[time_idx, :]
        
        # Get fitted parameters
        amplitude = results['amplitudes'][time_idx]
        center = results['centers'][time_idx]
        sigma_left = results['sigma_lefts'][time_idx]
        sigma_right = results['sigma_rights'][time_idx]
        offset = results['offsets'][time_idx]
        rmse = results['rmse_values'][time_idx]
        r_squared = results['r_squared_values'][time_idx]
        
        # Plot actual data
        ax.plot(radial_positions, gradients, 'o', color=colors[i], 
               markersize=4, alpha=0.7, label=f't={time:.2e}s (data)')
        
        # Plot fitted split normal
        fitted_values = split_normal_function(radial_positions, amplitude, center, sigma_left, sigma_right, offset)
        ax.plot(radial_positions, fitted_values, '-', color=colors[i], 
               linewidth=2, alpha=0.8, 
               label=f't={time:.2e}s (fit, RMSE={rmse:.2e}, R²={r_squared:.3f})')
    
    ax.set_xlabel('Radial Position (m)', fontsize=12)
    ax.set_ylabel('Radial Temperature Gradient (K/m)', fontsize=12)
    ax.set_title('Split Normal Fit Comparison at Selected Time Points', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Fit comparison plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig, ax


def save_fit_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save split normal fitting results to a CSV file.
    
    Parameters:
    -----------
    results : dict
        Results from analyze_split_normal_fits
    output_path : str
        Path to save the CSV file
    """
    df = pd.DataFrame({
        'time': results['time_values'],
        'amplitude': results['amplitudes'],
        'center': results['centers'],
        'sigma_left': results['sigma_lefts'],
        'sigma_right': results['sigma_rights'],
        'offset': results['offsets'],
        'rmse': results['rmse_values'],
        'r_squared': results['r_squared_values']
    })
    
    df.to_csv(output_path, index=False)
    print(f"Split Normal fit results saved to: {output_path}")


def plot_comparison_raw_vs_amp_only(plotter: RadialGradientPlotter, raw_results: Dict[str, Any], amp_only_results: Dict[str, Any], time_indices: List[int], save_path: Optional[str] = None, show_plot: bool = True):
    """
    Plot data, raw fit, and amplitude-only fit for selected time steps.
    """
    radial_positions = np.array(plotter.radial_positions)
    gradient_data = plotter.data.iloc[:, 1:].values
    colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(time_indices)))
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, time_idx in enumerate(time_indices):
        if time_idx >= len(raw_results['time_values']):
            continue
        time = raw_results['time_values'][time_idx]
        gradients = gradient_data[time_idx, :]
        # Raw fit params
        amp_r = raw_results['amplitudes'][time_idx]
        ctr_r = raw_results['centers'][time_idx]
        sig_left_r = raw_results['sigma_lefts'][time_idx]
        sig_right_r = raw_results['sigma_rights'][time_idx]
        off_r = raw_results['offsets'][time_idx]
        rmse_r = raw_results['rmse_values'][time_idx]
        # Amplitude-only fit params
        amp_a = amp_only_results['amplitudes'][time_idx]
        ctr_a = amp_only_results['center']
        sig_left_a = amp_only_results['sigma_left']
        sig_right_a = amp_only_results['sigma_right']
        off_a = amp_only_results['offset']
        rmse_a = amp_only_results['rmse_values'][time_idx]
        # Plot data
        ax.scatter(radial_positions, gradients, color=colors[i], s=18, alpha=0.6, label=f't={time:.2e}s (data)')
        # Plot raw fit
        fit_raw = split_normal_function(radial_positions, amp_r, ctr_r, sig_left_r, sig_right_r, off_r)
        ax.plot(radial_positions, fit_raw, color=colors[i], linestyle='-', linewidth=2, alpha=0.8, label=f't={time:.2e}s (raw, RMSE={rmse_r:.1e})')
        # Plot amplitude-only fit
        fit_amp = split_normal_function(radial_positions, amp_a, ctr_a, sig_left_a, sig_right_a, off_a)
        ax.plot(radial_positions, fit_amp, color=colors[i], linestyle='--', linewidth=2, alpha=0.8, label=f't={time:.2e}s (amp-only, RMSE={rmse_a:.1e})')
    ax.set_xlabel('Radial Position (m)', fontsize=12)
    ax.set_ylabel('Radial Temperature Gradient (K/m)', fontsize=12)
    ax.set_title('Raw vs Amplitude-Only Split Normal Fit Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Raw vs amplitude-only comparison plot saved to: {save_path}")
    if show_plot:
        plt.show()
    return fig, ax


def save_fitted_curves_csv(time_values, radial_positions, fitted_matrix, output_path):
    """
    Save fitted values as a CSV with the same format as the original radial gradient data.
    Columns: time, r1, r2, ...
    Each row: time, fitted values at each radius
    """
    df = pd.DataFrame(fitted_matrix, columns=[f"{r:.6e}" for r in radial_positions])
    df.insert(0, 'time', time_values)
    df.to_csv(output_path, index=False)
    print(f"Saved fitted curves to: {output_path}")


def plot_residual_analysis(raw_data, fitted_data, radial_positions, time_values, prefix='split_normal_fit'):
    """
    Plot average, max, and min residuals (raw - fit) over all time at each radius,
    and the sum of raw and fitted gradients over radius at each time.
    """
    residuals = raw_data - fitted_data
    avg_residual = np.mean(residuals, axis=0)
    max_residual = np.max(residuals, axis=0)
    min_residual = np.min(residuals, axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(radial_positions, avg_residual, label='Average Residual')
    plt.plot(radial_positions, max_residual, label='Max Residual', linestyle='--')
    plt.plot(radial_positions, min_residual, label='Min Residual', linestyle='--')
    plt.xlabel('Radial Position (m)')
    plt.ylabel('Residual (raw - fit) (K/m)')
    plt.title('Residuals of Split Normal Fit Over All Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{prefix}_residuals.png', dpi=300)
    print(f"Saved: {prefix}_residuals.png")
    # Integral (sum) of raw and fitted gradients over radius at each time
    sum_raw = np.sum(raw_data, axis=1)
    sum_fit = np.sum(fitted_data, axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(time_values, sum_raw, label='Sum of Raw Gradients')
    plt.plot(time_values, sum_fit, label='Sum of Fitted Gradients')
    plt.xlabel('Time (s)')
    plt.ylabel('Sum of Gradients (K/m)')
    plt.title('Sum of Gradients Over Radius at Each Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{prefix}_sum_comparison.png', dpi=300)
    print(f"Saved: {prefix}_sum_comparison.png")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Gaussian fitting analysis for radial gradient data')
    
    # Required arguments
    parser.add_argument('data_path', type=str,
                       help='Path to radial gradient CSV file')
    
    # Optional arguments
    parser.add_argument('--save-results', type=str,
                       help='Path to save fitting results CSV')
    parser.add_argument('--save-analysis-plot', type=str,
                       help='Path to save analysis plot')
    parser.add_argument('--save-comparison-plot', type=str,
                       help='Path to save fit comparison plot')
    parser.add_argument('--time-indices', type=int, nargs='+', default=[0, 10, 20, 30],
                       help='Time indices for comparison plot')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots')
    parser.add_argument('--compare-steps', type=int, nargs='+',
                       help='Time indices for raw vs amplitude-only comparison plot (default: every 5th step)')
    parser.add_argument('--save-compare-plot', type=str,
                       help='Path to save raw vs amplitude-only comparison plot')
    parser.add_argument('--save-fitted-csv-full', type=str, help='Path to save full-parameter split normal fit as CSV')
    parser.add_argument('--save-fitted-csv-amp', type=str, help='Path to save amplitude-only split normal fit as CSV')
    parser.add_argument('--fit-method', type=str, choices=['rmse', 'maxerr'], default='rmse', help='Fitting method: rmse (default) or maxerr (minimize maximum absolute error)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.data_path).exists():
        print(f"Error: File {args.data_path} does not exist")
        sys.exit(1)
    
    # Create plotter and load data
    plotter = RadialGradientPlotter(args.data_path)
    
    # Perform split normal fitting analysis
    results = analyze_split_normal_fits(plotter, fit_method=args.fit_method)
    
    # Print summary statistics
    print("\nSplit Normal Fitting Summary:")
    print(f"  Average RMSE: {np.mean(results['rmse_values']):.2e} K/m")
    print(f"  Average R²: {np.mean(results['r_squared_values']):.3f}")
    print(f"  Best fit time: t={results['time_values'][np.argmax(results['r_squared_values'])]:.2e}s")
    print(f"  Worst fit time: t={results['time_values'][np.argmin(results['r_squared_values'])]:.2e}s")
    
    total_rmse = np.sum(results['rmse_values'])
    print(f"Total RMSE summed across all time steps: {total_rmse:.2e} K/m")
    
    # Calculate averages (excluding amplitude)
    avg_center = np.mean(results['centers'])
    avg_sigma_left = np.mean(results['sigma_lefts'])
    avg_sigma_right = np.mean(results['sigma_rights'])
    avg_offset = np.mean(results['offsets'])
    print(f"\nAveraged parameters (excluding amplitude):")
    print(f"  center: {avg_center:.3e}, sigma_left: {avg_sigma_left:.3e}, sigma_right: {avg_sigma_right:.3e}, offset: {avg_offset:.3e}")

    # Second pass: fit only amplitude
    amp_only_results = analyze_split_normal_fits_amplitude_only(plotter, avg_center, avg_sigma_left, avg_sigma_right, avg_offset)
    total_rmse_amp_only = np.sum(amp_only_results['rmse_values'])
    print(f"Total RMSE (amplitude-only fit): {total_rmse_amp_only:.2e} K/m")
    
    # Create analysis plots
    plot_split_normal_analysis(
        results, 
        save_path=args.save_analysis_plot,
        show_plot=not args.no_show
    )
    
    # Create comparison plot
    plot_fit_comparison(
        plotter, results, args.time_indices,
        save_path=args.save_comparison_plot,
        show_plot=not args.no_show
    )
    
    # Determine comparison steps
    if args.compare_steps:
        compare_indices = args.compare_steps
    else:
        compare_indices = list(range(0, len(results['time_values']), 5))
    plot_comparison_raw_vs_amp_only(
        plotter, results, amp_only_results, compare_indices,
        save_path=args.save_compare_plot,
        show_plot=not args.no_show
    )
    
    # Residual analysis for full fit
    raw_data = plotter.data.iloc[:, 1:].values
    fitted_full = np.array([
        split_normal_function(np.array(plotter.radial_positions),
                          results['amplitudes'][i],
                          results['centers'][i],
                          results['sigma_lefts'][i],
                          results['sigma_rights'][i],
                          results['offsets'][i])
        for i in range(len(results['time_values']))
    ])
    plot_residual_analysis(raw_data, fitted_full, plotter.radial_positions, results['time_values'], prefix='split_normal_fit_full')
    # Residual analysis for amplitude-only fit
    fitted_amp = np.array([
        split_normal_function(np.array(plotter.radial_positions),
                          amp_only_results['amplitudes'][i],
                          amp_only_results['center'],
                          amp_only_results['sigma_left'],
                          amp_only_results['sigma_right'],
                          amp_only_results['offset'])
        for i in range(len(amp_only_results['time_values']))
    ])
    plot_residual_analysis(raw_data, fitted_amp, plotter.radial_positions, amp_only_results['time_values'], prefix='split_normal_fit_amp_only')
    
    # Save results if requested
    if args.save_results:
        save_fit_results(results, args.save_results)

    # Save fitted curves as CSVs if requested
    if args.save_fitted_csv_full:
        # Reconstruct fitted values for each time step (full fit)
        fitted_full = []
        for i in range(len(results['time_values'])):
            vals = split_normal_function(
                np.array(plotter.radial_positions),
                results['amplitudes'][i],
                results['centers'][i],
                results['sigma_lefts'][i],
                results['sigma_rights'][i],
                results['offsets'][i]
            )
            fitted_full.append(vals)
        save_fitted_curves_csv(results['time_values'], plotter.radial_positions, fitted_full, args.save_fitted_csv_full)
    if args.save_fitted_csv_amp:
        # Reconstruct fitted values for each time step (amp-only fit)
        fitted_amp = []
        for i in range(len(amp_only_results['time_values'])):
            vals = split_normal_function(
                np.array(plotter.radial_positions),
                amp_only_results['amplitudes'][i],
                amp_only_results['center'],
                amp_only_results['sigma_left'],
                amp_only_results['sigma_right'],
                amp_only_results['offset']
            )
            fitted_amp.append(vals)
        save_fitted_curves_csv(amp_only_results['time_values'], plotter.radial_positions, fitted_amp, args.save_fitted_csv_amp)


if __name__ == '__main__':
    main() 