#!/usr/bin/env python3
"""
Parameter sweep script for heatflow simulations.

This script performs a grid search over:
- Laser FWHM (heating.fwhm)
- Sample thermal conductivity (mats.p_sample.k) 
- Sample width (mats.p_sample.z)

The script efficiently reuses meshes by:
1. Grouping runs by sample width (since mesh depends on sample dimensions)
2. Creating one mesh per sample width
3. Running all FWHM/k combinations for each sample width
4. Moving to the next sample width and repeating

Features:
- Multiprocessing support with single-threaded processes
- Efficient mesh reuse by sample width
- Progress tracking and error handling
- Watcher points positioned relative to mesh geometry (halfway through iridium layers)

Usage:
    python parameter_sweep.py --config base_config.yaml --output-dir sweep_results
"""

import os
import sys
import yaml
import argparse
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
from datetime import datetime
import shutil
import multiprocessing as mp
from functools import partial
import threading

# Import the simulation runner
from run_no_diamond import run_simulation


def set_single_thread():
    """Set environment variables to ensure single-threaded operation."""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['BLIS_NUM_THREADS'] = '1'


def initialize_worker():
    """Initialize worker process with proper MPI setup."""
    set_single_thread()
    
    # Import MPI and initialize if not already initialized
    try:
        from mpi4py import MPI
        if not MPI.Is_initialized():
            MPI.Init()
    except ImportError:
        pass


def get_watcher_points(cfg):
    """
    Calculate watcher points positioned halfway through the iridium coupler layers.
    These points are positioned relative to the mesh geometry.
    
    Parameters:
    -----------
    cfg : dict
        Configuration dictionary
        
    Returns:
    --------
    dict : Watcher points with keys 'pside' and 'oside', values are (z, r) coordinates
    """
    # Extract material dimensions from config
    r_sample = float(cfg['mats']['p_sample']['r'])
    z_sample = float(cfg['mats']['p_sample']['z'])
    z_ins_pside = float(cfg['mats']['p_ins']['z'])
    z_ins_oside = float(cfg['mats']['o_ins']['z'])
    z_coupler = float(cfg['mats']['p_coupler']['z'])
    
    # Check if diamond is present in the configuration
    has_diamond = 'p_diam' in cfg['mats']
    
    if has_diamond:
        z_diam = float(cfg['mats']['p_diam']['z'])
        # Calculate mesh boundaries with diamond
        mesh_zmin = -(z_sample/2) - z_ins_pside - z_coupler - z_diam
        mesh_zmax = (z_sample/2) + z_ins_oside + z_coupler + z_diam
        
        # Calculate coupler boundary positions with diamond
        bnd_p_ins_end = mesh_zmin + z_diam + z_ins_pside
        bnd_o_ins_start = mesh_zmax - z_diam - z_ins_oside
    else:
        # Calculate mesh boundaries without diamond
        mesh_zmin = -(z_sample/2) - z_ins_pside - z_coupler
        mesh_zmax = (z_sample/2) + z_ins_oside + z_coupler
        
        # Calculate coupler boundary positions without diamond
        bnd_p_ins_end = mesh_zmin + z_ins_pside
        bnd_o_ins_start = mesh_zmax - z_ins_oside
    
    # Calculate watcher points at the center of each coupler (halfway through iridium layers)
    pside_coupler_z = bnd_p_ins_end + z_coupler/2
    oside_coupler_z = bnd_o_ins_start - z_coupler/2
    
    watcher_points = {
        'pside': (pside_coupler_z, 0.0),  # (z, r) coordinates
        'oside': (oside_coupler_z, 0.0)   # (z, r) coordinates
    }
    
    return watcher_points


def run_single_simulation(args):
    """
    Run a single simulation with the given parameters.
    This function is designed to be called by multiprocessing workers.
    
    Parameters:
    -----------
    args : tuple
        (combo, base_config, mesh_folder, output_dir, write_xdmf, suppress_print, run_id)
        
    Returns:
    --------
    dict : Results dictionary with status and timing information
    """
    # Set single thread environment
    set_single_thread()
    
    combo, base_config, mesh_folder, output_dir, write_xdmf, suppress_print, run_id = args
    
    fwhm, k, width = combo['fwhm'], combo['k'], combo['width']
    
    # Create run-specific output directory
    run_name = f"fwhm_{fwhm:.2e}_k_{k:.2f}_width_{width:.2e}".replace('+', '').replace('-0', '-')
    run_output_dir = os.path.join(output_dir, run_name)
    
    # Modify configuration for this parameter combination
    config = modify_config_for_parameters(base_config, fwhm, k, width)
    
    # Calculate watcher points for this configuration
    watcher_points = get_watcher_points(config)
    
    try:
        # Run simulation
        start_time = time.time()
        run_simulation(
            cfg=config,
            mesh_folder=mesh_folder,
            rebuild_mesh=False,  # Mesh should already exist
            visualize_mesh=False,
            output_folder=run_output_dir,
            watcher_points=watcher_points,
            write_xdmf=write_xdmf,
            suppress_print=suppress_print
        )
        end_time = time.time()
        
        return {
            'run_id': run_id,
            'run_name': run_name,
            'fwhm': fwhm,
            'k': k,
            'width': width,
            'output_dir': run_output_dir,
            'runtime': end_time - start_time,
            'status': 'success',
            'error': None
        }
        
    except Exception as e:
        return {
            'run_id': run_id,
            'run_name': run_name,
            'fwhm': fwhm,
            'k': k,
            'width': width,
            'output_dir': run_output_dir,
            'runtime': 0.0,
            'status': 'failed',
            'error': str(e)
        }


def create_parameter_grid(fwhm_range, k_range, width_range, num_points):
    """
    Create a parameter grid for the sweep.
    
    Parameters:
    -----------
    fwhm_range : tuple
        (min_fwhm, max_fwhm) in meters
    k_range : tuple  
        (min_k, max_k) in W/m/K
    width_range : tuple
        (min_width, max_width) in meters
    num_points : tuple
        (num_fwhm, num_k, num_width) number of points for each parameter
        
    Returns:
    --------
    list of dicts, each containing parameter combinations
    """
    fwhm_min, fwhm_max = fwhm_range
    k_min, k_max = k_range
    width_min, width_max = width_range
    
    num_fwhm, num_k, num_width = num_points
    
    # Create parameter arrays
    fwhm_vals = np.logspace(np.log10(fwhm_min), np.log10(fwhm_max), num_fwhm)
    k_vals = np.logspace(np.log10(k_min), np.log10(k_max), num_k)
    width_vals = np.linspace(width_min, width_max, num_width)
    
    # Group by width first (for mesh reuse)
    parameter_combinations = []
    for width in width_vals:
        for fwhm, k in itertools.product(fwhm_vals, k_vals):
            parameter_combinations.append({
                'fwhm': fwhm,
                'k': k, 
                'width': width
            })
    
    return parameter_combinations, fwhm_vals, k_vals, width_vals


def modify_config_for_parameters(base_config, fwhm, k, width):
    """
    Modify a base configuration with new parameter values.
    
    Parameters:
    -----------
    base_config : dict
        Base configuration dictionary
    fwhm : float
        Laser FWHM in meters
    k : float
        Sample thermal conductivity in W/m/K
    width : float
        Sample width/thickness in meters
        
    Returns:
    --------
    dict : Modified configuration
    """
    config = base_config.copy()
    
    # Update heating parameters (convert to native Python types)
    config['heating']['fwhm'] = float(fwhm)
    
    # Update sample material properties (convert to native Python types)
    config['mats']['p_sample']['k'] = float(k)
    config['mats']['p_sample']['z'] = float(width)
    
    return config


def get_mesh_folder_for_width(base_mesh_folder, width):
    """
    Generate a mesh folder name based on sample width.
    
    Parameters:
    -----------
    base_mesh_folder : str
        Base mesh folder path
    width : float
        Sample width in meters
        
    Returns:
    --------
    str : Mesh folder path for this width
    """
    # Convert width to a reasonable folder name (e.g., 1.84e-06 -> width_1.84e-06)
    width_str = f"{width:.3e}".replace('+', '').replace('-0', '-')
    return os.path.join(base_mesh_folder, f"width_{width_str}")


def run_parameter_sweep(base_config_path, output_dir, fwhm_range, k_range, width_range, 
                       num_points, base_mesh_folder="meshes", write_xdmf=False, suppress_print=True, num_processes=None):
    """
    Run the parameter sweep with multiprocessing support.
    
    Parameters:
    -----------
    base_config_path : str
        Path to base configuration file
    output_dir : str
        Directory to save all results
    fwhm_range : tuple
        (min_fwhm, max_fwhm) in meters
    k_range : tuple
        (min_k, max_k) in W/m/K  
    width_range : tuple
        (min_width, max_width) in meters
    num_points : tuple
        (num_fwhm, num_k, num_width) number of points for each parameter
    base_mesh_folder : str
        Base directory for mesh storage
    write_xdmf : bool
        Whether to write XDMF output files
    suppress_print : bool
        Whether to suppress print output during simulations
    num_processes : int, optional
        Number of processes to use. If None, uses CPU count.
    """
    
    # Set single thread environment for main process
    set_single_thread()
    
    # Set multiprocessing start method to spawn to avoid MPI issues
    if mp.get_start_method(allow_none=True) != 'spawn':
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set, continue
            pass
    
    # Load base configuration
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create parameter grid
    parameter_combinations, fwhm_vals, k_vals, width_vals = create_parameter_grid(
        fwhm_range, k_range, width_range, num_points
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save sweep metadata
    sweep_metadata = {
        'base_config': base_config_path,
        'fwhm_range': fwhm_range,
        'k_range': k_range, 
        'width_range': width_range,
        'num_points': num_points,
        'fwhm_values': fwhm_vals.tolist(),
        'k_values': k_vals.tolist(),
        'width_values': width_vals.tolist(),
        'total_runs': len(parameter_combinations),
        'num_processes': num_processes or mp.cpu_count(),
        'timestamp': datetime.now().isoformat(),
        'watcher_points': {
            'description': 'Temperature monitoring points positioned halfway through iridium coupler layers',
            'locations': {
                'pside': 'Center of p-side iridium coupler (r=0)',
                'oside': 'Center of o-side iridium coupler (r=0)'
            },
            'coordinates': 'Relative to mesh geometry, calculated for each parameter combination'
        }
    }
    
    with open(os.path.join(output_dir, 'sweep_metadata.json'), 'w') as f:
        json.dump(sweep_metadata, f, indent=2)
    
    # Group combinations by width for efficient mesh reuse
    width_groups = {}
    for combo in parameter_combinations:
        width = combo['width']
        if width not in width_groups:
            width_groups[width] = []
        width_groups[width].append(combo)
    
    # Results tracking
    results = []
    failed_runs = []
    total_completed = 0
    
    print(f"Starting parameter sweep with {len(parameter_combinations)} total runs")
    print(f"Parameters: {len(fwhm_vals)} FWHM values, {len(k_vals)} k values, {len(width_vals)} width values")
    print(f"Grouped into {len(width_groups)} width groups for mesh reuse")
    print(f"Using {num_processes or mp.cpu_count()} processes")
    print(f"Output directory: {output_dir}")
    print(f"Watcher points: Temperature monitoring at iridium coupler centers (pside, oside)")
    print("-" * 80)
    
    # Determine number of processes
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Run simulations grouped by width
    for width_idx, (width, combinations) in enumerate(width_groups.items()):
        print(f"\nProcessing width group {width_idx + 1}/{len(width_groups)}: width = {width:.2e} m")
        print(f"  {len(combinations)} runs for this width")
        
        # Create mesh folder for this width
        mesh_folder = get_mesh_folder_for_width(base_mesh_folder, width)
        os.makedirs(mesh_folder, exist_ok=True)
        
        # Check if mesh already exists
        mesh_file = os.path.join(mesh_folder, 'mesh.msh')
        mesh_cfg_file = os.path.join(mesh_folder, 'mesh_cfg.yaml')
        rebuild_mesh = not (os.path.exists(mesh_file) and os.path.exists(mesh_cfg_file))
        
        if rebuild_mesh:
            print(f"  Building new mesh for width {width:.2e} m")
            # Build mesh using single process
            config = modify_config_for_parameters(base_config, combinations[0]['fwhm'], combinations[0]['k'], width)
            run_simulation(
                cfg=config,
                mesh_folder=mesh_folder,
                rebuild_mesh=True,
                visualize_mesh=False,
                output_folder=None,  # Don't save output for mesh building
                watcher_points=None,
                write_xdmf=False,
                suppress_print=suppress_print
            )
        else:
            print(f"  Reusing existing mesh for width {width:.2e} m")
        
        # Prepare arguments for multiprocessing
        run_args = []
        for run_id, combo in enumerate(combinations):
            run_args.append((
                combo, base_config, mesh_folder, output_dir, 
                write_xdmf, suppress_print, 
                total_completed + run_id + 1 
            ))
        
        # Run simulations in parallel
        print(f"  Starting {len(combinations)} simulations with {num_processes} processes...")
        
        try:
            with mp.Pool(processes=num_processes, initializer=initialize_worker) as pool:
                # Use imap to get results as they complete
                for result in pool.imap_unordered(run_single_simulation, run_args):
                    total_completed += 1
                    
                    if result['status'] == 'success':
                        results.append(result)
                        print(f"  ✓ [{total_completed}/{len(parameter_combinations)}] FWHM={result['fwhm']:.2e}m, k={result['k']:.2f}W/m/K, width={result['width']:.2e}m - Completed in {result['runtime']:.2f}s")
                    else:
                        failed_runs.append(result)
                        print(f"  ✗ [{total_completed}/{len(parameter_combinations)}] FWHM={result['fwhm']:.2e}m, k={result['k']:.2f}W/m/K, width={result['width']:.2e}m - Failed: {result['error']}")
        except Exception as e:
            print(f"  Error in multiprocessing: {e}")
            print("  Falling back to single-threaded execution...")
            
            # Fallback to single-threaded execution
            for run_id, combo in enumerate(combinations):
                fwhm, k = combo['fwhm'], combo['k']
                
                # Create run-specific output directory
                run_name = f"fwhm_{fwhm:.2e}_k_{k:.2f}_width_{width:.2e}".replace('+', '').replace('-0', '-')
                run_output_dir = os.path.join(output_dir, run_name)
                
                # Modify configuration for this parameter combination
                config = modify_config_for_parameters(base_config, fwhm, k, width)
                
                # Calculate watcher points for this configuration
                watcher_points = get_watcher_points(config)
                
                try:
                    # Run simulation
                    start_time = time.time()
                    run_simulation(
                        cfg=config,
                        mesh_folder=mesh_folder,
                        rebuild_mesh=False,
                        visualize_mesh=False,
                        output_folder=run_output_dir,
                        watcher_points=watcher_points,
                        write_xdmf=write_xdmf,
                        suppress_print=suppress_print
                    )
                    end_time = time.time()
                    
                    total_completed += 1
                    results.append({
                        'run_id': total_completed,
                        'run_name': run_name,
                        'fwhm': fwhm,
                        'k': k,
                        'width': width,
                        'output_dir': run_output_dir,
                        'runtime': end_time - start_time,
                        'status': 'success',
                        'error': None
                    })
                    
                    print(f"  ✓ [{total_completed}/{len(parameter_combinations)}] FWHM={fwhm:.2e}m, k={k:.2f}W/m/K, width={width:.2e}m - Completed in {end_time - start_time:.2f}s")
                    
                except Exception as sim_error:
                    total_completed += 1
                    failed_runs.append({
                        'run_id': total_completed,
                        'run_name': run_name,
                        'fwhm': fwhm,
                        'k': k,
                        'width': width,
                        'output_dir': run_output_dir,
                        'runtime': 0.0,
                        'status': 'failed',
                        'error': str(sim_error)
                    })
                    
                    print(f"  ✗ [{total_completed}/{len(parameter_combinations)}] FWHM={fwhm:.2e}m, k={k:.2f}W/m/K, width={width:.2e}m - Failed: {str(sim_error)}")
    
    # Save results summary
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.to_csv(os.path.join(output_dir, 'successful_runs.csv'), index=False)
    
    failed_df = pd.DataFrame(failed_runs)
    if not failed_df.empty:
        failed_df.to_csv(os.path.join(output_dir, 'failed_runs.csv'), index=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("PARAMETER SWEEP COMPLETE")
    print("=" * 80)
    print(f"Total runs: {len(parameter_combinations)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(failed_runs)}")
    print(f"Results saved to: {output_dir}")
    
    if results:
        avg_runtime = np.mean([r['runtime'] for r in results])
        total_runtime = sum([r['runtime'] for r in results])
        print(f"Average runtime per simulation: {avg_runtime:.2f}s")
        print(f"Total simulation time: {total_runtime:.2f}s")
        print(f"Speedup factor: {total_runtime / (avg_runtime * len(parameter_combinations)):.2f}x")
    
    return results, failed_runs


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Parameter sweep for heatflow simulations')
    
    # Required arguments
    parser.add_argument('--config', type=str, required=True,
                       help='Path to base configuration file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save all results')
    
    # Parameter ranges
    parser.add_argument('--fwhm-range', type=float, nargs=2, default=[1e-6, 1e-4],
                       help='FWHM range in meters (min max)')
    parser.add_argument('--k-range', type=float, nargs=2, default=[1.0, 100.0],
                       help='Thermal conductivity range in W/m/K (min max)')
    parser.add_argument('--width-range', type=float, nargs=2, default=[1e-6, 10e-6],
                       help='Sample width range in meters (min max)')
    
    # Number of points
    parser.add_argument('--num-points', type=int, nargs=3, default=[5, 5, 3],
                       help='Number of points for each parameter (fwhm k width)')
    
    # Optional arguments
    parser.add_argument('--mesh-folder', type=str, default='meshes',
                       help='Base directory for mesh storage')
    parser.add_argument('--write-xdmf', action='store_true',
                       help='Write XDMF output files (increases storage significantly)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output during simulations')
    parser.add_argument('--num-processes', type=int, default=None,
                       help='Number of processes to use (default: CPU count)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if any(x <= 0 for x in args.num_points):
        parser.error("Number of points must be positive")
    
    if args.fwhm_range[0] <= 0 or args.fwhm_range[1] <= 0:
        parser.error("FWHM range must be positive")
    
    if args.k_range[0] <= 0 or args.k_range[1] <= 0:
        parser.error("Thermal conductivity range must be positive")
    
    if args.width_range[0] <= 0 or args.width_range[1] <= 0:
        parser.error("Width range must be positive")
    
    if args.num_processes is not None and args.num_processes <= 0:
        parser.error("Number of processes must be positive")
    
    # Run the parameter sweep
    run_parameter_sweep(
        base_config_path=args.config,
        output_dir=args.output_dir,
        fwhm_range=tuple(args.fwhm_range),
        k_range=tuple(args.k_range),
        width_range=tuple(args.width_range),
        num_points=tuple(args.num_points),
        base_mesh_folder=args.mesh_folder,
        write_xdmf=args.write_xdmf,
        suppress_print=not args.verbose,
        num_processes=args.num_processes
    )


if __name__ == '__main__':
    main() 