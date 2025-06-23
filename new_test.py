import run_with_diamond as run
import yaml
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the configuration
with open('simulation_template.yaml', 'r') as f:
    cfg = yaml.safe_load(f)


# Extract parameters from config
r_sample = float(cfg['mats']['p_sample']['r'])
z_sample = float(cfg['mats']['p_sample']['z'])
z_ins_pside = float(cfg['mats']['p_ins']['z'])
z_ins_oside = float(cfg['mats']['o_ins']['z'])
z_coupler = float(cfg['mats']['p_coupler']['z'])
z_diam = float(cfg['mats']['p_diam']['z'])

# Calculate mesh boundaries (same as in run_with_diamond.py)
mesh_zmin = -(z_sample/2) - z_ins_pside - z_coupler - z_diam
mesh_zmax = (z_sample/2) + z_ins_oside + z_coupler + z_diam

# Calculate coupler boundary positions
bnd_p_ins_end = mesh_zmin + z_diam + z_ins_pside
bnd_o_ins_start = mesh_zmax - z_diam - z_ins_oside

pside_coupler_z = bnd_p_ins_end + z_coupler/2  # Middle of pside coupler
oside_coupler_z = bnd_o_ins_start - z_coupler/2  # Middle of oside coupler

# Define watcher points at the center of each coupler (r=0 for centerline)
watcher_points = {
    'pside': (pside_coupler_z, 0.0),  # (z, r) coordinates
    'oside': (oside_coupler_z, 0.0)   # (z, r) coordinates
}

# Run the simulation
run.run_simulation(
    cfg=cfg,
    mesh_folder='meshes/test1',
    rebuild_mesh=False,  # Use existing mesh
    visualize_mesh=False,
    output_folder='outputs/test1',
    watcher_points=watcher_points,
    write_xdmf=False  # No XDMF output as requested
)

print("Simulation completed! Check outputs/test1/ for results.")

# Plot the normalized temperature curves
def plot_normalized_temperature_curves():
    """Plot normalized temperature curves for pside and oside watcher points."""
    
    # Load simulation watcher data
    watcher_csv_path = 'outputs/test1/watcher_points.csv'
    if not os.path.exists(watcher_csv_path):
        print(f"Warning: Watcher data file not found at {watcher_csv_path}")
        return
    df_sim = pd.read_csv(watcher_csv_path)

    # Normalize simulation data to simulated pside
    sim_pside_min = df_sim['pside'].min()
    sim_pside_max = df_sim['pside'].max()
    df_sim['pside_normed'] = (df_sim['pside'] - df_sim['pside'].iloc[0]) / (sim_pside_max - sim_pside_min)
    df_sim['oside_normed'] = (df_sim['oside'] - df_sim['oside'].iloc[0]) / (sim_pside_max - sim_pside_min)

    # Load experimental data
    df_exp = pd.read_csv('experimental_data/heat_data.csv')
    df_exp['time'] = df_exp['time']
    exp_pside_min = df_exp['temp'].min()
    exp_pside_max = df_exp['temp'].max()

    # Downshift experimental oside to start from ic_temp
    ic_temp = cfg['heating']['ic_temp']
    oside_initial = df_exp['oside'].iloc[0]
    df_exp['oside'] = df_exp['oside'] - oside_initial + ic_temp

    df_exp['pside_normed'] = (df_exp['temp'] - df_exp['temp'].iloc[0]) / (exp_pside_max - exp_pside_min)
    df_exp['oside_normed'] = (df_exp['oside'] - df_exp['oside'].iloc[0]) / (exp_pside_max - exp_pside_min)

    # Plot
    plt.figure(figsize=(12, 8))
    # Simulated curves
    plt.plot(df_sim['time'], df_sim['pside_normed'], 'b-', linewidth=2, label='Sim P-side (norm)')
    plt.plot(df_sim['time'], df_sim['oside_normed'], 'r-', linewidth=2, label='Sim O-side (norm)')
    # Experimental points
    plt.scatter(df_exp['time'], df_exp['pside_normed'], color='blue', marker='o', s=40, label='Exp P-side (norm)')
    plt.scatter(df_exp['time'], df_exp['oside_normed'], color='red', marker='o', s=40, label='Exp O-side (norm)')

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Normalized Temperature', fontsize=12)
    plt.title('Normalized Temperature: Simulation vs Experiment', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plot_path = 'outputs/test1/temperature_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Temperature curves plot saved to: {plot_path}")
    plt.show()

# Generate the plot
plot_normalized_temperature_curves()






