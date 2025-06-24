import multiprocessing
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["PETSC_NUM_THREADS"] = "1"
import run_with_diamond as run
import yaml
import pandas as pd
import numpy as np
import analysis_utils as au
import shutil
import time

# Load the configuration
def load_cfg():
    with open('simulation_template.yaml', 'r') as f:
        return yaml.safe_load(f)

# Extract watcher points from config (same as before)
def get_watcher_points(cfg):
    r_sample = float(cfg['mats']['p_sample']['r'])
    z_sample = float(cfg['mats']['p_sample']['z'])
    z_ins_pside = float(cfg['mats']['p_ins']['z'])
    z_ins_oside = float(cfg['mats']['o_ins']['z'])
    z_coupler = float(cfg['mats']['p_coupler']['z'])
    z_diam = float(cfg['mats']['p_diam']['z'])

    mesh_zmin = -(z_sample/2) - z_ins_pside - z_coupler - z_diam
    mesh_zmax = (z_sample/2) + z_ins_oside + z_coupler + z_diam

    bnd_p_ins_end = mesh_zmin + z_diam + z_ins_pside
    bnd_o_ins_start = mesh_zmax - z_diam - z_ins_oside

    pside_coupler_z = bnd_p_ins_end + z_coupler/2
    oside_coupler_z = bnd_o_ins_start - z_coupler/2

    watcher_points = {
        'pside': (pside_coupler_z, 0.0),
        'oside': (oside_coupler_z, 0.0)
    }
    return watcher_points

# Sweep parameters
def get_k_values():
    k0 = 3.8
    k_min = k0 - 0.5
    k_max = k0 + 0.5
    step = 0.02
    return np.round(np.arange(k_min, k_max + step, step), 4)

# Function to run a single simulation and compute RMSE
def run_single_k(k):
    import time
    start_time = time.time()
    cfg = load_cfg()
    cfg['mats']['p_sample']['k'] = float(k)
    watcher_points = get_watcher_points(cfg)
    outdir = f'outputs/sweep_test/{k:.2f}'
    os.makedirs(outdir, exist_ok=True)
    watcher_csv_path = os.path.join(outdir, 'watcher_points.csv')
    if os.path.exists(watcher_csv_path):
        os.remove(watcher_csv_path)
    run.run_simulation(
        cfg=cfg,
        mesh_folder='meshes/test1',
        rebuild_mesh=False,
        visualize_mesh=False,
        output_folder=outdir,
        watcher_points=watcher_points,
        write_xdmf=False,
        suppress_print=True
    )
    if not os.path.exists(watcher_csv_path):
        return {'k': k, 'rmse': np.nan}
    df_sim = pd.read_csv(watcher_csv_path)
    df_exp = pd.read_csv('experimental_data/heat_data.csv')
    sim_pside_normed = (df_sim['pside'] - df_sim['pside'].iloc[0]) / (df_sim['pside'].max() - df_sim['pside'].min())
    sim_oside_normed = (df_sim['oside'] - df_sim['oside'].iloc[0]) / (df_sim['pside'].max() - df_sim['pside'].min())
    exp_pside_normed = (df_exp['temp'] - df_exp['temp'].iloc[0]) / (df_exp['temp'].max() - df_exp['temp'].min())
    ic_temp = cfg['heating']['ic_temp']
    oside_initial = df_exp['oside'].iloc[0]
    exp_oside_shifted = df_exp['oside'] - oside_initial + ic_temp
    exp_oside_normed = (exp_oside_shifted - exp_oside_shifted.iloc[0]) / (df_exp['temp'].max() - df_exp['temp'].min())
    oside_rmse = au.calculate_rmse(
        exp_time=df_exp['time'],
        exp_data=exp_oside_normed,
        sim_time=df_sim['time'],
        sim_data=sim_oside_normed
    )
    print(f"[k={k}] Total time: {time.time() - start_time:.2f}s")
    return {'k': k, 'rmse': oside_rmse}

def main():
    k_values = get_k_values()
    sweep_dir = 'outputs/sweep_test'
    if os.path.exists(sweep_dir):
        shutil.rmtree(sweep_dir)
    os.makedirs(sweep_dir, exist_ok=True)
    total_start_time = time.time()

    # Set up multiprocessing pool
    num_workers = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker) as pool:
        results = pool.map(run_single_k, k_values)

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(sweep_dir, 'rmse_summary.csv'), index=False)
    min_row = df_results.loc[df_results['rmse'].idxmin()]
    total_elapsed = time.time() - total_start_time
    print(f'Lowest RMSE: {min_row["rmse"]:.6f} at k = {min_row["k"]:.2f}')
    print(f'Total sweep time: {total_elapsed:.2f}s')
    print(f'Average time per run: {total_elapsed/len(k_values):.2f}s')

def init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["PETSC_NUM_THREADS"] = "1"

if __name__ == '__main__':
    main()






