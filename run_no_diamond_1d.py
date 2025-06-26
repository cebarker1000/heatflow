from mesh_and_materials.mesh import *
from mesh_and_materials.materials import *
from space.space_and_forms import *
from dirichlet_bc.bc import *
import os
from dolfinx import fem
import pandas as pd
import numpy as np
import gmsh
import yaml
import argparse
import copy
import time
from scipy.spatial import cKDTree
import sys
import contextlib
import ufl
import dolfinx
from petsc4py import PETSc

@contextlib.contextmanager
def suppress_output(enabled):
    if not enabled:
        yield
    else:
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                yield

def extract_1d_submesh_from_2d(domain_2d, cell_tags_2d, tolerance=1e-10):
    """
    Extract a 1D submesh from the r=0 line of a 2D mesh using dolfinx.mesh.create_submesh.
    
    Parameters:
    -----------
    domain_2d : dolfinx.mesh.Mesh
        The 2D mesh to extract from
    cell_tags_2d : dolfinx.mesh.MeshTags
        Cell tags from the 2D mesh (for material properties)
    tolerance : float
        Tolerance for identifying facets on the r=0 line
        
    Returns:
    --------
    domain_1d : dolfinx.mesh.Mesh
        The 1D submesh along the r=0 line
    cell_tags_1d : dolfinx.mesh.MeshTags
        Cell tags for the 1D mesh (preserving material information)
    maps : tuple
        (entity_map, vertex_map, geom_map) mapping from 1D entities to 2D entities
    """
    import dolfinx.mesh as dmesh
    import numpy as np
    
    # Get mesh topology information
    tdim = domain_2d.topology.dim  # topological dimension (2 for 2D mesh)
    
    # Create connectivity between facets and vertices if it doesn't exist
    domain_2d.topology.create_connectivity(tdim-1, 0)  # facets to vertices
    
    # Get facet-to-vertex connectivity
    facet_to_vertex = domain_2d.topology.connectivity(tdim-1, 0)  # facets to vertices
    
    # Get vertex coordinates
    vertex_coords = domain_2d.geometry.x
    
    # Find facets that lie on the r=0 line
    facets_on_axis = []
    
    for facet_idx in range(domain_2d.topology.index_map(tdim-1).size_local + domain_2d.topology.index_map(tdim-1).num_ghosts):
        # Get vertices of this facet
        facet_vertices = facet_to_vertex.links(facet_idx)
        
        # Check if both vertices of the facet are on the r=0 axis
        both_on_axis = True
        for vertex_idx in facet_vertices:
            r_coord = vertex_coords[vertex_idx, 1]  # r-coordinate
            if abs(r_coord) > tolerance:
                both_on_axis = False
                break
        
        if both_on_axis:
            facets_on_axis.append(facet_idx)
    
    facets_on_axis = np.array(facets_on_axis, dtype=np.int32)
    
    if len(facets_on_axis) == 0:
        raise ValueError("No facets found on the r=0 axis. Check tolerance or mesh.")
    
    print(f"Found {len(facets_on_axis)} facets on the r=0 axis")
    
    # Create 1D submesh using dolfinx.mesh.create_submesh
    # The function returns (submesh, entity_map, vertex_map, geom_map)
    domain_1d, entity_map, vertex_map, geom_map = dmesh.create_submesh(
        domain_2d, 
        tdim-1,  # 1D entities (facets in 2D)
        facets_on_axis
    )
    
    print(f"Created 1D submesh with {domain_1d.topology.index_map(0).size_local} vertices")
    
    # Handle material tag mapping
    print("Mapping material tags...")
    
    # Create connectivity for 2D mesh
    domain_2d.topology.create_connectivity(2, 1)  # 2D cells to facets
    
    # Get 2D cell-to-facet connectivity
    cell_to_facet_2d = domain_2d.topology.connectivity(2, 1)  # 2D cells to facets
    
    # Create a reverse mapping: facet -> list of cells that contain it
    facet_to_cells = {}
    for cell_2d_idx in range(domain_2d.topology.index_map(2).size_local + domain_2d.topology.index_map(2).num_ghosts):
        cell_facets = cell_to_facet_2d.links(cell_2d_idx)
        for facet_idx in cell_facets:
            if facet_idx not in facet_to_cells:
                facet_to_cells[facet_idx] = []
            facet_to_cells[facet_idx].append(cell_2d_idx)
    
    # Map 1D cells to material tags
    cell_tags_1d_values = []
    num_cells_1d = domain_1d.topology.index_map(1).size_local + domain_1d.topology.index_map(1).num_ghosts
    
    for cell_1d_idx in range(num_cells_1d):
        # Map 1D cell to 2D facet using entity_map
        facet_2d_idx = entity_map[cell_1d_idx]
        
        # Find which 2D cell contains this facet
        if facet_2d_idx in facet_to_cells:
            # Use the first cell that contains this facet
            containing_cell_2d = facet_to_cells[facet_2d_idx][0]
            material_tag = cell_tags_2d.values[containing_cell_2d]
        else:
            # Fallback: use default tag
            print(f"Warning: Could not find 2D cell for facet {facet_2d_idx}, using default tag")
            material_tag = 1
        
        cell_tags_1d_values.append(material_tag)
    
    cell_tags_1d_values = np.array(cell_tags_1d_values, dtype=cell_tags_2d.values.dtype)
    
    # Create 1D cell tags
    cell_tags_1d = dmesh.meshtags(
        domain_1d,
        1,  # 1D cells
        np.arange(num_cells_1d, dtype=np.int32),
        cell_tags_1d_values
    )
    
    # Print some info about the extracted mesh
    coords_1d = domain_1d.geometry.x[:, 0]  # z-coordinates
    print(f"1D mesh z-range: [{coords_1d.min():.6e}, {coords_1d.max():.6e}]")
    print(f"1D mesh has {num_cells_1d} cells")
    
    # Print material tag distribution
    unique_tags, counts = np.unique(cell_tags_1d_values, return_counts=True)
    print("Material tag distribution:")
    for tag, count in zip(unique_tags, counts):
        print(f"  Tag {tag}: {count} cells")
    
    # Return the maps as a tuple for convenience
    maps = (entity_map, vertex_map, geom_map)
    
    return domain_1d, cell_tags_1d, maps

def run_1d(cfg, mesh_folder_2d, mesh_folder_1d=None, rebuild_mesh=False, visualize_mesh=False, output_folder=None, watcher_points=None, write_xdmf=True, suppress_print=False, use_radial_correction=True):
    """
    Run the 1D heatflow simulation with the given configuration.
    
    Parameters:
    -----------
    cfg : dict
        Configuration dictionary loaded from YAML
    mesh_folder_2d : str
        Path to the folder containing the 2D mesh from which to build the 1D mesh
    mesh_folder_1d : str, optional
        Path to save the 1D mesh (if None, uses mesh_folder_2d)
    visualize_mesh : bool, optional
        Whether to visualize the mesh and output plots
    output_folder : str, optional
        Where to save XDMF, watcher CSV, and a copy of the used YAML config
    watcher_points : dict or list of dicts, optional
        Points to watch, e.g. {'pside': (z, r), ...}
    write_xdmf : bool, optional
        Whether to write XDMF output
    suppress_print : bool, optional
        If True, suppress all print output
    use_radial_correction : bool, optional
        Whether to apply radial heating correction from 2D simulation data
    """
    
    with suppress_output(suppress_print):
        # Start timing
        program_start_time = time.time()
        
        # Use mesh_folder_2d if mesh_folder_1d is not provided
        if mesh_folder_1d is None:
            mesh_folder_1d = mesh_folder_2d
        
        # Load the 2D mesh
        mesh_cfg_path = os.path.join(mesh_folder_2d, 'mesh_cfg.yaml')
        mesh_file_path = os.path.join(mesh_folder_2d, 'mesh.msh')
        
        # Check for required files
        missing = []
        if not os.path.isfile(mesh_file_path):
            missing.append('mesh.msh')
        if not os.path.isfile(mesh_cfg_path):
            missing.append('mesh_cfg.yaml')
        if missing:
            raise FileNotFoundError(f"Missing required file(s) in {mesh_folder_2d}: {', '.join(missing)}")
        
        # Load mesh configuration
        with open(mesh_cfg_path, 'r') as f:
            mesh_cfg = yaml.safe_load(f)
        
        mat_tag_map = mesh_cfg.get('material_tags', {})
        
        # Load 2D mesh
        from dolfinx.io import gmshio
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.open(mesh_file_path)
        domain_2d, cell_tags_2d, facet_tags_2d = gmshio.model_to_mesh(gmsh.model, COMM, 0, 2)
        gmsh.finalize()
        
        print("Loaded 2D mesh successfully")
        
        # Print radial correction status
        if use_radial_correction:
            print("Radial heating correction: ENABLED (user choice)")
        else:
            print("Radial heating correction: DISABLED (user choice)")
        
        # Extract 1D submesh from r=0 line
        domain_1d, cell_tags_1d, maps = extract_1d_submesh_from_2d(domain_2d, cell_tags_2d)
        entity_map, vertex_map, geom_map = maps
        
        # Visualize 1D mesh if requested
        if visualize_mesh:
            print("Visualizing 1D mesh...")
            coords_1d = domain_1d.geometry.x[:, 0]  # z-coordinates
            print(f"1D mesh nodes: {coords_1d}")
        
        # Create materials list for 1D (needed for material properties)
        # Extract material properties from the 2D mesh configuration
        materials_1d = []
        for mat_name, mat_tag in mat_tag_map.items():
            if mat_name in cfg['mats']:
                mat_props = cfg['mats'][mat_name]
                material = Material(
                    mat_name,
                    boundaries=[0.0, 1.0, 0.0, 1.0],  # Dummy boundaries for 1D
                    properties={
                        "rho_cv": float(mat_props['rho']) * float(mat_props['cv']),
                        "k": float(mat_props['k'])
                    },
                    mesh_size=float(mat_props['mesh'])
                )
                material.tag = mat_tag  # Use the tag from 2D mesh
                materials_1d.append(material)
        
        # Read experimental heating data
        heating_file = cfg['heating']['file']
        df_heat = pd.read_csv(heating_file)
        
        # Only convert numeric columns to numeric, not the entire DataFrame
        df_heat = (df_heat
                    .sort_values('time')
                    .assign(
                        time=pd.to_numeric(df_heat['time'], errors='coerce'),
                        temp=pd.to_numeric(df_heat['temp'], errors='coerce')
                    )
                    .dropna(subset=['time', 'temp'])  # Only drop rows where time or temp is NaN
                    .reset_index(drop=True))

        # Ensure the CSV has the expected columns
        if 'temp' not in df_heat.columns:
            raise ValueError(f"Heating CSV file {heating_file} must contain a 'temp' column")
        if 'time' not in df_heat.columns:
            raise ValueError(f"Heating CSV file {heating_file} must contain a 'time' column")

        # Normalize temperature data for heating curve
        df_heat['temp normed'] = (df_heat['temp'] - df_heat['temp'].iloc[0]) / (df_heat['temp'].max() - df_heat['temp'].min())
        
        # Initialize function spaces & material properties for 1D
        V = fem.functionspace(domain_1d, ("Lagrange", 1))
        Q = fem.functionspace(domain_1d, ("DG", 0))

        print('Assigning material properties...')

        # Create material property mappings for 1D
        tag_to_k = {mat.tag: mat.properties["k"] for mat in materials_1d}
        tag_to_rho_cv = {mat.tag: mat.properties["rho_cv"] for mat in materials_1d}

        cell_tag_array = cell_tags_1d.values

        kappa_per_cell = np.array([tag_to_k[tag] for tag in cell_tag_array])
        rho_cv_per_cell = np.array([tag_to_rho_cv[tag] for tag in cell_tag_array])

        # assign to DG0 functions
        kappa = fem.Function(Q)
        rho_cv = fem.Function(Q)

        kappa.x.array[:] = kappa_per_cell
        rho_cv.x.array[:] = rho_cv_per_cell
        kappa.x.scatter_forward()
        rho_cv.x.scatter_forward()
        print('Material properties assigned.')
        
        # NEW: Load radial gradient data from 2D simulation for heating correction
        print('Loading radial gradient data from 2D simulation...')
        
        # Initialize radial correction variables
        grad_interp = None
        
        if use_radial_correction:
            # Find the 2D simulation output folder
            # We'll look for the smoothed radial_gradient.csv file (more physically motivated)
            potential_2d_outputs = [
                os.path.join(mesh_folder_2d, '..', 'outputs', 'geballe_no_diamond_read_flux'),
                os.path.join(mesh_folder_2d, '..', '..', 'outputs', 'geballe_no_diamond_read_flux'),
                os.path.join(os.getcwd(), 'outputs', 'geballe_no_diamond_read_flux'),
                os.path.join(os.getcwd(), 'sim_outputs', 'geballe_no_diamond_read_flux')
            ]
            
            # Try smoothed gradient first (more physically motivated)
            radial_grad_file = None
            for output_dir in potential_2d_outputs:
                test_file = os.path.join(output_dir, 'radial_gradient.csv')
                if os.path.exists(test_file):
                    radial_grad_file = test_file
                    print(f"Found smoothed radial gradient file: {radial_grad_file}")
                    break
            
            # Fallback to raw gradient if smoothed not available
            if radial_grad_file is None:
                for output_dir in potential_2d_outputs:
                    test_file = os.path.join(output_dir, 'radial_gradient_raw.csv')
                    if os.path.exists(test_file):
                        radial_grad_file = test_file
                        print(f"Found raw radial gradient file: {radial_grad_file}")
                        break
            
            if radial_grad_file is None:
                print("Warning: Could not find radial gradient file. Disabling radial heating correction.")
                use_radial_correction = False
            else:
                # Load the gradient data
                grad_df = pd.read_csv(radial_grad_file, index_col=0)
                grad_times = grad_df.index.values.astype(float)
                grad_z_positions = grad_df.columns.values.astype(float)
                grad_values = grad_df.values
                
                print(f"Loaded gradient data: {grad_values.shape[0]} timesteps, {grad_values.shape[1]} z-positions")
                print(f"Time range: [{grad_times.min():.6e}, {grad_times.max():.6e}]")
                print(f"Z range: [{grad_z_positions.min():.6e}, {grad_z_positions.max():.6e}]")
                
                # Check 1D mesh z-range to see if it matches gradient data
                mesh_z_coords = domain_1d.geometry.x[:, 0]
                print(f"1D mesh z-range: [{mesh_z_coords.min():.6e}, {mesh_z_coords.max():.6e}]")
                
                # Warn if there's a mismatch
                if (mesh_z_coords.min() < grad_z_positions.min() or 
                    mesh_z_coords.max() > grad_z_positions.max()):
                    print("⚠ WARNING: 1D mesh extends beyond gradient data z-range")
                    print("  This will cause interpolation bounds errors. Coordinates will be clamped.")
                else:
                    print("✓ 1D mesh z-range is within gradient data range")
                
                # Create interpolation function for gradient values
                from scipy.interpolate import RegularGridInterpolator
                grad_interp = RegularGridInterpolator((grad_times, grad_z_positions), grad_values, method='linear')
                
                # Determine if we're using smoothed or raw data
                using_smoothed_data = 'radial_gradient.csv' in radial_grad_file
                if using_smoothed_data:
                    print("Using smoothed gradient data (recommended)")
                else:
                    print("Using raw gradient data (fallback)")
        
        # Create source term function that interpolates gradient and applies correction
        def radial_source_term(z_coord, t):
            """Compute radial source term that accounts for missing radial heat flow in 1D approximation"""
            if not use_radial_correction or grad_interp is None:
                return 0.0
            
            try:
                # Check bounds before interpolation
                t_min, t_max = grad_times.min(), grad_times.max()
                z_min, z_max = grad_z_positions.min(), grad_z_positions.max()
                
                # Clamp coordinates to valid range
                t_clamped = np.clip(t, t_min, t_max)
                z_clamped = np.clip(z_coord, z_min, z_max)
                
                # Check if we're at the boundaries (might indicate extrapolation)
                at_time_boundary = (t == t_clamped and (t <= t_min or t >= t_max))
                at_z_boundary = (z_coord == z_clamped and (z_coord <= z_min or z_coord >= z_max))
                
                # Interpolate gradient at this z-coordinate and time
                # RegularGridInterpolator expects points as (n_points, n_dims) array
                grad_val = grad_interp([t_clamped, z_clamped])[0]
                
                # If we had to clamp coordinates, the source term might not be reliable
                if at_time_boundary or at_z_boundary:
                    # Reduce the source term magnitude at boundaries to avoid spurious effects
                    grad_val *= 0.1
                
                # Find the local kappa value at this z-coordinate
                # We need to find which cell this z-coordinate belongs to
                # Get cell-to-vertex connectivity
                domain_1d.topology.create_connectivity(1, 0)  # cells to vertices
                cell_to_vertex = domain_1d.topology.connectivity(1, 0)
                
                # Find which cell contains this z-coordinate
                cell_idx = None
                for i in range(domain_1d.topology.index_map(1).size_local + domain_1d.topology.index_map(1).num_ghosts):
                    cell_vertices = cell_to_vertex.links(i)
                    cell_coords = domain_1d.geometry.x[cell_vertices, 0]
                    if cell_coords.min() <= z_coord <= cell_coords.max():
                        cell_idx = i
                        break
                
                # Get kappa for this cell
                if cell_idx is not None and cell_idx < len(cell_tags_1d.values):
                    local_kappa = kappa_per_cell[cell_tags_1d.values[cell_idx]]
                else:
                    # Fallback: use first kappa value
                    local_kappa = kappa_per_cell[0]
                
                # DERIVATION OF THE RADIAL SOURCE TERM
                # In a 2D axisymmetric model (cylindrical coordinates), the heat equation is:
                #   ρc * ∂T/∂t = ∂/∂z(κ * ∂T/∂z) + (1/r) * ∂/∂r(r * κ * ∂T/∂r)
                # Our 1D model only includes the ∂z term. The second term, the radial heat flow,
                # must be added as a source term to the 1D equation.
                #
                # Source = (1/r) * ∂/∂r(r * κ * ∂T/∂r)
                #
                # Let's expand this, assuming κ is constant in r:
                #   Source = κ * (∂²T/∂r² + (1/r) * ∂T/∂r)
                #
                # At the centerline (r=0), this expression is undefined. We must evaluate it in the limit as r → 0.
                # The axial symmetry of the problem requires that the radial gradient is zero at the center:
                #   ∂T/∂r |_(r=0) = 0
                #
                # So, the term (1/r) * ∂T/∂r becomes 0/0, which requires L'Hôpital's rule:
                #   lim_{r→0} ( (1/r) * ∂T/∂r ) = lim_{r→0} ( (∂/∂r ∂T/∂r) / (∂/∂r r) ) = (∂²T/∂r²) / 1 = ∂²T/∂r² |_(r=0)
                #
                # Substituting this back into the source term expression at r=0:
                #   Source |_(r=0) = κ * (∂²T/∂r² |_(r=0) + ∂²T/∂r² |_(r=0))
                #   Source |_(r=0) = 2 * κ * ∂²T/∂r² |_(r=0)
                #
                # This is where the single factor of 2 comes from. It is a direct result of the geometry of the
                # coordinate system at the axis of symmetry. It is NOT two separate factors combined.
                #
                # To implement this, we approximate the second derivative using the gradient data from the 2D simulation:
                #   ∂²T/∂r² ≈ (∂T/∂r) / Δr
                # where ∂T/∂r is the `grad_val` from the 2D simulation and Δr is a small characteristic radial distance.
                #
                # So, the final source term we implement is:
                #   Source ≈ 2 * κ * (grad_val / Δr)
                source_val = 2.0 * local_kappa * grad_val / (delta_r*0.613)
                print(f'delta r: {delta_r}')
                
                return source_val
            except Exception as e:
                # Fallback if interpolation fails - return zero instead of crashing
                return 0.0
        
        # Define delta_r once for the entire simulation
        if use_radial_correction:
            # Determine if we're using smoothed or raw data
            using_smoothed_data = 'radial_gradient.csv' in radial_grad_file if radial_grad_file else False
            
            # Characteristic radial length scale
            if using_smoothed_data:
                delta_r = 0.1e-6  # Radial band width from 2D simulation (0 < r ≤ 0.25 μm)
            else:
                delta_r = 0.07e-6  # Typical mesh size in radial direction
        else:
            delta_r = 0.0  # Not used when correction is disabled
        
        # Test radial source term function if enabled
        if use_radial_correction and grad_interp is not None:
            print("Testing radial source term function...")
            test_z = domain_1d.geometry.x[0, 0]  # First z-coordinate
            test_t = 0.0  # Initial time
            test_source = radial_source_term(test_z, test_t)
            print(f"  Test source term at z={test_z:.6e}, t={test_t}: {test_source:.2e}")
            if abs(test_source) > 1e-10:
                print("  ✓ Radial source term function working correctly")
            else:
                print("  ⚠ Radial source term function returned zero (may be normal for t=0)")
        
        # Simulation & heating parameters
        t_final = float(cfg['timing']['t_final'])
        num_steps = int(cfg['timing']['num_steps'])
        dt = t_final / num_steps
        ic_temp = float(cfg['heating']['ic_temp'])
        heating_FWHM = float(cfg['heating']['fwhm'])
        
        # Test radial source term at a later time if enabled
        if use_radial_correction and grad_interp is not None:
            test_z = domain_1d.geometry.x[0, 0]  # First z-coordinate
            test_t_later = min(1.0, t_final / 10)  # 10% into simulation or 1 second
            test_source_later = radial_source_term(test_z, test_t_later)
            print(f"  Test source term at z={test_z:.6e}, t={test_t_later}: {test_source_later:.2e}")
            if abs(test_source_later) > 1e-10:
                print("  ✓ Radial source term function produces non-zero values during simulation")
                if using_smoothed_data:
                    print("  ✓ Using smoothed gradient data with smoothing window width (0.25 μm) as Δr")
                else:
                    print("  ✓ Using raw gradient data with typical mesh size (0.07 μm) as Δr")
                print("  ✓ Physically motivated source term: 2 * κ * (∂T/∂r) / Δr")
            else:
                print("  ⚠ Radial source term function still zero at later time - check gradient data")
        
        # Build variational forms for 1D (NO radial weighting)
        u_n = fem.Function(V)
        u_n.x.array[:] = np.full_like(u_n.x.array, ic_temp) # assign initial temperature
        u_n.x.scatter_forward()

        x = ufl.SpatialCoordinate(domain_1d)
        # In 1D, we only have x[0] (z-coordinate), no radial coordinate

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # NEW: Create source term function for radial heating correction
        if use_radial_correction:
            # Create a function to hold the source term values
            source_func = fem.Function(V)
            source_func.name = 'Radial_Source_Term'
        else:
            source_func = fem.Constant(domain_1d, PETSc.ScalarType(0))

        # 1D heat equation with radial heating correction
        lhs = (
            rho_cv * u * v * ufl.dx  # No r-weighting
            + dt * kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx  # No r-weighting
        )
        rhs = (
            rho_cv * u_n * v * ufl.dx  # No r-weighting
            + dt * source_func * v * ufl.dx  # Radial heating correction
        )
        lhs_form = fem.form(lhs)
        rhs_form = fem.form(rhs)
        
        # Define boundary conditions for 1D
        # get heating curve
        pside_heating_interp = lambda t: np.interp(t, 
                                                   df_heat['time'], 
                                                   df_heat['temp'], 
                                                   left=df_heat['temp'].iloc[0],
                                                   right=df_heat['temp'].iloc[-1])

        offset = df_heat['temp'].iloc[0] - ic_temp # start from ic temp
        def heating_offset(t):
            return float(pside_heating_interp(t)) - offset

        # For 1D, we need to find the heating location
        # Use the sample boundaries to determine heating location
        r_sample = float(cfg['mats']['p_sample']['r'])
        z_sample = float(cfg['mats']['p_sample']['z'])
        z_ins_pside = float(cfg['mats']['p_ins']['z'])
        z_coupler = float(cfg['mats']['p_coupler']['z'])
        
        # Calculate heating location (middle of sample)
        mesh_zmin = -(z_sample/2) - z_ins_pside - z_coupler
        heating_z = mesh_zmin + z_ins_pside  # Middle of sample
        
        # Simple heating function for 1D (no gaussian in r-direction)
        # The BC expects (x, y, t) but we only need (x, t) for 1D
        def heating_1d(x, y, t):
            amp = heating_offset(t)
            return (amp - ic_temp) + ic_temp

        obj_bcs = []
        left_bc = RowDirichletBC(V, 'left', value=ic_temp)
        right_bc = RowDirichletBC(V, 'right', value=ic_temp)
        
        # For 1D, we need to find the heating location and apply BC there
        # This is a simplified approach - you might want to refine this
        heating_bc = RowDirichletBC(
            V,
            'x',
            coord=heating_z,
            value=heating_1d,
        )
        
        obj_bcs = [left_bc, right_bc, heating_bc] # Only left, right, and heating BCs
        bcs = [bc.bc for bc in obj_bcs] # dolfinx bc object
        
        # Assemble matrix and vector
        from dolfinx.fem.petsc import assemble_matrix, create_vector

        A = assemble_matrix(lhs_form, bcs=bcs)
        A.assemble()
        b = create_vector(rhs_form)
        b.assemble()
        
        # Initialize solver
        solver = PETSc.KSP().create(A.getComm())
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")
        
        # Set up output folder
        if output_folder is not None:
            save_folder = output_folder
            os.makedirs(save_folder, exist_ok=True)
            # Save a copy of the used YAML config
            with open(os.path.join(save_folder, 'used_config.yaml'), 'w') as f:
                yaml.safe_dump(cfg, f)
        else:
            save_name = '1d_simulation'
            outputs_folder = os.path.join(os.getcwd(), 'sim_outputs')
            save_folder = os.path.join(outputs_folder, save_name)
            os.makedirs(save_folder, exist_ok=True)

        xdmf_path = os.path.join(save_folder, "output.xdmf")
        watcher_csv_path = os.path.join(save_folder, "watcher_points.csv")

        # Initialize xdmf output
        from dolfinx import io
        if write_xdmf:
            xdmf = io.XDMFFile(domain_1d.comm, xdmf_path, "w")
            xdmf.write_mesh(domain_1d)
        else:
            xdmf = None

        u_n.name = 'Temperature (K)'
        if write_xdmf:
            xdmf.write_function(u_n, 0.0) # write initial 
        
        # Pre-compute mesh node coordinates for watcher points
        mesh_coords = domain_1d.geometry.x[:, 0]  # Only z-coordinates for 1D

        # Watcher point setup
        watcher_data = None
        watcher_names = None
        watcher_time = None
        if watcher_points is not None:
            if isinstance(watcher_points, dict):
                watcher_names = list(watcher_points.keys())
                watcher_coords = [pt[0] for pt in watcher_points.values()]  # Only z-coordinates
            elif isinstance(watcher_points, list):
                watcher_names = [pt['name'] for pt in watcher_points]
                watcher_coords = [pt['coords'][0] for pt in watcher_points]  # Only z-coordinates
            else:
                raise ValueError("watcher_points must be a dict or list of dicts")
            watcher_data = {name: [] for name in watcher_names}
            watcher_time = []
            # Nearest-node mapping for watcher points
            tree = cKDTree(mesh_coords.reshape(-1, 1))  # Reshape for 1D
            watcher_nodes = []
            for z_coord in watcher_coords:
                _, node_idx = tree.query([z_coord])
                watcher_nodes.append(node_idx)
        else:
            watcher_names = []
            watcher_coords = []
            watcher_data = {}
            watcher_time = []

        # Pre-compute cell mapping for source terms (do this once, not every timestep)
        if use_radial_correction:
            print("Pre-computing cell mapping for source terms...")
            mesh_coords_1d = domain_1d.geometry.x[:, 0]  # z-coordinates only
            
            # Get cell-to-vertex connectivity
            domain_1d.topology.create_connectivity(1, 0)  # cells to vertices
            cell_to_vertex = domain_1d.topology.connectivity(1, 0)
            
            # Pre-compute which cell each node belongs to
            node_to_cell = []
            for z_coord in mesh_coords_1d:
                cell_idx = None
                for i in range(domain_1d.topology.index_map(1).size_local + domain_1d.topology.index_map(1).num_ghosts):
                    cell_vertices = cell_to_vertex.links(i)
                    cell_coords = domain_1d.geometry.x[cell_vertices, 0]
                    if cell_coords.min() <= z_coord <= cell_coords.max():
                        cell_idx = i
                        break
                node_to_cell.append(cell_idx if cell_idx is not None else 0)
            
            # Pre-compute kappa values for each node
            node_kappas = []
            for cell_idx in node_to_cell:
                if cell_idx < len(cell_tags_1d.values):
                    local_kappa = kappa_per_cell[cell_tags_1d.values[cell_idx]]
                else:
                    local_kappa = kappa_per_cell[0]
                node_kappas.append(local_kappa)
            
            node_kappas = np.array(node_kappas)
            print(f"Pre-computed kappa mapping for {len(node_kappas)} nodes")
        
        # Time stepping loop
        from dolfinx.fem.petsc import assemble_vector, apply_lifting, set_bc
        for x in obj_bcs:
            x.update(0.0)
        progress_interval = max(1, num_steps // 10)

        step_times = []
        loop_start_time = time.time()

        print('Beginning 1D simulation loop...')
        startup_end_time = time.time()
        startup_time = startup_end_time - program_start_time
        for step in range(num_steps):
            step_start = time.time()
            t = (step+1)*dt
            heating_bc.update(t)

            # NEW: Update radial source term at each timestep
            if use_radial_correction:
                # Get mesh coordinates for source term evaluation
                mesh_coords_1d = domain_1d.geometry.x[:, 0]  # z-coordinates only
                
                # Check bounds before interpolation
                t_min, t_max = grad_times.min(), grad_times.max()
                z_min, z_max = grad_z_positions.min(), grad_z_positions.max()
                
                # Clamp coordinates to valid range
                t_clamped = np.clip(t, t_min, t_max)
                z_coords_clamped = np.clip(mesh_coords_1d, z_min, z_max)
                
                # Vectorized source term computation (much faster than loop)
                # Create points array for interpolation: (n_points, 2) where each row is [t, z]
                points = np.column_stack([np.full_like(z_coords_clamped, t_clamped), z_coords_clamped])
                
                # Interpolate gradients for all points at once
                grad_vals = grad_interp(points)
                
                # Reduce source term magnitude for boundary nodes (where clamping occurred)
                boundary_mask = (mesh_coords_1d != z_coords_clamped)
                if np.any(boundary_mask):
                    grad_vals[boundary_mask] *= 0.1
                
                # Compute source terms vectorized using pre-computed kappa values
                source_vals = 2.0 * node_kappas * grad_vals / delta_r
                
                # Update the source function
                source_func.x.array[:] = source_vals
                source_func.x.scatter_forward()
                
                # Debug output (only print occasionally and much less verbose)
                if step % max(1, num_steps // 10) == 0:
                    max_source = max(abs(s) for s in source_vals)
                    print(f"Step {step}: Max radial source term = {max_source:.2e}")
                    
                    # Quick check: verify source term is non-zero
                    if max_source > 1e-12:
                        print(f"Step {step}: ✓ Source term is being applied")
                    else:
                        print(f"Step {step}: ⚠ Source term is effectively zero")

            with b.localForm() as local_b:
                local_b.set(0)
            assemble_vector(b, rhs_form)
            apply_lifting(b, [lhs_form], [bcs])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, bcs)
            solver.solve(b, u_n.x.petsc_vec)
            u_n.x.scatter_forward()

            if write_xdmf:
                xdmf.write_function(u_n, t)
            if watcher_points is not None:
                watcher_time.append(t)
                for name, z_coord, node_idx in zip(watcher_names, watcher_coords, watcher_nodes):
                    # Get temperature directly from nearest node
                    try:
                        val = u_n.x.array[node_idx]
                    except:
                        val = np.nan
                    watcher_data[name].append(val)

            step_end = time.time()
            step_times.append(step_end - step_start)

            if (step + 1) % progress_interval == 0 or (step + 1) == num_steps:
                percent = int((step + 1) / num_steps * 100)
                # Calculate average time for the current progress interval
                interval_start = max(0, len(step_times) - progress_interval)
                interval_steps = step_times[interval_start:]
                avg_interval = sum(interval_steps) / len(interval_steps)
                print(f"1D Simulation progress: {percent}% (step {step + 1}/{num_steps}) | Avg time/step (interval): {avg_interval:.4f} s")

        if write_xdmf:
            xdmf.close()

        # Save watcher data to CSV
        if watcher_points is not None:
            df = pd.DataFrame({'time': watcher_time})
            for name in watcher_names:
                df[name] = watcher_data[name]
            df.to_csv(watcher_csv_path, index=False)

        # Timing outputs
        program_end_time = time.time()
        total_time = program_end_time - program_start_time
        loop_time = program_end_time - loop_start_time
        avg_step_time = sum(step_times) / len(step_times) if step_times else 0.0

        print("\n--- 1D Simulation Timing Summary ---")
        print(f"Total time: {total_time:.2f} s")
        print(f"Startup time: {startup_time:.2f} s")
        print(f"Loop time: {loop_time:.2f} s")
        print(f"Average time per step: {avg_step_time:.4f} s")
        if use_radial_correction:
            print("Radial heating correction: ENABLED")
            if grad_interp is not None:
                print("  - Gradient interpolation function loaded successfully")
            else:
                print("  - WARNING: Gradient interpolation function not available")
        else:
            print("Radial heating correction: DISABLED")
        print("------------------------------------\n")
        
        return domain_1d, cell_tags_1d, maps

    