# Codebase Summary

Generated 2025-06-15T16:06:20.220054

## analysis_and_visualization/__init__.py


## analysis_and_visualization/plotting.py

* **Imports:**
  - import os
  - import numpy as np
  - import matplotlib.pyplot as plt
  - from matplotlib import colors
  - import ipywidgets as widgets
  - from io_utilities.xdmf_extract import extract_point_timeseries_xdmf
* **Functions:**
  - interactive_line_plot(xdmf_path, axis='x', coord=0.0, line_range, samples=200, function_name='Temperature (K)') - Interactively plot solution values along a line at different times.
  - line_difference_colormap(xdmf_a, xdmf_b, axis='x', coord=0.0, line_range, samples=200, function_name='Temperature (K)', label_a='sim A', label_b='sim B') - Plot temperature differences between two simulations along a line.

## dirichlet_bc/bc.py

* **Imports:**
  - import numpy as np
  - from dolfinx import fem
  - from petsc4py import PETSc
* **Classes:**
  - RowDirichletBC - Dirichlet BC along one edge/line, optionally clipped to a centred segment.

## io_utilities/xdmf_extract.py

* **Imports:**
  - import meshio
  - import numpy as np
  - from scipy.spatial import cKDTree
  - from scipy.interpolate import griddata
* **Functions:**
  - extract_point_timeseries_xdmf(xdmf_path, function_name, query_points, method='nearest') - Extract time-series of a nodal field from an XDMF time-series file using meshio.

## io_utilities/xdmf_utils.py

* **Imports:**
  - import os
  - from dolfinx import io
* **Functions:**
  - init_xdmf(domain, sim_folder, output_name) - Initialize an XDMF file for time-dependent output.
  - save_params(sim_folder, params_dict) - Save simulation parameters to a text file for reproducibility.

## mesh_and_materials/__init__.py


## mesh_and_materials/materials.py

* **Classes:**
  - Material - Representation of a rectangular material region in the mesh.

## mesh_and_materials/mesh.py

* **Imports:**
  - import gmsh
* **Classes:**
  - Mesh - Container for a gmsh mesh over a rectangular domain with multiple materials.
* **Globals:**
  - SCALE = 1000000.0

## mesh_and_materials/mesh_utils.py

* **Imports:**
  - import gmsh
* **Classes:**
  - Material - Representation of a rectangular material region in the mesh.
  - Mesh - Container for a gmsh mesh over a rectangular domain with multiple materials.

## repo_summary.py

* **Imports:**
  - import os
  - import ast
  - import sys
  - from datetime import datetime
* **Functions:**
  - is_excluded(path)
  - get_expr(expr)
  - format_signature(args)
  - parse_file(path)
  - walk_repo(root)
  - write_summary(summary, out_path)
  - locate_symbol(name) - Return (filepath, line) of symbol definition if known.
* **Globals:**
  - EXCLUDE_DIRS = {'.git', 'venv', '__pycache__'}

## simulate/run_sim.py


## space/space_and_forms.py

* **Imports:**
  - import numpy as np
  - import ufl
  - from dolfinx import fem
  - from petsc4py import PETSc
* **Classes:**
  - Space - High-level wrapper for FEniCSx heat-equation workspaces.

