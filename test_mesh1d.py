import yaml
import matplotlib.pyplot as plt
from mesh_and_materials.mesh import Mesh1D
from mesh_and_materials.materials import Material
import gmsh
import numpy as np

# Load config
yaml_path = 'cfgs/geballe_no_diamond_read_flux.yaml'
with open(yaml_path, 'r') as f:
    cfg = yaml.safe_load(f)

# Build material list for 1D (zmin, zmax)
# Use explicit stacking order to match 2D mesh
order = ["p_ins", "p_coupler", "p_sample", "o_coupler", "o_ins"]
zmin = 0.0
mats = []
for name in order:
    mat = cfg['mats'][name]
    dz = float(mat['z'])
    zmax = zmin + dz
    mats.append(Material(name, [zmin, zmax, 0.0, 1.0], properties=mat, mesh_size=float(mat['mesh'])))
    zmin = zmax

zmin = mats[0].boundaries[0]
zmax = mats[-1].boundaries[1]

# Build mesh
mesh1d = Mesh1D('test1d', [zmin, zmax], mats)
mesh1d.build_mesh()

# Open in gmsh GUI
print("Opening mesh in GMSH GUI. Close the window to continue.")
gmsh.fltk.initialize()
gmsh.fltk.run()

# After gmsh.finalize(), convert to dolfinx mesh and inspect nodes
mesh, cell_tags, facet_tags = mesh1d.to_dolfinx()
coords = mesh.geometry.x[:, 0]  # 1D node positions

print(f"Number of mesh nodes: {len(coords)}")
print("Node positions (z):")
print(coords)

# Optionally, plot the nodes as points
plt.figure(figsize=(8, 2))
plt.plot(coords, np.zeros_like(coords), 'o', label='FEM nodes')
plt.xlabel('z (m)')
plt.yticks([])
plt.title('FEM Node Positions in 1D Mesh')
plt.legend()
plt.tight_layout()
plt.show()
