# Main simulation script scaffold for heatflow
# Fill in each section with your own code as you develop the simulation

# Imports ---------------------------------------------------------------------
import gmsh
from dolfinx import fem
import pandas as pd
import numpy as np
import os

from mesh_and_materials.mesh import *
from mesh_and_materials.materials import *
from space.space_and_forms import *
from dirichlet_bc.bc import *
# -----------------------------------------------------------------------------


# Create mesh and material boundaries ------------------------------------------
# r_ prefix corresponds to width in the r (radial), z_ prefix corresponds to width in the z (axial) direction
r_sample    = 20e-6 
r_gasket    = 75e-6 
r_ins_gside = 5e-6
r_diamond   = r_sample + r_gasket + r_ins_gside # diamond covers full z-extent

r_ins_oside = r_sample
r_ins_pside = r_sample
r_coupler   = r_sample

z_ins_oside = 6.3e-6
z_ins_pside = 3.2e-6
z_sample    = 1.84e-6
z_coupler   = 0.062e-6
z_diam      = 40e-6
z_gasket    = z_sample + z_ins_pside + z_ins_oside + 2*z_coupler

# Derive mesh boundaries (pside is negative z, oside is positive z)
# Bottom neumann boundary is at r=0, top neumann boundary is at z=r_diamond
mesh_zmin = -(z_sample/2) - z_ins_pside - z_coupler - z_diam
mesh_zmax = (z_sample/2) + z_ins_oside + z_coupler + z_diam
mesh_rmin = 0.0
mesh_rmax = r_diamond

# build material boundary lists as [zmin, zmax, rmin, rmax]
bnd_p_diam      = [mesh_zmin, mesh_zmin + z_diam, mesh_rmin, mesh_rmax]
bnd_o_diam      = [mesh_zmax-z_diam, mesh_zmax, mesh_rmin, mesh_rmax]

bnd_p_ins       = [bnd_p_diam[1], bnd_p_diam[1] + z_ins_pside, mesh_rmin, mesh_rmin + r_ins_pside]
bnd_o_ins       = [bnd_o_diam[0]-z_ins_oside, bnd_o_diam[0], mesh_rmin, mesh_rmin + r_ins_oside]

bnd_p_coupler   = [bnd_p_ins[1], bnd_p_ins[1] + z_coupler, mesh_rmin, mesh_rmin + r_coupler]
bnd_o_coupler   = [bnd_o_ins[0]-z_coupler, bnd_o_ins[0], mesh_rmin, mesh_rmin + r_coupler]

bnd_sample      = [bnd_p_coupler[1], bnd_p_coupler[1] + z_sample, mesh_rmin, mesh_rmin + r_sample]

bnd_g_ins       = [bnd_p_diam[1], bnd_o_diam[0], mesh_rmin + r_sample, mesh_rmin + r_sample + r_ins_gside]
bnd_gasket      = [bnd_p_diam[1], bnd_o_diam[0], bnd_g_ins[3], mesh_rmax]


# init materials
p_diam      = Material(
                "p_diam",
                boundaries=bnd_p_diam,
                properties={"rho_cv": 3500 * 510, "k": 2000},
                mesh_size=1e-6
)
p_ins       = Material(
                "p_ins",
                boundaries=bnd_p_ins,
                properties={"rho_cv": 4131 * 668, "k": 10},
                mesh_size=0.1e-6
)
p_coupler   = Material(
                "p_coupler",
                boundaries=bnd_p_coupler,
                properties={"rho_cv": 26504 * 130, "k": 352},
                mesh_size=0.02e-6
)
p_sample    = Material(
                "p_sample",
                boundaries=bnd_sample,
                properties={"rho_cv": 5164 * 1158, "k": 3.8},
                mesh_size=0.08e-6
)
o_coupler   = Material(
                "o_coupler",
                boundaries=bnd_o_coupler,
                properties={"rho_cv": 26504 * 130, "k": 352},
                mesh_size=0.02e-6
)
o_ins       = Material(
                "o_ins",
                boundaries=bnd_o_ins,
                properties={"rho_cv": 4131 * 668, "k": 10},
                mesh_size=0.1e-6
)
o_diam      = Material(
                "o_diam",
                boundaries=bnd_o_diam,
                properties={"rho_cv": 3500 * 510, "k": 2000},
                mesh_size=1e-6
)
gasket      = Material(
                "gasket",
                boundaries=bnd_gasket,
                properties={"rho_cv": 21000 * 140, "k": 100},
                mesh_size=1e-6
)
g_ins       = Material(
                "g_ins",
                boundaries=bnd_g_ins,
                properties={"rho_cv": 4131 * 668, "k": 10},
                mesh_size=0.1e-6
)

materials = [p_diam, p_ins, p_coupler, p_sample, o_coupler, o_ins, o_diam, gasket, g_ins]

# mesh name
mesh_name = 'with_gask'

# mesh save path
mesh_save_path = os.path.join(os.path.dirname(__file__), 'meshes', mesh_name + '.msh')

# build mesh
gmsh_domain = Mesh(
    name=mesh_name,
    boundaries=[mesh_zmin, mesh_zmax, mesh_rmin, mesh_rmax],
    materials=materials
)

# build and visualize mesh
gmsh_domain.build_mesh()

gmsh.initialize()
gmsh_domain.write(mesh_save_path)

# visualize mesh
visualize_mesh = True
if visualize_mesh:
    gmsh.initialize()
    gmsh.open(mesh_save_path)
    gmsh.fltk.initialize()
    gmsh.fltk.run()
    gmsh.finalize()

# convert mesh to dolfinx ------------------------------------------------------
from dolfinx.io import gmshio
gmsh.initialize()
gmsh.open(mesh_save_path)
domain, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, COMM, 0, 2)
gmsh.finalize()
# -----------------------------------------------------------------------------

# Read experimental heating data  -----------------------------------------------
# for now, clean data in here. later, however, expect a standardized format
df = pd.read_csv('experimental_data/raw_temp_time_curve.csv')
df_heat = df.copy().reset_index(drop=True)
df_heat.columns = ['time', 'pside temperature', 'oside temperature']

df_heat = (df_heat
            .sort_values('time')
            .apply(pd.to_numeric)
            .dropna()
            .reset_index(drop=True))

df_heat['pside normed'] = (df_heat['pside temperature'] - df_heat['pside temperature'][0]) / (df_heat['pside temperature'].max() - df_heat['pside temperature'].min())
df_heat['oside normed'] = (df_heat['oside temperature'] - df_heat['oside temperature'][0]) / (df_heat['pside temperature'].max() - df_heat['pside temperature'].min())
df_heat['time'] = df_heat['time'] * 10**-6
# -----------------------------------------------------------------------------

# Initialize function spaces ----------------------------------------------------
# and build material property functions ----------------------------------------
V = fem.functionspace(domain, ("Lagrange", 1))
Q = fem.functionspace(domain, ("DG", 0))

print('Assigning material properties...')
mat_names = [mat.name for mat in materials]
mat_tags = [mat.tag for mat in materials]
mat_tag_map = dict(zip(mat_names, mat_tags))

# assumes each Material object has a .tag and .properties dict
tag_to_k = {mat.tag: mat.properties["k"] for mat in materials}
tag_to_rho_cv = {mat.tag: mat.properties["rho_cv"] for mat in materials}

cell_tag_array = cell_tags.values

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
# -----------------------------------------------------------------------------

# Simulation parameters --------------------------------------------------------
t_final = 7e-6
num_steps = 100
dt = t_final / num_steps
ic_temp = 300.0

heating_FWHM = 13.2e-6
# -----------------------------------------------------------------------------

# Build variational forms ------------------------------------------------------
u_n = fem.Function(V)
u_n.x.array[:] = np.full_like(u_n.x.array, ic_temp) # assign initial temperature
u_n.x.scatter_forward()

x = ufl.SpatialCoordinate(domain)
r = x[1] # y-coord is radial direction

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(0))

lhs = (
    rho_cv * u * v *  r * ufl.dx
    + dt * kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) *  r * ufl.dx  
)
rhs = (
    rho_cv * u_n * v *  r * ufl.dx
    + dt * f * v * r *  ufl.dx
)
lhs_form = fem.form(lhs)
rhs_form = fem.form(rhs)
# -----------------------------------------------------------------------------

# Define boundary conditions ---------------------------------------------------
# get heating curve
pside_heating_interp = lambda t: np.interp(t, 
                                           df_heat['time'], 
                                           df_heat['pside temperature'], 
                                           left=df_heat['pside temperature'].iloc[0],
                                           right=df_heat['pside temperature'].iloc[-1])

offset = df_heat['pside temperature'].iloc[0] - ic_temp # start from ic temp
def heating_offset(t):
    return float(pside_heating_interp(t)) - offset

# gaussian profile for inner boundary
coeff = -4.0 * np.log(2.0) / heating_FWHM**2
y_center = 0.0

def gaussian(x, y, t):
    amp = heating_offset(t)
    return (amp - ic_temp) * np.exp(coeff * (y - y_center)**2) + ic_temp

obj_bcs = []
left_bc = RowDirichletBC(V, 'left', value=ic_temp)
right_bc = RowDirichletBC(V, 'right', value=ic_temp)
bottom_bc = RowDirichletBC(V, 'top', value=ic_temp)
inner_bc = RowDirichletBC(
    V,
    'x',
    coord=p_ins.boundaries[0],
    length=abs(r_sample)*2,
    center=0.0,
    value=gaussian,
)
obj_bcs = [left_bc, right_bc, bottom_bc, inner_bc] # custom object
bcs = [bc.bc for bc in obj_bcs] # dolfinx bc object
# -----------------------------------------------------------------------------

# Assemble matrix and vector ---------------------------------------------------
from dolfinx.fem.petsc import assemble_matrix, create_vector

A = assemble_matrix(lhs_form, bcs=bcs)
A.assemble()
b = create_vector(rhs_form)
b.assemble()
# -----------------------------------------------------------------------------

# Initialize solver ------------------------------------------------------------
solver = PETSc.KSP().create(A.getComm())
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
pc = solver.getPC()
pc.setType(PETSc.PC.Type.LU)
pc.setFactorSolverType("mumps")
# -----------------------------------------------------------------------------

# Initalize xmdf output --------------------------------------------------------
from dolfinx import io
save_name = 'refactor_test'
outputs_folder = os.path.join(os.getcwd(), 'sim_outputs')
save_folder = os.path.join(outputs_folder, save_name)
os.makedirs(save_folder, exist_ok=True)

xdmf_path = os.path.join(save_folder, f"{save_name}.xdmf")
xdmf = io.XDMFFile(domain.comm, xdmf_path, "w")
xdmf.write_mesh(domain)

u_n.name = 'Temperature (K)'
xdmf.write_function(u_n, 0.0) # write initial 
# -----------------------------------------------------------------------------

# Time stepping loop -----------------------------------------------------------
from dolfinx.fem.petsc import assemble_vector, apply_lifting, set_bc
for x in obj_bcs:
    x.update(0.0)
progress_interval = max(1, num_steps // 5)

for step in range(num_steps):
    t = (step+1)*dt
    inner_bc.update(t)

    with b.localForm() as local_b:
        local_b.set(0)
    assemble_vector(b, rhs_form)
    apply_lifting(b, [lhs_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)
    solver.solve(b, u_n.x.petsc_vec)
    u_n.x.scatter_forward()

    xdmf.write_function(u_n, t)
    if (step + 1) % progress_interval == 0 or (step + 1) == num_steps:
        percent = int((step + 1) / num_steps * 100)
        print(f"Simulation progress: {percent}% (step {step + 1}/{num_steps})")

xdmf.close()
# -----------------------------------------------------------------------------

visualize_output = True

if visualize_output:

    from io_utilities.xdmf_extract import *
    time, data = extract_point_timeseries_xdmf(
        xdmf_path,
        function_name='Temperature (K)',
        query_points=[(p_ins.boundaries[0], 0), (o_ins.boundaries[1], 0)]
    )
    sim_df = pd.DataFrame({'time': time,
                           'pside': data[0],
                           'oside': data[1]})
    sim_df['normed pside'] = (sim_df['pside'] - sim_df['pside'].iloc[0]) / (sim_df['pside'].max() - sim_df['pside'].min())
    sim_df['normed oside'] = (sim_df['oside'] - sim_df['oside'].iloc[0]) / (sim_df['pside'].max() - sim_df['pside'].min())

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_title('Normed Temperatures with Diamond')
    ax.plot(df_heat['time'], df_heat['pside normed'], label='Geballe pside')
    ax.scatter(df_heat['time'], df_heat['oside normed'], label='Geballe oside')
    ax.plot(sim_df['time'], sim_df['normed pside'], label='sim pside')
    ax.plot(sim_df['time'], sim_df['normed oside'], label='sim oside')
    ax.grid(True, ls = '--', color = 'lightgray')
    ax.legend()
    plt.show()
# -----------------------------------------------------------------------------
