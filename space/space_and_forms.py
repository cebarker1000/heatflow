import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc


class Space:
    """High‑level wrapper for FEniCSx heat‑equation workspaces.

    The class owns the function spaces (CG for the solution, DG0 for material
    data), builds variational forms for transient heat conduction, and
    provides convenience helpers for:
      - material-property assignment (piecewise constants on the DG0 space)
      - initial-condition construction
      - system assembly (assemble_matrix / assemble_vector)

    Parameters:
        mesh_and_tags: Either a dolfinx.mesh.Mesh, or a tuple
            (mesh, cell_tags, facet_tags) as returned by mesh generators.
        V_family: String name of the finite element family for temperature (default 'Lagrange').
        V_degree: Polynomial degree for the temperature space (default 1).
        Q_family: String name of the finite element family for coefficients (default 'DG').
        Q_degree: Polynomial degree for the coefficient space (default 0).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self, mesh_and_tags, V_family="Lagrange", V_degree=1, Q_family="DG", Q_degree=0
    ):
        """Initialize the Space by creating function spaces on the given mesh or mesh+tags tuple.

        Parameters:
            mesh_and_tags: Either a dolfinx.mesh.Mesh or a tuple
                (mesh, cell_tags, facet_tags) returned by mesh generators.
            V_family: Finite element family for temperature space.
            V_degree: Polynomial degree for temperature space.
            Q_family: Finite element family for coefficient space.
            Q_degree: Polynomial degree for coefficient space.
        """
        # Unpack mesh and optional tags
        if isinstance(mesh_and_tags, tuple) and len(mesh_and_tags) >= 1:
            self.mesh = mesh_and_tags[0]
            self.cell_tags = mesh_and_tags[1] if len(mesh_and_tags) > 1 else None
            self.facet_tags = mesh_and_tags[2] if len(mesh_and_tags) > 2 else None
        else:
            self.mesh = mesh_and_tags
            self.cell_tags = None
            self.facet_tags = None

        # Build Lagrange and DG spaces
        self.V = fem.functionspace(self.mesh, (V_family, V_degree))
        self.Q = fem.functionspace(self.mesh, (Q_family, Q_degree))

        # Forms cache
        self.a_form = None
        self.L_form = None

    # ------------------------------------------------------------------
    # Variational forms
    # ------------------------------------------------------------------
    def build_variational_forms(self, rho_c, kappa, u_n, dt, f=None):
        """Assemble and store the bilinear and linear forms for the heat equation.

        Parameters:
            rho_c: Volumetric heat capacity field (Function or Constant).
            kappa: Thermal conductivity field (Function or Constant).
            u_n: Solution at the previous time step (Function).
            dt: Time step size (float).
            f: Source term (Function or Constant), defaults to zero.

        Returns:
            a_form, L_form: The assembled bilinear and linear forms.
        """
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        if f is None:
            f = fem.Constant(self.mesh, PETSc.ScalarType(0))

        a = (
            rho_c * u * v * ufl.dx
            + dt * kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        )
        L = rho_c * u_n * v * ufl.dx + dt * f * v * ufl.dx

        self.a_form = fem.form(a)
        self.L_form = fem.form(L)
        return self.a_form, self.L_form

    # ------------------------------------------------------------------
    # Assembly helpers
    # ------------------------------------------------------------------
    def assemble_matrix(self, bcs):
        """Assemble and return the global matrix A for the bilinear form.

        Parameters:
            bcs: List of Dolfinx DirichletBC objects.
        """
        A = fem.petsc.assemble_matrix(self.a_form, bcs=bcs)
        A.assemble()
        return A

    def assemble_vector(self, bcs):
        """Assemble and return the global RHS vector b for the linear form.

        Parameters:
            bcs: List of Dolfinx DirichletBC objects.
        """
        b = fem.petsc.assemble_vector(self.L_form)
        fem.petsc.apply_lifting(b, [self.a_form], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, bcs)
        return b

    # ------------------------------------------------------------------
    # Material → DG0 helper
    # ------------------------------------------------------------------
    def assign_material_property(self, materials, property_name):
        """Build a DG0 Function containing piecewise constants for a material property.

        Parameters:
            materials: List of Material objects with .tag and .properties dict.
            property_name: Name of the property to map.

        Returns:
            A Dolfinx Function in Q with the property assigned per cell.
        """
        if self.cell_tags is None:
            raise RuntimeError("cell_tags not present – cannot map materials.")
        from dolfinx.fem import locate_dofs_topological

        # Map tags to values
        tag_to_val = {}
        for mat in materials:
            if not hasattr(mat, "tag"):
                raise AttributeError("Material object must have a .tag attribute.")
            if property_name not in mat.properties:
                raise KeyError(f"Property '{property_name}' not found in {mat}.")
            tag_to_val[mat.tag] = mat.properties[property_name]

        values = np.zeros(self.Q.dofmap.index_map.size_local, dtype=PETSc.ScalarType)
        tdim = self.mesh.topology.dim
        self.mesh.topology.create_connectivity(tdim, tdim)

        tags = self.cell_tags.values
        cells = self.cell_tags.indices
        for tag, cell in zip(tags, cells):
            val = tag_to_val.get(tag, 0)
            dofs = locate_dofs_topological(self.Q, tdim, [cell])
            values[dofs] = val

        f_prop = fem.Function(self.Q)
        f_prop.x.array[:] = values
        f_prop.x.scatter_forward()
        return f_prop

    # ------------------------------------------------------------------
    # Initial condition helper
    # ------------------------------------------------------------------
    def initial_condition(self, init, *, name="u0"):
        """Create an initial-condition Function in V from various inputs.

        Parameters:
            init: float/int for constant, \
                  callable f(x,y) for pointwise, \
                  or array-like of DOF values.
            name: Optional function name (default 'u0').

        Returns:
            A Dolfinx Function populated with the initial data.
        """
        f_ic = fem.Function(self.V)
        f_ic.name = name

        if isinstance(init, (int, float, np.number)):
            f_ic.interpolate(
                lambda x: np.full(x.shape[1], init, dtype=PETSc.ScalarType)
            )
            return f_ic

        if callable(init):
            try:
                f_ic.interpolate(init)
            except Exception:
                wrapper = self.vectorize_callable(init)
                f_ic.interpolate(wrapper)
            return f_ic

        arr = np.asarray(init, dtype=PETSc.ScalarType)
        if arr.size != f_ic.x.array.size:
            raise ValueError("Array length does not match number of DOFs.")
        f_ic.x.array[:] = arr
        f_ic.x.scatter_forward()
        return f_ic

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def vectorize_callable(func):
        """Wrap a scalar function func(x,y) so it accepts point arrays.

        Returns a function matching interpolate's signature.
        """

        def wrapper(points):
            xx, yy = points[0], points[1]
            return np.array(
                [func(xi, yi) for xi, yi in zip(xx, yy)], dtype=PETSc.ScalarType
            )

        return wrapper
