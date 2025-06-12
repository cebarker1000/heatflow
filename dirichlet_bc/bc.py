import numpy as np
from dolfinx import fem
from petsc4py import PETSc


class RowDirichletBC:
    """Dirichlet BC along one edge/line, optionally clipped to a centred segment.

    Locations
    ---------
    left, right, bottom, top, outer, x, y  (see below)

    Parameters
    ----------
    V        : FunctionSpace on which the BC acts (typically CG space)
    location : str
        "left", "right", "bottom", "top", "outer", "x", or "y".
    coord    : float, optional
        Required for inner-line cases (location "x" or "y"). Ignored otherwise.
    length   : float, optional
        If provided, only dofs whose **orthogonal** coordinate lies within
        ±length/2 of the domain midpoint are clamped, keeping the patch centred.
        When None (default) the entire row is clamped.
    width    : float, optional
        Geometric tolerance when comparing coordinates (default 1e‑12).
    value    : float | callable(x,y,t) -> scalar, optional
        Boundary value. If callable, call update(t) each time step.
    """

    def __init__(self, V, location, *, coord=None, length=None, width=1e-10, value=0.0):
        self.V = V
        self.mesh = V.mesh
        self.width = float(width)
        self.length = length

        # Domain extents
        verts = self.mesh.geometry.x
        xmin, ymin = verts[:, 0].min(), verts[:, 1].min()
        xmax, ymax = verts[:, 0].max(), verts[:, 1].max()
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)
        half = None if length is None else 0.5 * length

        # Helper: centred-length mask along an axis array
        def centred_mask(axis_vals, center):
            if half is None:
                return np.ones_like(axis_vals, dtype=bool)
            return np.abs(axis_vals - center) <= half + 1e-14

        # Build vectorised predicate ------------------------------------------------
        if location == "left":
            def pred(x):
                return np.logical_and(
                    np.isclose(x[0], xmin, atol=self.width),
                    centred_mask(x[1], ymid))
        elif location == "right":
            def pred(x):
                return np.logical_and(
                    np.isclose(x[0], xmax, atol=self.width),
                    centred_mask(x[1], ymid))
        elif location == "bottom":
            def pred(x):
                return np.logical_and(
                    np.isclose(x[1], ymin, atol=self.width),
                    centred_mask(x[0], xmid))
        elif location == "top":
            def pred(x):
                return np.logical_and(
                    np.isclose(x[1], ymax, atol=self.width),
                    centred_mask(x[0], xmid))
        elif location == "outer":
            def pred(x):
                mask_left = np.logical_and(np.isclose(x[0], xmin, atol=self.width), centred_mask(x[1], ymid))
                mask_right = np.logical_and(np.isclose(x[0], xmax, atol=self.width), centred_mask(x[1], ymid))
                mask_bottom = np.logical_and(np.isclose(x[1], ymin, atol=self.width), centred_mask(x[0], xmid))
                mask_top = np.logical_and(np.isclose(x[1], ymax, atol=self.width), centred_mask(x[0], xmid))
                return mask_left | mask_right | mask_bottom | mask_top
        elif location == "x":
            if coord is None:
                raise ValueError("coord required when location='x'.")
            c = float(coord)
            def pred(x):
                return np.logical_and(
                    np.isclose(x[0], c, atol=self.width),
                    centred_mask(x[1], ymid))
        elif location == "y":
            if coord is None:
                raise ValueError("coord required when location='y'.")
            c = float(coord)
            def pred(x):
                return np.logical_and(
                    np.isclose(x[1], c, atol=self.width),
                    centred_mask(x[0], xmid))
        else:
            raise ValueError("Unknown location keyword.")

        # Locate DOFs ------------------------------------------------------------
        self.row_dofs = fem.locate_dofs_geometrical(self.V, pred)
        if self.row_dofs.size == 0:
            raise RuntimeError("No DOFs found for requested BC location/length.")

        self.dof_coords = self.V.tabulate_dof_coordinates()[self.row_dofs]

        # Storage function and DirichletBC object --------------------------------
        self._g = fem.Function(self.V)
        self._bc = fem.dirichletbc(self._g, self.row_dofs)

        # Wrap constant into callable if needed ----------------------------------
        if callable(value):
            self._value_callable = value
        else:
            self._value_callable = lambda x, y, t, c=value: c

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def bc(self):
        """dolfinx DirichletBC ready for assembly."""
        return self._bc

    def update(self, t):
        """Call each time step if the BC value depends on *t*."""
        vals = np.array([self._value_callable(x, y, t) for x, y in self.dof_coords],
                        dtype=PETSc.ScalarType)
        self._g.x.array[self.row_dofs] = vals
        self._g.x.scatter_forward()

    # ------------------------------------------------------------------
    # Convenience constant BC
    # ------------------------------------------------------------------
    @staticmethod
    def constant(V, location, value, *, coord=None, length=None, width=1e-12):
        bc = RowDirichletBC(V, location, coord=coord, length=length, width=width, value=value)
        bc.update(0.0)
        return bc
