import gmsh

# Optional MPI support – falls back to serial if mpi4py is unavailable
try:
    from mpi4py import MPI  # noqa: F401

    COMM = MPI.COMM_WORLD
except (ModuleNotFoundError, ImportError):

    class _SerialComm:
        rank = 0
        size = 1

        def Barrier(self):
            pass

    COMM = _SerialComm()


class Material:
    """
    A class to store materials within the mesh, incorporating their
    boundaries, desired mesh fineness within the mesh, and any material
    properties (kappa, rho, c_v, etc).

    Attributes:
        name (str):
            Unique name of the material.
        boundaries (list of float):
            [xmin, xmax, ymin, ymax] coordinates defining the rectangular region.
        mesh_size (float):
            Desired target mesh element size within this region.
        properties (dict):
            Dictionary mapping property names to values.
    """

    def __init__(
        self, name, boundaries, properties=None, mesh_size=None, material_tag=None
    ):
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name)}")
        self.name = name
        if not hasattr(boundaries, "__len__") or len(boundaries) != 4:
            raise ValueError("boundaries must be [xmin,xmax,ymin,ymax]")
        xmin, xmax, ymin, ymax = boundaries
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(f"Invalid boundaries {boundaries}")
        self.boundaries = [float(xmin), float(xmax), float(ymin), float(ymax)]
        if mesh_size is not None and not isinstance(mesh_size, (int, float)):
            raise TypeError(f"mesh_size must be a number, got {type(mesh_size)}")
        self.mesh_size = float(mesh_size) if mesh_size is not None else None
        self.properties = {} if properties is None else dict(properties)

    def contains(self, x, y):
        """Return True if (x,y) lies inside this material region."""
        xmin, xmax, ymin, ymax = self.boundaries
        return xmin <= x <= xmax and ymin <= y <= ymax

    def __repr__(self):
        return (
            f"Material({self.name!r}, bounds={self.boundaries}, size={self.mesh_size})"
        )


class Mesh:
    """
    Stores and builds a gmsh mesh over a rectangular domain with multiple materials.

    Attributes:
        name (str): name of the mesh
        boundaries (list): [xmin,xmax,ymin,ymax] of domain
        materials (list[Material])
        default_mesh_size (float)
        material_tags (dict): maps name->physical group tag
    """

    # Construction ---------------------------------------------------------------------------
    def __init__(self, name, boundaries, materials, default_mesh_size=0.1):
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        self.name = name
        if len(boundaries) != 4:
            raise ValueError("boundaries must be 4 floats")
        self.boundaries = [float(b) for b in boundaries]
        self.materials = list(materials)
        self.default_mesh_size = float(default_mesh_size)
        self.material_tags = {}
        self.mesh = None

    # Mesh Generation ------------------------------------------------------------------------
    def build_mesh(self):
        """Builds a 2D mesh with piecewise-constant sizes via Box fields and groups background."""
        gmsh.initialize()
        if COMM.rank == 0:
            # 1) create base rectangle and material rectangles
            xmin, xmax, ymin, ymax = self.boundaries
            base = gmsh.model.occ.addRectangle(xmin, ymin, 0, xmax - xmin, ymax - ymin)
            surfaces = [(2, base)]
            for mat in self.materials:
                bx, BX, by, BY = mat.boundaries
                tag = gmsh.model.occ.addRectangle(bx, by, 0, BX - bx, BY - by)
                mat._tag = tag
                surfaces.append((2, tag))

            # 2) fragment all so that surfaces align and split
            gmsh.model.occ.fragment(surfaces, surfaces)
            gmsh.model.occ.synchronize()

            # 3) create a Box field per material
            box_fields = []
            for mat in self.materials:
                bf = gmsh.model.mesh.field.add("Box")
                bx, BX, by, BY = mat.boundaries
                gmsh.model.mesh.field.setNumber(bf, "XMin", bx)
                gmsh.model.mesh.field.setNumber(bf, "XMax", BX)
                gmsh.model.mesh.field.setNumber(bf, "YMin", by)
                gmsh.model.mesh.field.setNumber(bf, "YMax", BY)
                size_in = (
                    mat.mesh_size
                    if mat.mesh_size is not None
                    else self.default_mesh_size
                )
                gmsh.model.mesh.field.setNumber(bf, "VIn", size_in)
                gmsh.model.mesh.field.setNumber(bf, "VOut", self.default_mesh_size)
                box_fields.append(bf)

            # 4) combine via Min field, set as background
            if box_fields:
                minf = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(minf, "FieldsList", box_fields)
                gmsh.model.mesh.field.setAsBackgroundMesh(minf)

            # 5) assign physical groups by centroid classification
            all_surfs = gmsh.model.occ.getEntities(2)
            bg_tags = []
            for dim, tag in all_surfs:
                x, y, z = gmsh.model.occ.getCenterOfMass(dim, tag)
                assigned = False
                for mat in self.materials:
                    if mat.contains(x, y):
                        pg = self.material_tags.get(
                            mat.name
                        ) or gmsh.model.addPhysicalGroup(2, [tag])
                        gmsh.model.setPhysicalName(2, pg, mat.name)
                        self.material_tags[mat.name] = pg
                        assigned = True
                        break
                if not assigned:
                    bg_tags.append(tag)
            # background group
            pg_bg = gmsh.model.addPhysicalGroup(2, bg_tags)
            gmsh.model.setPhysicalName(2, pg_bg, "background")
            self.material_tags["background"] = pg_bg

        COMM.Barrier()
        gmsh.model.mesh.generate(2)
        self.mesh = gmsh
        COMM.Barrier()

    # FEniCs Interoperability ------------------------------------------------------------
    def to_dolfinx(self, *, comm=COMM, gdim: int = 2, rank: int = 0):
        """Convert *in‑memory* Gmsh model to a DOLFINx mesh (no files).

        Returns:

        mesh: dolfinx.mesh.Mesh
        cell_tags: dolfinx.mesh.MeshTags
        facet_tags: dolfinx.mesh.MeshTags
        """
        if self.mesh is None:
            raise RuntimeError("Mesh not built – call build_mesh() first.")

        from dolfinx.io import gmshio  # local import to avoid hard dependency if unused

        return gmshio.model_to_mesh(gmsh.model, comm, rank, gdim)

    @staticmethod
    def msh_to_dolfinx(filename: str, *, comm=COMM, gdim: int = 2, rank: int = 0):
        """Load a ``.msh`` file *filename* and convert to DOLFINx mesh.

        This is a convenience wrapper around gmsh.open + gmshio.model_to_mesh.
        """
        gmsh.initialize()
        gmsh.open(filename)
        from dolfinx.io import gmshio

        mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, comm, rank, gdim)
        gmsh.finalize()
        return mesh, cell_tags, facet_tags

    # I/O Helpers ------------------------------------------------------------------------
    def write(self, filename: str):
        """Write the current Gmsh mesh to disk (e.g., ``.msh``)."""
        if self.mesh is None:
            raise RuntimeError("Mesh not built – call build_mesh() first.")
        self.mesh.write(filename)
