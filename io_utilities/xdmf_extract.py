import meshio
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata


def extract_point_timeseries_xdmf(
    xdmf_path: str,
    function_name: str,
    query_points: list[tuple[float, float]],
    method: str = "nearest",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts time-series of a nodal field from an XDMF time-series file via meshio.

    Params:
    xdmf_path      (str):   Path to your 'solution.xdmf' file.
    function_name  (str):   The name of the field in your XDMF (e.g. "Temperature (K)").
    query_points   (list):  [(x1,y1), (x2,y2), …] points to sample.
    method         (str):   "nearest" (default) for nearest-vertex lookup,
                           or "linear" for true barycentric interpolation.

    Returns:
    times          (np.ndarray): shape (Nsteps,) sorted time-values.
    data           (np.ndarray): shape (n_points, Nsteps) sampled values.
    """
    # 1) open with meshio
    with meshio.xdmf.TimeSeriesReader(xdmf_path) as reader:
        # Read the static mesh once
        points, _ = reader.read_points_cells()  # (N_pts, 3)
        pts2d = points[:, :2]  # drop z if 2D
        tree = cKDTree(pts2d)

        n_pts = len(query_points)
        n_steps = reader.num_steps
        times = np.empty(n_steps, dtype=float)
        data = np.empty((n_pts, n_steps), dtype=float)

        # 2) loop over time-steps
        for i in range(n_steps):
            t, point_data, _ = reader.read_data(i)
            times[i] = t
            vals = point_data[function_name]  # shape (N_pts,)

            if method == "nearest":
                # pick nearest mesh-vertex
                for j, qp in enumerate(query_points):
                    idx = tree.query(qp)[1]
                    data[j, i] = vals[idx]
            else:
                # linear interpolation within the triangles
                for j, qp in enumerate(query_points):
                    data[j, i] = griddata(pts2d, vals, qp, method="linear")

    # 3) ensure times are sorted (meshio gives them in order, but just in case)
    sort_idx = np.argsort(times)
    return times[sort_idx], data[:, sort_idx]
