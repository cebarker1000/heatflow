import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from io_utilities.xdmf_extract import extract_point_timeseries_xdmf


def line_plot(xdmf_path: str, *, axis: str = "x", coord: float = 0.0,
              line_range: tuple[float, float], samples: int = 200,
              time: float | None = None, time_index: int | None = None,
              function_name: str = "Temperature (K)"):
    """Plot solution values along a line at a specific time.

    Parameters
    ----------
    xdmf_path : str
        Path to the ``solution.xdmf`` file.
    axis : {"x", "y"}, optional
        Direction along which the line varies.
    coord : float, optional
        Fixed coordinate of the orthogonal direction (e.g. ``y`` if ``axis='x'``).
    line_range : tuple[float, float]
        ``(min, max)`` coordinate bounds of the line.
    samples : int, optional
        Number of sample points along the line.
    time : float, optional
        Desired time in seconds. The closest available time step will be used.
    time_index : int, optional
        Index of the desired time step if ``time`` is ``None``.
    function_name : str, optional
        Name of the field stored in the XDMF file.
    """
    if line_range is None:
        raise ValueError("line_range must be provided")
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")

    # Build query points along the requested line
    coords = np.linspace(line_range[0], line_range[1], samples)
    if axis == "x":
        query_points = [(x, coord) for x in coords]
        xlabel = "x position"
    else:
        query_points = [(coord, y) for y in coords]
        xlabel = "y position"

    times, data = extract_point_timeseries_xdmf(
        xdmf_path, function_name, query_points)

    if time is not None:
        idx = int(np.abs(times - time).argmin())
    else:
        idx = 0 if time_index is None else int(time_index)

    fig, ax = plt.subplots()
    ax.plot(coords, data[:, idx], color="black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(function_name)
    ax.set_title(f"{function_name} along {axis}={coord} at t={times[idx]:.3e}s")
    return fig, ax


def line_difference_colormap(xdmf_a: str, xdmf_b: str, *, axis: str = "x",
                              coord: float = 0.0,
                              line_range: tuple[float, float], samples: int = 200,
                              function_name: str = "Temperature (K)",
                              label_a: str = "sim A", label_b: str = "sim B"):
    """Plot temperature differences between two simulations along a line.

    Parameters
    ----------
    xdmf_a, xdmf_b : str
        Paths to the ``solution.xdmf`` files of the two simulations.
    axis : {"x", "y"}
        Axis along which the slice varies.
    coord : float
        Fixed coordinate of the orthogonal axis.
    line_range : tuple[float, float]
        Bounds ``(min, max)`` of the varying coordinate.
    samples : int
        Number of sample points along the slice.
    function_name : str
        Name of the field stored in the XDMF files.
    label_a, label_b : str
        Labels describing the simulations; used for the colorbar annotation.
    """
    if line_range is None:
        raise ValueError("line_range must be provided")
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")

    coords = np.linspace(line_range[0], line_range[1], samples)
    if axis == "x":
        qpts = [(x, coord) for x in coords]
        xlabel = "x position"
    else:
        qpts = [(coord, y) for y in coords]
        xlabel = "y position"

    times_a, data_a = extract_point_timeseries_xdmf(xdmf_a, function_name, qpts)
    times_b, data_b = extract_point_timeseries_xdmf(xdmf_b, function_name, qpts)

    if not np.allclose(times_a, times_b):
        raise ValueError("Simulations have different time grids")
    diff = data_b - data_a

    fig, ax = plt.subplots()
    t_mesh, x_mesh = np.meshgrid(times_a, coords, indexing="ij")
    pcm = ax.pcolormesh(x_mesh, t_mesh, diff.T,
                        cmap="coolwarm",
                        norm=colors.TwoSlopeNorm(vcenter=0.0))
    cbar = fig.colorbar(pcm, ax=ax, label=f"{function_name} difference ({label_b} - {label_a})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("time [s]")
    ax.set_title(f"{label_b} minus {label_a} along {axis}={coord}")
    return fig, ax, cbar
