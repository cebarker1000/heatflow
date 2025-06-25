import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np


def main(csv_path: str, initial_step: int = 0):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(csv_path, index_col=0)
    times = df.index.values  # time array
    # Convert column names to float and ensure ascending order
    col_float = df.columns.astype(float)
    sort_idx = np.argsort(col_float)
    z_positions = col_float[sort_idx]
    data = df.to_numpy()[:, sort_idx]  # reorder columns accordingly

    # Clamp initial_step
    initial_step = max(0, min(initial_step, len(times) - 1))

    # --- Matplotlib figure setup ---
    plt.rcParams.update({"figure.dpi": 120})
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Initial plot
    line, = ax.plot(z_positions, data[initial_step, :], lw=2)
    ax.set_xlabel("z position")
    ax.set_ylabel("∂T/∂r  (smoothed)")

    # Set y-limits once to encompass all data (with small padding)
    global_min = np.nanmin(data)
    global_max = np.nanmax(data)
    if global_max == global_min:
        # Avoid zero range
        global_max += 1.0
    pad = 0.05 * (global_max - global_min)
    ax.set_ylim(global_min - pad, global_max + pad)

    # Slider axis (positioned at bottom)
    slider_ax = fig.add_axes((0.15, 0.1, 0.7, 0.03))
    time_slider = Slider(
        ax=slider_ax,
        label="Time index",
        valmin=0,
        valmax=len(times) - 1,
        valinit=initial_step,
        valstep=1,
        color="blue",
    )

    # Next/Prev buttons ------------------------------------------------------
    button_ax_prev = fig.add_axes((0.15, 0.02, 0.1, 0.05))
    button_ax_next = fig.add_axes((0.27, 0.02, 0.1, 0.05))
    btn_prev = Button(button_ax_prev, "← Prev")
    btn_next = Button(button_ax_next, "Next →")

    def update_plot(step_idx):
        idx = int(step_idx)
        line.set_ydata(data[idx, :])
        ax.set_title(f"Radial gradient @ t = {times[idx]*10**6:.2f} μs")
        fig.canvas.draw_idle()

    # Callback for slider
    def on_slider(val):
        update_plot(val)

    time_slider.on_changed(on_slider)

    # Callbacks for buttons
    def prev(event):
        current = int(time_slider.val)
        if current > 0:
            time_slider.set_val(current - 1)

    def next_(event):
        current = int(time_slider.val)
        if current < len(times) - 1:
            time_slider.set_val(current + 1)

    btn_prev.on_clicked(prev)
    btn_next.on_clicked(next_)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive plot of radial gradient over time.")
    parser.add_argument("csv", nargs="?", default="outputs/geballe_no_diamond_read_flux/radial_gradient.csv", help="Path to radial_gradient.csv")
    parser.add_argument("--step", type=int, default=0, help="Initial timestep index to display")
    args = parser.parse_args()

    main(args.csv, args.step)
