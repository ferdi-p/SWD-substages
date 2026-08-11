from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


TEMPERATURE_CMAP = "viridis"


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def temperature_palette(temperatures: pd.Series | list[float]) -> dict[float, tuple]:
    """Return stable colors keyed by temperature."""

    temps = sorted(float(temp) for temp in pd.Series(temperatures).dropna().unique())
    cmap = plt.get_cmap(TEMPERATURE_CMAP)
    if len(temps) == 1:
        return {temps[0]: cmap(0.55)}
    return {temp: cmap(index / (len(temps) - 1)) for index, temp in enumerate(temps)}


def clean_axis(ax: Axes) -> None:
    """Apply the common plot style used across project figures."""

    ax.grid(False)
    if ax.get_yscale() == "linear" and not _has_negative_y_data(ax):
        _, upper = ax.get_ylim()
        if np.isfinite(upper) and upper > 0:
            ax.set_ylim(bottom=0, top=upper * 1.06)
    ax.tick_params(direction="out", length=3, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _has_negative_y_data(ax: Axes) -> bool:
    values: list[np.ndarray] = []
    for line in ax.lines:
        values.append(np.asarray(line.get_ydata(), dtype=float).ravel())
    for collection in ax.collections:
        offsets = collection.get_offsets()
        if len(offsets):
            values.append(np.asarray(offsets[:, 1], dtype=float).ravel())
            continue
        for path in collection.get_paths():
            vertices = path.vertices
            if len(vertices):
                values.append(np.asarray(vertices[:, 1], dtype=float).ravel())
    for patch in ax.patches:
        vertices = patch.get_path().vertices
        if len(vertices):
            transformed = patch.get_patch_transform().transform(vertices)
            values.append(np.asarray(transformed[:, 1], dtype=float).ravel())
    if not values:
        return False
    finite = np.concatenate(values)
    finite = finite[np.isfinite(finite)]
    return bool(len(finite) and np.nanmin(finite) < 0)


def save_figure(fig: Figure, output: Path | str) -> None:
    """Save a figure with project defaults and close it."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_temperature_metric(
    rates: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    output: Path | str,
    critical_value: float | None = None,
) -> None:
    """Plot one demographic metric against temperature."""

    output = Path(output)
    data = rates.sort_values("temperature")

    fig, ax = plt.subplots(figsize=(6.5, 4.25), constrained_layout=True)
    ax.plot(data["temperature"], data[metric], marker="o", linewidth=1.8)
    if critical_value is not None:
        ax.axhline(
            critical_value,
            color="#333333",
            linewidth=0.9,
            linestyle=":",
        )
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(data["temperature"])
    clean_axis(ax)

    save_figure(fig, output)


def plot_lines_by_temperature(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str,
    xlabel: str,
    ylabel: str,
    output: Path | str,
) -> None:
    """Plot one time-series line per temperature."""

    fig, ax = plt.subplots(figsize=(7.25, 4.75), constrained_layout=True)
    colors = temperature_palette(data[group])

    for temp, temp_data in data.groupby(group, sort=True):
        temp_data = temp_data.sort_values(x)
        ax.plot(
            temp_data[x],
            temp_data[y],
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            color=colors[float(temp)],
            label=f"{temp:g} C",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    clean_axis(ax)
    ax.legend(title="Temperature", fontsize=8, title_fontsize=9, ncols=2, frameon=False)
    save_figure(fig, output)


def plot_temperature_summary(
    summary: pd.DataFrame,
    *,
    value: str,
    ylabel: str,
    output: Path | str,
    category: str | None = None,
    xlabel: str = "Temperature (°C)",
) -> None:
    """Plot mean values against temperature, optionally split by category."""

    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)

    if category is None:
        data = summary.sort_values("temperature")
        ax.plot(data["temperature"], data[value], marker="o", linewidth=1.8)
    else:
        for label, data in summary.groupby(category, sort=False, observed=False):
            data = data.sort_values("temperature")
            ax.plot(
                data["temperature"],
                data[value],
                marker="o",
                linewidth=1.8,
                label=str(label),
            )
        ax.legend(frameon=False)

    temps = sorted(summary["temperature"].dropna().unique())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(temps)
    clean_axis(ax)
    save_figure(fig, output)


def plot_temperature_histograms(
    data: pd.DataFrame,
    *,
    value: str,
    temperature: str = "temperature",
    xlabel: str,
    ylabel: str,
    output: Path | str,
    bins: int = 24,
) -> None:
    """Overlay per-temperature histograms for a duration or count variable."""

    plot_data = data[[temperature, value]].dropna()
    fig, ax = plt.subplots(figsize=(7.25, 4.75), constrained_layout=True)
    colors = temperature_palette(plot_data[temperature])

    for temp, temp_data in plot_data.groupby(temperature, sort=True):
        ax.hist(
            temp_data[value],
            bins=bins,
            histtype="step",
            linewidth=1.4,
            density=True,
            color=colors[float(temp)],
            label=f"{temp:g} C",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    clean_axis(ax)
    ax.legend(title="Temperature", fontsize=8, title_fontsize=9, ncols=2, frameon=False)
    save_figure(fig, output)
