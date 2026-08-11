"""Shared model metadata and figure styling for the paper workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from r_r0_pop.population_model import LifeHistoryParameters


PAPER_MODEL_KEYS = (
    "m1_single_stage",
    "m2_substage_transit",
    "m3_adult_exit_chain_fecundity",
)

MODEL_LABELS = {
    "m1_single_stage": "M1",
    "m2_substage_transit": "M2",
    "m3_adult_exit_chain_fecundity": "M3",
}

MODEL_COLORS = {
    "m1_single_stage": "#2364aa",
    "m2_substage_transit": "#d49f00",
    "m3_adult_exit_chain_fecundity": "#c44536",
}

MS_TEXT_WIDTH = 6.9
MS_ONE_COLUMN_WIDTH = 3.4
MS_TWO_COLUMN_WIDTH = 5.2
MS_PANEL_HEIGHT = 2.35
MS_MATURATION_SURVIVAL_PANEL_HEIGHT = 1.9
MS_LEGEND_Y = 1.01
MS_LEGEND_GAP_INCHES = {
    "tight": 0.30,
    "normal": 0.42,
    "loose": 0.54,
}

STAGE_DURATION_GAP2 = (3, 2)
STAGE_DURATION_GAP3 = (0.5, 2)


@dataclass(frozen=True)
class ModelSpec:
    """A fitted paper model together with its stage-chain structure."""

    key: str
    label: str
    scope: str
    stage_counts: dict[str, int]
    parameters: LifeHistoryParameters
    fitted_parameter_count: int = 0


def model_style(key: str) -> tuple[str, str]:
    """Return the color and line style used in diagnostic plots."""

    styles = {
        "m1_single_stage": (MODEL_COLORS["m1_single_stage"], "-"),
        "m2_substage_transit": (MODEL_COLORS["m2_substage_transit"], "--"),
        "m3_adult_exit_chain_fecundity": (
            MODEL_COLORS["m3_adult_exit_chain_fecundity"],
            "-",
        ),
    }
    try:
        return styles[key]
    except KeyError as error:
        raise ValueError(f"Unknown paper model: {key}") from error


def manuscript_figsize(
    columns: int,
    rows: int,
    *,
    panel_height: float = MS_PANEL_HEIGHT,
) -> tuple[float, float]:
    """Return a manuscript-sized figure width and height in inches."""

    width = {
        1: MS_ONE_COLUMN_WIDTH,
        2: MS_TWO_COLUMN_WIDTH,
    }.get(columns, MS_TEXT_WIDTH)
    return width, rows * panel_height


def manuscript_legend_layout_top(
    fig: Figure,
    *,
    legend_y: float = MS_LEGEND_Y,
    legend_gap: str = "normal",
) -> float:
    """Return the constrained-layout top edge beneath a figure legend."""

    if legend_gap not in MS_LEGEND_GAP_INCHES:
        raise ValueError(f"Unknown legend gap: {legend_gap}")
    return legend_y - MS_LEGEND_GAP_INCHES[legend_gap] / fig.get_figheight()


def manuscript_model_style(key: str) -> dict[str, object]:
    """Return the line styling used consistently in manuscript figures."""

    styles = {
        "m1_single_stage": {
            "color": MODEL_COLORS["m1_single_stage"],
            "linestyle": "-",
            "legend_linestyle": "-",
            "linewidth": 2.6,
            "marker": None,
            "dash_capstyle": "butt",
        },
        "m2_substage_transit": {
            "color": MODEL_COLORS["m2_substage_transit"],
            "linestyle": (0.0, STAGE_DURATION_GAP2),
            "legend_linestyle": (0.0, STAGE_DURATION_GAP2),
            "linewidth": 3.2,
            "marker": None,
            "dash_capstyle": "butt",
        },
        "m3_adult_exit_chain_fecundity": {
            "color": MODEL_COLORS["m3_adult_exit_chain_fecundity"],
            "linestyle": (10.0, STAGE_DURATION_GAP3),
            "legend_linestyle": (0.0, STAGE_DURATION_GAP3),
            "linewidth": 3.2,
            "marker": None,
            "dash_capstyle": "round",
        },
    }
    try:
        return styles[key]
    except KeyError as error:
        raise ValueError(f"Unknown paper model: {key}") from error


def manuscript_model_key_label(key: str) -> str:
    """Return the short model label used in the manuscript."""

    try:
        return MODEL_LABELS[key]
    except KeyError as error:
        raise ValueError(f"Unknown paper model: {key}") from error
