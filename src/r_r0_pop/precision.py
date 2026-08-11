from __future__ import annotations

import math

import pandas as pd


PARAMETER_SIGNIFICANT_FIGURES = 3
PARAMETER_COLUMNS = ("c1", "c2", "c3", "c4", "c5", "c6", "c7")


def round_parameter(
    value: float,
    significant_figures: int = PARAMETER_SIGNIFICANT_FIGURES,
) -> float:
    """Round a finite model parameter to the canonical significant figures."""

    value = float(value)
    if significant_figures < 1:
        raise ValueError("significant_figures must be at least 1")
    if value == 0.0 or not math.isfinite(value):
        return value
    decimal_places = significant_figures - 1 - math.floor(math.log10(abs(value)))
    return round(value, decimal_places)


def round_parameter_table(table: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with model coefficient columns at canonical precision."""

    rounded = table.copy()
    for column in PARAMETER_COLUMNS:
        if column in rounded.columns:
            rounded[column] = rounded[column].map(
                lambda value: round_parameter(value) if pd.notna(value) else value
            )
    return rounded
