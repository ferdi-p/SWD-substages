"""Check that the rebuilt analysis is complete and internally consistent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from r_r0_pop.paper_config import PAPER_MODEL_KEYS
from r_r0_pop.paths import (
    FIGURE_DIR,
    OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    SUPPLEMENTARY_FIGURE_DIR,
)


DIRECT_METRICS = ("R0", "r_euler", "generation_time_euler")
PROCESSED_SCHEMAS = {
    "development.csv": {"temperature", "specimen", "E", "L1", "L2", "L3", "P"},
    "adult_survival.csv": {"temperature", "specimen", "AM", "AF"},
    "fertility.csv": {
        "temperature",
        "female",
        "preoviposition_days",
        "adult_day",
        "eggs",
    },
}
EXPECTED_FIGURES = (
    "mean_stage_durations.pdf",
    "maturation_survival.pdf",
    "juvenile_survival_mortality_temperature.pdf",
    "fecundity_temperature.pdf",
    "adult_reproduction_time.pdf",
    "demographic_metrics.pdf",
    "seasonal_simulation_composite.pdf",
    "mortality_intervention_r_20C.pdf",
    "m3_adult_fecundity_profile.pdf",
)
EXPECTED_SUPPLEMENTARY_FIGURES = (
    "Figure_S1_maturation_survival_all_temperatures.pdf",
    "Figure_S2_adult_reproduction_all_temperatures.pdf",
)


def main() -> None:
    verify_processed_data()
    verify_direct_rates()
    verify_model_outputs()
    verify_intervention_outputs()
    figure_count = verify_figures()
    print(
        "Verified three processed datasets, direct/model demographic agreement, "
        f"all {len(PAPER_MODEL_KEYS)} models, intervention outputs, and "
        f"{figure_count} generated figures."
    )


def verify_processed_data() -> None:
    for filename, required_columns in PROCESSED_SCHEMAS.items():
        path = PROCESSED_DATA_DIR / filename
        frame = read_required_csv(path)
        missing = required_columns.difference(frame.columns)
        if missing:
            raise RuntimeError(
                f"{path} is missing required columns: {sorted(missing)}"
            )
        if frame.empty:
            raise RuntimeError(f"{path} contains no rows")


def verify_direct_rates() -> None:
    direct = read_required_csv(OUTPUT_DIR / "baser_direct_rates.csv")
    model_direct = read_required_csv(OUTPUT_DIR / "model_complexity" / "direct_rates.csv")
    merged = direct.merge(
        model_direct,
        on="temperature",
        suffixes=("_direct_script", "_model_script"),
        validate="one_to_one",
    )
    if len(merged) != len(direct) or len(merged) != len(model_direct):
        raise RuntimeError("Direct-rate scripts did not evaluate the same temperatures")
    for metric in DIRECT_METRICS:
        np.testing.assert_allclose(
            merged[f"{metric}_direct_script"],
            merged[f"{metric}_model_script"],
            rtol=0.0,
            atol=1e-12,
            err_msg=f"Direct {metric} differs between analysis entry points",
        )


def verify_model_outputs() -> None:
    summary = read_required_csv(OUTPUT_DIR / "model_complexity" / "summary.csv")
    observed_keys = tuple(summary["model"])
    if observed_keys != PAPER_MODEL_KEYS:
        raise RuntimeError(
            f"Expected models {PAPER_MODEL_KEYS}; found {observed_keys}"
        )
    if (summary["state_count"] <= 0).any():
        raise RuntimeError("Every model must contain at least one state")
    for metric in (
        "r_relative_sse",
        "R0_relative_sse",
        "generation_time_relative_sse",
    ):
        if not np.isfinite(summary[metric]).all():
            raise RuntimeError(f"Model summary contains nonfinite {metric} values")


def verify_intervention_outputs() -> None:
    thresholds = read_required_csv(
        OUTPUT_DIR / "mortality_interventions" / "critical_mortality_thresholds.csv"
    )
    observed_models = set(thresholds["model"])
    if observed_models != set(PAPER_MODEL_KEYS):
        raise RuntimeError(
            "Mortality-intervention output does not contain exactly the expected models"
        )
    if set(thresholds["intervention"]) != {"juvenile", "adult"}:
        raise RuntimeError("Expected both juvenile and adult intervention results")


def verify_figures() -> int:
    expected = tuple(FIGURE_DIR / name for name in EXPECTED_FIGURES) + tuple(
        SUPPLEMENTARY_FIGURE_DIR / name
        for name in EXPECTED_SUPPLEMENTARY_FIGURES
    )
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing generated figures: {missing}")
    return len(expected)


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise RuntimeError(f"Missing required output: {path}")
    return pd.read_csv(path)


if __name__ == "__main__":
    main()
