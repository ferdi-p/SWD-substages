"""Check that the rebuilt analysis is complete and internally consistent."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from r_r0_pop.paper_config import PAPER_MODEL_KEYS
from r_r0_pop.paths import MANUSCRIPT_DIR, OUTPUT_DIR, PROCESSED_DATA_DIR


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify model outputs and, optionally, manuscript figures."
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Verify the model and data workflow without requiring manuscript files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_processed_data()
    verify_direct_rates()
    verify_model_outputs()
    verify_intervention_outputs()
    if args.model_only:
        print(
            "Verified three processed datasets, direct/model demographic "
            f"agreement, all {len(PAPER_MODEL_KEYS)} paper models, and "
            "intervention outputs."
        )
        return
    figure_count = verify_manuscript_figures()
    print(
        "Verified three processed datasets, direct/model demographic agreement, "
        f"all {len(PAPER_MODEL_KEYS)} paper models, intervention outputs, and "
        f"{figure_count} manuscript figure references."
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
            f"Expected paper models {PAPER_MODEL_KEYS}; found {observed_keys}"
        )
    if (summary["state_count"] <= 0).any():
        raise RuntimeError("Every paper model must contain at least one state")
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
            "Mortality-intervention output does not contain exactly the paper models"
        )
    if set(thresholds["intervention"]) != {"juvenile", "adult"}:
        raise RuntimeError("Expected both juvenile and adult intervention results")


def verify_manuscript_figures() -> int:
    manuscript = MANUSCRIPT_DIR / "popModels.tex"
    text = manuscript.read_text(encoding="utf-8")
    references = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    missing = [
        reference
        for reference in references
        if not (MANUSCRIPT_DIR / reference).is_file()
    ]
    if missing:
        raise RuntimeError(f"Missing manuscript figures: {missing}")

    supplementary = (
        MANUSCRIPT_DIR
        / "supplementary_figures"
        / "Figure_S1_maturation_survival_all_temperatures.pdf",
        MANUSCRIPT_DIR
        / "supplementary_figures"
        / "Figure_S2_adult_reproduction_all_temperatures.pdf",
    )
    missing_supplementary = [str(path) for path in supplementary if not path.is_file()]
    if missing_supplementary:
        raise RuntimeError(
            f"Missing supplementary figures: {missing_supplementary}"
        )
    return len(references) + len(supplementary)


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise RuntimeError(f"Missing required output: {path}")
    return pd.read_csv(path)


if __name__ == "__main__":
    main()
