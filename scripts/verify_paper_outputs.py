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

MAIN_FIGURE_LABELS = (
    "fig:diagram",
    "fig:stage-duration-functions",
    "fig:stage-duration-survival",
    "fig:juvenile-survival-mortality-temperature",
    "fig:fecundity-temperature",
    "fig:adult-reproduction-time",
    "fig:demographic-metrics",
    "fig:seasonal-simulation-composite",
    "fig:mortality-intervention-r",
)


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

    for intervention in ("juvenile", "adult"):
        grid = read_required_csv(
            OUTPUT_DIR
            / "mortality_interventions"
            / f"{intervention}_mortality_intervention_r.csv"
        )
        baseline = grid.loc[np.isclose(grid["added_mortality_per_day"], 0.0)]
        for row in baseline.itertuples(index=False):
            rates = read_required_csv(
                OUTPUT_DIR
                / "model_complexity"
                / f"{row.model}_demographic_rates.csv"
            )
            expected = rates.loc[
                np.isclose(rates["temperature"], row.temperature), "r_model"
            ]
            if len(expected) != 1:
                raise RuntimeError(
                    f"Could not find the {row.temperature:g} C baseline for {row.model}"
                )
            np.testing.assert_allclose(
                row.r,
                expected.iloc[0],
                rtol=0.0,
                atol=1e-12,
                err_msg=(
                    f"{intervention} intervention baseline differs from the main "
                    f"model for {row.model} at {row.temperature:g} C"
                ),
            )


def verify_manuscript_figures() -> int:
    sources = (
        MANUSCRIPT_DIR / "popModels.tex",
        MANUSCRIPT_DIR / "supporting_information_methods.tex",
        MANUSCRIPT_DIR / "supporting_information_figures.tex",
    )
    missing_sources = [str(path) for path in sources if not path.is_file()]
    if missing_sources:
        raise RuntimeError(f"Missing manuscript sources: {missing_sources}")

    main_text = sources[0].read_text(encoding="utf-8")
    if r"\appendix" in main_text:
        raise RuntimeError("The main manuscript still contains an appendix boundary")

    bibliography_position = main_text.find(r"\printbibliography")
    legends_position = main_text.find(r"\section*{Figure Legends}")
    if bibliography_position < 0 or legends_position < bibliography_position:
        raise RuntimeError("The complete figure-legend section must follow the bibliography")

    main_body = main_text[:legends_position]
    active_main_body = "\n".join(
        line for line in main_body.splitlines() if not line.lstrip().startswith("%")
    )
    if r"\includegraphics" in active_main_body:
        raise RuntimeError("The main manuscript still embeds at least one figure image")

    legend_text = main_text[legends_position:]
    legend_labels = tuple(re.findall(r"\\label\{(fig:[^}]+)\}", legend_text))
    if legend_labels != MAIN_FIGURE_LABELS:
        raise RuntimeError(
            "Main figure legends are missing or out of order: "
            f"expected {MAIN_FIGURE_LABELS}, found {legend_labels}"
        )
    first_reference_positions = []
    for label in MAIN_FIGURE_LABELS:
        match = re.search(rf"\\ref\{{{re.escape(label)}\}}", main_body)
        if match is None:
            raise RuntimeError(f"Main text does not cite {label}")
        first_reference_positions.append(match.start())
    if first_reference_positions != sorted(first_reference_positions):
        raise RuntimeError("Main figures are not first cited in numerical order")

    required_supporting_citations = (
        "Section~S1",
        "Section~S2",
        "Section~S3",
        "Sections~S4--S6",
        "Section~S7",
        "Table~S1",
        "Figure~S1",
        "Figure~S2",
        "Figure~S3",
    )
    missing_supporting_citations = [
        citation for citation in required_supporting_citations if citation not in main_body
    ]
    if missing_supporting_citations:
        raise RuntimeError(
            "Main text is missing Supporting Information citations: "
            f"{missing_supporting_citations}"
        )

    for source in sources:
        text = source.read_text(encoding="utf-8")
        if re.search(r"\bAppend(?:ix|ices)\b", text, flags=re.IGNORECASE):
            raise RuntimeError(f"{source} still uses Appendix terminology")

    expected_main_figures = tuple(
        MANUSCRIPT_DIR / "figures" / f"Figure {number}.pdf"
        for number in range(1, 10)
    )
    expected_supporting_figures = (
        MANUSCRIPT_DIR / "supplementary_figures" / "Figure S1.pdf",
        MANUSCRIPT_DIR / "supplementary_figures" / "Figure S2.pdf",
        MANUSCRIPT_DIR / "figures" / "Figure S3.pdf",
    )
    expected_figure_files = expected_main_figures + expected_supporting_figures
    missing_figure_files = [
        str(path) for path in expected_figure_files if not path.is_file()
    ]
    if missing_figure_files:
        raise RuntimeError(f"Missing numbered figure files: {missing_figure_files}")

    upload_figure_dir = MANUSCRIPT_DIR.parent / "output" / "submission_figures"
    expected_upload_figures = tuple(
        upload_figure_dir / f"Figure {number}.eps" for number in range(1, 10)
    )
    missing_upload_figures = [
        str(path) for path in expected_upload_figures if not path.is_file()
    ]
    if missing_upload_figures:
        raise RuntimeError(
            f"Missing upload-ready EPS figures: {missing_upload_figures}"
        )

    references: list[str] = []
    for source in sources:
        text = source.read_text(encoding="utf-8")
        references.extend(
            re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
        )
    missing = [
        reference
        for reference in references
        if not (MANUSCRIPT_DIR / reference).is_file()
    ]
    if missing:
        raise RuntimeError(f"Missing manuscript/SI figures: {missing}")
    return len(expected_figure_files)


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise RuntimeError(f"Missing required output: {path}")
    return pd.read_csv(path)


if __name__ == "__main__":
    main()
