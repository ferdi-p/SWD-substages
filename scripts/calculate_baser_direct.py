from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from r_r0_pop.data import (
    BaserPaths,
    female_preadult_distribution,
    female_preadult_summary,
    load_baser_fertility,
    load_baser_life_history,
    load_or_create_baser_processed_data,
    pooled_adult_female_fraction,
)
from r_r0_pop.demography import build_reproduction_schedule, summarize_rates
from r_r0_pop.paths import OUTPUT_DIR, PROCESSED_DATA_DIR, SOURCE_DATA_DIR
from r_r0_pop.plotting import plot_temperature_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate direct R0 and Euler-Lotka r from Baser et al. data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SOURCE_DATA_DIR,
        help="Directory containing Baser workbook files.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory containing processed Baser CSV files.",
    )
    parser.add_argument(
        "--from-excel",
        action="store_true",
        help="Read the original Excel workbooks instead of processed CSV files.",
    )
    parser.add_argument(
        "--rebuild-processed",
        action="store_true",
        help="Rebuild processed CSV files from Excel before calculating.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "baser_direct_rates.csv",
        help="CSV path for temperature-level results.",
    )
    parser.add_argument(
        "--schedule-output",
        type=Path,
        default=OUTPUT_DIR / "baser_reproduction_schedule.csv",
        help="CSV path for the age-specific schedule used in the calculations.",
    )
    parser.add_argument(
        "--mean-shift-sensitivity-output",
        type=Path,
        default=OUTPUT_DIR / "baser_direct_rates_mean_shift_sensitivity.csv",
        help=(
            "CSV path comparing the empirical preadult-duration distribution "
            "with the former single-mean timing shift."
        ),
    )
    parser.add_argument(
        "--offspring-female-fraction",
        type=float,
        default=None,
        help=(
            "Optional female-offspring fraction override. By default, use the "
            "female fraction among BASER adults pooled over temperature."
        ),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=OUTPUT_DIR / "plots",
        help="Directory for PNG plots.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=200,
        help="Number of bootstrap samples used for direct-rate error bars.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260701,
        help="Random seed used for direct-rate bootstrap samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = BaserPaths.from_data_dir(args.data_dir)

    warnings.filterwarnings(
        "ignore",
        message="Conditional Formatting extension is not supported and will be removed",
        category=UserWarning,
        module="openpyxl",
    )

    if args.from_excel:
        development, adult_survival = load_baser_life_history(paths.life_tables)
        fertility = load_baser_fertility(paths.fertility)
    else:
        development, adult_survival, fertility = load_or_create_baser_processed_data(
            paths,
            args.processed_dir,
            rebuild=args.rebuild_processed,
        )
    offspring_female_fraction = (
        pooled_adult_female_fraction(adult_survival)
        if args.offspring_female_fraction is None
        else args.offspring_female_fraction
    )
    female_preadult = female_preadult_distribution(development, adult_survival)
    schedule = build_reproduction_schedule(
        fertility,
        female_preadult,
        offspring_female_fraction=offspring_female_fraction,
    )
    rates = summarize_rates(schedule)
    mean_shift_schedule = build_reproduction_schedule(
        fertility,
        female_preadult_summary(development, adult_survival),
        offspring_female_fraction=offspring_female_fraction,
    )
    mean_shift_rates = summarize_rates(mean_shift_schedule)
    sensitivity = rates.merge(
        mean_shift_rates,
        on="temperature",
        suffixes=("_empirical_distribution", "_mean_shift"),
    )
    for metric in ("R0", "r_euler", "generation_time_euler"):
        sensitivity[f"{metric}_percent_difference_mean_shift"] = (
            sensitivity[f"{metric}_mean_shift"]
            / sensitivity[f"{metric}_empirical_distribution"]
            - 1.0
        ) * 100.0
    if args.bootstrap_samples > 0:
        bootstrap = bootstrap_direct_rate_errors(
            development,
            adult_survival,
            fertility,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
            offspring_female_fraction=offspring_female_fraction,
        )
        rates = rates.merge(bootstrap, on="temperature", how="left")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rates.to_csv(args.output, index=False)
    schedule.to_csv(args.schedule_output, index=False)
    args.mean_shift_sensitivity_output.parent.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(args.mean_shift_sensitivity_output, index=False)
    plot_temperature_metric(
        rates,
        metric="R0",
        ylabel="Net reproduction rate, R0",
        output=args.plot_dir / "baser_direct_R0_by_temperature.png",
        critical_value=1.0,
    )
    plot_temperature_metric(
        rates,
        metric="r_euler",
        ylabel="Intrinsic rate of increase, r",
        output=args.plot_dir / "baser_direct_r_by_temperature.png",
    )

    print(f"Pooled BASER adult female fraction: {offspring_female_fraction:.6g}")
    print(rates.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"\nWrote {args.output}")
    print(f"Wrote {args.schedule_output}")
    print(f"Wrote {args.mean_shift_sensitivity_output}")
    print(f"Wrote {args.plot_dir / 'baser_direct_R0_by_temperature.png'}")
    print(f"Wrote {args.plot_dir / 'baser_direct_r_by_temperature.png'}")


def bootstrap_direct_rate_errors(
    development: pd.DataFrame,
    adult_survival: pd.DataFrame,
    fertility: pd.DataFrame,
    *,
    samples: int,
    seed: int,
    offspring_female_fraction: float,
) -> pd.DataFrame:
    """Estimate direct demographic-rate uncertainty by resampling specimens.

    The direct rate calculation combines two individual-level data sources:
    the 50-specimen life-table cohort and the 10-female fertility cohort. Each
    bootstrap replicate resamples both cohorts within temperature, rebuilds the
    temperature-level reproduction schedule, and recalculates R0, r, and
    generation time.
    """

    rng = np.random.default_rng(seed)
    rows = []
    for temp, fertility_temp in fertility.groupby("temperature", sort=True):
        development_temp = development.loc[
            (development["temperature"] == temp) & (development["specimen"] <= 50)
        ]
        adult_temp = adult_survival.loc[
            (adult_survival["temperature"] == temp) & (adult_survival["specimen"] <= 50)
        ]
        specimen_ids = adult_temp["specimen"].drop_duplicates().to_numpy()
        female_ids = fertility_temp["female"].drop_duplicates().to_numpy()
        if len(specimen_ids) == 0 or len(female_ids) == 0:
            continue

        replicate_rates = []
        for _ in range(samples):
            sampled_specimens = rng.choice(specimen_ids, size=len(specimen_ids), replace=True)
            sampled_females = rng.choice(female_ids, size=len(female_ids), replace=True)
            sampled_development, sampled_adults = _resample_life_table_cohort(
                development_temp,
                adult_temp,
                sampled_specimens,
            )
            sampled_fertility = _resample_fertility_cohort(
                fertility_temp,
                sampled_females,
            )
            preadult = female_preadult_distribution(
                sampled_development,
                sampled_adults,
                cohort_size=len(sampled_specimens),
            )
            schedule = build_reproduction_schedule(
                sampled_fertility,
                preadult,
                offspring_female_fraction=offspring_female_fraction,
            )
            if schedule.empty:
                continue
            rates = summarize_rates(schedule)
            if not rates.empty:
                replicate_rates.append(rates.iloc[0])

        if not replicate_rates:
            continue
        replicate_frame = pd.DataFrame(replicate_rates)
        rows.append(
            {
                "temperature": temp,
                "R0_se": replicate_frame["R0"].std(ddof=1),
                "r_euler_se": replicate_frame["r_euler"].std(ddof=1),
                "generation_time_euler_se": replicate_frame[
                    "generation_time_euler"
                ].std(ddof=1),
            }
        )
    return pd.DataFrame(rows)


def _resample_life_table_cohort(
    development: pd.DataFrame,
    adult_survival: pd.DataFrame,
    sampled_specimens: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    development_index = development.set_index("specimen", drop=False)
    adult_index = adult_survival.set_index("specimen", drop=False)
    development_rows = []
    adult_rows = []
    for sample_id, specimen in enumerate(sampled_specimens, start=1):
        development_row = development_index.loc[specimen].copy()
        adult_row = adult_index.loc[specimen].copy()
        development_row["specimen"] = sample_id
        adult_row["specimen"] = sample_id
        development_rows.append(development_row)
        adult_rows.append(adult_row)
    return pd.DataFrame(development_rows), pd.DataFrame(adult_rows)


def _resample_fertility_cohort(
    fertility: pd.DataFrame,
    sampled_females: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for sample_id, female in enumerate(sampled_females, start=1):
        female_rows = fertility.loc[fertility["female"] == female].copy()
        female_rows["female"] = f"B{sample_id}"
        female_rows["female_index"] = sample_id
        rows.append(female_rows)
    if not rows:
        return pd.DataFrame(columns=fertility.columns)
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    main()
