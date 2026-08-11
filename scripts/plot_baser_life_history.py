from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from r_r0_pop.data import BaserPaths, load_or_create_baser_processed_data
from r_r0_pop.paths import OUTPUT_DIR, PROCESSED_DATA_DIR, REPORT_DIR, SOURCE_DATA_DIR
from r_r0_pop.plotting import (
    clean_axis,
    plot_lines_by_temperature,
    plot_temperature_histograms,
    plot_temperature_summary,
    save_figure,
)


PLOT_GROUPS = {
    "maturation": (
        "maturation_means",
        "maturation_rates",
        "maturation_distributions",
        "adult_lifetime_means",
        "adult_lifetime_distribution",
    ),
    "reproduction": (
        "lifetime_reproduction",
        "daily_reproduction",
        "reproduction_schedule",
    ),
    "survival": ("adult_emergence",),
}
PLOT_CHOICES = tuple(PLOT_GROUPS) + tuple(
    plot for plots in PLOT_GROUPS.values() for plot in plots
)


@dataclass(frozen=True)
class PlotRecord:
    key: str
    title: str
    path: Path
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot unfitted Baser et al. life-history observations."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SOURCE_DATA_DIR,
        help="Directory containing the original Baser workbook files.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory containing processed Baser CSV files.",
    )
    parser.add_argument(
        "--rebuild-processed",
        action="store_true",
        help="Rebuild processed CSV files from Excel before plotting.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=OUTPUT_DIR / "plots" / "life_history",
        help="Directory for plot PNGs.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_DIR / "life_history_data_plots.md",
        help="Markdown file collecting generated plots.",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=("all",) + PLOT_CHOICES,
        default=["all"],
        help=(
            "Plot groups or individual plots to run. Groups: maturation, "
            "reproduction, survival."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_paths = BaserPaths.from_data_dir(args.data_dir)
    development, adult_survival, fertility = load_or_create_baser_processed_data(
        data_paths,
        args.processed_dir,
        rebuild=args.rebuild_processed,
    )

    selected = expand_plot_selection(args.plots)
    plot_records = all_plot_records(args.plot_dir)

    if "maturation_means" in selected:
        plot_maturation_means(development, fertility, plot_records["maturation_means"].path)
    if "maturation_rates" in selected:
        plot_maturation_rates(
            development, fertility, plot_records["maturation_rates"].path
        )
    if "maturation_distributions" in selected:
        plot_maturation_distributions(
            development, fertility, plot_records["maturation_distributions"].path
        )
    if "adult_lifetime_means" in selected:
        plot_adult_lifetime_means(
            adult_survival, plot_records["adult_lifetime_means"].path
        )
    if "adult_lifetime_distribution" in selected:
        adult_lifetimes = adult_lifetime_observations(adult_survival)
        plot_temperature_histograms(
            adult_lifetimes,
            value="adult_lifetime_days",
            xlabel="Adult lifetime (days)",
            ylabel="Density",
            output=plot_records["adult_lifetime_distribution"].path,
            bins=22,
        )
    if "lifetime_reproduction" in selected:
        plot_lifetime_reproduction(
            fertility, plot_records["lifetime_reproduction"].path
        )
    if "daily_reproduction" in selected:
        plot_daily_reproduction(fertility, plot_records["daily_reproduction"].path)
    if "reproduction_schedule" in selected:
        plot_reproduction_schedule(fertility, plot_records["reproduction_schedule"].path)
    if "adult_emergence" in selected:
        plot_adult_emergence(adult_survival, plot_records["adult_emergence"].path)

    write_report(args.report, plot_records, selected)
    print(f"Wrote {args.report}")
    for record in plot_records.values():
        if record.key in selected:
            print(f"Wrote {record.path}")


def expand_plot_selection(requested: list[str]) -> set[str]:
    if "all" in requested:
        return {plot for plots in PLOT_GROUPS.values() for plot in plots}

    selected: set[str] = set()
    for item in requested:
        if item in PLOT_GROUPS:
            selected.update(PLOT_GROUPS[item])
        else:
            selected.add(item)
    return selected


def all_plot_records(plot_dir: Path) -> dict[str, PlotRecord]:
    records = [
        PlotRecord(
            "maturation_means",
            "Mean Maturation Delays",
            plot_dir / "baser_maturation_means_by_temperature.png",
            (
                "Egg and pupa use positive observed stage durations; larva is "
                "L1+L2+L3 for individuals with all three larval stages observed; "
                "pre-oviposition is an early-adult zero-fecundity interval."
            ),
        ),
        PlotRecord(
            "maturation_rates",
            "Mean Maturation Rates",
            plot_dir / "baser_maturation_rates_by_temperature.png",
            "Rates are calculated as 1 / mean observed maturation delay for each stage and temperature.",
        ),
        PlotRecord(
            "maturation_distributions",
            "Maturation Delay Distributions",
            plot_dir / "baser_maturation_delay_distributions.png",
            "Observed duration distributions by temperature, without fitted curves.",
        ),
        PlotRecord(
            "adult_lifetime_means",
            "Mean Adult Lifetime",
            plot_dir / "baser_adult_lifetime_means_by_temperature.png",
            "Mean positive adult longevity values from the life-history workbook.",
        ),
        PlotRecord(
            "adult_lifetime_distribution",
            "Adult Lifetime Distribution",
            plot_dir / "baser_adult_lifetime_distribution.png",
            "Observed positive adult lifetimes, pooled across adult sex.",
        ),
        PlotRecord(
            "lifetime_reproduction",
            "Lifetime Reproduction",
            plot_dir / "baser_lifetime_reproduction_by_temperature.png",
            "Total eggs per female over the observed fertility trial.",
        ),
        PlotRecord(
            "daily_reproduction",
            "Mean Daily Reproduction",
            plot_dir / "baser_mean_daily_reproduction_by_temperature.png",
            "Mean eggs per live female-day at each temperature.",
        ),
        PlotRecord(
            "reproduction_schedule",
            "Daily Reproduction Schedule",
            plot_dir / "baser_daily_reproduction_schedule.png",
            "Mean eggs per live female by adult day; each line is a temperature.",
        ),
        PlotRecord(
            "adult_emergence",
            "Adult Emergence",
            plot_dir / "baser_adult_emergence_by_temperature.png",
            "Counts of observed male and female adults out of the initial cohort.",
        ),
    ]
    return {record.key: record for record in records}


def maturation_observations(
    development: pd.DataFrame, fertility: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    stage_specs = [
        ("Egg", development.loc[development["E"] > 0, ["temperature", "E"]], "E"),
        (
            "Larva",
            development.loc[
                (development["L1"] > 0)
                & (development["L2"] > 0)
                & (development["L3"] > 0),
                ["temperature", "L1", "L2", "L3"],
            ].assign(larva_days=lambda data: data[["L1", "L2", "L3"]].sum(axis=1)),
            "larva_days",
        ),
        ("Pupa", development.loc[development["P"] > 0, ["temperature", "P"]], "P"),
        (
            "Pre-oviposition",
            fertility.drop_duplicates(["temperature", "female"])[
                ["temperature", "preoviposition_days"]
            ].dropna(),
            "preoviposition_days",
        ),
    ]

    for stage, data, column in stage_specs:
        stage_data = data[["temperature", column]].rename(columns={column: "days"})
        stage_data = stage_data.dropna()
        stage_data = stage_data.loc[stage_data["days"] > 0].copy()
        stage_data["stage"] = stage
        rows.append(stage_data)
    return pd.concat(rows, ignore_index=True)


def plot_maturation_means(
    development: pd.DataFrame, fertility: pd.DataFrame, output: Path
) -> None:
    summary = summarize_maturation_delays(development, fertility)
    plot_temperature_summary(
        summary,
        value="mean_days",
        category="stage",
        ylabel="Mean duration (days)",
        output=output,
    )


def plot_maturation_rates(
    development: pd.DataFrame, fertility: pd.DataFrame, output: Path
) -> None:
    summary = summarize_maturation_delays(development, fertility)
    summary["maturation_rate"] = 1 / summary["mean_days"]
    plot_temperature_summary(
        summary,
        value="maturation_rate",
        category="stage",
        ylabel="Maturation rate (1 / day)",
        output=output,
    )


def summarize_maturation_delays(
    development: pd.DataFrame, fertility: pd.DataFrame
) -> pd.DataFrame:
    observations = maturation_observations(development, fertility)
    return (
        observations.groupby(["stage", "temperature"], as_index=False)
        .agg(mean_days=("days", "mean"), n=("days", "size"))
        .sort_values(["stage", "temperature"])
    )


def plot_maturation_distributions(
    development: pd.DataFrame, fertility: pd.DataFrame, output: Path
) -> None:
    observations = maturation_observations(development, fertility)
    stages = ["Egg", "Larva", "Pupa", "Pre-oviposition"]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.75), constrained_layout=True)

    for ax, stage in zip(axes.flat, stages):
        stage_data = observations.loc[observations["stage"] == stage]
        for temp, temp_data in stage_data.groupby("temperature", sort=True):
            ax.hist(
                temp_data["days"],
                bins=18,
                histtype="step",
                density=True,
                linewidth=1.2,
                label=f"{temp:g} C",
            )
        ax.set_title(stage)
        ax.set_xlabel("Days")
        ax.set_ylabel("Density")
        clean_axis(ax)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Temperature", loc="outside right center")
    save_figure(fig, output)


def adult_lifetime_observations(adult_survival: pd.DataFrame) -> pd.DataFrame:
    male = adult_survival.loc[adult_survival["AM"] > 0, ["temperature", "specimen", "AM"]]
    male = male.rename(columns={"AM": "adult_lifetime_days"})
    male["sex"] = "Male"

    female = adult_survival.loc[
        adult_survival["AF"] > 0, ["temperature", "specimen", "AF"]
    ]
    female = female.rename(columns={"AF": "adult_lifetime_days"})
    female["sex"] = "Female"

    return pd.concat([female, male], ignore_index=True)


def plot_adult_lifetime_means(adult_survival: pd.DataFrame, output: Path) -> None:
    observations = adult_lifetime_observations(adult_survival)
    all_observations = observations.copy()
    all_observations["sex"] = "All adults"
    observations = pd.concat([observations, all_observations], ignore_index=True)
    summary = (
        observations.groupby(["sex", "temperature"], as_index=False)
        .agg(mean_days=("adult_lifetime_days", "mean"), n=("adult_lifetime_days", "size"))
        .sort_values(["sex", "temperature"])
    )
    sex_order = pd.CategoricalDtype(["All adults", "Female", "Male"], ordered=True)
    summary["sex"] = summary["sex"].astype(sex_order)
    summary = summary.sort_values(["sex", "temperature"])
    plot_temperature_summary(
        summary,
        value="mean_days",
        category="sex",
        ylabel="Mean adult lifetime (days)",
        output=output,
    )


def lifetime_reproduction(fertility: pd.DataFrame) -> pd.DataFrame:
    return (
        fertility.groupby(["temperature", "female"], as_index=False)
        .agg(total_eggs=("eggs", "sum"), observed_days=("eggs", "count"))
        .sort_values(["temperature", "female"])
    )


def plot_lifetime_reproduction(fertility: pd.DataFrame, output: Path) -> None:
    totals = lifetime_reproduction(fertility)
    summary = (
        totals.groupby("temperature", as_index=False)
        .agg(mean_total_eggs=("total_eggs", "mean"), n=("female", "size"))
        .sort_values("temperature")
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.scatter(
        totals["temperature"],
        totals["total_eggs"],
        alpha=0.35,
        s=24,
        label="Females",
    )
    ax.plot(
        summary["temperature"],
        summary["mean_total_eggs"],
        marker="o",
        linewidth=1.9,
        color="black",
        label="Mean",
    )
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Lifetime eggs per female")
    ax.set_xticks(summary["temperature"])
    clean_axis(ax)
    ax.legend(frameon=False)
    save_figure(fig, output)


def daily_reproduction_summary(fertility: pd.DataFrame) -> pd.DataFrame:
    return (
        fertility.dropna(subset=["eggs"])
        .groupby("temperature", as_index=False)
        .agg(mean_daily_eggs=("eggs", "mean"), live_female_days=("eggs", "size"))
        .sort_values("temperature")
    )


def plot_daily_reproduction(fertility: pd.DataFrame, output: Path) -> None:
    summary = daily_reproduction_summary(fertility)
    plot_temperature_summary(
        summary,
        value="mean_daily_eggs",
        ylabel="Mean eggs per live female-day",
        output=output,
    )


def plot_reproduction_schedule(fertility: pd.DataFrame, output: Path) -> None:
    daily = (
        fertility.dropna(subset=["eggs"])
        .groupby(["temperature", "adult_day"], as_index=False)
        .agg(mean_eggs=("eggs", "mean"), live_females=("eggs", "size"))
    )
    plot_lines_by_temperature(
        daily,
        x="adult_day",
        y="mean_eggs",
        group="temperature",
        xlabel="Adult day",
        ylabel="Mean eggs per live female",
        output=output,
    )


def plot_adult_emergence(adult_survival: pd.DataFrame, output: Path) -> None:
    counts = (
        adult_survival.assign(
            female=adult_survival["AF"] > 0,
            male=adult_survival["AM"] > 0,
        )
        .groupby("temperature", as_index=False)
        .agg(female_adults=("female", "sum"), male_adults=("male", "sum"))
        .melt(
            id_vars="temperature",
            value_vars=["female_adults", "male_adults"],
            var_name="sex",
            value_name="adults",
        )
    )
    counts["sex"] = counts["sex"].map(
        {"female_adults": "Female adults", "male_adults": "Male adults"}
    )
    plot_temperature_summary(
        counts,
        value="adults",
        category="sex",
        ylabel="Adult count",
        output=output,
    )


def write_report(
    report_path: Path, plot_records: dict[str, PlotRecord], selected: set[str]
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Baser Life-History Plots",
        "",
        "Unfitted summaries of the processed Baser et al. life-history data.",
        "",
    ]
    for record in plot_records.values():
        if record.key not in selected and not record.path.exists():
            continue
        relative_plot_path = os.path.relpath(record.path, start=report_path.parent)
        lines.extend(
            [
                f"## {record.title}",
                "",
                record.note,
                "",
                f"![{record.title}]({Path(relative_plot_path).as_posix()})",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
