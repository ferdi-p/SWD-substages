from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigvals

from r_r0_pop.life_history_fits import FUNCTION_PARAMETER_NAMES
from r_r0_pop.paper_config import (
    MS_LEGEND_Y,
    PAPER_MODEL_KEYS,
    ModelSpec,
    manuscript_figsize,
    manuscript_legend_layout_top,
    manuscript_model_key_label,
    manuscript_model_style,
    model_style,
)
from r_r0_pop.paths import MANUSCRIPT_FIGURE_DIR, OUTPUT_DIR, REPORT_DIR
from r_r0_pop.plotting import clean_axis, save_figure, temperature_palette
from r_r0_pop.population_model import (
    LifeHistoryParameters,
    TemperatureResponse,
    fixed_temperature_matrix,
    life_history_parameters_from_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-temperature mortality interventions on main model r."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=OUTPUT_DIR / "model_complexity",
        help="Directory with fitted model-complexity CSV outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "mortality_interventions",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=OUTPUT_DIR / "plots" / "mortality_interventions",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_DIR / "mortality_interventions.md",
    )
    parser.add_argument(
        "--publication-figure-dir",
        type=Path,
        default=MANUSCRIPT_FIGURE_DIR,
        help=(
            "Directory for publication-formatted figures. The model-only "
            "workflow redirects this to outputs/."
        ),
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[16.0, 20.0, 24.0, 28.0],
        help="Fixed temperatures to facet.",
    )
    parser.add_argument("--juvenile-max", type=float, default=1.0)
    parser.add_argument("--adult-max", type=float, default=3.0)
    parser.add_argument("--manuscript-temperature", type=float, default=20.0)
    parser.add_argument("--points", type=int, default=101)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    models = load_main_models(args.model_dir)
    temperatures = np.asarray(args.temperatures, dtype=float)
    juvenile_added = np.linspace(0.0, args.juvenile_max, args.points)
    adult_added = np.linspace(0.0, args.adult_max, args.points)

    juvenile = intervention_grid(
        models,
        temperatures=temperatures,
        added_mortalities=juvenile_added,
        stage_group="juvenile",
    )
    adult = intervention_grid(
        models,
        temperatures=temperatures,
        added_mortalities=adult_added,
        stage_group="adult",
    )
    summary = intervention_threshold_summary(pd.concat([juvenile, adult]))

    juvenile.to_csv(args.output_dir / "juvenile_mortality_intervention_r.csv", index=False)
    adult.to_csv(args.output_dir / "adult_mortality_intervention_r.csv", index=False)
    summary.to_csv(args.output_dir / "critical_mortality_thresholds.csv", index=False)

    plot_faceted_intervention(
        juvenile,
        y="r",
        xlabel="Added juvenile mortality (1/day)",
        ylabel="Intrinsic rate of increase, r (1/day)",
        output=args.plot_dir / "juvenile_mortality_intervention_r.png",
    )
    plot_faceted_intervention(
        adult,
        y="r",
        xlabel="Added adult mortality (1/day)",
        ylabel="Intrinsic rate of increase, r (1/day)",
        output=args.plot_dir / "adult_mortality_intervention_r.png",
    )
    plot_faceted_intervention(
        juvenile,
        y="delta_r",
        xlabel="Added juvenile mortality (1/day)",
        ylabel="Change in r from model baseline (1/day)",
        output=args.plot_dir / "juvenile_mortality_intervention_delta_r.png",
    )
    plot_faceted_intervention(
        adult,
        y="delta_r",
        xlabel="Added adult mortality (1/day)",
        ylabel="Change in r from model baseline (1/day)",
        output=args.plot_dir / "adult_mortality_intervention_delta_r.png",
    )
    composite_delta_plot = args.plot_dir / "mortality_intervention_delta_r_composite.png"
    plot_intervention_delta_r_composite(
        juvenile,
        adult,
        models=models,
        output=composite_delta_plot,
    )
    manuscript_r_plot = args.plot_dir / "mortality_intervention_r_20C_manuscript.png"
    plot_manuscript_mortality_intervention_r(
        juvenile,
        adult,
        temperature=args.manuscript_temperature,
        models=models,
        output=manuscript_r_plot,
    )
    manuscript_pdf = (
        args.publication_figure_dir / "Figure 9.pdf"
    )
    plot_manuscript_mortality_intervention_r(
        juvenile,
        adult,
        temperature=args.manuscript_temperature,
        models=models,
        output=manuscript_pdf,
    )
    write_report(
        args.report,
        juvenile_plot=args.plot_dir / "juvenile_mortality_intervention_r.png",
        adult_plot=args.plot_dir / "adult_mortality_intervention_r.png",
        juvenile_delta_plot=args.plot_dir / "juvenile_mortality_intervention_delta_r.png",
        adult_delta_plot=args.plot_dir / "adult_mortality_intervention_delta_r.png",
        composite_delta_plot=composite_delta_plot,
        manuscript_r_plot=manuscript_r_plot,
        summary=summary,
    )

    print(f"Wrote {args.output_dir / 'juvenile_mortality_intervention_r.csv'}")
    print(f"Wrote {args.output_dir / 'adult_mortality_intervention_r.csv'}")
    print(f"Wrote {args.output_dir / 'critical_mortality_thresholds.csv'}")
    print(f"Wrote {manuscript_r_plot}")
    print(f"Wrote {manuscript_pdf}")


def load_main_models(model_dir: Path) -> tuple[ModelSpec, ...]:
    parameter_table = pd.read_csv(model_dir / "base_temperature_parameters.csv")
    base_parameters = life_history_parameters_from_table(parameter_table)
    juvenile_fits = pd.read_csv(model_dir / "juvenile_survival_fit_parameters.csv")

    def parameters_for_juvenile_fit(model_key: str) -> LifeHistoryParameters:
        row = juvenile_fits.loc[juvenile_fits["model"] == model_key].iloc[0]
        function = str(row["function"])
        parameter_names = FUNCTION_PARAMETER_NAMES[function]
        juvenile_mortality = TemperatureResponse(
            name="Juvenile mortality rate",
            function=function,
            parameters=tuple(float(row[name]) for name in parameter_names),
        )
        return replace(base_parameters, juvenile_mortality=juvenile_mortality)

    single_parameters = parameters_for_juvenile_fit("m1_single_stage")
    substage_parameters = parameters_for_juvenile_fit("m2_m3_substage")
    count_table = pd.read_csv(model_dir / "stage_substage_counts.csv")
    substage_counts = {
        str(row.stage_key): int(row.substage_count)
        for row in count_table.itertuples(index=False)
    }
    single_counts = {key: 1 for key in substage_counts}
    juvenile_substage_counts = {**substage_counts, "adult": 1}

    exit_profile_path = model_dir / "adult_exit_chain_fecundity_profile.csv"
    exit_profile = pd.read_csv(exit_profile_path)
    exit_fecundity_profile_values = exit_profile["fecundity_profile"].to_numpy(
        dtype=float, copy=True
    )
    exit_fecundity_profile_values /= float(
        np.mean(exit_fecundity_profile_values)
    )
    exit_fecundity_profile = tuple(float(value) for value in exit_fecundity_profile_values)

    all_models = (
        ModelSpec(
            key="m1_single_stage",
            label="M1 single stages",
            scope="main",
            stage_counts=single_counts,
            parameters=single_parameters,
        ),
        ModelSpec(
            key="m2_substage_transit",
            label="M2 juvenile substages, single adult",
            scope="main",
            stage_counts=juvenile_substage_counts,
            parameters=substage_parameters,
        ),
        ModelSpec(
            key="m3_adult_exit_chain_fecundity",
            label="M3 adult exit chain + fecundity timing",
            scope="main",
            stage_counts=substage_counts,
            parameters=replace(
                substage_parameters,
                daily_fecundity_response=None,
                adult_fecundity_profile=exit_fecundity_profile,
                adult_mortality_weights=None,
            ),
        ),
    )
    return tuple(model for model in all_models if model.key in PAPER_MODEL_KEYS)


def intervention_grid(
    models: tuple[ModelSpec, ...],
    *,
    temperatures: np.ndarray,
    added_mortalities: np.ndarray,
    stage_group: str,
) -> pd.DataFrame:
    rows = []
    for model in models:
        for temperature in temperatures:
            baseline = perturbed_intrinsic_growth_rate(
                model.parameters,
                float(temperature),
                model.stage_counts,
                stage_group=stage_group,
                added_mortality=0.0,
            )
            for added_mortality in added_mortalities:
                r = perturbed_intrinsic_growth_rate(
                    model.parameters,
                    float(temperature),
                    model.stage_counts,
                    stage_group=stage_group,
                    added_mortality=float(added_mortality),
                )
                rows.append(
                    {
                        "intervention": stage_group,
                        "model": model.key,
                        "label": model.label,
                        "temperature": float(temperature),
                        "added_mortality_per_day": float(added_mortality),
                        "r": r,
                        "baseline_r": baseline,
                        "delta_r": r - baseline,
                    }
                )
    return pd.DataFrame(rows)


def perturbed_intrinsic_growth_rate(
    parameters: LifeHistoryParameters,
    temperature: float,
    stage_counts: dict[str, int],
    *,
    stage_group: str,
    added_mortality: float,
) -> float:
    matrix = fixed_temperature_matrix(parameters, temperature, stage_counts).copy()
    stage_slice = intervention_slice(stage_counts, stage_group)
    indices = np.arange(stage_slice.start, stage_slice.stop)
    matrix[indices, indices] -= added_mortality
    eigenvalues = np.linalg.eigvals(matrix)
    return float(np.max(eigenvalues.real))


def intervention_slice(stage_counts: dict[str, int], stage_group: str) -> slice:
    egg = stage_counts["egg"]
    larva = stage_counts["larva"]
    pupa = stage_counts["pupa"]
    adult = stage_counts["adult"]
    if stage_group == "egg":
        return slice(0, egg)
    if stage_group == "larva":
        return slice(egg, egg + larva)
    if stage_group == "pupa":
        return slice(egg + larva, egg + larva + pupa)
    if stage_group == "juvenile":
        return slice(0, egg + larva + pupa)
    if stage_group == "adult":
        return slice(egg + larva + pupa, egg + larva + pupa + adult)
    if stage_group == "broad":
        return slice(0, egg + larva + pupa + adult)
    raise ValueError(f"Unsupported stage group: {stage_group}")


def intervention_threshold_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["intervention", "model", "label", "temperature"]
    for keys, group in data.groupby(group_cols, sort=False):
        group = group.sort_values("added_mortality_per_day")
        threshold = critical_added_mortality(
            group["added_mortality_per_day"].to_numpy(dtype=float),
            group["r"].to_numpy(dtype=float),
        )
        rows.append(
            dict(
                zip(group_cols, keys),
                baseline_r=float(group["baseline_r"].iloc[0]),
                min_r=float(group["r"].iloc[-1]),
                critical_added_mortality_per_day=threshold,
            )
        )
    return pd.DataFrame(rows)


def critical_added_mortality(added: np.ndarray, r: np.ndarray) -> float:
    if r[0] <= 0:
        return 0.0
    crossing = np.flatnonzero(r <= 0)
    if len(crossing) == 0:
        return np.nan
    high = int(crossing[0])
    low = high - 1
    return float(np.interp(0.0, [r[high], r[low]], [added[high], added[low]]))

def plot_faceted_intervention(
    data: pd.DataFrame,
    *,
    y: str,
    xlabel: str,
    ylabel: str,
    output: Path,
) -> None:
    temperatures = sorted(data["temperature"].unique())
    columns = 2
    rows = int(np.ceil(len(temperatures) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(8.8, 3.6 * rows),
        constrained_layout=True,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax, temperature in zip(axes.ravel(), temperatures):
        temp_data = data.loc[data["temperature"] == temperature]
        for model, model_data in temp_data.groupby("model", sort=False):
            color, linestyle = model_style(str(model))
            model_data = model_data.sort_values("added_mortality_per_day")
            ax.plot(
                model_data["added_mortality_per_day"],
                model_data[y],
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=str(model_data["label"].iloc[0]),
            )
        ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle=":")
        ax.set_title(f"{temperature:g} C")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        clean_axis(ax)
    for ax in axes.ravel()[len(temperatures) :]:
        ax.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    save_figure(fig, output)


def plot_intervention_delta_r_composite(
    juvenile: pd.DataFrame,
    adult: pd.DataFrame,
    *,
    models: tuple[ModelSpec, ...],
    output: Path,
) -> None:
    figure_data = {
        "juvenile": {
            "data": juvenile,
            "ylabel": "Juvenile\nDelta r (1/day)",
            "xlabel": "Added juvenile mortality (1/day)",
        },
        "adult": {
            "data": adult,
            "ylabel": "Adult\nDelta r (1/day)",
            "xlabel": "Added adult mortality (1/day)",
        },
    }
    temperatures = sorted(
        pd.concat([juvenile["temperature"], adult["temperature"]]).unique()
    )
    colors = temperature_palette(temperatures)
    y_min = min(float(juvenile["delta_r"].min()), float(adult["delta_r"].min()))
    y_max = max(float(juvenile["delta_r"].max()), float(adult["delta_r"].max()))
    y_padding = max((y_max - y_min) * 0.08, 0.015)

    fig, axes = plt.subplots(
        2,
        len(models),
        figsize=manuscript_figsize(len(models), 2, panel_height=1.72),
        constrained_layout=True,
        sharey=True,
        squeeze=False,
    )
    for row_index, (intervention, settings) in enumerate(figure_data.items()):
        data = settings["data"]
        for col_index, model in enumerate(models):
            ax = axes[row_index, col_index]
            model_data = data.loc[data["model"] == model.key]
            if row_index == 0:
                ax.set_title(manuscript_model_key_label(model.key), fontweight="bold")
            for temperature in temperatures:
                temp_data = model_data.loc[
                    model_data["temperature"] == temperature
                ].sort_values("added_mortality_per_day")
                ax.plot(
                    temp_data["added_mortality_per_day"],
                    temp_data["delta_r"],
                    color=colors[float(temperature)],
                    linewidth=1.6,
                    label=f"{temperature:g} C",
                )
            ax.axhline(0.0, color="#333333", linewidth=0.9, linestyle=":")
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
            ax.set_xlabel(str(settings["xlabel"]))
            if col_index == 0:
                ax.set_ylabel(str(settings["ylabel"]))
            clean_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=len(temperatures),
        frameon=False,
        title="Temperature",
    )
    save_figure(fig, output)


def plot_manuscript_mortality_intervention_r(
    juvenile: pd.DataFrame,
    adult: pd.DataFrame,
    *,
    temperature: float,
    models: tuple[ModelSpec, ...],
    output: Path,
) -> None:
    panels = (
        (juvenile, "Added juvenile mortality (1/day)", "Intrinsic growth rate $r$"),
        (adult, "Added adult mortality (1/day)", "Intrinsic growth rate $r$"),
    )
    panel_data = [
        data.loc[np.isclose(data["temperature"].to_numpy(dtype=float), temperature)]
        for data, _, _ in panels
    ]
    if any(data.empty for data in panel_data):
        raise ValueError(f"No intervention data found for {temperature:g} C.")

    y_values = np.concatenate(
        [data["r"].to_numpy(dtype=float) for data in panel_data]
    )
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))
    y_padding = max((y_max - y_min) * 0.08, 0.02)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=manuscript_figsize(2, 1, panel_height=2.55),
        constrained_layout=False,
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.22,
        top=manuscript_legend_layout_top(fig, legend_y=MS_LEGEND_Y),
        wspace=0.18,
    )
    for ax, data, (_, xlabel, title) in zip(axes, panel_data, panels):
        for model in models:
            model_data = data.loc[data["model"] == model.key].sort_values(
                "added_mortality_per_day"
            )
            style = manuscript_model_style(model.key)
            ax.plot(
                model_data["added_mortality_per_day"],
                model_data["r"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                dash_capstyle=style["dash_capstyle"],
                label=manuscript_model_key_label(model.key),
            )
        ax.axhline(0.0, color="#333333", linewidth=0.9, linestyle=":")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylim(min(y_min - y_padding, -0.02), y_max + y_padding)
        clean_axis(ax)

    axes[0].set_ylabel("$r$ (1/day)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, MS_LEGEND_Y),
        ncols=len(models),
        frameon=False,
    )
    save_figure(fig, output)

def write_report(
    report: Path,
    *,
    juvenile_plot: Path,
    adult_plot: Path,
    juvenile_delta_plot: Path,
    adult_delta_plot: Path,
    composite_delta_plot: Path,
    manuscript_r_plot: Path,
    summary: pd.DataFrame,
) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Mortality Intervention Effects on r",
        "",
        "Interventions are post-fit perturbations of the fixed-temperature stage matrix.",
        "Added juvenile mortality is applied to egg, larval, and pupal substages.",
        "Added adult mortality is applied to adult substages.",
        "",
        image_line(report, composite_delta_plot, "Composite mortality intervention delta r"),
        "",
        image_line(report, manuscript_r_plot, "Manuscript 20 C mortality intervention r"),
        "",
        image_line(report, juvenile_plot, "Juvenile mortality intervention"),
        "",
        image_line(report, juvenile_delta_plot, "Juvenile mortality intervention delta r"),
        "",
        image_line(report, adult_plot, "Adult mortality intervention"),
        "",
        image_line(report, adult_delta_plot, "Adult mortality intervention delta r"),
        "",
        "## Critical Added Mortality",
        "",
        markdown_table(summary),
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


def image_line(report_path: Path, image_path: Path, alt: str) -> str:
    relative = os.path.relpath(image_path, start=report_path.parent)
    return f"![{alt}]({Path(relative).as_posix()})"


def markdown_table(data: pd.DataFrame) -> str:
    formatted = data.reset_index(drop=True).copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: "not reached" if pd.isna(value) else f"{value:.6g}"
            )
        else:
            formatted[column] = formatted[column].astype(str)
    headers = list(formatted.columns)
    rows = formatted.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(":--" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
