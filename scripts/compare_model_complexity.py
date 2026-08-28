from __future__ import annotations

import argparse
import os
import statistics
import tempfile
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.linalg import expm
from scipy.optimize import curve_fit, minimize
from scipy.stats import gamma

from r_r0_pop.data import (
    BaserPaths,
    female_preadult_distribution,
    load_or_create_baser_processed_data,
    pooled_adult_female_fraction,
)
from r_r0_pop.demography import build_reproduction_schedule, summarize_rates
from r_r0_pop.life_history_fits import (
    FUNCTION_PARAMETER_NAMES,
    FitResult,
    STAGE_COUNT_KEYS,
    STAGES,
    adult_delay_summary,
    adult_daily_reproduction_summary,
    adult_substage_occupancy,
    adult_substage_occupancy_with_mortality,
    competing_risk_transition_rate,
    daily_fecundity_summary,
    fit_q10_deactivation_response,
    fit_stage_delay_q10_deactivation,
    fit_juvenile_mortality,
    fit_skew_lifetime_fecundity,
    juvenile_mortality_summary_for_stage_chain,
    erlang_stage_counts,
    lifetime_fecundity_summary,
    maturation_delay_summary,
    stage_duration_observations,
)
from r_r0_pop.paper_config import (
    MS_LEGEND_Y,
    MS_MATURATION_SURVIVAL_PANEL_HEIGHT,
    ModelSpec,
    manuscript_figsize,
    manuscript_legend_layout_top,
    manuscript_model_key_label,
    manuscript_model_style,
    model_style,
)
from r_r0_pop.paths import (
    MANUSCRIPT_FIGURE_DIR,
    OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    REPORT_DIR,
    SOURCE_DATA_DIR,
    SUPPLEMENTARY_FIGURE_DIR,
)
from r_r0_pop.plotting import clean_axis, save_figure
from r_r0_pop.population_model import (
    LifeHistoryParameters,
    MANUSCRIPT_M1_STAGE_COUNTS,
    MANUSCRIPT_M2_STAGE_COUNTS,
    MANUSCRIPT_M3_STAGE_COUNTS,
    SimulationConfig,
    TemperatureResponse,
    life_history_parameters_from_table,
    model_demographic_rates,
    model_reproduction_kernel,
    simulate_single_season,
    temperature_sinusoid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare predefined model-complexity variants for paper figures."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SOURCE_DATA_DIR,
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
    )
    parser.add_argument("--rebuild-processed", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "model_complexity",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=OUTPUT_DIR / "plots" / "model_complexity",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_DIR / "model_complexity.md",
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
        "--supplementary-figure-dir",
        type=Path,
        default=SUPPLEMENTARY_FIGURE_DIR,
        help=(
            "Directory for publication-formatted supplementary figures. "
            "The model-only workflow redirects this to outputs/."
        ),
    )
    parser.add_argument("--temperature-min", type=float, default=10.0)
    parser.add_argument("--temperature-max", type=float, default=32.0)
    parser.add_argument("--temperature-step", type=float, default=0.1)
    parser.add_argument(
        "--max-substages",
        type=int,
        default=40,
        help="Upper bound for variance-informed substage counts.",
    )
    parser.add_argument(
        "--reuse-fits",
        action="store_true",
        help="Reuse fitted CSV outputs and regenerate plots/report without refitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    development, adult_survival, fertility = load_or_create_baser_processed_data(
        BaserPaths.from_data_dir(args.data_dir),
        args.processed_dir,
        rebuild=args.rebuild_processed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    count_path = args.output_dir / "stage_substage_counts.csv"
    durations = stage_duration_observations(development, adult_survival, fertility)
    if args.reuse_fits:
        count_table = pd.read_csv(count_path)
        substage_counts = {
            str(row.stage_key): int(row.substage_count)
            for row in count_table.itertuples(index=False)
        }
    else:
        substage_counts, count_table = erlang_stage_counts(
            durations, maximum=args.max_substages
        )
        count_table.to_csv(count_path, index=False)

    summaries = {
        "stage": maturation_delay_summary(development, fertility),
        "juvenile": juvenile_mortality_summary_for_stage_chain(
            development, adult_survival, substage_counts
        ),
        "juvenile_stage_survival": juvenile_stage_survival_observations(
            development, adult_survival
        ),
        "adult": adult_delay_summary(adult_survival),
        "fecundity": lifetime_fecundity_summary(fertility),
        "daily": daily_fecundity_summary(fertility),
        "adult_time": adult_daily_reproduction_summary(fertility).rename(
            columns={"mean_eggs": "value"}
        ),
    }
    parameter_path = args.output_dir / "base_temperature_parameters.csv"
    juvenile_survival_parameters_path = (
        args.output_dir / "juvenile_survival_fit_parameters.csv"
    )
    exit_fecundity_profile_path = (
        args.output_dir / "adult_exit_chain_fecundity_profile.csv"
    )
    exit_fecundity_predictions_path = (
        args.output_dir / "adult_exit_chain_fecundity_predictions.csv"
    )
    exit_fecundity_parameters_path = (
        args.output_dir / "adult_exit_chain_fecundity_parameters.csv"
    )
    direct_rates_path = args.output_dir / "direct_rates.csv"

    if args.reuse_fits:
        parameter_table = pd.read_csv(parameter_path)
        juvenile_survival_parameters = pd.read_csv(juvenile_survival_parameters_path)
    else:
        parameter_table = fit_base_parameter_table(summaries)
        single_counts = {key: 1 for key in substage_counts}
        stage_fits = fit_stage_parameters(summaries["stage"])
        juvenile_survival_parameters = fit_model_specific_juvenile_survival_parameters(
            parameter_table,
            summaries["juvenile"],
            stage_fits=stage_fits,
            single_counts=single_counts,
            substage_counts=substage_counts,
        )
        parameter_table = parameter_table_with_juvenile_fit(
            parameter_table,
            juvenile_survival_parameters.loc[
                juvenile_survival_parameters["model"] == "m2_m3_substage"
            ].iloc[0],
        )
        parameter_table.to_csv(parameter_path, index=False)
        juvenile_survival_parameters.to_csv(
            juvenile_survival_parameters_path, index=False
        )
    base_parameters = life_history_parameters_from_table(parameter_table)
    base_fitted_parameter_count = fitted_coefficient_count(parameter_table)

    if args.reuse_fits:
        exit_fecundity_profile = pd.read_csv(exit_fecundity_profile_path)
        exit_fecundity_parameters = pd.read_csv(exit_fecundity_parameters_path)
        exit_fecundity_profile_values = exit_fecundity_profile[
            "fecundity_profile"
        ].to_numpy(dtype=float)
        profile_parameters = exit_fecundity_parameters.iloc[0]
        exit_fecundity_profile_values = gaussian_adult_fecundity_profile(
            peak_location=float(profile_parameters["peak_location"]),
            sigma=float(profile_parameters["sigma"]),
            adult_stage_count=substage_counts["adult"],
        )
        exit_fecundity_profile_values /= float(np.mean(exit_fecundity_profile_values))
    else:
        exit_fecundity_fit = fit_exit_chain_fecundity_profile(
            summaries["adult_time"],
            summaries["fecundity"],
            base_parameters.adult_delay,
            base_parameters.lifetime_fecundity,
            adult_stage_count=substage_counts["adult"],
        )
        exit_fecundity_profile = exit_fecundity_fit["profile"]
        exit_fecundity_predictions = exit_fecundity_fit["predictions"]
        exit_fecundity_parameters = exit_fecundity_fit["parameters"]
        exit_fecundity_profile.to_csv(exit_fecundity_profile_path, index=False)
        exit_fecundity_predictions.to_csv(exit_fecundity_predictions_path, index=False)
        exit_fecundity_parameters.to_csv(exit_fecundity_parameters_path, index=False)
        exit_fecundity_profile_values = exit_fecundity_fit["fecundity_profile"]

    single_counts = dict(MANUSCRIPT_M1_STAGE_COUNTS)
    juvenile_substage_counts = {
        **dict(MANUSCRIPT_M2_STAGE_COUNTS),
        "egg": substage_counts["egg"],
        "larva": substage_counts["larva"],
        "pupa": substage_counts["pupa"],
    }
    substage_counts = {
        **dict(MANUSCRIPT_M3_STAGE_COUNTS),
        "egg": substage_counts["egg"],
        "larva": substage_counts["larva"],
        "pupa": substage_counts["pupa"],
        "adult": substage_counts["adult"],
    }
    single_parameters = life_history_parameters_from_table(
        parameter_table_with_juvenile_fit(
            parameter_table,
            juvenile_survival_parameters.loc[
                juvenile_survival_parameters["model"] == "m1_single_stage"
            ].iloc[0],
        )
    )
    substage_parameters = life_history_parameters_from_table(
        parameter_table_with_juvenile_fit(
            parameter_table,
            juvenile_survival_parameters.loc[
                juvenile_survival_parameters["model"] == "m2_m3_substage"
            ].iloc[0],
        )
    )
    models = build_model_specs(
        substage_parameters,
        single_counts=single_counts,
        juvenile_substage_counts=juvenile_substage_counts,
        substage_counts=substage_counts,
        single_parameters=single_parameters,
        exit_fecundity_profile=tuple(float(x) for x in exit_fecundity_profile_values),
        base_fitted_parameter_count=base_fitted_parameter_count,
        adult_timing_fitted_parameter_count=fitted_coefficient_count(
            exit_fecundity_parameters
        ),
    )
    offspring_female_fraction = pooled_adult_female_fraction(adult_survival)
    models = tuple(
        replace(
            model,
            parameters=replace(
                model.parameters,
                female_fraction=offspring_female_fraction,
            ),
        )
        for model in models
    )
    temperatures = np.arange(
        args.temperature_min,
        args.temperature_max + args.temperature_step / 2,
        args.temperature_step,
    )
    female_preadult = female_preadult_distribution(development, adult_survival)
    reproduction_schedule = build_reproduction_schedule(
        fertility,
        female_preadult,
        offspring_female_fraction=offspring_female_fraction,
    )
    direct_rates = summarize_rates(reproduction_schedule)
    direct_rates.to_csv(direct_rates_path, index=False)

    curves = {}
    at_direct = {}
    summary_rows = []
    for model in models:
        curves[model.key] = model_demographic_rates(
            model.parameters, temperatures, model.stage_counts
        )
        at_direct[model.key] = model_demographic_rates(
            model.parameters,
            direct_rates["temperature"].to_numpy(dtype=float),
            model.stage_counts,
        )
        curves[model.key].assign(model=model.key).to_csv(
            args.output_dir / f"{model.key}_demographic_rates.csv", index=False
        )
        at_direct[model.key].assign(model=model.key).to_csv(
            args.output_dir / f"{model.key}_rates_at_direct_temperatures.csv",
            index=False,
        )
        summary_rows.append(summarize_model(model, at_direct[model.key], direct_rates))

    main_models = [model for model in models if model.scope == "main"]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    seasonal_speed = summarize_seasonal_simulation_speed(main_models)
    seasonal_speed.to_csv(
        args.output_dir / "seasonal_simulation_speed.csv", index=False
    )
    seasonal_summary = summarize_seasonal_simulations(main_models)
    seasonal_summary.to_csv(
        args.output_dir / "seasonal_simulation_summary.csv", index=False
    )

    r_plot = args.plot_dir / "main_model_complexity_r.png"
    R0_plot = args.plot_dir / "main_model_complexity_R0.png"
    generation_time_plot = args.plot_dir / "main_model_complexity_generation_time.png"
    counts_plot = args.plot_dir / "stage_substage_counts.png"

    plot_demographic_models(
        curves,
        direct_rates,
        models=[model for model in models if model.scope == "main"],
        model_metric="r_model",
        direct_metric="r_euler",
        ylabel="Intrinsic rate of increase, r (1/day)",
        output=r_plot,
    )
    plot_demographic_models(
        curves,
        direct_rates,
        models=[model for model in models if model.scope == "main"],
        model_metric="R0_model",
        direct_metric="R0",
        ylabel="Net reproductive rate, R0",
        output=R0_plot,
    )
    plot_demographic_models(
        curves,
        direct_rates,
        models=[model for model in models if model.scope == "main"],
        model_metric="generation_time_model",
        direct_metric="generation_time_euler",
        ylabel="Euler-Lotka generation time (days)",
        output=generation_time_plot,
    )
    plot_substage_counts(count_table, output=counts_plot)
    shared_figures = write_shared_model_figures(
        summaries=summaries,
        durations=durations,
        plot_dir=args.plot_dir,
        main_models=main_models,
        publication_figure_dir=args.publication_figure_dir,
        supplementary_figure_dir=args.supplementary_figure_dir,
    )
    shared_figures["seasonal_simulation"] = (
        args.plot_dir / "seasonal_simulation_composite.png"
    )
    plot_seasonal_simulation_composite(
        main_models,
        output=shared_figures["seasonal_simulation"],
    )
    model_figures = {
        model.key: write_model_figures(
            model,
            summaries=summaries,
            direct_rates=direct_rates,
            reproduction_schedule=reproduction_schedule,
            curve=curves[model.key],
            plot_dir=args.plot_dir / model.key,
        )
        for model in main_models
    }
    manuscript_figures = write_manuscript_comparison_figures(
        summaries=summaries,
        curves=curves,
        direct_rates=direct_rates,
        main_models=main_models,
        figure_dir=args.publication_figure_dir,
        supplementary_figure_dir=args.supplementary_figure_dir,
    )

    write_report(
        args.report,
        output_dir=args.output_dir,
        summary=summary,
        count_table=count_table,
        plots={
            "main_r": r_plot,
            "main_R0": R0_plot,
            "generation_time": generation_time_plot,
            "counts": counts_plot,
        },
        main_models=main_models,
        shared_figures=shared_figures,
        model_figures=model_figures,
        manuscript_figures=manuscript_figures,
        seasonal_speed=seasonal_speed,
    )
    print(f"Wrote {args.output_dir / 'summary.csv'}")
    print(f"Wrote {args.report}")


def fit_base_parameter_table(
    summaries: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    stage_fits = fit_stage_parameters(summaries["stage"])
    fits: list[FitResult] = [
        *stage_fits.values(),
        fit_juvenile_mortality(summaries["juvenile"]),
        fit_q10_deactivation_response(
            summaries["adult"],
            name="Adult duration",
        ),
        fit_skew_lifetime_fecundity(summaries["fecundity"]),
    ]
    return pd.DataFrame([fit.as_dict() for fit in fits])


def fit_stage_parameters(
    stage_data: pd.DataFrame,
) -> dict[str, FitResult]:
    """Fit the stage-specific Q10 responses used by every paper model."""

    return {
        stage: fit_stage_delay_q10_deactivation(
            stage_data.loc[stage_data["stage"] == stage],
            stage,
        )
        for stage in STAGES
    }


def fit_model_specific_juvenile_survival_parameters(
    parameter_table: pd.DataFrame,
    juvenile_data: pd.DataFrame,
    *,
    stage_fits: dict[str, FitResult],
    single_counts: dict[str, int],
    substage_counts: dict[str, int],
) -> pd.DataFrame:
    rows = []
    for model, counts in (
        ("m1_single_stage", single_counts),
        ("m2_m3_substage", substage_counts),
    ):
        fit = fit_juvenile_mortality(
            juvenile_data,
            stage_fits=stage_fits,
            stage_counts=counts,
        )
        rows.append(
            {
                **fit.as_dict(),
                "model": model,
                "egg_substages": counts["egg"],
                "larva_substages": counts["larva"],
                "pupa_substages": counts["pupa"],
            }
        )
    return pd.DataFrame(rows)


def juvenile_stage_survival_observations(
    development: pd.DataFrame,
    adult_survival: pd.DataFrame,
    *,
    cohort_size: int = 50,
) -> pd.DataFrame:
    merged = adult_survival.merge(
        development, on=["temperature", "specimen"], how="inner"
    )
    merged["death_larva"] = (
        (merged["death_L1"] > 0) | (merged["death_L2"] > 0) | (merged["death_L3"] > 0)
    )
    merged["adult"] = (merged["AF"] > 0) | (merged["AM"] > 0)

    rows = []
    for temperature, data in merged.groupby("temperature", sort=True):
        egg_deaths = int((data["death_E"] > 0).sum())
        larva_deaths = int(data["death_larva"].sum())
        adults = int(data["adult"].sum())
        entered = {
            "Egg": cohort_size,
            "Larva": cohort_size - egg_deaths,
            "Pupa": cohort_size - egg_deaths - larva_deaths,
        }
        survived = {
            "Egg": cohort_size - egg_deaths,
            "Larva": cohort_size - egg_deaths - larva_deaths,
            "Pupa": adults,
        }
        for stage in STAGES:
            stage_entered = entered[stage]
            rows.append(
                {
                    "temperature": float(temperature),
                    "stage": stage,
                    "entered": stage_entered,
                    "survived": survived[stage],
                    "survival_probability": (
                        survived[stage] / stage_entered if stage_entered > 0 else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["stage", "temperature"])


def parameter_table_with_juvenile_fit(
    parameter_table: pd.DataFrame, juvenile_fit: pd.Series
) -> pd.DataFrame:
    updated = parameter_table.copy()
    functions = set(updated["function"].dropna()) | {juvenile_fit["function"]}
    response_columns = {
        column
        for function in functions
        for columns in (FUNCTION_PARAMETER_NAMES[function],)
        for column in columns
    }
    for column in ("function", *sorted(response_columns), "rss", "r2", "n"):
        if column not in updated.columns:
            updated[column] = np.nan
        updated.loc[updated["name"] == "Juvenile mortality rate", column] = (
            juvenile_fit.get(column, np.nan)
        )
    return updated


def fit_exit_chain_fecundity_profile(
    adult_time: pd.DataFrame,
    lifetime_fecundity: pd.DataFrame,
    adult_delay: TemperatureResponse,
    lifetime_fecundity_response: TemperatureResponse,
    *,
    adult_stage_count: int,
) -> dict[str, object]:
    daily = adult_time.dropna(subset=["value"]).copy()
    daily_temperatures = daily["temperature"].to_numpy(dtype=float)
    adult_days = daily["adult_day"].to_numpy(dtype=float)
    observed_daily = daily["value"].to_numpy(dtype=float)
    live_females = daily["live_females"].to_numpy(dtype=float)
    lifetime = lifetime_fecundity.loc[lifetime_fecundity["value"] > 0].copy()
    lifetime_temperatures = lifetime["temperature"].to_numpy(dtype=float)
    x0 = np.asarray(
        [
            0.45,
            np.log(0.18),
        ],
        dtype=float,
    )
    bounds = [
        (0.0, 1.0),
        (np.log(0.04), np.log(0.8)),
    ]

    def profile_from_vector(vector: np.ndarray) -> np.ndarray:
        raw_profile = gaussian_adult_fecundity_profile(
            peak_location=float(vector[0]),
            sigma=float(np.exp(vector[1])),
            adult_stage_count=adult_stage_count,
        )
        return raw_profile / float(np.mean(raw_profile))

    def daily_predictions(vector: np.ndarray) -> np.ndarray:
        profile = profile_from_vector(vector)
        daily_adult_delay = np.asarray(adult_delay(daily_temperatures), dtype=float)
        occupancy = adult_substage_occupancy(
            adult_days,
            daily_adult_delay,
            adult_stage_count=adult_stage_count,
        )
        base_daily = (
            np.asarray(lifetime_fecundity_response(daily_temperatures), dtype=float)
            / daily_adult_delay
        )
        return base_daily * occupancy.dot(profile)

    def objective(vector: np.ndarray) -> float:
        try:
            fitted_daily = daily_predictions(vector)
            if not np.all(np.isfinite(fitted_daily)):
                return 1e9
            return float(
                np.sum(live_females * (observed_daily - fitted_daily) ** 2)
                / np.sum(live_females * observed_daily**2)
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            return 1e9

    result = minimize(
        objective,
        x0=x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 250, "maxfun": 900, "ftol": 1e-9, "maxls": 20},
    )
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"Adult exit-chain fecundity fit failed: {result.message}")

    peak_location = float(result.x[0])
    sigma = float(np.exp(result.x[1]))
    raw_profile = gaussian_adult_fecundity_profile(
        peak_location=peak_location,
        sigma=sigma,
        adult_stage_count=adult_stage_count,
    )
    profile = raw_profile / float(np.mean(raw_profile))
    daily_adult_delay = np.asarray(adult_delay(daily_temperatures), dtype=float)
    daily_occupancy = adult_substage_occupancy(
        adult_days,
        daily_adult_delay,
        adult_stage_count=adult_stage_count,
    )
    fitted_daily = (
        np.asarray(lifetime_fecundity_response(daily_temperatures), dtype=float)
        / daily_adult_delay
        * daily_occupancy.dot(profile)
    )
    lifetime_predictions = np.asarray(
        lifetime_fecundity_response(lifetime_temperatures), dtype=float
    )
    canonical_objective = float(
        np.sum(live_females * (observed_daily - fitted_daily) ** 2)
        / np.sum(live_females * observed_daily**2)
    )
    profile_frame = pd.DataFrame(
        {
            "adult_substage": np.arange(1, adult_stage_count + 1),
            "fecundity_profile": profile,
        }
    )
    prediction_frame = pd.concat(
        [
            daily.assign(
                fitted_value=fitted_daily,
                target="daily_fecundity",
            ),
            lifetime.assign(
                adult_day=np.nan,
                live_females=np.nan,
                fitted_value=lifetime_predictions,
                target="lifetime_fecundity",
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    parameter_frame = pd.DataFrame(
        [
            {
                "name": "Adult fecundity profile",
                "function": "normalized_gaussian_adult_fecundity_profile",
                "peak_location": peak_location,
                "sigma": sigma,
                "objective": canonical_objective,
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
            }
        ]
    )
    return {
        "fecundity_profile": profile,
        "profile": profile_frame,
        "predictions": prediction_frame,
        "parameters": parameter_frame,
    }


def gaussian_adult_fecundity_profile(
    peak_location: float,
    sigma: float,
    *,
    adult_stage_count: int,
) -> np.ndarray:
    x = (np.arange(adult_stage_count, dtype=float) + 0.5) / adult_stage_count
    return np.exp(-0.5 * ((x - peak_location) / sigma) ** 2)


def build_model_specs(
    base_parameters: LifeHistoryParameters,
    *,
    single_counts: dict[str, int],
    juvenile_substage_counts: dict[str, int],
    substage_counts: dict[str, int],
    single_parameters: LifeHistoryParameters | None = None,
    exit_fecundity_profile: tuple[float, ...],
    base_fitted_parameter_count: int = 0,
    adult_timing_fitted_parameter_count: int = 0,
) -> tuple[ModelSpec, ...]:
    m1_parameters = base_parameters if single_parameters is None else single_parameters
    return (
        ModelSpec(
            key="m1_single_stage",
            label="M1 single stages",
            scope="main",
            stage_counts=single_counts,
            parameters=m1_parameters,
            fitted_parameter_count=base_fitted_parameter_count,
        ),
        ModelSpec(
            key="m2_substage_transit",
            label="M2 juvenile substages, single adult",
            scope="main",
            stage_counts=juvenile_substage_counts,
            parameters=base_parameters,
            fitted_parameter_count=base_fitted_parameter_count,
        ),
        ModelSpec(
            key="m3_adult_exit_chain_fecundity",
            label="M3 adult exit chain + fecundity timing",
            scope="main",
            stage_counts=substage_counts,
            parameters=replace(
                base_parameters,
                daily_fecundity_response=None,
                adult_fecundity_profile=exit_fecundity_profile,
                adult_mortality_weights=None,
            ),
            fitted_parameter_count=(
                base_fitted_parameter_count + adult_timing_fitted_parameter_count
            ),
        ),
    )


def fitted_coefficient_count(parameter_table: pd.DataFrame) -> int:
    """Count nonmissing fitted response coefficients in a parameter table."""

    return sum(
        sum(
            pd.notna(getattr(row, column))
            for column in FUNCTION_PARAMETER_NAMES[row.function]
        )
        for row in parameter_table.itertuples(index=False)
    )


def summarize_model(
    model: ModelSpec, rates: pd.DataFrame, direct_rates: pd.DataFrame
) -> dict[str, float | int | str]:
    model_r = rates["r_model"].to_numpy(dtype=float)
    model_R0 = rates["R0_model"].to_numpy(dtype=float)
    direct_r = direct_rates["r_euler"].to_numpy(dtype=float)
    direct_R0 = direct_rates["R0"].to_numpy(dtype=float)
    model_generation_time = rates["generation_time_model"].to_numpy(dtype=float)
    direct_generation_time = direct_rates["generation_time_euler"].to_numpy(dtype=float)
    return {
        "model": model.key,
        "label": model.label,
        "scope": model.scope,
        "state_count": int(sum(model.stage_counts.values())),
        "egg_substages": model.stage_counts["egg"],
        "larva_substages": model.stage_counts["larva"],
        "pupa_substages": model.stage_counts["pupa"],
        "adult_substages": model.stage_counts["adult"],
        "fitted_parameter_count": model.fitted_parameter_count,
        "r_relative_sse": relative_sse(model_r, direct_r),
        "R0_relative_sse": relative_sse(model_R0, direct_R0),
        "generation_time_relative_sse": relative_sse(
            model_generation_time, direct_generation_time
        ),
        "mean_R0_ratio_model_over_direct": float(np.mean(model_R0 / direct_R0)),
        "max_r_model": float(np.max(model_r)),
        "max_R0_model": float(np.max(model_R0)),
    }


def summarize_seasonal_simulation_speed(
    models: list[ModelSpec],
    *,
    repetitions: int = 10,
) -> pd.DataFrame:
    rows = []
    config_kwargs = {
        "start_day": 182.0,
        "end_day": 334.0,
        "initial_eggs": 1.0,
        "initial_adults": 0.0,
        "temperature_mean": 11.5,
        "temperature_amplitude": 16.5,
        "temperature_phase": -104.75,
    }
    for model in models:
        config = SimulationConfig(stage_counts=model.stage_counts, **config_kwargs)
        simulate_single_season(model.parameters, config)
        timings = []
        for _ in range(repetitions):
            start = time.perf_counter()
            simulate_single_season(model.parameters, config)
            timings.append(time.perf_counter() - start)
        rows.append(
            {
                "model": model.key,
                "label": model.label,
                "state_count": int(sum(model.stage_counts.values())),
                "egg_substages": model.stage_counts["egg"],
                "larva_substages": model.stage_counts["larva"],
                "pupa_substages": model.stage_counts["pupa"],
                "adult_substages": model.stage_counts["adult"],
                "repetitions": repetitions,
                "median_seconds": float(statistics.median(timings)),
                "mean_seconds": float(statistics.mean(timings)),
                "min_seconds": float(min(timings)),
                "max_seconds": float(max(timings)),
            }
        )
    return pd.DataFrame(rows)


def summarize_seasonal_simulations(models: list[ModelSpec]) -> pd.DataFrame:
    """Summarize the illustrative seasonal trajectories used in Figure 8."""

    rows = []
    for model in models:
        simulation = seasonal_simulation_for_model(model)
        total = simulation[["eggs", "larvae", "pupae", "adults"]].sum(axis=1)
        peak_position = int(np.argmax(total.to_numpy(dtype=float)))
        rows.append(
            {
                "model": model.key,
                "label": model.label,
                "peak_expected_female_abundance": float(total.iloc[peak_position]),
                "day_of_peak": float(simulation.iloc[peak_position]["day"]),
                "final_expected_female_abundance": float(total.iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def relative_sse(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Return SSE relative to the observed sum of squares."""

    return float(np.sum((predicted - observed) ** 2) / np.sum(observed**2))


STAGE_DURATION_LEGEND_FONTSIZE = 8.0
STAGE_DURATION_LEGEND_HANDLE_POINTS = 30
STAGE_DURATION_LEGEND_TEXT_PAD = 0.8
STAGE_DURATION_SURVIVAL_X_LIMIT_MULTIPLIER = 1.175
STAGE_DURATION_SURVIVAL_Y_PADDING = 0.055
STAGE_DURATION_DATA_COLOR = "black"
STAGE_DURATION_MEAN_DATA_COLOR = "black"
STAGE_DURATION_MEAN_MARKER_SIZE = 44.0
STAGE_DURATION_MEAN_LEGEND_MARKER_SIZE = 6.2
STAGE_DURATION_DATA_LINEWIDTH = 2.2
STAGE_DURATION_DATA_ALPHA = 0.78
MANUSCRIPT_CURVE_LINEWIDTH_SCALE = 1.0
MAIN_TEXT_MATURATION_SURVIVAL_TEMPERATURES = (13.0, 25.0, 29.0)
MAIN_TEXT_ADULT_REPRODUCTION_TEMPERATURES = (13.0, 25.0, 29.0)
STAGE_DURATION_LEGEND_HANDLE_LENGTH = (
    STAGE_DURATION_LEGEND_HANDLE_POINTS / STAGE_DURATION_LEGEND_FONTSIZE
)


class RightAlignedMarkerHandler(HandlerBase):
    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        marker = Line2D(
            [xdescent + width],
            [ydescent + 0.5 * height],
            marker=orig_handle.get_marker(),
            markersize=orig_handle.get_markersize(),
            markerfacecolor=orig_handle.get_markerfacecolor(),
            markeredgecolor=orig_handle.get_markeredgecolor(),
            markeredgewidth=orig_handle.get_markeredgewidth(),
            linestyle="None",
            transform=trans,
        )
        return [marker]


def manuscript_model_legend_handles(
    models: list[ModelSpec],
    *,
    data_color: str = STAGE_DURATION_DATA_COLOR,
    data_markersize: float = 6.0,
    data_handle: str = "marker",
    linewidth_scale: float = MANUSCRIPT_CURVE_LINEWIDTH_SCALE,
) -> list[Line2D]:
    if data_handle == "line":
        data_line = Line2D(
            [0],
            [0],
            color=data_color,
            linewidth=STAGE_DURATION_DATA_LINEWIDTH,
            alpha=STAGE_DURATION_DATA_ALPHA,
            label="Data",
        )
    else:
        data_line = Line2D(
            [0],
            [0],
            marker="o",
            color=data_color,
            markerfacecolor=data_color,
            markeredgecolor="white",
            markeredgewidth=0.45,
            linestyle="None",
            markersize=data_markersize,
            label="Data",
        )
    handles = [data_line]
    for model in models:
        style = manuscript_model_style(model.key)
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["legend_linestyle"],
                linewidth=float(style["linewidth"]) * linewidth_scale,
                dash_capstyle=style["dash_capstyle"],
                label=manuscript_model_key_label(model.key),
            )
        )
    return handles


def stage_row_label(stage: str) -> str:
    return stage


def add_composite_legend(
    fig: plt.Figure,
    models: list[ModelSpec],
    *,
    left: float = 0.0,
    bottom: float = 0.0,
    right: float = 0.985,
    data_color: str = STAGE_DURATION_DATA_COLOR,
    data_markersize: float = 6.0,
    data_handle: str = "marker",
    layout_top: float | None = None,
    legend_y: float = MS_LEGEND_Y,
    legend_gap: str = "normal",
    linewidth_scale: float = MANUSCRIPT_CURVE_LINEWIDTH_SCALE,
    w_pad: float | None = None,
    h_pad: float | None = None,
    wspace: float | None = None,
    hspace: float | None = None,
) -> None:
    if layout_top is None:
        layout_top = manuscript_legend_layout_top(
            fig,
            legend_y=legend_y,
            legend_gap=legend_gap,
        )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_kwargs: dict[str, float | tuple[float, float, float, float]] = {
            "rect": (left, bottom, right, layout_top)
        }
        if w_pad is not None:
            layout_kwargs["w_pad"] = w_pad
        if h_pad is not None:
            layout_kwargs["h_pad"] = h_pad
        if wspace is not None:
            layout_kwargs["wspace"] = wspace
        if hspace is not None:
            layout_kwargs["hspace"] = hspace
        layout_engine.set(**layout_kwargs)
    handles = manuscript_model_legend_handles(
        models,
        data_color=data_color,
        data_markersize=data_markersize,
        data_handle=data_handle,
        linewidth_scale=linewidth_scale,
    )
    handler_map = (
        {handles[0]: RightAlignedMarkerHandler()} if data_handle == "marker" else None
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=4,
        frameon=False,
        fontsize=STAGE_DURATION_LEGEND_FONTSIZE,
        handlelength=STAGE_DURATION_LEGEND_HANDLE_LENGTH,
        handletextpad=STAGE_DURATION_LEGEND_TEXT_PAD,
        handler_map=handler_map,
    )


def add_stage_row_labels(fig: plt.Figure, stages: list[str]) -> None:
    for row, stage in enumerate(stages):
        fig.text(
            0.012,
            1.0 - (row + 0.5) / len(stages),
            stage_row_label(stage),
            rotation="vertical",
            ha="center",
            va="center",
            fontsize=11,
        )


def add_aligned_stage_row_labels(
    fig: plt.Figure,
    axes: np.ndarray,
    stages: list[str],
    *,
    x: float = 0.012,
) -> None:
    fig.canvas.draw()
    for row, stage in enumerate(stages):
        visible_axes = [ax for ax in axes[row, :] if ax.get_visible()]
        if not visible_axes:
            continue
        bbox = visible_axes[0].get_position()
        fig.text(
            x,
            bbox.y0 + 0.5 * bbox.height,
            stage_row_label(stage),
            rotation="vertical",
            ha="center",
            va="center",
            fontsize=11,
        )


def add_temperature_row_labels(
    fig: plt.Figure,
    temperatures: list[float],
    *,
    top: float = 0.93,
    bottom: float = 0.0,
) -> None:
    plot_height = top - bottom
    for row, temperature in enumerate(temperatures):
        fig.text(
            0.012,
            top - (row + 0.5) * plot_height / len(temperatures),
            f"{temperature:g} °C",
            rotation="vertical",
            ha="center",
            va="center",
            fontsize=9,
        )


def add_aligned_stage_column_labels(
    fig: plt.Figure,
    axes: np.ndarray,
    stages: list[str],
    *,
    pad: float = 0.006,
) -> None:
    fig.canvas.draw()
    top = max(axes[0, col].get_position().y1 for col in range(axes.shape[1]))
    y = top + pad
    for col, stage in enumerate(stages):
        bbox = axes[0, col].get_position()
        fig.text(
            bbox.x0 + 0.5 * bbox.width,
            y,
            stage_row_label(stage),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def add_bottom_visible_xlabels(axes: np.ndarray, label: str) -> None:
    for col in range(axes.shape[1]):
        for row in reversed(range(axes.shape[0])):
            ax = axes[row, col]
            if ax.get_visible():
                ax.set_xlabel(label)
                break


def hide_inner_tick_labels(
    axes: np.ndarray,
    *,
    hide_x: bool = True,
    hide_y: bool = True,
) -> None:
    for row in range(axes.shape[0]):
        visible_cols = [
            col for col in range(axes.shape[1]) if axes[row, col].get_visible()
        ]
        last_visible_col = visible_cols[-1] if visible_cols else None
        for col in range(axes.shape[1]):
            ax = axes[row, col]
            if not ax.get_visible():
                continue
            if hide_x and row < axes.shape[0] - 1:
                ax.tick_params(labelbottom=False)
            if hide_y and col > 0:
                ax.tick_params(labelleft=False)
            if hide_x and last_visible_col is not None and row == axes.shape[0] - 1:
                ax.tick_params(labelbottom=True)
            if hide_y and col == 0:
                ax.tick_params(labelleft=True)


def hide_repeated_row_tick_labels(axes: np.ndarray) -> None:
    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            ax = axes[row, col]
            if not ax.get_visible():
                continue
            if col > 0:
                ax.tick_params(labelbottom=False, labelleft=False)
            else:
                ax.tick_params(labelbottom=True, labelleft=True)


def plot_demographic_models(
    curves: dict[str, pd.DataFrame],
    direct: pd.DataFrame,
    *,
    models: list[ModelSpec],
    model_metric: str,
    direct_metric: str,
    ylabel: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 4.55), constrained_layout=True)
    if model_metric == "R0_model":
        ax.axhline(1.0, color="#333333", linewidth=0.9, linestyle=":")
    for model in models:
        color, linestyle = model_style(model.key)
        curve = curves[model.key]
        ax.plot(
            curve["temperature"],
            curve[model_metric],
            color=color,
            linestyle=linestyle,
            linewidth=2.1,
            label=model.label,
        )
    ax.scatter(
        direct["temperature"],
        direct[direct_metric],
        color="black",
        edgecolor="white",
        linewidth=0.7,
        s=48,
        zorder=3,
        label="Direct data",
    )
    first_curve = curves[models[0].key]
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(first_curve["temperature"].min(), first_curve["temperature"].max())
    clean_axis(ax)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, output)


def plot_substage_counts(count_table: pd.DataFrame, *, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    data = count_table.copy()
    ax.bar(data["stage"], data["substage_count"], color="#2364aa")
    ax.set_ylabel("Predefined substages")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    clean_axis(ax)
    save_figure(fig, output)


def plot_adult_timing_weights(weights: pd.DataFrame, *, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.75), constrained_layout=True)
    axes[0].plot(
        weights["adult_substage"],
        weights["reproduction_weight"],
        color="#00847e",
        marker="o",
        linewidth=2.0,
    )
    axes[1].plot(
        weights["adult_substage"],
        weights["mortality_weight"],
        color="#7b5ea7",
        marker="o",
        linewidth=2.0,
    )
    axes[0].set_title("Adult fecundity")
    axes[1].set_title("Adult mortality")
    for ax in axes:
        ax.axhline(1.0, color="#888888", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Adult substage")
        ax.set_ylabel("Relative weight")
        clean_axis(ax)
    save_figure(fig, output)


def plot_adult_reproduction_timing(
    predictions: pd.DataFrame,
    parameters: LifeHistoryParameters,
    reproduction_weights: np.ndarray,
    *,
    output: Path,
) -> None:
    daily = predictions.dropna(subset=["mean_eggs"]).copy()
    temperatures = sorted(daily["temperature"].dropna().unique())
    columns = 3
    rows = int(np.ceil(len(temperatures) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12.0, 2.85 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, temperature in zip(axes.flat, temperatures):
        temp_data = daily.loc[daily["temperature"] == temperature].sort_values(
            "adult_day"
        )
        adult_days = temp_data["adult_day"].to_numpy(dtype=float)
        adult_delay = np.repeat(
            float(parameters.adult_delay(float(temperature))), len(adult_days)
        )
        occupancy = adult_substage_occupancy(
            adult_days,
            adult_delay,
            adult_stage_count=len(reproduction_weights),
        )
        fitted = float(parameters.daily_fecundity(float(temperature))) * occupancy.dot(
            reproduction_weights
        )
        ax.scatter(
            temp_data["adult_day"],
            temp_data["mean_eggs"],
            s=16 + 1.5 * temp_data["live_females"],
            color="black",
            alpha=0.36,
            label="Observed",
        )
        ax.plot(adult_days, fitted, color="#00847e", linewidth=2.0, label="Adult aging")
        ax.set_title(f"{temperature:g} C")
        ax.set_xlabel("Adult day")
        ax.set_ylabel("Eggs per live female")
        clean_axis(ax)
    for ax in axes.flat[len(temperatures) :]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8)
    save_figure(fig, output)


def plot_adult_survival_timing(predictions: pd.DataFrame, *, output: Path) -> None:
    data = predictions.dropna(subset=["survival_fraction"]).copy()
    temperatures = sorted(data["temperature"].unique())
    columns = 3
    rows = int(np.ceil(len(temperatures) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12.0, 2.85 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for panel_index, (ax, temperature) in enumerate(zip(axes.flat, temperatures)):
        temp_data = data.loc[data["temperature"] == temperature].sort_values(
            "adult_day"
        )
        ax.scatter(
            temp_data["adult_day"],
            temp_data["survival_fraction"],
            s=18 + 1.4 * temp_data["initial_females"],
            color="black",
            alpha=0.42,
            label="Observed",
        )
        ax.plot(
            temp_data["adult_day"],
            temp_data["fitted_survival_fraction"],
            color="#7b5ea7",
            linewidth=2.0,
            label="Adult aging",
        )
        ax.set_title(f"{temperature:g} C")
        ax.set_xlabel("Adult day")
        ax.set_ylabel("Live fraction")
        ax.set_ylim(-0.03, 1.05)
        clean_axis(ax)
    for ax in axes.flat[len(temperatures) :]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8)
    save_figure(fig, output)


def plot_reproduction_kernel_panel(
    schedule: pd.DataFrame,
    model: ModelSpec,
    *,
    output: Path,
) -> None:
    temperatures = sorted(schedule["temperature"].dropna().unique())
    columns = 3
    rows = int(np.ceil(len(temperatures) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12.0, 2.95 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    color = model_style(model.key)[0]
    for ax, temperature in zip(axes.flat, temperatures):
        temp_schedule = schedule.loc[
            schedule["temperature"] == temperature
        ].sort_values("age_days")
        if temp_schedule.empty:
            ax.set_visible(False)
            continue
        min_age = max(0.0, float(temp_schedule["age_days"].min()) - 5.0)
        max_age = float(temp_schedule["age_days"].max()) + 8.0
        age_grid = np.linspace(min_age, max_age, 350)
        kernel = model_reproduction_kernel(
            model.parameters,
            float(temperature),
            age_grid,
            model.stage_counts,
        )
        ax.scatter(
            temp_schedule["age_days"],
            temp_schedule["lx"] * temp_schedule["mx"],
            s=18 + 1.7 * temp_schedule["live_females"],
            color="black",
            alpha=0.45,
            label="Direct schedule",
        )
        ax.plot(age_grid, kernel, color=color, linewidth=2.0, label=model.label)
        ax.set_title(f"{temperature:g} C")
        ax.set_xlabel("Age from egg (days)")
        ax.set_ylabel("Female offspring per egg-day")
        clean_axis(ax)
    for ax in axes.flat[len(temperatures) :]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8)
    save_figure(fig, output)


def write_shared_model_figures(
    *,
    summaries: dict[str, pd.DataFrame],
    durations: pd.DataFrame,
    plot_dir: Path,
    main_models: list[ModelSpec],
    publication_figure_dir: Path = MANUSCRIPT_FIGURE_DIR,
    supplementary_figure_dir: Path = SUPPLEMENTARY_FIGURE_DIR,
) -> dict[str, Path]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    publication_figure_dir.mkdir(parents=True, exist_ok=True)
    supplementary_figure_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "stage_durations": plot_dir / "mean_stage_durations.png",
        "manuscript_stage_durations": publication_figure_dir
        / "mean_stage_durations.pdf",
        "delay_distributions": plot_dir / "delay_distributions.png",
        "delay_distributions_flipped": plot_dir / "delay_distributions_flipped.png",
        "delay_distributions_main_stage_xrange": plot_dir
        / "delay_distributions_main_stage_xrange.png",
        "delay_distributions_main_juvenile_stage_xrange": plot_dir
        / "delay_distributions_main_juvenile_stage_xrange.png",
        "maturation_survival": plot_dir / "maturation_survival.png",
        "maturation_survival_flipped": plot_dir / "maturation_survival_flipped.png",
        "maturation_survival_main_stage_xrange": plot_dir
        / "maturation_survival_main_stage_xrange.png",
        "manuscript_maturation_survival": publication_figure_dir
        / "maturation_survival.pdf",
        "si_maturation_survival": supplementary_figure_dir
        / "Figure_S1_maturation_survival_all_temperatures.pdf",
    }
    plot_stage_duration_panel(summaries, main_models, output=paths["stage_durations"])
    plot_stage_duration_panel(
        summaries,
        main_models,
        output=paths["manuscript_stage_durations"],
        figsize=manuscript_figsize(2, 2),
    )
    plot_delay_distributions(
        durations, main_models, output=paths["delay_distributions"]
    )
    plot_delay_distributions(
        durations,
        main_models,
        output=paths["delay_distributions_flipped"],
        transpose=True,
    )
    plot_delay_distributions(
        durations,
        main_models,
        output=paths["delay_distributions_main_stage_xrange"],
        temperature_subset=MAIN_TEXT_MATURATION_SURVIVAL_TEMPERATURES,
        share_x_by_stage=True,
    )
    plot_delay_distributions(
        durations,
        main_models,
        output=paths["delay_distributions_main_juvenile_stage_xrange"],
        temperature_subset=MAIN_TEXT_MATURATION_SURVIVAL_TEMPERATURES,
        share_x_by_stage=True,
        stage_subset=("Egg", "Larva"),
    )
    plot_maturation_survival(
        durations, main_models, output=paths["maturation_survival"]
    )
    plot_maturation_survival(
        durations,
        main_models,
        output=paths["maturation_survival_flipped"],
        transpose=True,
    )
    plot_maturation_survival(
        durations,
        main_models,
        output=paths["maturation_survival_main_stage_xrange"],
        temperature_subset=MAIN_TEXT_MATURATION_SURVIVAL_TEMPERATURES,
        share_x_by_stage=True,
    )
    plot_maturation_survival(
        durations,
        main_models,
        output=paths["manuscript_maturation_survival"],
        temperature_subset=MAIN_TEXT_MATURATION_SURVIVAL_TEMPERATURES,
        share_x_by_stage=True,
        figsize=manuscript_figsize(
            3, 4, panel_height=MS_MATURATION_SURVIVAL_PANEL_HEIGHT
        ),
    )
    plot_maturation_survival(
        durations,
        main_models,
        output=paths["si_maturation_survival"],
        figsize=(18.0, 7.2),
    )
    return paths


def write_model_figures(
    model: ModelSpec,
    *,
    summaries: dict[str, pd.DataFrame],
    direct_rates: pd.DataFrame,
    reproduction_schedule: pd.DataFrame,
    curve: pd.DataFrame,
    plot_dir: Path,
) -> dict[str, Path]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "juvenile": plot_dir / "juvenile_maturation_and_mortality.png",
        "daily_fecundity": plot_dir / "daily_fecundity.png",
        "lifetime_fecundity": plot_dir / "lifetime_adult_fecundity.png",
        "adult_fecundity_time": plot_dir / "adult_fecundity_over_time.png",
        "adult_timing_profile": plot_dir / "adult_fecundity_mortality_timing.png",
        "reproduction_kernel": plot_dir / "reproduction_kernel.png",
        "demographics": plot_dir / "demographics_r_R0.png",
        "simulation": plot_dir / "seasonal_simulation.png",
    }
    plot_juvenile_panel(summaries["juvenile"], model, output=paths["juvenile"])
    if shows_single_daily_fecundity(model):
        plot_daily_fecundity_panel(
            summaries["daily"], model, output=paths["daily_fecundity"]
        )
    else:
        paths.pop("daily_fecundity")
    plot_lifetime_fecundity_panel(
        summaries["fecundity"], model, output=paths["lifetime_fecundity"]
    )
    plot_adult_fecundity_time_panel(
        summaries["adult_time"], model, output=paths["adult_fecundity_time"]
    )
    if has_adult_timing_profile(model):
        plot_adult_timing_profile_panel(model, output=paths["adult_timing_profile"])
    else:
        paths.pop("adult_timing_profile")
    plot_reproduction_kernel_panel(
        reproduction_schedule,
        model,
        output=paths["reproduction_kernel"],
    )
    plot_model_demographics(curve, direct_rates, model, output=paths["demographics"])
    plot_model_simulation(model, output=paths["simulation"])
    return paths


def write_manuscript_comparison_figures(
    *,
    summaries: dict[str, pd.DataFrame],
    curves: dict[str, pd.DataFrame],
    direct_rates: pd.DataFrame,
    main_models: list[ModelSpec],
    figure_dir: Path = MANUSCRIPT_FIGURE_DIR,
    supplementary_figure_dir: Path = SUPPLEMENTARY_FIGURE_DIR,
) -> dict[str, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    supplementary_figure_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "juvenile_temperature": figure_dir
        / "juvenile_survival_mortality_temperature.pdf",
        "fecundity_temperature": figure_dir / "fecundity_temperature.pdf",
        "adult_reproduction_time": figure_dir / "adult_reproduction_time.pdf",
        "adult_reproduction_time_full": supplementary_figure_dir
        / "Figure_S2_adult_reproduction_all_temperatures.pdf",
        "demographic_metrics": figure_dir / "demographic_metrics.pdf",
        "seasonal_simulation": figure_dir / "seasonal_simulation_composite.pdf",
    }
    plot_juvenile_temperature_comparison(
        summaries["juvenile"],
        summaries["juvenile_stage_survival"],
        main_models,
        output=paths["juvenile_temperature"],
        show_mortality_data=False,
    )
    plot_fecundity_temperature_comparison(
        daily=summaries["daily"],
        lifetime=summaries["fecundity"],
        models=main_models,
        output=paths["fecundity_temperature"],
    )
    plot_adult_reproduction_time_comparison(
        summaries["adult_time"],
        main_models,
        output=paths["adult_reproduction_time"],
        temperature_subset=MAIN_TEXT_ADULT_REPRODUCTION_TEMPERATURES,
    )
    plot_adult_reproduction_time_comparison(
        summaries["adult_time"],
        main_models,
        output=paths["adult_reproduction_time_full"],
    )
    plot_demographic_metric_row(
        curves,
        direct_rates,
        models=main_models,
        output=paths["demographic_metrics"],
    )
    plot_seasonal_simulation_composite(
        main_models,
        output=paths["seasonal_simulation"],
    )
    return paths


def shows_single_daily_fecundity(model: ModelSpec) -> bool:
    profile = adult_fecundity_profile_values(model)
    return (
        model.stage_counts["adult"] == 1
        or profile is None
        or np.allclose(profile, profile[0])
    )


def has_adult_timing_profile(model: ModelSpec) -> bool:
    return (
        adult_fecundity_profile_values(model) is not None
        or model.parameters.adult_mortality_weights is not None
    )


def adult_fecundity_profile_values(model: ModelSpec) -> np.ndarray | None:
    profile = model.parameters.adult_fecundity_profile
    if profile is None:
        return None
    return np.asarray(profile, dtype=float)


def manuscript_model_label(model: ModelSpec) -> str:
    return manuscript_model_key_label(model.key)


def plot_model_curve(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    model: ModelSpec,
    *,
    label: str | None = None,
    linewidth_scale: float = 1.0,
) -> None:
    style = manuscript_model_style(model.key)
    ax.plot(
        x,
        y,
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=float(style["linewidth"]) * linewidth_scale,
        solid_capstyle="butt",
        dash_capstyle=style["dash_capstyle"],
        label=label or manuscript_model_label(model),
    )


def juvenile_survival_probability(
    model: ModelSpec, temperatures: np.ndarray
) -> np.ndarray:
    mortality = model.parameters.juvenile_mortality(temperatures)
    egg_rate = competing_risk_transition_rate(
        model.stage_counts["egg"],
        model.parameters.egg_delay(temperatures),
        mortality,
    )
    larva_rate = competing_risk_transition_rate(
        model.stage_counts["larva"],
        model.parameters.larva_delay(temperatures),
        mortality,
    )
    pupa_rate = competing_risk_transition_rate(
        model.stage_counts["pupa"],
        model.parameters.pupa_delay(temperatures),
        mortality,
    )
    return (
        (egg_rate / (egg_rate + mortality)) ** model.stage_counts["egg"]
        * (larva_rate / (larva_rate + mortality)) ** model.stage_counts["larva"]
        * (pupa_rate / (pupa_rate + mortality)) ** model.stage_counts["pupa"]
    )


def juvenile_stage_survival_probability(
    model: ModelSpec, stage: str, temperatures: np.ndarray
) -> np.ndarray:
    stage_key = STAGE_COUNT_KEYS[stage]
    stage_count = model.stage_counts[stage_key]
    mortality = model.parameters.juvenile_mortality(temperatures)
    delay_response = getattr(model.parameters, f"{stage_key}_delay")
    stage_rate = competing_risk_transition_rate(
        stage_count, delay_response(temperatures), mortality
    )
    return (stage_rate / (stage_rate + mortality)) ** stage_count


def model_mean_daily_fecundity(
    model: ModelSpec, temperatures: np.ndarray
) -> np.ndarray:
    daily = np.asarray(model.parameters.daily_fecundity(temperatures), dtype=float)
    profile = adult_fecundity_profile_values(model)
    if profile is None:
        return daily
    return daily * float(np.mean(profile))


def plot_juvenile_temperature_comparison(
    data: pd.DataFrame,
    stage_data: pd.DataFrame,
    models: list[ModelSpec],
    *,
    output: Path,
    show_mortality_data: bool = True,
) -> None:
    data = data.sort_values("temperature")
    stage_data = stage_data.sort_values(["stage", "temperature"])
    x_min = float(data["temperature"].min())
    x_max = float(data["temperature"].max())
    x_padding = 0.035 * (x_max - x_min)
    x_limits = (x_min - x_padding, x_max + x_padding)
    x_grid = np.linspace(x_limits[0], x_limits[1], 300)
    fig = plt.figure(figsize=manuscript_figsize(3, 2), constrained_layout=True)
    grid = fig.add_gridspec(2, 6)
    total_ax = fig.add_subplot(grid[0, 1:3])
    mortality_ax = fig.add_subplot(grid[0, 3:5])
    stage_axes = {
        "Egg": fig.add_subplot(grid[1, 0:2]),
        "Larva": fig.add_subplot(grid[1, 2:4]),
        "Pupa": fig.add_subplot(grid[1, 4:6]),
    }
    total_ax.scatter(
        data["temperature"],
        data["survival_probability"],
        color=STAGE_DURATION_MEAN_DATA_COLOR,
        edgecolor="white",
        linewidth=0.45,
        s=STAGE_DURATION_MEAN_MARKER_SIZE,
        zorder=4,
        label="Data",
    )
    if show_mortality_data:
        mortality_ax.scatter(
            data["temperature"],
            data["value"],
            color=STAGE_DURATION_MEAN_DATA_COLOR,
            edgecolor="white",
            linewidth=0.45,
            s=STAGE_DURATION_MEAN_MARKER_SIZE,
            zorder=4,
            label="Data",
        )
    for stage, ax in stage_axes.items():
        observed = stage_data.loc[stage_data["stage"] == stage]
        ax.scatter(
            observed["temperature"],
            observed["survival_probability"],
            color=STAGE_DURATION_MEAN_DATA_COLOR,
            edgecolor="white",
            linewidth=0.45,
            s=STAGE_DURATION_MEAN_MARKER_SIZE,
            zorder=4,
            label="Data",
        )
    for model in models:
        plot_model_curve(
            total_ax,
            x_grid,
            juvenile_survival_probability(model, x_grid),
            model,
            linewidth_scale=MANUSCRIPT_CURVE_LINEWIDTH_SCALE,
        )
        plot_model_curve(
            mortality_ax,
            x_grid,
            model.parameters.juvenile_mortality(x_grid),
            model,
            linewidth_scale=MANUSCRIPT_CURVE_LINEWIDTH_SCALE,
        )
        for stage, ax in stage_axes.items():
            plot_model_curve(
                ax,
                x_grid,
                juvenile_stage_survival_probability(model, stage, x_grid),
                model,
                linewidth_scale=MANUSCRIPT_CURVE_LINEWIDTH_SCALE,
            )
    total_ax.set_title("Total juvenile survival")
    total_ax.set_ylabel("Egg-to-adult survival probability")
    mortality_ax.set_title("Juvenile mortality")
    mortality_ax.set_ylabel("Mortality rate (1/day)")
    mortality_ax.set_ylim(0.0, 0.2)
    for stage, ax in stage_axes.items():
        ax.set_title(f"{stage} survival")
        ax.set_ylabel("Conditional survival")
        ax.set_ylim(-0.03, 1.03)
    for ax in [total_ax, mortality_ax, *stage_axes.values()]:
        ax.set_xlabel("Temperature (°C)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_xlim(*x_limits)
        clean_axis(ax)
    add_composite_legend(
        fig,
        models,
        data_color=STAGE_DURATION_MEAN_DATA_COLOR,
        data_markersize=STAGE_DURATION_MEAN_LEGEND_MARKER_SIZE,
    )
    save_figure(fig, output)


def plot_fecundity_temperature_comparison(
    *,
    daily: pd.DataFrame,
    lifetime: pd.DataFrame,
    models: list[ModelSpec],
    output: Path,
) -> None:
    daily = daily.sort_values("temperature")
    lifetime = lifetime.sort_values("temperature")
    x_min = float(min(daily["temperature"].min(), lifetime["temperature"].min()))
    x_max = float(max(daily["temperature"].max(), lifetime["temperature"].max()))
    x_grid = np.linspace(x_min, x_max, 300)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=manuscript_figsize(2, 1),
        constrained_layout=True,
    )
    panels = [
        (
            axes[0],
            lifetime,
            "Total adult fecundity",
            "Eggs per female",
            lambda model, x: model.parameters.lifetime_fecundity(x),
        ),
        (
            axes[1],
            daily,
            "Daily reproduction",
            "Eggs per live female-day",
            model_mean_daily_fecundity,
        ),
    ]
    for ax, data, title, ylabel, response in panels:
        ax.scatter(
            data["temperature"],
            data["value"],
            color=STAGE_DURATION_MEAN_DATA_COLOR,
            edgecolor="white",
            linewidth=0.45,
            s=STAGE_DURATION_MEAN_MARKER_SIZE,
            zorder=4,
            label="Data",
        )
        for model in models:
            plot_model_curve(
                ax,
                x_grid,
                np.asarray(response(model, x_grid), dtype=float),
                model,
                linewidth_scale=MANUSCRIPT_CURVE_LINEWIDTH_SCALE,
            )
        ax.set_title(title)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        clean_axis(ax)
    add_composite_legend(
        fig,
        models,
        data_color=STAGE_DURATION_MEAN_DATA_COLOR,
        data_markersize=STAGE_DURATION_MEAN_LEGEND_MARKER_SIZE,
    )
    save_figure(fig, output)


def plot_adult_reproduction_time_comparison(
    data: pd.DataFrame,
    models: list[ModelSpec],
    *,
    output: Path,
    temperature_subset: tuple[float, ...] | None = None,
) -> None:
    temperatures = selected_temperatures(data, temperature_subset)
    data = data.loc[data["temperature"].isin(temperatures)]
    x_min = 0.0
    x_max = float(np.ceil(data["adult_day"].max()))
    adult_day_grid = np.linspace(x_min, x_max, 300)
    y_limit = (0.0, 15.0)
    columns = 3
    rows = int(np.ceil(len(temperatures) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=manuscript_figsize(columns, rows, panel_height=2.45),
        constrained_layout=True,
        squeeze=False,
    )
    for panel_index, (ax, temperature) in enumerate(zip(axes.flat, temperatures)):
        temp_data = data.loc[data["temperature"] == temperature].sort_values(
            "adult_day"
        )
        adult_days = temp_data["adult_day"].to_numpy(dtype=float)
        in_range = temp_data["value"] <= y_limit[1]
        in_range_data = temp_data.loc[in_range]
        outlier_data = temp_data.loc[~in_range]
        ax.scatter(
            in_range_data["adult_day"],
            in_range_data["value"],
            s=30,
            alpha=STAGE_DURATION_DATA_ALPHA,
            color=STAGE_DURATION_DATA_COLOR,
            edgecolor="white",
            linewidth=0.35,
            label="Data",
            zorder=5,
        )
        if not outlier_data.empty:
            clipped_y = np.repeat(y_limit[1] * 0.985, len(outlier_data))
            ax.scatter(
                outlier_data["adult_day"],
                clipped_y,
                marker="^",
                s=48,
                alpha=0.92,
                color=STAGE_DURATION_DATA_COLOR,
                edgecolor="white",
                linewidth=0.4,
                zorder=6,
            )
            for row in outlier_data.itertuples(index=False):
                ax.text(
                    float(row.adult_day),
                    y_limit[1] * 0.91,
                    f"{float(row.value):.0f}",
                    ha="center",
                    va="top",
                    fontsize=7,
                    color=STAGE_DURATION_DATA_COLOR,
                )
        for model in models:
            plot_model_curve(
                ax,
                adult_day_grid,
                adult_time_fecundity(model, float(temperature), adult_day_grid),
                model,
                linewidth_scale=MANUSCRIPT_CURVE_LINEWIDTH_SCALE,
            )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(*y_limit)
        ax.set_title(f"{temperature:g} °C", fontsize=8.5, fontweight="bold")
        clean_axis(ax)
    for ax in axes.flat[len(temperatures) :]:
        ax.set_visible(False)
    hide_inner_tick_labels(axes)
    fig.supxlabel("Days")
    fig.supylabel("Eggs per live female")
    add_composite_legend(
        fig,
        models,
        left=0.055,
        bottom=0.055,
        w_pad=0.04,
        h_pad=0.08,
        wspace=0.1,
        legend_gap="loose" if temperature_subset is None else "normal",
        data_color=STAGE_DURATION_DATA_COLOR,
        data_markersize=7.0,
    )
    save_figure(fig, output)


def plot_demographic_metric_row(
    curves: dict[str, pd.DataFrame],
    direct_rates: pd.DataFrame,
    *,
    models: list[ModelSpec],
    output: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=manuscript_figsize(3, 1),
        constrained_layout=True,
    )
    panels = [
        (
            axes[0],
            "R0_model",
            "R0",
            "Net reproductive rate $R_0$",
            "$R_0$",
        ),
        (
            axes[1],
            "r_model",
            "r_euler",
            "Intrinsic growth rate $r$",
            "$r$ (1/day)",
        ),
        (
            axes[2],
            "generation_time_model",
            "generation_time_euler",
            r"Generation time $G$",
            r"$G$ (days)",
        ),
    ]
    observed_temperature_min = float(direct_rates["temperature"].min())
    observed_temperature_max = float(direct_rates["temperature"].max())
    first_curve = curves[models[0].key]
    curve_temperature_min = float(first_curve["temperature"].min())
    curve_temperature_max = float(first_curve["temperature"].max())
    for ax, model_metric, direct_metric, title, ylabel in panels:
        ax.axvspan(
            curve_temperature_min,
            observed_temperature_min,
            color="#eeeeee",
            zorder=0,
        )
        ax.axvspan(
            observed_temperature_max,
            curve_temperature_max,
            color="#eeeeee",
            zorder=0,
        )
        if model_metric == "r_model":
            ax.axhline(0.0, color="#333333", linewidth=0.9, linestyle=":")
        elif model_metric == "R0_model":
            ax.axhline(1.0, color="#333333", linewidth=0.9, linestyle=":")
        for model in models:
            curve = curves[model.key].sort_values("temperature")
            plot_model_curve(
                ax,
                curve["temperature"].to_numpy(dtype=float),
                curve[model_metric].to_numpy(dtype=float),
                model,
                linewidth_scale=MANUSCRIPT_CURVE_LINEWIDTH_SCALE,
            )
        ax.scatter(
            direct_rates["temperature"],
            direct_rates[direct_metric],
            color=STAGE_DURATION_MEAN_DATA_COLOR,
            edgecolor="white",
            linewidth=0.45,
            s=STAGE_DURATION_MEAN_MARKER_SIZE,
            zorder=4,
            label="Data",
        )
        ax.set_title(title)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel(ylabel)
        if model_metric == "generation_time_model":
            ax.set_ylim(0, 60)  # change tau y range here
        clean_axis(ax)
    add_composite_legend(
        fig,
        models,
        data_color=STAGE_DURATION_MEAN_DATA_COLOR,
        data_markersize=STAGE_DURATION_MEAN_LEGEND_MARKER_SIZE,
    )
    save_figure(fig, output)


def plot_stage_duration_panel(
    summaries: dict[str, pd.DataFrame],
    models: list[ModelSpec],
    *,
    output: Path,
    figsize: tuple[float, float] = (9.2, 6.25),
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    panels = [
        (
            "Egg",
            summaries["stage"].loc[summaries["stage"]["stage"] == "Egg"],
            "egg_delay",
            "days",
        ),
        (
            "Larva",
            summaries["stage"].loc[summaries["stage"]["stage"] == "Larva"],
            "larva_delay",
            "days",
        ),
        (
            "Pupa",
            summaries["stage"].loc[summaries["stage"]["stage"] == "Pupa"],
            "pupa_delay",
            "days",
        ),
        ("Adult", summaries["adult"], "adult_delay", "days"),
    ]
    all_temperatures = np.concatenate(
        [data["temperature"].dropna().to_numpy(dtype=float) for _, data, _, _ in panels]
    )
    x_min = float(np.min(all_temperatures))
    x_max = float(np.max(all_temperatures))
    x_padding = 0.04 * (x_max - x_min)
    x_limits = (x_min - x_padding, x_max + x_padding)
    for ax, (title, data, response_name, unit) in zip(axes.flat, panels):
        data = data.sort_values("temperature")
        x_grid = np.linspace(x_limits[0], x_limits[1], 300)
        y_for_limits = data["value"].to_numpy(dtype=float)
        ax.scatter(
            data["temperature"],
            data["value"],
            color=STAGE_DURATION_MEAN_DATA_COLOR,
            edgecolor="white",
            linewidth=0.45,
            s=STAGE_DURATION_MEAN_MARKER_SIZE,
            zorder=3,
            label="Mean data",
        )
        for model in models:
            style = manuscript_model_style(model.key)
            response = getattr(model.parameters, response_name)
            ax.plot(
                x_grid,
                response(x_grid),
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4.5,
                markevery=15,
                linewidth=float(style["linewidth"]) * MANUSCRIPT_CURVE_LINEWIDTH_SCALE,
                solid_capstyle="butt",
                dash_capstyle=style["dash_capstyle"],
                label=manuscript_model_label(model),
            )
        ax.set_title(title)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel(f"Mean duration ({unit})")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_xlim(*x_limits)
        clean_axis(ax)
        y_top = float(np.nanmax(y_for_limits))
        if np.isfinite(y_top) and y_top > 0:
            ax.set_ylim(bottom=0, top=y_top * 1.12)
    add_composite_legend(
        fig,
        models,
        data_color=STAGE_DURATION_MEAN_DATA_COLOR,
        data_markersize=STAGE_DURATION_MEAN_LEGEND_MARKER_SIZE,
    )
    save_figure(fig, output)


def selected_temperatures(
    durations: pd.DataFrame,
    temperature_subset: tuple[float, ...] | None,
) -> list[float]:
    temperatures = sorted(durations["temperature"].dropna().unique())
    if temperature_subset is None:
        return temperatures
    return [
        temperature
        for requested in temperature_subset
        for temperature in temperatures
        if np.isclose(temperature, requested)
    ]


def stage_duration_x_max_by_stage(
    durations: pd.DataFrame,
    stages: list[str],
    temperatures: list[float],
    *,
    multiplier: float = 1.0,
) -> dict[str, float]:
    x_max_by_stage = {}
    for stage in stages:
        stage_values = durations.loc[
            (durations["stage"] == stage)
            & (durations["temperature"].isin(temperatures)),
            "duration",
        ].dropna()
        if stage_values.empty:
            x_max_by_stage[stage] = 2.0
            continue
        x_max_by_stage[stage] = max(
            float(np.ceil(stage_values.max() * multiplier)), 2.0
        )
    return x_max_by_stage


def plot_delay_distributions(
    durations: pd.DataFrame,
    models: list[ModelSpec],
    *,
    output: Path,
    transpose: bool = False,
    temperature_subset: tuple[float, ...] | None = None,
    share_x_by_stage: bool = False,
    stage_subset: tuple[str, ...] | None = None,
) -> None:
    stages = (
        list(stage_subset)
        if stage_subset is not None
        else ["Egg", "Larva", "Pupa", "Adult"]
    )
    temperatures = selected_temperatures(durations, temperature_subset)
    stage_x_max = stage_duration_x_max_by_stage(durations, stages, temperatures)
    n_rows = len(temperatures) if transpose else len(stages)
    n_cols = len(stages) if transpose else len(temperatures)
    figsize = (
        (2.35 * len(stages), 1.45 * len(temperatures))
        if transpose
        else (1.9 * len(temperatures), 2.35 * len(stages))
    )
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        constrained_layout=True,
        squeeze=False,
    )
    for stage_index, stage in enumerate(stages):
        for temp_index, temperature in enumerate(temperatures):
            row, col = (
                (temp_index, stage_index) if transpose else (stage_index, temp_index)
            )
            ax = axes[row, col]
            stage_data = durations.loc[
                (durations["stage"] == stage)
                & (durations["temperature"] == temperature),
                "duration",
            ].dropna()
            if stage_data.empty:
                ax.set_visible(False)
                continue
            max_day = int(np.ceil(stage_data.max()))
            x_max = stage_x_max[stage] if share_x_by_stage else max(max_day + 1.0, 2.0)
            bins = np.arange(0.5, x_max + 0.6, 1.0)
            ax.hist(
                stage_data,
                bins=bins,
                color="#b8c7d9",
                edgecolor=STAGE_DURATION_DATA_COLOR,
                linewidth=0.55,
                alpha=0.82,
                label="Data",
            )
            x_grid = np.linspace(0.01, x_max, 300)
            for model in models:
                style = manuscript_model_style(model.key)
                pdf = delay_pdf(stage, float(temperature), x_grid, model)
                ax.plot(
                    x_grid,
                    pdf * len(stage_data),
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=4.5,
                    markevery=15,
                    linewidth=float(style["linewidth"])
                    * MANUSCRIPT_CURVE_LINEWIDTH_SCALE,
                    solid_capstyle="butt",
                    dash_capstyle=style["dash_capstyle"],
                    label=manuscript_model_label(model),
                )
            if row == 0 and not transpose:
                ax.set_title(f"{temperature:g} °C", fontsize=10, fontweight="bold")
            if row == n_rows - 1:
                ax.set_xlabel("Delay (days)")
            ax.set_xlim(0.0, x_max)
            clean_axis(ax)
    if transpose:
        hide_inner_tick_labels(axes)
        add_bottom_visible_xlabels(axes, "Delay (days)")
        add_temperature_row_labels(fig, temperatures)
        add_composite_legend(fig, models, left=0.035, right=0.955)
        add_aligned_stage_column_labels(fig, axes, stages)
    else:
        hide_inner_tick_labels(axes)
        add_composite_legend(fig, models, left=0.025)
        add_aligned_stage_row_labels(fig, axes, stages)
    save_figure(fig, output)


def plot_maturation_survival(
    durations: pd.DataFrame,
    models: list[ModelSpec],
    *,
    output: Path,
    transpose: bool = False,
    temperature_subset: tuple[float, ...] | None = None,
    share_x_by_stage: bool = False,
    figsize: tuple[float, float] | None = None,
) -> None:
    stages = ["Egg", "Larva", "Pupa", "Adult"]
    temperatures = selected_temperatures(durations, temperature_subset)
    stage_x_max = stage_duration_x_max_by_stage(
        durations,
        stages,
        temperatures,
        multiplier=STAGE_DURATION_SURVIVAL_X_LIMIT_MULTIPLIER,
    )
    n_rows = len(temperatures) if transpose else len(stages)
    n_cols = len(stages) if transpose else len(temperatures)
    if figsize is None:
        figsize = (
            (2.35 * len(stages), 1.45 * len(temperatures))
            if transpose
            else (1.9 * len(temperatures), 2.35 * len(stages))
        )
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        constrained_layout=True,
        squeeze=False,
    )
    for stage_index, stage in enumerate(stages):
        for temp_index, temperature in enumerate(temperatures):
            row, col = (
                (temp_index, stage_index) if transpose else (stage_index, temp_index)
            )
            ax = axes[row, col]
            stage_data = durations.loc[
                (durations["stage"] == stage)
                & (durations["temperature"] == temperature),
                "duration",
            ].dropna()
            if stage_data.empty:
                ax.set_visible(False)
                continue
            x_step, y_step = empirical_one_minus_cdf(stage_data)
            max_day = float(stage_data.max())
            x_max = (
                stage_x_max[stage]
                if share_x_by_stage
                else max(max_day * STAGE_DURATION_SURVIVAL_X_LIMIT_MULTIPLIER, 2.0)
            )
            if len(x_step) and x_step[-1] < x_max:
                x_step = np.append(x_step, x_max)
                y_step = np.append(y_step, y_step[-1])
            x_grid = np.linspace(0.0, x_max, 300)
            ax.step(
                x_step,
                y_step,
                where="post",
                color=STAGE_DURATION_DATA_COLOR,
                linewidth=STAGE_DURATION_DATA_LINEWIDTH,
                alpha=STAGE_DURATION_DATA_ALPHA,
                zorder=1,
                label="Data",
            )
            for model in models:
                style = manuscript_model_style(model.key)
                ax.plot(
                    x_grid,
                    model_maturation_survival(stage, float(temperature), x_grid, model),
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=4.5,
                    markevery=15,
                    linewidth=float(style["linewidth"])
                    * MANUSCRIPT_CURVE_LINEWIDTH_SCALE,
                    solid_capstyle="butt",
                    dash_capstyle=style["dash_capstyle"],
                    zorder=3,
                    label=manuscript_model_label(model),
                )
            if row == 0 and not transpose:
                ax.set_title(f"{temperature:g} °C", fontsize=10, fontweight="bold")
            if row == n_rows - 1:
                ax.set_xlabel("Days")
            ax.set_xlim(0.0, x_max)
            clean_axis(ax)
            ax.set_ylim(
                0.0,
                1.0 + STAGE_DURATION_SURVIVAL_Y_PADDING,
            )
            ax.spines["bottom"].set_zorder(0)
    if transpose:
        hide_inner_tick_labels(axes)
        add_bottom_visible_xlabels(axes, "Days")
        add_temperature_row_labels(fig, temperatures)
        add_composite_legend(fig, models, left=0.035, right=0.955, data_handle="line")
        add_aligned_stage_column_labels(fig, axes, stages)
    else:
        hide_inner_tick_labels(axes)
        add_composite_legend(
            fig,
            models,
            left=0.04,
            right=0.93,
            data_handle="line",
            legend_y=0.995,
            w_pad=0.035,
            h_pad=0.045,
            wspace=0.09,
            hspace=0.08,
        )
        add_aligned_stage_row_labels(fig, axes, stages, x=0.018)
    save_figure(fig, output)


def empirical_one_minus_cdf(durations: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    values_sorted = np.sort(durations.to_numpy(dtype=float))
    unique_values = np.unique(values_sorted)
    n = len(values_sorted)
    times = [0.0]
    values = [1.0]
    for duration in unique_values:
        proportion_longer = float(np.mean(values_sorted > duration))
        times.append(float(duration))
        values.append(proportion_longer)
    return np.asarray(times, dtype=float), np.asarray(values, dtype=float)


def model_maturation_survival(
    stage: str,
    temperature: float,
    time_in_stage: np.ndarray,
    model: ModelSpec,
) -> np.ndarray:
    if stage == "Egg":
        return gamma.sf(
            time_in_stage,
            a=model.stage_counts["egg"],
            scale=float(model.parameters.egg_delay(temperature))
            / model.stage_counts["egg"],
        )
    if stage == "Larva":
        return gamma.sf(
            time_in_stage,
            a=model.stage_counts["larva"],
            scale=float(model.parameters.larva_delay(temperature))
            / model.stage_counts["larva"],
        )
    if stage == "Pupa":
        return gamma.sf(
            time_in_stage,
            a=model.stage_counts["pupa"],
            scale=float(model.parameters.pupa_delay(temperature))
            / model.stage_counts["pupa"],
        )
    if stage == "Adult":
        mortality_weights = model.parameters.adult_mortality_weights
        if mortality_weights is None:
            return gamma.sf(
                time_in_stage,
                a=model.stage_counts["adult"],
                scale=float(model.parameters.adult_delay(temperature))
                / model.stage_counts["adult"],
            )
        adult_delay = np.repeat(
            float(model.parameters.adult_delay(temperature)), len(time_in_stage)
        )
        _, survival = adult_substage_occupancy_with_mortality(
            time_in_stage,
            adult_delay,
            mortality_weights=mortality_weights,
        )
        return survival
    raise ValueError(f"Unknown stage: {stage}")


def delay_pdf(
    stage: str,
    temperature: float,
    delay_days: np.ndarray,
    model: ModelSpec,
) -> np.ndarray:
    if stage == "Egg":
        return gamma_delay_pdf(
            delay_days,
            stage_count=model.stage_counts["egg"],
            mean_delay=float(model.parameters.egg_delay(temperature)),
        )
    if stage == "Larva":
        return gamma_delay_pdf(
            delay_days,
            stage_count=model.stage_counts["larva"],
            mean_delay=float(model.parameters.larva_delay(temperature)),
        )
    if stage == "Pupa":
        return gamma_delay_pdf(
            delay_days,
            stage_count=model.stage_counts["pupa"],
            mean_delay=float(model.parameters.pupa_delay(temperature)),
        )
    if stage == "Adult":
        return adult_lifetime_pdf(delay_days, temperature, model)
    raise ValueError(f"Unknown stage: {stage}")


def gamma_delay_pdf(
    delay_days: np.ndarray,
    *,
    stage_count: int,
    mean_delay: float,
) -> np.ndarray:
    return gamma.pdf(delay_days, a=stage_count, scale=mean_delay / stage_count)


def adult_lifetime_pdf(
    delay_days: np.ndarray,
    temperature: float,
    model: ModelSpec,
) -> np.ndarray:
    adult_count = model.stage_counts["adult"]
    adult_delay = float(model.parameters.adult_delay(temperature))
    mortality_weights = model.parameters.adult_mortality_weights
    if mortality_weights is None:
        return gamma_delay_pdf(
            delay_days,
            stage_count=adult_count,
            mean_delay=adult_delay,
        )

    transition_rate = adult_count / adult_delay
    mortality_rates = np.asarray(mortality_weights, dtype=float) / adult_delay
    generator = np.zeros((adult_count, adult_count), dtype=float)
    for index in range(adult_count):
        if index < adult_count - 1:
            generator[index, index] -= transition_rate + mortality_rates[index]
            generator[index + 1, index] += transition_rate
        else:
            generator[index, index] -= mortality_rates[index]
    initial = np.zeros(adult_count, dtype=float)
    initial[0] = 1.0
    return np.asarray(
        [
            float(np.dot(mortality_rates, expm(generator * day).dot(initial)))
            for day in delay_days
        ],
        dtype=float,
    )


def plot_juvenile_panel(data: pd.DataFrame, model: ModelSpec, *, output: Path) -> None:
    data = data.sort_values("temperature")
    x_grid = np.linspace(
        float(data["temperature"].min()), float(data["temperature"].max()), 300
    )
    mortality = model.parameters.juvenile_mortality(x_grid)
    egg_rate = competing_risk_transition_rate(
        model.stage_counts["egg"],
        model.parameters.egg_delay(x_grid),
        mortality,
    )
    larva_rate = competing_risk_transition_rate(
        model.stage_counts["larva"],
        model.parameters.larva_delay(x_grid),
        mortality,
    )
    pupa_rate = competing_risk_transition_rate(
        model.stage_counts["pupa"],
        model.parameters.pupa_delay(x_grid),
        mortality,
    )
    probability = (
        (egg_rate / (egg_rate + mortality)) ** model.stage_counts["egg"]
        * (larva_rate / (larva_rate + mortality)) ** model.stage_counts["larva"]
        * (pupa_rate / (pupa_rate + mortality)) ** model.stage_counts["pupa"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.95), constrained_layout=True)
    axes[0].scatter(
        data["temperature"],
        data["survival_probability"],
        color="black",
        s=34,
        label="Data",
    )
    axes[0].plot(
        x_grid,
        probability,
        color=model_style(model.key)[0],
        linewidth=2.0,
        label=model.label,
    )
    axes[0].set_ylabel("Total juvenile maturation probability")
    axes[1].plot(
        x_grid,
        mortality,
        color=model_style(model.key)[0],
        linewidth=2.0,
        label=model.label,
    )
    axes[1].set_ylabel("Juvenile mortality (1/day)")
    for ax in axes:
        ax.set_xlabel("Temperature (°C)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        clean_axis(ax)
    axes[0].legend(frameon=False, fontsize=8)
    save_figure(fig, output)


def plot_daily_fecundity_panel(
    data: pd.DataFrame, model: ModelSpec, *, output: Path
) -> None:
    data = data.sort_values("temperature")
    x_grid = np.linspace(
        float(data["temperature"].min()), float(data["temperature"].max()), 300
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    ax.scatter(data["temperature"], data["value"], color="black", s=34, label="Data")
    ax.plot(
        x_grid,
        model.parameters.daily_fecundity(x_grid),
        color=model_style(model.key)[0],
        linewidth=2.0,
        label=model.label,
    )
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Daily eggs per live female")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    clean_axis(ax)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, output)


def plot_lifetime_fecundity_panel(
    data: pd.DataFrame, model: ModelSpec, *, output: Path
) -> None:
    data = data.sort_values("temperature")
    x_grid = np.linspace(
        float(data["temperature"].min()), float(data["temperature"].max()), 300
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    ax.scatter(data["temperature"], data["value"], color="black", s=34, label="Data")
    ax.plot(
        x_grid,
        model.parameters.lifetime_fecundity(x_grid),
        color=model_style(model.key)[0],
        linewidth=2.0,
        label=model.label,
    )
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Total lifetime adult fecundity")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    clean_axis(ax)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, output)


def plot_adult_fecundity_time_panel(
    data: pd.DataFrame, model: ModelSpec, *, output: Path
) -> None:
    temperatures = sorted(data["temperature"].dropna().unique())
    columns = 3
    rows = int(np.ceil(len(temperatures) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12.0, 2.85 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    color = model_style(model.key)[0]
    for ax, temperature in zip(axes.flat, temperatures):
        temp_data = data.loc[data["temperature"] == temperature].sort_values(
            "adult_day"
        )
        ax.scatter(
            temp_data["adult_day"],
            temp_data["value"],
            s=14,
            alpha=0.34,
            color="black",
        )
        adult_days = temp_data["adult_day"].to_numpy(dtype=float)
        temp = float(temperature)
        fecundity = adult_time_fecundity(model, temp, adult_days)
        ax.plot(adult_days, fecundity, color=color, linewidth=1.8, label="Model")
        ax.set_title(f"{temp:g} C")
        ax.set_xlabel("Adult day")
        ax.set_ylabel("Eggs per live female")
        clean_axis(ax)
    for ax in axes.flat[len(temperatures) :]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8)
    save_figure(fig, output)


def adult_time_fecundity(
    model: ModelSpec, temperature: float, adult_days: np.ndarray
) -> np.ndarray:
    profile = adult_fecundity_profile_values(model)
    if profile is None:
        return np.repeat(
            float(model.parameters.daily_fecundity(temperature)), len(adult_days)
        )
    adult_delay = np.repeat(
        float(model.parameters.adult_delay(temperature)), len(adult_days)
    )
    occupancy = adult_substage_occupancy(
        adult_days,
        adult_delay,
        adult_stage_count=model.stage_counts["adult"],
    )
    return float(model.parameters.daily_fecundity(temperature)) * occupancy.dot(profile)


def plot_adult_timing_profile_panel(model: ModelSpec, *, output: Path) -> None:
    adult_count = model.stage_counts["adult"]
    substages = np.arange(1, adult_count + 1)
    fecundity_profile = (
        np.ones(adult_count, dtype=float)
        if adult_fecundity_profile_values(model) is None
        else adult_fecundity_profile_values(model)
    )
    mortality_weights = (
        np.ones(adult_count, dtype=float)
        if model.parameters.adult_mortality_weights is None
        else np.asarray(model.parameters.adult_mortality_weights, dtype=float)
    )

    temperatures = np.asarray([18.0, 24.0, 28.0], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.7), constrained_layout=True)

    axes[0, 0].plot(
        substages,
        fecundity_profile,
        color="#00847e",
        marker="o",
        linewidth=2.0,
    )
    axes[0, 0].axhline(1.0, color="#888888", linestyle=":", linewidth=1.0)
    axes[0, 0].set_xlabel("Adult substage")
    axes[0, 0].set_ylabel("Fecundity profile")
    axes[0, 0].set_title("Fecundity by substage")

    axes[0, 1].plot(
        substages,
        mortality_weights,
        color="#7b5ea7",
        marker="o",
        linewidth=2.0,
    )
    axes[0, 1].axhline(1.0, color="#888888", linestyle=":", linewidth=1.0)
    axes[0, 1].set_xlabel("Adult substage")
    axes[0, 1].set_ylabel("Relative mortality")
    axes[0, 1].set_title("Mortality by substage")

    colors = {18.0: "#2364aa", 24.0: "#00847e", 28.0: "#c44536"}
    for temperature in temperatures:
        adult_delay = float(model.parameters.adult_delay(float(temperature)))
        adult_days = np.linspace(0.0, max(2.0, adult_delay * 1.75), 220)
        fecundity = adult_time_fecundity(model, float(temperature), adult_days)
        mortality = adult_time_mortality_hazard(model, float(temperature), adult_days)
        axes[1, 0].plot(
            adult_days,
            fecundity,
            color=colors[float(temperature)],
            linewidth=2.0,
            label=f"{temperature:g} C",
        )
        axes[1, 1].plot(
            adult_days,
            mortality,
            color=colors[float(temperature)],
            linewidth=2.0,
            label=f"{temperature:g} C",
        )

    axes[1, 0].set_xlabel("Adult age (days)")
    axes[1, 0].set_ylabel("Eggs per live female-day")
    axes[1, 0].set_title("Realized fecundity over adult age")
    axes[1, 1].set_xlabel("Adult age (days)")
    axes[1, 1].set_ylabel("Exit or mortality hazard (1/day)")
    axes[1, 1].set_title("Realized adult exit over adult age")
    axes[1, 0].legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        clean_axis(ax)
    save_figure(fig, output)


def adult_time_mortality_hazard(
    model: ModelSpec, temperature: float, adult_days: np.ndarray
) -> np.ndarray:
    mortality_weights = model.parameters.adult_mortality_weights
    adult_delay = float(model.parameters.adult_delay(temperature))
    if mortality_weights is None:
        density = gamma.pdf(
            adult_days,
            a=model.stage_counts["adult"],
            scale=adult_delay / model.stage_counts["adult"],
        )
        survival = gamma.sf(
            adult_days,
            a=model.stage_counts["adult"],
            scale=adult_delay / model.stage_counts["adult"],
        )
        return density / np.maximum(survival, 1e-12)
    base_mortality = 1.0 / adult_delay
    adult_delay_array = np.repeat(adult_delay, len(adult_days))
    occupancy, _ = adult_substage_occupancy_with_mortality(
        adult_days,
        adult_delay_array,
        mortality_weights=mortality_weights,
    )
    return base_mortality * occupancy.dot(np.asarray(mortality_weights, dtype=float))


def plot_model_demographics(
    curve: pd.DataFrame,
    direct_rates: pd.DataFrame,
    model: ModelSpec,
    *,
    output: Path,
) -> None:
    color, _ = model_style(model.key)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.95), constrained_layout=True)
    axes[0].plot(
        curve["temperature"],
        curve["r_model"],
        color=color,
        linewidth=2.0,
        label=model.label,
    )
    axes[0].scatter(
        direct_rates["temperature"],
        direct_rates["r_euler"],
        color="black",
        s=34,
        label="Data",
    )
    axes[0].set_ylabel("r (1/day)")
    axes[1].plot(
        curve["temperature"],
        curve["R0_model"],
        color=color,
        linewidth=2.0,
        label=model.label,
    )
    axes[1].scatter(
        direct_rates["temperature"],
        direct_rates["R0"],
        color="black",
        s=34,
        label="Data",
    )
    axes[1].axhline(1.0, color="#333333", linewidth=0.9, linestyle=":")
    axes[1].set_ylabel("R0")
    for ax in axes:
        ax.set_xlabel("Temperature (°C)")
        clean_axis(ax)
    axes[0].legend(frameon=False, fontsize=8)
    save_figure(fig, output)


def plot_model_simulation(model: ModelSpec, *, output: Path) -> None:
    simulation = simulate_single_season(
        model.parameters,
        SimulationConfig(
            start_day=182.0,
            end_day=334.0,
            initial_eggs=1.0,
            initial_adults=0.0,
            stage_counts=model.stage_counts,
            temperature_mean=11.5,
            temperature_amplitude=16.5,
            temperature_phase=-104.75,
        ),
    )
    stages = ["eggs", "larvae", "pupae", "adults"]
    colors = ["#6f9ceb", "#5a9f68", "#d08b3e", "#c44e52"]
    fig, axes = plt.subplots(
        4, 1, figsize=(9.4, 9.2), constrained_layout=True, sharex=True
    )
    axes[0].plot(
        simulation["day"], simulation["temperature"], color="#333333", linewidth=2.0
    )
    axes[0].set_ylabel("°C")
    for stage, color in zip(stages, colors):
        axes[1].plot(
            simulation["day"],
            simulation[stage],
            color=color,
            linewidth=1.7,
            label=stage.title(),
        )
        axes[2].plot(
            simulation["day"],
            simulation[stage].clip(lower=1e-12),
            color=color,
            linewidth=1.7,
            label=stage.title(),
        )
    composition = (
        simulation[stages]
        .div(simulation[stages].sum(axis=1).replace(0, np.nan), axis=0)
        .fillna(0.0)
    )
    axes[3].stackplot(
        simulation["day"],
        [composition[stage] for stage in stages],
        colors=colors,
        labels=[stage.title() for stage in stages],
    )
    axes[1].set_ylabel("Expected female abundance")
    axes[2].set_ylabel("Expected female abundance")
    axes[2].set_yscale("log")
    axes[3].set_ylabel("Composition")
    axes[3].set_ylim(0, 1)
    axes[3].set_xlabel("Day of year")
    for ax in axes:
        clean_axis(ax)
    axes[1].legend(frameon=False, fontsize=8, ncols=4)
    save_figure(fig, output)


def seasonal_simulation_for_model(
    model: ModelSpec,
    *,
    display_start_day: float = 167.0,
    release_day: float = 182.0,
    end_day: float = 334.0,
) -> pd.DataFrame:
    simulation = simulate_single_season(
        model.parameters,
        SimulationConfig(
            start_day=release_day,
            end_day=end_day,
            initial_eggs=1.0,
            initial_adults=0.0,
            stage_counts=model.stage_counts,
            temperature_mean=11.5,
            temperature_amplitude=16.5,
            temperature_phase=-104.75,
        ),
    )
    if display_start_day >= release_day:
        return simulation

    pre_release_days = np.arange(display_start_day, release_day)
    pre_release_temperature = temperature_sinusoid(
        pre_release_days,
        mean=11.5,
        amplitude=16.5,
        phase=-104.75,
    )
    pre_release = pd.DataFrame(
        {
            "day": pre_release_days,
            "temperature": pre_release_temperature,
            "eggs": 0.0,
            "larvae": 0.0,
            "pupae": 0.0,
            "adults": 0.0,
            "total": 0.0,
            "daily_fecundity": model.parameters.daily_fecundity(
                pre_release_temperature
            ),
            "egg_production_rate": 0.0,
        }
    )
    return pd.concat([pre_release, simulation], ignore_index=True)


def log_population_series(
    simulation: pd.DataFrame,
    stage: str,
    *,
    log_min: float,
    introduced_stage: str = "eggs",
) -> pd.Series:
    values = simulation[stage].astype(float).copy()
    if stage == introduced_stage:
        positive = values > 0
        if positive.any():
            first_positive_day = float(simulation.loc[positive, "day"].iloc[0])
            values = values.mask(simulation["day"] < first_positive_day)
    return values.clip(lower=log_min)


def one_e_tick_label(value: float, _position: int | None = None) -> str:
    if not np.isfinite(value):
        return ""
    if abs(value) < 1e-12:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    coefficient = value / 10**exponent
    return f"{coefficient:g}e{exponent}"


def plot_seasonal_simulation_composite(
    models: list[ModelSpec],
    *,
    output: Path,
) -> None:
    simulations = {model.key: seasonal_simulation_for_model(model) for model in models}
    stages = ["eggs", "larvae", "pupae", "adults"]
    stage_labels = ["Eggs", "Larvae", "Pupae", "Adults"]
    stage_colors = ["#6f9ceb", "#5a9f68", "#d08b3e", "#c44e52"]
    stage_linewidth = 2.2
    total_by_model = {
        key: simulation[stages].sum(axis=1) for key, simulation in simulations.items()
    }
    stage_max_by_model = {
        key: max(float(simulation[stage].max()) for stage in stages)
        for key, simulation in simulations.items()
    }
    global_max = max(float(total.max()) for total in total_by_model.values())
    positive_values = np.concatenate(
        [
            simulation[stages].to_numpy(dtype=float).ravel()
            for simulation in simulations.values()
        ]
    )
    positive_values = positive_values[positive_values > 0]
    log_min = max(float(np.nanmin(positive_values)) * 0.75, 1e-3)
    log_max = (
        max(
            float(simulation[stages].max().max()) for simulation in simulations.values()
        )
        * 1.45
    )
    first_simulation = next(iter(simulations.values()))
    month_ticks, month_labels = month_tick_labels(
        float(first_simulation["day"].min()),
        float(first_simulation["day"].max()),
    )

    population_tick_formatter = FuncFormatter(one_e_tick_label)

    fig = plt.figure(figsize=(6.55, 5.55), constrained_layout=True)
    grid = fig.add_gridspec(
        3,
        len(models),
        height_ratios=[0.72, 1.25, 1.25],
    )
    temp_ax = fig.add_subplot(grid[0, 1])
    temp_ax.plot(
        first_simulation["day"],
        first_simulation["temperature"],
        color="#333333",
        linewidth=2.2,
    )
    temp_ax.set_ylabel("°C")
    temp_ax.set_xticklabels([])
    clean_axis(temp_ax)

    axes = np.empty((2, len(models)), dtype=object)
    for col, model in enumerate(models):
        simulation = simulations[model.key]
        total = total_by_model[model.key]

        linear_ax = fig.add_subplot(grid[1, col], sharex=temp_ax)
        log_ax = fig.add_subplot(grid[2, col], sharex=temp_ax)
        axes[:, col] = [linear_ax, log_ax]

        model_label = manuscript_model_key_label(model.key)
        linear_ax.set_title(model_label, color="black", fontweight="bold")
        for stage, color, label in zip(stages, stage_colors, stage_labels):
            linear_ax.plot(
                simulation["day"],
                simulation[stage],
                color=color,
                linewidth=stage_linewidth,
                label=label,
            )
        stage_max = stage_max_by_model[model.key]
        peak_fraction = np.clip((stage_max / global_max) ** 0.12, 0.22, 0.95)
        linear_upper = stage_max * 1.08 / peak_fraction
        linear_ax.set_ylim(0, linear_upper)
        linear_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        linear_ax.yaxis.set_major_formatter(population_tick_formatter)

        for stage, color, label in zip(stages, stage_colors, stage_labels):
            log_ax.plot(
                simulation["day"],
                log_population_series(simulation, stage, log_min=log_min),
                color=color,
                linewidth=stage_linewidth,
                label=label,
            )
        log_ax.set_yscale("log")
        log_ax.set_ylim(log_min, log_max)
        log_ax.yaxis.set_major_formatter(population_tick_formatter)
        log_ax.yaxis.set_minor_formatter(FuncFormatter(lambda _value, _position: ""))

    row_labels = [
        "Expected female abundance",
        "Expected female abundance\n(log)",
    ]
    for row_index, label in enumerate(row_labels):
        axes[row_index, 0].set_ylabel(label)
        for ax in axes[row_index, 1:]:
            ax.set_ylabel("")
    temp_ax.tick_params(labelbottom=False)
    for ax in axes[:-1, :].ravel():
        ax.tick_params(labelbottom=False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Month")
        ax.set_xticks(month_ticks)
        ax.set_xticklabels(month_labels)
    for ax in [temp_ax, *axes.ravel()]:
        ax.set_xlim(first_simulation["day"].min(), first_simulation["day"].max())
        add_month_guides(ax, month_ticks)
        clean_axis(ax)
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(
            rect=(
                0.0,
                0.0,
                0.985,
                manuscript_legend_layout_top(fig, legend_y=MS_LEGEND_Y),
            )
        )
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, MS_LEGEND_Y),
        ncols=4,
        frameon=False,
        fontsize=9,
        handlelength=2.6,
        handletextpad=0.8,
    )
    save_figure(fig, output)


def add_month_guides(ax: plt.Axes, month_ticks: list[int]) -> None:
    for tick in month_ticks:
        ax.axvline(tick, color="#d0d0d0", linewidth=0.55, alpha=0.55, zorder=0)


def month_tick_labels(start_day: float, end_day: float) -> tuple[list[int], list[str]]:
    month_starts = [
        (1, "Jan."),
        (32, "Feb."),
        (60, "Mar."),
        (91, "Apr."),
        (121, "May"),
        (152, "Jun."),
        (182, "Jul."),
        (213, "Aug."),
        (244, "Sep."),
        (274, "Oct."),
        (305, "Nov."),
        (335, "Dec."),
        (366, "Jan."),
    ]
    ticks = [day for day, _ in month_starts if start_day <= day <= end_day]
    labels = [label for day, label in month_starts if start_day <= day <= end_day]
    if len(ticks) > 7:
        ticks = ticks[::2]
        labels = labels[::2]
    return ticks, labels


def write_report(
    report: Path,
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    count_table: pd.DataFrame,
    plots: dict[str, Path],
    main_models: list[ModelSpec],
    shared_figures: dict[str, Path],
    model_figures: dict[str, dict[str, Path]],
    manuscript_figures: dict[str, Path],
    seasonal_speed: pd.DataFrame,
) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Model Complexity Comparison",
        "",
        "The substage models use fixed stage counts chosen before model comparison.",
        "Counts are selected by Erlang profile likelihood from the recorded durations,",
        "then bounded by the scripted maximum substage count for numerical tractability.",
        "Egg, larva, and pupa curves are fitted on the development-rate scale",
        "and parameterized by mean stage duration at 20 C.",
        "",
        "## Predefined Substage Counts",
        "",
        markdown_table(count_table),
        "",
        image_line(
            report, plots["counts"], "Recorded-duration, capped substage counts"
        ),
        "",
        "## Main-Paper Models",
        "",
        markdown_table(summary.loc[summary["scope"] == "main"]),
        "",
        image_line(report, plots["main_r"], "Main model r comparison"),
        "",
        image_line(report, plots["main_R0"], "Main model R0 comparison"),
        "",
        image_line(
            report, plots["generation_time"], "Main model generation time comparison"
        ),
        "",
        "## Seasonal ODE Simulation Speed",
        "",
        "Timings use the same seasonal setup as the model panels: one initial female egg, no initial adults, and sinusoidal temperature forcing. M2 and M3 use the substage counts shown above.",
        "",
        markdown_table(seasonal_speed),
        "",
        "## Composite M1/M2/M3 Panels",
        "",
        "These panels overlay the three main model variants in one figure.",
        "",
        image_line(
            report, shared_figures["stage_durations"], "M1/M2/M3 mean stage durations"
        ),
        "",
        image_line(
            report,
            shared_figures["delay_distributions"],
            "M1/M2/M3 delay distributions",
        ),
        "",
        image_line(
            report,
            shared_figures["delay_distributions_main_stage_xrange"],
            "M1/M2/M3 delay distributions with stage-shared x ranges at main-text temperatures",
        ),
        "",
        image_line(
            report,
            shared_figures["delay_distributions_main_juvenile_stage_xrange"],
            "M1/M2/M3 egg and larva delay distributions with stage-shared x ranges at main-text temperatures",
        ),
        "",
        image_line(
            report,
            shared_figures["maturation_survival"],
            "M1/M2/M3 stage-duration 1-CDF curves",
        ),
        "",
        image_line(
            report,
            shared_figures["maturation_survival_main_stage_xrange"],
            "M1/M2/M3 stage-duration 1-CDF curves with stage-shared x ranges at main-text temperatures",
        ),
        "",
        "Juvenile mortality is fitted by binomial likelihood on observed total adult-emergence counts, including zero-emergence temperatures, with a single Gaussian-inverse mortality response across Egg+Larva+Pupa. M1 uses the single-stage juvenile survival fit, while M2 and M3 use the recorded-duration, capped juvenile substage fit; mortality panels therefore show only model-implied hazard curves.",
        "",
        image_line(
            report,
            manuscript_figures["juvenile_temperature"],
            "M1/M2/M3 juvenile survival and mortality temperature responses",
        ),
        "",
        image_line(
            report,
            manuscript_figures["fecundity_temperature"],
            "M1/M2/M3 fecundity temperature responses",
        ),
        "",
        image_line(
            report,
            manuscript_figures["adult_reproduction_time"],
            "M1/M2/M3 adult reproduction through adult life",
        ),
        "",
        image_line(
            report,
            manuscript_figures["demographic_metrics"],
            "M1/M2/M3 demographic metric comparison",
        ),
        "",
        image_line(
            report,
            shared_figures["seasonal_simulation"],
            "M1/M2/M3 seasonal simulation comparison",
        ),
        "",
        "## Per-Model Panels",
        "",
        "Each section contains diagnostics specific to one main model variant. Daily fecundity is omitted when adult substages have different fecundity profiles.",
        "",
    ]
    for model in main_models:
        figures = model_figures[model.key]
        lines.extend(
            [
                f"### {model.label}",
                "",
                image_line(
                    report,
                    figures["juvenile"],
                    f"{model.label} juvenile maturation and mortality",
                ),
                "",
            ]
        )
        if "daily_fecundity" in figures:
            lines.extend(
                [
                    image_line(
                        report,
                        figures["daily_fecundity"],
                        f"{model.label} daily fecundity",
                    ),
                    "",
                ]
            )
        lines.extend(
            [
                image_line(
                    report,
                    figures["lifetime_fecundity"],
                    f"{model.label} lifetime adult fecundity",
                ),
                "",
                image_line(
                    report,
                    figures["adult_fecundity_time"],
                    f"{model.label} adult fecundity over time",
                ),
                "",
            ]
        )
        if "adult_timing_profile" in figures:
            lines.extend(
                [
                    image_line(
                        report,
                        figures["adult_timing_profile"],
                        f"{model.label} adult fecundity and mortality timing",
                    ),
                    "",
                ]
            )
        lines.extend(
            [
                image_line(
                    report,
                    figures["reproduction_kernel"],
                    f"{model.label} reproduction kernel",
                ),
                "",
                image_line(report, figures["demographics"], f"{model.label} r and R0"),
                "",
                image_line(
                    report, figures["simulation"], f"{model.label} seasonal simulation"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Outputs",
            "",
            f"- Summary: `{output_dir / 'summary.csv'}`",
            f"- Seasonal simulation speed: `{output_dir / 'seasonal_simulation_speed.csv'}`",
            f"- Substage counts: `{output_dir / 'stage_substage_counts.csv'}`",
            f"- Adult exit-chain fecundity profile: `{output_dir / 'adult_exit_chain_fecundity_profile.csv'}`",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")


def image_line(report_path: Path, image_path: Path, alt: str) -> str:
    relative = os.path.relpath(image_path, start=report_path.parent)
    return f"![{alt}]({Path(relative).as_posix()})"


def markdown_table(data: pd.DataFrame) -> str:
    formatted = data.reset_index(drop=True).copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{value:.6g}")
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
