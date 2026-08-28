import math

import pandas as pd
import pytest

from r_r0_pop.life_history_fits import (
    adult_substage_occupancy_with_mortality,
    adult_delay_summary,
    fit_juvenile_mortality,
    fit_adult_mortality_rate,
    fit_q10_deactivation_response,
    gaussinv,
    fit_stage_development_rate,
    fit_stage_delay_q10_deactivation,
    fit_shared_stage_development_rates,
    fit_shared_stage_delays_log,
    FitResult,
    juvenile_survival_from_mortality_response,
    juvenile_mortality_summary,
    juvenile_mortality_summary_for_stage_chain,
    juvenile_mortality_rate_for_stage_chain,
    maturation_delay_summary,
    parametric_adult_timing_weights,
    predict,
    q10_deactivation_delay,
    q10_deactivation_inv,
    q10_deactivation_response,
    skew_gauss_peak,
    stage_duration_observations,
    variance_matched_stage_counts,
)
from r_r0_pop.population_model import (
    DEFAULT_STAGE_COUNTS,
    LifeHistoryParameters,
    MANUSCRIPT_M1_STAGE_COUNTS,
    MANUSCRIPT_M2_STAGE_COUNTS,
    MANUSCRIPT_M3_STAGE_COUNTS,
    SimulationConfig,
    TemperatureResponse,
    fixed_temperature_matrix,
    life_history_parameters_from_table,
    model_demographic_rates,
    model_generation_time_euler,
    model_intrinsic_growth_rate,
    model_net_reproductive_rate,
    model_reproduction_kernel,
    simulate_single_season,
    temperature_sinusoid,
)


def simple_life_history_parameters() -> LifeHistoryParameters:
    delay = TemperatureResponse("delay", "gauss", (2.0, 24.0, 1e9))
    juvenile_mortality = TemperatureResponse(
        "juvenile mortality", "gauss", (0.02, 24.0, 1e9)
    )
    adult_delay = TemperatureResponse("adult delay", "gauss", (10.0, 24.0, 1e9))
    lifetime_fecundity = TemperatureResponse(
        "fecundity", "gauss", (20.0, 24.0, 1e9)
    )
    return LifeHistoryParameters(
        egg_delay=delay,
        larva_delay=delay,
        pupa_delay=delay,
        juvenile_mortality=juvenile_mortality,
        adult_delay=adult_delay,
        lifetime_fecundity=lifetime_fecundity,
    )


def test_stage_duration_summaries_use_recorded_durations():
    development = pd.DataFrame(
        {
            "temperature": [20.0, 20.0],
            "E": [1.0, 2.0],
            "L1": [1.0, 1.0],
            "L2": [1.0, 1.0],
            "L3": [1.0, 1.0],
            "P": [2.0, 3.0],
        }
    )
    adult_survival = pd.DataFrame(
        {
            "temperature": [20.0, 20.0],
            "AM": [4.0, 0.0],
            "AF": [0.0, 6.0],
        }
    )
    fertility = pd.DataFrame()

    juvenile = maturation_delay_summary(development, fertility).set_index("stage")
    adult = adult_delay_summary(adult_survival).iloc[0]
    observations = stage_duration_observations(development, adult_survival, fertility)

    assert juvenile.loc["Egg", "value"] == pytest.approx(1.5)
    assert juvenile.loc["Larva", "value"] == pytest.approx(3.0)
    assert juvenile.loc["Pupa", "value"] == pytest.approx(2.5)
    assert adult["value"] == pytest.approx(6.0)
    assert observations.loc[
        observations["stage"] == "Larva", "duration"
    ].tolist() == pytest.approx([3.0, 3.0])


def test_default_stage_counts_match_manuscript_m2_baseline():
    assert DEFAULT_STAGE_COUNTS == MANUSCRIPT_M2_STAGE_COUNTS
    assert MANUSCRIPT_M1_STAGE_COUNTS == {
        "egg": 1,
        "larva": 1,
        "pupa": 1,
        "adult": 1,
    }
    assert MANUSCRIPT_M2_STAGE_COUNTS == {
        "egg": 14,
        "larva": 40,
        "pupa": 40,
        "adult": 1,
    }
    assert MANUSCRIPT_M3_STAGE_COUNTS == {
        "egg": 14,
        "larva": 40,
        "pupa": 40,
        "adult": 15,
    }


def test_temperature_sinusoid_matches_wolfram_formula():
    assert math.isclose(
        float(temperature_sinusoid(0)),
        15 + 10 * math.sin(2 * math.pi * 240 / 365),
    )


def test_single_season_simulation_has_expected_columns():
    parameters = simple_life_history_parameters()
    result = simulate_single_season(
        parameters,
        SimulationConfig(start_day=59, end_day=62, initial_adults=10),
    )

    assert list(result["day"]) == [59, 60, 61, 62]
    assert {
        "temperature",
        "eggs",
        "larvae",
        "pupae",
        "adults",
        "egg_production_rate",
    }.issubset(result.columns)
    assert (result[["eggs", "larvae", "pupae", "adults"]] >= -1e-9).all().all()


def test_single_season_can_initialize_first_adult_substage():
    parameters = simple_life_history_parameters()
    result = simulate_single_season(
        parameters,
        SimulationConfig(
            start_day=59,
            end_day=60,
            initial_adults=1.0,
            initial_adult_substage=0,
            stage_counts={"egg": 1, "larva": 1, "pupa": 1, "adult": 4},
        ),
    )

    assert math.isclose(float(result["adults"].iloc[0]), 1.0)


def test_fixed_temperature_model_rates_are_finite():
    parameters = simple_life_history_parameters()

    r = model_intrinsic_growth_rate(parameters, 24.0)
    R0 = model_net_reproductive_rate(parameters, 24.0)

    assert math.isfinite(r)
    assert math.isfinite(R0)
    assert R0 > 0


def test_net_reproductive_rate_counts_all_adult_substages():
    delay = TemperatureResponse("delay", "gauss", (2.0, 24.0, 1e9))
    no_mortality = TemperatureResponse(
        "no mortality", "gauss", (0.0, 24.0, 1e9)
    )
    lifetime_fecundity = TemperatureResponse(
        "fecundity", "gauss", (8.0, 24.0, 1e9)
    )
    parameters = LifeHistoryParameters(
        egg_delay=delay,
        larva_delay=delay,
        pupa_delay=delay,
        juvenile_mortality=no_mortality,
        adult_delay=delay,
        lifetime_fecundity=lifetime_fecundity,
        female_fraction=0.5,
    )
    counts = {
        "egg": 1,
        "larva": 1,
        "pupa": 1,
        "adult": 4,
    }

    R0 = model_net_reproductive_rate(parameters, 24.0, counts)

    assert math.isclose(R0, 4.0)


def test_model_generation_time_matches_single_reproductive_delay():
    delay = TemperatureResponse("delay", "gauss", (1.0, 24.0, 1e9))
    no_mortality = TemperatureResponse(
        "no mortality", "gauss", (0.0, 24.0, 1e9)
    )
    lifetime_fecundity = TemperatureResponse(
        "fecundity", "gauss", (4.0, 24.0, 1e9)
    )
    parameters = LifeHistoryParameters(
        egg_delay=delay,
        larva_delay=delay,
        pupa_delay=delay,
        juvenile_mortality=no_mortality,
        adult_delay=delay,
        lifetime_fecundity=lifetime_fecundity,
    )
    counts = {"egg": 1, "larva": 1, "pupa": 1, "adult": 1}

    generation_time = model_generation_time_euler(parameters, 24.0, counts)
    kernel = model_reproduction_kernel(parameters, 24.0, [0.0, 4.0], counts)

    assert math.isfinite(generation_time)
    assert kernel[0] == 0.0
    assert kernel[1] > 0.0


def test_fixed_temperature_matrix_has_expected_shape():
    parameters = simple_life_history_parameters()

    matrix = fixed_temperature_matrix(parameters, 24.0)

    expected_size = sum(DEFAULT_STAGE_COUNTS.values())
    assert matrix.shape == (expected_size, expected_size)


def test_fixed_temperature_matrix_uses_adult_substage_fecundity_profile():
    response = TemperatureResponse("constant", "gauss", (1.0, 24.0, 1e9))
    peak_response = TemperatureResponse("peak", "gauss", (3.0, 24.0, 1e9))
    parameters = LifeHistoryParameters(
        egg_delay=response,
        larva_delay=response,
        pupa_delay=response,
        juvenile_mortality=response,
        adult_delay=response,
        lifetime_fecundity=response,
        daily_fecundity_response=peak_response,
        female_fraction=0.5,
        adult_fecundity_profile=(0.25, 1.0),
    )
    counts = {
        "egg": 1,
        "larva": 1,
        "pupa": 1,
        "adult": 2,
    }

    matrix = fixed_temperature_matrix(parameters, 24.0, counts)
    adult_slice = slice(3, 5)

    assert list(matrix[0, adult_slice]) == [0.375, 1.5]


def test_daily_fecundity_response_overrides_lifetime_divided_by_adult_delay():
    response = TemperatureResponse("constant", "gauss", (1.0, 24.0, 1e9))
    daily_response = TemperatureResponse("daily", "gauss", (3.0, 24.0, 1e9))
    parameters = LifeHistoryParameters(
        egg_delay=response,
        larva_delay=response,
        pupa_delay=response,
        juvenile_mortality=response,
        adult_delay=response,
        lifetime_fecundity=response,
        daily_fecundity_response=daily_response,
        female_fraction=0.5,
    )
    counts = {"egg": 1, "larva": 1, "pupa": 1, "adult": 1}

    matrix = fixed_temperature_matrix(parameters, 24.0, counts)

    assert parameters.daily_fecundity(24.0) == 3.0
    assert matrix[0, 3] == 1.5


def test_model_demographic_rates_returns_requested_temperatures():
    parameters = simple_life_history_parameters()

    rates = model_demographic_rates(parameters, [18.0, 24.0])

    assert list(rates["temperature"]) == [18.0, 24.0]
    assert {"R0_model", "r_model"}.issubset(rates.columns)


def test_life_history_parameter_table_preserves_skew_parameter():
    rows = [
        {
            "name": name,
            "function": "gaussinv",
            "minimum": 1.0,
            "minimum_temperature": 24.0,
            "sigma": 10.0,
        }
        for name in ("Egg", "Larva", "Pupa")
    ]
    rows.extend(
        [
            {
                "name": "Juvenile mortality rate",
                "function": "skew_gaussinv",
                "scale": 1.0,
                "location_temperature": 24.0,
                "sigma": 10.0,
                "skew": 2.0,
            },
            {
                "name": "Adult duration",
                "function": "skew_gauss",
                "scale": 1.0,
                "location_temperature": 24.0,
                "sigma": 10.0,
                "skew": 2.0,
            },
            {
                "name": "Lifetime fecundity",
                "function": "skew_gauss_peak",
                "maximum": 1.0,
                "optimum_temperature": 24.0,
                "sigma": 10.0,
                "skew": 2.0,
            },
        ]
    )

    parameters = life_history_parameters_from_table(pd.DataFrame(rows))

    assert parameters.adult_delay.function == "skew_gauss"
    assert parameters.adult_delay.parameters[3] == 2.0
    assert math.isfinite(float(parameters.adult_delay(24.0)))
    assert float(parameters.lifetime_fecundity(24.0)) == pytest.approx(1.0)


def test_inverse_gaussian_fit_serializes_width_as_sigma():
    fit = FitResult(
        name="Juvenile mortality rate",
        function="gaussinv",
        parameters=(0.02, 16.0, 10.0),
        rss=0.0,
        r2=1.0,
        n=3,
    )

    serialized = fit.as_dict()

    assert serialized["sigma"] == 10.0
    assert "temperature_breadth" not in serialized


def test_adult_substage_occupancy_with_mortality_is_normalized():
    occupancy, survival = adult_substage_occupancy_with_mortality(
        [1.0, 5.0],
        [10.0, 10.0],
        mortality_weights=[0.5, 1.0, 1.2, 1.3],
    )

    assert occupancy.shape == (2, 4)
    assert all(math.isclose(float(row.sum()), 1.0) for row in occupancy)
    assert (survival > 0).all()
    assert survival[1] < survival[0]


def test_parametric_adult_timing_weights_are_normalized_and_monotone_mortality():
    reproduction, mortality = parametric_adult_timing_weights(
        [0.45, math.log(0.2), math.log(2.0)],
        adult_stage_count=12,
    )

    assert math.isclose(float(reproduction.mean()), 1.0)
    assert math.isclose(float(mortality.mean()), 1.0)
    assert (reproduction > 0).all()
    assert (mortality > 0).all()
    assert all(mortality[i] <= mortality[i + 1] for i in range(len(mortality) - 1))


def test_juvenile_mortality_summary_uses_total_adult_emergence():
    development = pd.DataFrame(
        {
            "temperature": [20.0, 20.0, 20.0, 20.0],
            "specimen": [1, 2, 3, 4],
            "E": [1.0, 1.0, 1.0, 0.0],
            "L1": [1.0, 1.0, 1.0, 0.0],
            "L2": [1.0, 1.0, 1.0, 0.0],
            "L3": [1.0, 1.0, 1.0, 0.0],
            "P": [1.0, 1.0, 1.0, 0.0],
        }
    )
    adult_survival = pd.DataFrame(
        {
            "temperature": [20.0, 20.0, 20.0, 20.0],
            "specimen": [1, 2, 3, 4],
            "AM": [2.0, 0.0, 0.0, 0.0],
            "AF": [0.0, 3.0, 0.0, 0.0],
        }
    )

    summary = juvenile_mortality_summary(
        development, adult_survival, cohort_size=10
    ).iloc[0]

    assert math.isclose(summary["survival_probability"], 0.2)
    assert math.isclose(summary["value"], -math.log(0.2) / 5.0)


def test_juvenile_mortality_rate_for_stage_chain_matches_target_survival():
    counts = {"egg": 2, "larva": 3, "pupa": 4}
    mean_delays = {"egg": 2.0, "larva": 3.0, "pupa": 4.0}
    mortality = juvenile_mortality_rate_for_stage_chain(0.25, mean_delays, counts)
    survival = 1.0
    for stage in ("egg", "larva", "pupa"):
        rate = counts[stage] / mean_delays[stage] - mortality
        survival *= (rate / (rate + mortality)) ** counts[stage]

    assert math.isclose(survival, 0.25)


def test_variance_matched_stage_counts_uses_mean_temperature_cv2():
    observations = pd.DataFrame(
        {
            "stage": ["Egg"] * 6,
            "temperature": [20.0, 20.0, 20.0, 25.0, 25.0, 25.0],
            "duration": [8.0, 10.0, 12.0, 8.0, 10.0, 12.0],
        }
    )

    counts, table = variance_matched_stage_counts(observations)

    assert counts == {"egg": 25}
    assert table.loc[0, "substage_count"] == 25


def test_log_shared_stage_delay_fit_returns_positive_stage_responses():
    observations = pd.DataFrame(
        {
            "stage": ["Egg", "Egg", "Larva", "Larva", "Pupa", "Pupa"],
            "temperature": [20.0, 25.0, 20.0, 25.0, 20.0, 25.0],
            "value": [6.0, 3.0, 12.0, 6.0, 8.0, 4.0],
        }
    )

    fits = fit_shared_stage_delays_log(observations)

    assert set(fits) == {"Egg", "Larva", "Pupa"}
    assert all(fit.function == "gaussinv" for fit in fits.values())
    assert all(fit.parameters[0] > 0 for fit in fits.values())


def test_shared_stage_rate_fit_returns_equivalent_delay_responses():
    observations = pd.DataFrame(
        {
            "stage": ["Egg", "Egg", "Larva", "Larva", "Pupa", "Pupa"],
            "temperature": [20.0, 25.0, 20.0, 25.0, 20.0, 25.0],
            "value": [6.0, 3.0, 12.0, 6.0, 8.0, 4.0],
        }
    )

    fits = fit_shared_stage_development_rates(observations)

    assert set(fits) == {"Egg", "Larva", "Pupa"}
    assert all(fit.function == "gaussinv" for fit in fits.values())
    assert all(fit.parameters[0] > 0 for fit in fits.values())


def test_stage_specific_rate_fits_return_delay_responses():
    observations = pd.DataFrame(
        {
            "temperature": [15.0, 20.0, 25.0, 30.0, 32.0],
            "value": [10.0, 6.0, 4.0, 5.0, 8.0],
        }
    )

    gauss_fit = fit_stage_development_rate(observations, "Egg")
    q10_fit = fit_stage_delay_q10_deactivation(observations, "Egg")

    assert gauss_fit.function == "gaussinv"
    assert q10_fit.function == "q10_deactivation_delay"
    assert gauss_fit.parameters[0] > 0
    assert q10_fit.parameters[0] > 0
    assert q10_fit.parameters[1] > 0
    assert q10_fit.parameters[3] > 0
    assert math.isclose(float(predict(q10_fit, 20.0)), q10_fit.parameters[0])


def test_q10_deactivation_temperature_responses_round_trip_from_table():
    table = pd.DataFrame(
        [
            {
                "name": "Egg",
                "function": "q10_deactivation_delay",
                "delay_at_20": 5.0,
                "q10": 2.0,
                "deactivation_temperature": 28.0,
                "deactivation_steepness": 0.3,
            },
            {
                "name": "Larva",
                "function": "q10_deactivation_delay",
                "delay_at_20": 12.5,
                "q10": 3.0,
                "deactivation_temperature": 30.0,
                "deactivation_steepness": 0.4,
            },
            {
                "name": "Pupa",
                "function": "gaussinv",
                "minimum": 4.0,
                "minimum_temperature": 25.0,
                "sigma": 10.0,
            },
            {
                "name": "Juvenile mortality rate",
                "function": "gaussinv",
                "minimum": 0.01,
                "minimum_temperature": 16.0,
                "sigma": 10.0,
            },
            {
                "name": "Adult duration",
                "function": "gauss",
                "maximum": 20.0,
                "optimum_temperature": 25.0,
                "sigma": 10.0,
            },
            {
                "name": "Lifetime fecundity",
                "function": "gauss",
                "maximum": 100.0,
                "optimum_temperature": 25.0,
                "sigma": 10.0,
            },
        ]
    )

    parameters = life_history_parameters_from_table(table)

    assert parameters.egg_delay(20.0) > 0
    assert parameters.larva_delay(20.0) > 0
    assert parameters.juvenile_mortality(20.0) > 0


def test_q10_deactivation_response_is_normalized_at_20_c():
    response = q10_deactivation_response(20.0, 0.25, 2.5, 28.0, 0.4)
    inverse = q10_deactivation_inv(20.0, 0.25, 2.5, 28.0, 0.4)
    delay = q10_deactivation_delay(20.0, 4.0, 2.5, 28.0, 0.4)

    assert math.isclose(float(response), 0.25)
    assert math.isclose(float(inverse), 4.0)
    assert math.isclose(float(delay), 4.0)


def test_skew_gaussian_peak_parameters_are_the_actual_peak():
    maximum = 210.0
    optimum_temperature = 23.8

    at_optimum = skew_gauss_peak(
        optimum_temperature,
        maximum,
        optimum_temperature,
        10.6,
        -4.5,
    )
    nearby = skew_gauss_peak(
        [optimum_temperature - 0.1, optimum_temperature + 0.1],
        maximum,
        optimum_temperature,
        10.6,
        -4.5,
    )

    assert float(at_optimum) == pytest.approx(maximum)
    assert all(value < maximum for value in nearby)


def test_juvenile_mortality_fit_can_target_survival_probability():
    stage_fits = {
        stage: FitResult(
            name=stage,
            function="gauss",
            parameters=(2.0, 24.0, 1e9),
            rss=0.0,
            r2=1.0,
            n=4,
        )
        for stage in ("Egg", "Larva", "Pupa")
    }
    stage_counts = {"egg": 1, "larva": 1, "pupa": 1}
    temperatures = pd.Series([12.0, 15.0, 18.0, 21.0])
    true_params = (0.02, 15.0, 12.0)
    survival = juvenile_survival_from_mortality_response(
        temperatures.to_numpy(dtype=float),
        gaussinv,
        true_params,
        stage_fits=stage_fits,
        stage_counts=stage_counts,
    )
    observations = pd.DataFrame(
        {
            "temperature": temperatures,
            "value": gaussinv(temperatures, *true_params),
            "survival_probability": survival,
            "n": 100,
            "adult_count": survival * 100,
        }
    )
    observations = pd.concat(
        [
            observations,
            pd.DataFrame(
                {
                    "temperature": [6.0],
                    "value": [float("nan")],
                    "survival_probability": [0.0],
                    "n": [100],
                    "adult_count": [0],
                }
            ),
        ],
        ignore_index=True,
    )

    fit = fit_juvenile_mortality(
        observations,
        stage_fits=stage_fits,
        stage_counts=stage_counts,
    )

    assert fit.function == "gaussinv"
    assert fit.n == len(observations)
    fitted_survival = juvenile_survival_from_mortality_response(
        observations["temperature"].to_numpy(dtype=float),
        gaussinv,
        fit.parameters,
        stage_fits=stage_fits,
        stage_counts=stage_counts,
    )
    assert float(fitted_survival[-1]) < float(fitted_survival[:4].min())


def test_juvenile_mortality_summary_uses_cohort_size_as_binomial_trials():
    specimen_ids = range(1000)
    development = pd.DataFrame(
        {
            "temperature": [20.0] * 1000,
            "specimen": specimen_ids,
            "E": [1.0] * 1000,
            "L1": [1.0] * 1000,
            "L2": [1.0] * 1000,
            "L3": [1.0] * 1000,
            "P": [1.0] * 1000,
        }
    )
    adult_survival = pd.DataFrame(
        {
            "temperature": [20.0] * 1000,
            "specimen": specimen_ids,
            "AF": [1.0] * 20 + [0.0] * 980,
            "AM": [0.0] * 20 + [1.0] * 10 + [0.0] * 970,
        }
    )

    summary = juvenile_mortality_summary_for_stage_chain(
        development,
        adult_survival,
        {"egg": 1, "larva": 1, "pupa": 1},
        cohort_size=50,
    )

    row = summary.iloc[0]
    assert row["adult_count"] == 30
    assert row["n"] == 50
    assert math.isclose(row["survival_probability"], 0.6)


def test_adult_mortality_rate_fits_return_positive_rates():
    observations = pd.DataFrame(
        {
            "temperature": [15.0, 20.0, 25.0, 30.0, 32.0],
            "value": [40.0, 30.0, 20.0, 12.0, 8.0],
        }
    )

    gauss_fit = fit_adult_mortality_rate(observations)
    q10_fit = fit_q10_deactivation_response(
        observations,
        name="Adult duration",
    )

    assert gauss_fit.function == "gaussinv"
    assert q10_fit.function == "q10_deactivation_response"
    assert gauss_fit.parameters[0] > 0
    assert q10_fit.parameters[0] > 0
