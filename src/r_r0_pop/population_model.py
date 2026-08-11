from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.integrate import solve_ivp

from r_r0_pop.data import BASER_POOLED_ADULT_FEMALE_FRACTION
from r_r0_pop.life_history_fits import (
    competing_risk_transition_rate,
    double_logistic_mortality,
    gauss,
    gaussinv,
    gaussinvgentle,
    q10_deactivation_delay,
    q10_deactivation_inv,
    q10_deactivation_rate,
    skew_gauss,
    skew_gaussinv,
)


MANUSCRIPT_M1_STAGE_COUNTS = {
    "egg": 1,
    "larva": 1,
    "pupa": 1,
    "adult": 1,
}

MANUSCRIPT_M2_STAGE_COUNTS = {
    "egg": 20,
    "larva": 40,
    "pupa": 40,
    "adult": 1,
}

MANUSCRIPT_M3_STAGE_COUNTS = {
    "egg": 20,
    "larva": 40,
    "pupa": 40,
    "adult": 16,
}

MANUSCRIPT_STAGE_COUNTS = {
    "m1_single_stage": MANUSCRIPT_M1_STAGE_COUNTS,
    "m2_substage_transit": MANUSCRIPT_M2_STAGE_COUNTS,
    "m3_adult_exit_chain_fecundity": MANUSCRIPT_M3_STAGE_COUNTS,
}

DEFAULT_STAGE_COUNTS = MANUSCRIPT_M2_STAGE_COUNTS


@dataclass(frozen=True)
class TemperatureResponse:
    name: str
    function: str
    c1: float
    c2: float
    c3: float
    c4: float = np.nan
    c5: float = np.nan
    c6: float = np.nan
    c7: float = np.nan

    def __call__(self, temperature: float | np.ndarray) -> np.ndarray:
        functions = {
            "gauss": gauss,
            "gaussinv": gaussinv,
            "gaussinvgentle": gaussinvgentle,
            "q10_deactivation_delay": q10_deactivation_delay,
            "q10_deactivation_rate": q10_deactivation_rate,
            "q10_deactivation_inv": q10_deactivation_inv,
            "double_logistic_mortality": double_logistic_mortality,
            "skew_gauss": skew_gauss,
            "skew_gaussinv": skew_gaussinv,
        }
        if self.function == "double_logistic_mortality":
            return functions[self.function](
                temperature,
                self.c1,
                self.c2,
                self.c3,
                self.c4,
                self.c5,
                self.c6,
                self.c7,
            )
        if self.function in {
            "skew_gauss",
            "skew_gaussinv",
            "q10_deactivation_delay",
            "q10_deactivation_rate",
            "q10_deactivation_inv",
        }:
            return functions[self.function](
                temperature, self.c1, self.c2, self.c3, self.c4
            )
        return functions[self.function](temperature, self.c1, self.c2, self.c3)


@dataclass(frozen=True)
class LifeHistoryParameters:
    egg_delay: TemperatureResponse
    larva_delay: TemperatureResponse
    pupa_delay: TemperatureResponse
    juvenile_mortality: TemperatureResponse
    adult_delay: TemperatureResponse
    lifetime_fecundity: TemperatureResponse
    daily_fecundity_response: TemperatureResponse | None = None
    female_fraction: float = BASER_POOLED_ADULT_FEMALE_FRACTION
    adult_fecundity_profile: tuple[float, ...] | None = None
    adult_fecundity_weights: tuple[float, ...] | None = None
    adult_mortality_weights: tuple[float, ...] | None = None

    def daily_fecundity(self, temperature: float | np.ndarray) -> np.ndarray:
        if self.daily_fecundity_response is not None:
            return self.daily_fecundity_response(temperature)
        return self.lifetime_fecundity(temperature) / self.adult_delay(temperature)

    def adult_substage_fecundity(
        self, temperature: float | np.ndarray, adult_stage_count: int
    ) -> np.ndarray:
        profile = self.adult_fecundity_profile
        if profile is None:
            profile = self.adult_fecundity_weights
        profile_values = (
            np.ones(adult_stage_count, dtype=float)
            if profile is None
            else np.asarray(profile, dtype=float)
        )
        if len(profile_values) != adult_stage_count:
            raise ValueError(
                "adult fecundity profile length must match the adult stage count."
            )
        return (
            np.asarray(self.daily_fecundity(temperature), dtype=float)[..., None]
            * profile_values
        )

    def adult_substage_mortality(
        self, temperature: float | np.ndarray, adult_stage_count: int
    ) -> np.ndarray | None:
        if self.adult_mortality_weights is None:
            return None
        weights = np.asarray(self.adult_mortality_weights, dtype=float)
        if len(weights) != adult_stage_count:
            raise ValueError(
                "adult_mortality_weights length must match the adult stage count."
            )
        base_mortality = 1.0 / np.asarray(self.adult_delay(temperature), dtype=float)
        return base_mortality[..., None] * weights


@dataclass(frozen=True)
class SimulationConfig:
    start_day: float = 120.0
    end_day: float = 333.0
    initial_adults: float = 10.0
    initial_eggs: float = 0.0
    initial_adult_substage: int | None = None
    stage_counts: dict[str, int] | None = None
    temperature_mean: float = 15.0
    temperature_amplitude: float = 10.0
    temperature_phase: float = 240.0

    @property
    def counts(self) -> dict[str, int]:
        return DEFAULT_STAGE_COUNTS if self.stage_counts is None else self.stage_counts


def temperature_sinusoid(
    day: float | np.ndarray,
    *,
    mean: float = 15.0,
    amplitude: float = 10.0,
    phase: float = 240.0,
) -> np.ndarray:
    """Typical annual temperature curve from the Wolfram simple simulation."""

    day = np.asarray(day, dtype=float)
    return mean + amplitude * np.sin(2 * np.pi * (day + phase) / 365.0)


def load_life_history_parameters(path: Path | str) -> LifeHistoryParameters:
    return life_history_parameters_from_table(pd.read_csv(path))


def life_history_parameters_from_table(table: pd.DataFrame) -> LifeHistoryParameters:
    params = table.set_index("name")

    def response(name: str) -> TemperatureResponse:
        row = params.loc[name]
        return TemperatureResponse(
            name=name,
            function=str(row["function"]),
            c1=float(row["c1"]),
            c2=float(row["c2"]),
            c3=float(row["c3"]),
            c4=float(row["c4"]) if "c4" in row and pd.notna(row["c4"]) else np.nan,
            c5=float(row["c5"]) if "c5" in row and pd.notna(row["c5"]) else np.nan,
            c6=float(row["c6"]) if "c6" in row and pd.notna(row["c6"]) else np.nan,
            c7=float(row["c7"]) if "c7" in row and pd.notna(row["c7"]) else np.nan,
        )

    return LifeHistoryParameters(
        egg_delay=response("Egg"),
        larva_delay=response("Larva"),
        pupa_delay=response("Pupa"),
        juvenile_mortality=response("Juvenile mortality rate"),
        adult_delay=response("Adult mortality delay"),
        lifetime_fecundity=response("Lifetime fecundity"),
    )


def simulate_single_season(
    parameters: LifeHistoryParameters,
    config: SimulationConfig | None = None,
) -> pd.DataFrame:
    """Simulate one season with no fruit limitation or larval competition."""

    config = SimulationConfig() if config is None else config
    slices = _state_slices(config.counts)
    y0 = np.zeros(slices["total"], dtype=float)
    if config.initial_eggs:
        y0[slices["egg"].start] = config.initial_eggs
    if config.initial_adults:
        if config.initial_adult_substage is None:
            y0[slices["adult"]] = config.initial_adults / config.counts["adult"]
        else:
            adult_index = slices["adult"].start + config.initial_adult_substage
            if not slices["adult"].start <= adult_index < slices["adult"].stop:
                raise ValueError("initial_adult_substage is outside the adult stage range.")
            y0[adult_index] = config.initial_adults

    solution = solve_ivp(
        lambda t, y: _rhs(t, y, parameters, config, slices),
        (config.start_day, config.end_day),
        y0,
        t_eval=np.arange(config.start_day, config.end_day + 1),
        method="LSODA",
        rtol=1e-7,
        atol=1e-9,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    totals = {
        name: solution.y[slc].sum(axis=0)
        for name, slc in slices.items()
        if name != "total"
    }
    day = solution.t
    temperature = temperature_sinusoid(
        day,
        mean=config.temperature_mean,
        amplitude=config.temperature_amplitude,
        phase=config.temperature_phase,
    )
    adults = totals["adult"]

    return pd.DataFrame(
        {
            "day": day,
            "temperature": temperature,
            "eggs": totals["egg"],
            "larvae": totals["larva"],
            "pupae": totals["pupa"],
            "adults": adults,
            "total": sum(totals.values()),
            "daily_fecundity": parameters.daily_fecundity(temperature),
            "egg_production_rate": (
                parameters.female_fraction
                * adults
                * parameters.daily_fecundity(temperature)
            ),
        }
    )


def model_net_reproductive_rate(
    parameters: LifeHistoryParameters,
    temperature: float,
    stage_counts: dict[str, int] | None = None,
) -> float:
    """Calculate model-derived R0 for a fixed temperature."""

    counts = DEFAULT_STAGE_COUNTS if stage_counts is None else stage_counts
    juvenile_survival = model_juvenile_survival_probability(
        parameters, temperature, counts
    )
    adult_expected_eggs = _expected_adult_eggs(parameters, temperature, counts)
    return float(
        parameters.female_fraction * adult_expected_eggs * juvenile_survival
    )


def model_juvenile_survival_probability(
    parameters: LifeHistoryParameters,
    temperature: float,
    stage_counts: dict[str, int] | None = None,
) -> float:
    """Probability of surviving the fitted egg, larval, and pupal chains."""

    counts = DEFAULT_STAGE_COUNTS if stage_counts is None else stage_counts
    rates = _fixed_temperature_rates(parameters, temperature, counts)
    survival = 1.0
    for stage in ("egg", "larva", "pupa"):
        survival *= (
            rates[stage] / (rates[stage] + rates["juvenile_mortality"])
        ) ** counts[stage]
    return float(survival)


def model_intrinsic_growth_rate(
    parameters: LifeHistoryParameters,
    temperature: float,
    stage_counts: dict[str, int] | None = None,
) -> float:
    """Calculate model-derived r as the dominant eigenvalue at fixed temperature."""

    matrix = fixed_temperature_matrix(parameters, temperature, stage_counts)
    eigenvalues = np.linalg.eigvals(matrix)
    return float(np.max(eigenvalues.real))


def model_demographic_rates(
    parameters: LifeHistoryParameters,
    temperatures: np.ndarray,
    stage_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Calculate model-derived R0 and r over fixed temperatures."""

    rows = []
    for temperature in temperatures:
        rows.append(
            {
                "temperature": float(temperature),
                "R0_model": model_net_reproductive_rate(
                    parameters, float(temperature), stage_counts
                ),
                "r_model": model_intrinsic_growth_rate(
                    parameters, float(temperature), stage_counts
                ),
                "generation_time_model": model_generation_time_euler(
                    parameters, float(temperature), stage_counts
                ),
            }
        )
    return pd.DataFrame(rows)


def model_reproduction_kernel(
    parameters: LifeHistoryParameters,
    temperature: float,
    ages: np.ndarray,
    stage_counts: dict[str, int] | None = None,
) -> np.ndarray:
    """Expected female offspring rate by age for one initial female egg."""

    no_birth_matrix, fecundity = _fixed_temperature_no_birth_matrix_and_fecundity(
        parameters, temperature, stage_counts
    )
    initial = np.zeros(no_birth_matrix.shape[0], dtype=float)
    initial[0] = 1.0
    return np.asarray(
        [float(fecundity @ expm(no_birth_matrix * age) @ initial) for age in ages],
        dtype=float,
    )


def model_generation_time_euler(
    parameters: LifeHistoryParameters,
    temperature: float,
    stage_counts: dict[str, int] | None = None,
) -> float:
    """Mean reproductive age in the model's Euler-Lotka kernel."""

    no_birth_matrix, fecundity = _fixed_temperature_no_birth_matrix_and_fecundity(
        parameters, temperature, stage_counts
    )
    initial = np.zeros(no_birth_matrix.shape[0], dtype=float)
    initial[0] = 1.0
    r = model_intrinsic_growth_rate(parameters, temperature, stage_counts)
    resolvent_matrix = r * np.eye(no_birth_matrix.shape[0]) - no_birth_matrix
    try:
        discounted_state_time = np.linalg.solve(resolvent_matrix, initial)
    except np.linalg.LinAlgError:
        return np.nan
    denominator = float(fecundity @ discounted_state_time)
    if denominator <= 0 or not np.isfinite(denominator):
        return np.nan
    try:
        discounted_state_time2 = np.linalg.solve(
            resolvent_matrix, discounted_state_time
        )
    except np.linalg.LinAlgError:
        return np.nan
    numerator = float(fecundity @ discounted_state_time2)
    return numerator / denominator


def fixed_temperature_matrix(
    parameters: LifeHistoryParameters,
    temperature: float,
    stage_counts: dict[str, int] | None = None,
) -> np.ndarray:
    """Build the linear stage-chain matrix for one fixed temperature."""

    counts = DEFAULT_STAGE_COUNTS if stage_counts is None else stage_counts
    slices = _state_slices(counts)
    matrix = np.zeros((slices["total"], slices["total"]), dtype=float)
    rates = _fixed_temperature_rates(parameters, temperature, counts)

    _add_chain_to_matrix(
        matrix,
        slices["egg"],
        transition_rate=rates["egg"],
        mortality_rate=rates["juvenile_mortality"],
    )
    _add_chain_to_matrix(
        matrix,
        slices["larva"],
        transition_rate=rates["larva"],
        mortality_rate=rates["juvenile_mortality"],
    )
    _add_chain_to_matrix(
        matrix,
        slices["pupa"],
        transition_rate=rates["pupa"],
        mortality_rate=rates["juvenile_mortality"],
    )
    _add_chain_to_matrix(
        matrix,
        slices["adult"],
        transition_rate=rates["adult"],
        mortality_rate=0.0,
    ) if rates["adult_mortality"] is None else _add_adult_chain_with_mortality_to_matrix(
        matrix,
        slices["adult"],
        transition_rate=rates["adult"],
        mortality_rates=rates["adult_mortality"],
    )

    matrix[slices["larva"].start, slices["egg"].stop - 1] += rates["egg"]
    matrix[slices["pupa"].start, slices["larva"].stop - 1] += rates["larva"]
    matrix[slices["adult"].start, slices["pupa"].stop - 1] += rates["pupa"]
    matrix[slices["egg"].start, slices["adult"]] += rates["fecundity"]
    return matrix


def _fixed_temperature_no_birth_matrix_and_fecundity(
    parameters: LifeHistoryParameters,
    temperature: float,
    stage_counts: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    counts = DEFAULT_STAGE_COUNTS if stage_counts is None else stage_counts
    slices = _state_slices(counts)
    matrix = np.zeros((slices["total"], slices["total"]), dtype=float)
    rates = _fixed_temperature_rates(parameters, temperature, counts)

    _add_chain_to_matrix(
        matrix,
        slices["egg"],
        transition_rate=rates["egg"],
        mortality_rate=rates["juvenile_mortality"],
    )
    _add_chain_to_matrix(
        matrix,
        slices["larva"],
        transition_rate=rates["larva"],
        mortality_rate=rates["juvenile_mortality"],
    )
    _add_chain_to_matrix(
        matrix,
        slices["pupa"],
        transition_rate=rates["pupa"],
        mortality_rate=rates["juvenile_mortality"],
    )
    if rates["adult_mortality"] is None:
        _add_chain_to_matrix(
            matrix,
            slices["adult"],
            transition_rate=rates["adult"],
            mortality_rate=0.0,
        )
    else:
        _add_adult_chain_with_mortality_to_matrix(
            matrix,
            slices["adult"],
            transition_rate=rates["adult"],
            mortality_rates=rates["adult_mortality"],
        )

    matrix[slices["larva"].start, slices["egg"].stop - 1] += rates["egg"]
    matrix[slices["pupa"].start, slices["larva"].stop - 1] += rates["larva"]
    matrix[slices["adult"].start, slices["pupa"].stop - 1] += rates["pupa"]

    fecundity = np.zeros(slices["total"], dtype=float)
    fecundity[slices["adult"]] = rates["fecundity"]
    return matrix, fecundity


def _rhs(
    day: float,
    state: np.ndarray,
    parameters: LifeHistoryParameters,
    config: SimulationConfig,
    slices: dict[str, slice | int],
) -> np.ndarray:
    temperature = float(
        temperature_sinusoid(
            day,
            mean=config.temperature_mean,
            amplitude=config.temperature_amplitude,
            phase=config.temperature_phase,
        )
    )
    counts = config.counts
    dydt = np.zeros_like(state)

    egg = state[slices["egg"]]
    larva = state[slices["larva"]]
    pupa = state[slices["pupa"]]
    adult = state[slices["adult"]]

    juvenile_mortality = float(parameters.juvenile_mortality(temperature))
    egg_rate = float(
        competing_risk_transition_rate(
            counts["egg"],
            parameters.egg_delay(temperature),
            juvenile_mortality,
        )
    )
    larva_rate = float(
        competing_risk_transition_rate(
            counts["larva"],
            parameters.larva_delay(temperature),
            juvenile_mortality,
        )
    )
    pupa_rate = float(
        competing_risk_transition_rate(
            counts["pupa"],
            parameters.pupa_delay(temperature),
            juvenile_mortality,
        )
    )
    adult_rate = counts["adult"] / float(parameters.adult_delay(temperature))
    adult_fecundity = parameters.adult_substage_fecundity(
        temperature, counts["adult"]
    ).reshape(-1)
    adult_mortality = parameters.adult_substage_mortality(
        temperature, counts["adult"]
    )
    fecundity = parameters.female_fraction * float(np.dot(adult_fecundity, adult))

    _chain_derivative(dydt[slices["egg"]], egg, egg_rate, fecundity, juvenile_mortality)
    _chain_derivative(
        dydt[slices["larva"]], larva, larva_rate, egg_rate * egg[-1], juvenile_mortality
    )
    _chain_derivative(
        dydt[slices["pupa"]], pupa, pupa_rate, larva_rate * larva[-1], juvenile_mortality
    )
    if adult_mortality is None:
        _chain_derivative(
            dydt[slices["adult"]],
            adult,
            adult_rate,
            pupa_rate * pupa[-1],
            0.0,
        )
    else:
        _adult_chain_derivative_with_mortality(
            dydt[slices["adult"]],
            adult,
            adult_rate,
            pupa_rate * pupa[-1],
            adult_mortality.reshape(-1),
        )

    return dydt


def _fixed_temperature_rates(
    parameters: LifeHistoryParameters,
    temperature: float,
    counts: dict[str, int],
) -> dict[str, float]:
    juvenile_mortality = float(parameters.juvenile_mortality(temperature))
    return {
        "egg": float(
            competing_risk_transition_rate(
                counts["egg"],
                parameters.egg_delay(temperature),
                juvenile_mortality,
            )
        ),
        "larva": float(
            competing_risk_transition_rate(
                counts["larva"],
                parameters.larva_delay(temperature),
                juvenile_mortality,
            )
        ),
        "pupa": float(
            competing_risk_transition_rate(
                counts["pupa"],
                parameters.pupa_delay(temperature),
                juvenile_mortality,
            )
        ),
        "adult": counts["adult"] / float(parameters.adult_delay(temperature)),
        "adult_mortality": parameters.adult_substage_mortality(
            temperature, counts["adult"]
        ),
        "juvenile_mortality": juvenile_mortality,
        "fecundity": parameters.female_fraction
        * parameters.adult_substage_fecundity(temperature, counts["adult"]).reshape(-1),
    }


def _expected_adult_eggs(
    parameters: LifeHistoryParameters,
    temperature: float,
    counts: dict[str, int],
) -> float:
    adult_count = counts["adult"]
    adult_fecundity = parameters.adult_substage_fecundity(
        temperature, adult_count
    ).reshape(-1)
    adult_rate = adult_count / float(parameters.adult_delay(temperature))
    adult_mortality = parameters.adult_substage_mortality(temperature, adult_count)
    matrix = np.zeros((adult_count, adult_count), dtype=float)
    if adult_mortality is None:
        _add_chain_to_matrix(
            matrix,
            slice(0, adult_count),
            transition_rate=adult_rate,
            mortality_rate=0.0,
        )
    else:
        _add_adult_chain_with_mortality_to_matrix(
            matrix,
            slice(0, adult_count),
            transition_rate=adult_rate,
            mortality_rates=adult_mortality.reshape(-1),
        )
    initial = np.zeros(adult_count, dtype=float)
    initial[0] = 1.0
    expected_time = np.linalg.solve(-matrix, initial)
    return float(np.dot(expected_time, adult_fecundity))


def _add_chain_to_matrix(
    matrix: np.ndarray,
    stage_slice: slice,
    *,
    transition_rate: float,
    mortality_rate: float,
) -> None:
    indices = range(stage_slice.start, stage_slice.stop)
    for index in indices:
        matrix[index, index] -= transition_rate + mortality_rate
        if index > stage_slice.start:
            matrix[index, index - 1] += transition_rate


def _add_adult_chain_with_mortality_to_matrix(
    matrix: np.ndarray,
    stage_slice: slice,
    *,
    transition_rate: float,
    mortality_rates: np.ndarray,
) -> None:
    indices = list(range(stage_slice.start, stage_slice.stop))
    for local_index, index in enumerate(indices):
        mortality_rate = float(mortality_rates[local_index])
        if local_index < len(indices) - 1:
            matrix[index, index] -= transition_rate + mortality_rate
            matrix[index + 1, index] += transition_rate
        else:
            matrix[index, index] -= mortality_rate


def _chain_derivative(
    derivative: np.ndarray,
    state: np.ndarray,
    transition_rate: float,
    input_rate: float,
    mortality_rate: float,
) -> None:
    derivative[0] = input_rate - transition_rate * state[0] - mortality_rate * state[0]
    derivative[1:] = (
        transition_rate * state[:-1]
        - transition_rate * state[1:]
        - mortality_rate * state[1:]
    )


def _adult_chain_derivative_with_mortality(
    derivative: np.ndarray,
    state: np.ndarray,
    transition_rate: float,
    input_rate: float,
    mortality_rates: np.ndarray,
) -> None:
    derivative[0] = (
        input_rate
        - transition_rate * state[0]
        - mortality_rates[0] * state[0]
    )
    if len(state) > 2:
        derivative[1:-1] = (
            transition_rate * state[:-2]
            - transition_rate * state[1:-1]
            - mortality_rates[1:-1] * state[1:-1]
        )
    if len(state) > 1:
        derivative[-1] = transition_rate * state[-2] - mortality_rates[-1] * state[-1]


def _state_slices(counts: dict[str, int]) -> dict[str, slice | int]:
    start = 0
    slices: dict[str, slice | int] = {}
    for name in ("egg", "larva", "pupa", "adult"):
        end = start + counts[name]
        slices[name] = slice(start, end)
        start = end
    slices["total"] = start
    return slices
