from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import brentq
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.linalg import expm
from scipy.special import erf
from scipy.stats import gamma as gamma_distribution

STAGES = ("Egg", "Larva", "Pupa")
STAGE_COUNT_KEYS = {
    "Egg": "egg",
    "Larva": "larva",
    "Pupa": "pupa",
    "Adult": "adult",
}

FUNCTION_PARAMETER_NAMES = {
    "gauss": ("maximum", "optimum_temperature", "sigma"),
    "gaussinv": ("minimum", "minimum_temperature", "sigma"),
    "q10_deactivation_response": (
        "response_at_20",
        "q10",
        "deactivation_temperature",
        "deactivation_steepness",
    ),
    "q10_deactivation_inv": (
        "response_at_20",
        "q10",
        "deactivation_temperature",
        "deactivation_steepness",
    ),
    "q10_deactivation_delay": (
        "delay_at_20",
        "q10",
        "deactivation_temperature",
        "deactivation_steepness",
    ),
    "double_logistic_mortality": (
        "minimum",
        "cold_increment",
        "cold_midpoint",
        "cold_steepness",
        "hot_increment",
        "hot_midpoint",
        "hot_steepness",
    ),
    "skew_gauss": ("scale", "location_temperature", "sigma", "skew"),
    "skew_gauss_peak": ("maximum", "optimum_temperature", "sigma", "skew"),
    "skew_gaussinv": (
        "scale",
        "location_temperature",
        "sigma",
        "skew",
    ),
    "normalized_gaussian_adult_fecundity_profile": ("peak_location", "sigma"),
}


@dataclass(frozen=True)
class FitResult:
    name: str
    function: str
    parameters: tuple[float, ...]
    rss: float
    r2: float
    n: int

    def as_dict(self) -> dict[str, float | int | str]:
        parameter_names = FUNCTION_PARAMETER_NAMES[self.function]
        if len(parameter_names) != len(self.parameters):
            raise ValueError(
                f"Expected {len(parameter_names)} parameters for {self.function}; "
                f"found {len(self.parameters)}."
            )
        result: dict[str, float | int | str] = {
            "name": self.name,
            "function": self.function,
            "rss": self.rss,
            "r2": self.r2,
            "n": self.n,
        }
        result.update(zip(parameter_names, self.parameters))
        return result


@dataclass(frozen=True)
class AdultFecundityProfileFit:
    stage_count: int
    weights: np.ndarray
    rss: float
    n: int

    def weights_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "adult_substage": np.arange(1, self.stage_count + 1),
                "fecundity_weight": self.weights,
            }
        )


@dataclass(frozen=True)
class AdultTimingProfileFit:
    stage_count: int
    reproduction_weights: np.ndarray
    mortality_weights: np.ndarray
    egg_normalized_sse: float
    survival_normalized_sse: float
    smoothness_penalty: float
    total_normalized_sse: float
    n: int

    def weights_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "adult_substage": np.arange(1, self.stage_count + 1),
                "reproduction_weight": self.reproduction_weights,
                "mortality_weight": self.mortality_weights,
            }
        )


def gauss(
    temperature: np.ndarray | float,
    maximum: float,
    optimum_temperature: float,
    sigma: float,
) -> np.ndarray:
    temperature = np.asarray(temperature, dtype=float)
    return maximum * np.exp(-(((optimum_temperature - temperature) / sigma) ** 2))


def gaussinv(
    temperature: np.ndarray | float,
    minimum: float,
    minimum_temperature: float,
    sigma: float,
) -> np.ndarray:
    temperature = np.asarray(temperature, dtype=float)
    return minimum * np.exp(((minimum_temperature - temperature) / sigma) ** 2)


def q10_deactivation_response(
    temperature: np.ndarray | float,
    response_at_reference: float,
    q10: float,
    deactivation_temperature: float,
    deactivation_steepness: float,
) -> np.ndarray:
    """Positive Q10 response with normalized logistic high-temperature decline."""

    temperature = np.asarray(temperature, dtype=float)
    reference_temperature = 20.0
    q10 = np.maximum(q10, 1e-12)
    deactivation_steepness = np.maximum(deactivation_steepness, 1e-12)
    log_activation = np.log(q10) * (temperature - reference_temperature) / 10.0
    reference_deactivation = np.logaddexp(
        0.0,
        deactivation_steepness * (reference_temperature - deactivation_temperature),
    )
    deactivation = np.logaddexp(
        0.0, deactivation_steepness * (temperature - deactivation_temperature)
    )
    return response_at_reference * np.exp(
        log_activation + reference_deactivation - deactivation
    )


def q10_deactivation_inv(
    temperature: np.ndarray | float,
    response_at_reference: float,
    q10: float,
    deactivation_temperature: float,
    deactivation_steepness: float,
) -> np.ndarray:
    """Reciprocal Q10 response parameterized by the reference response rate."""

    rate = q10_deactivation_response(
        temperature,
        response_at_reference,
        q10,
        deactivation_temperature,
        deactivation_steepness,
    )
    return 1.0 / np.maximum(rate, 1e-12)


def q10_deactivation_delay(
    temperature: np.ndarray | float,
    delay_at_reference: float,
    q10: float,
    deactivation_temperature: float,
    deactivation_steepness: float,
) -> np.ndarray:
    """Mean delay whose reciprocal follows the Q10 development-rate response."""

    return q10_deactivation_inv(
        temperature,
        1.0 / np.maximum(delay_at_reference, 1e-12),
        q10,
        deactivation_temperature,
        deactivation_steepness,
    )


def double_logistic_mortality(
    temperature: np.ndarray | float,
    minimum: float,
    cold_increment: float,
    cold_midpoint: float,
    cold_steepness: float,
    hot_increment: float,
    hot_midpoint: float,
    hot_steepness: float,
) -> np.ndarray:
    temperature = np.asarray(temperature, dtype=float)
    cold = cold_increment / (
        1.0 + np.exp(cold_steepness * (temperature - cold_midpoint))
    )
    hot = hot_increment / (
        1.0 + np.exp(-hot_steepness * (temperature - hot_midpoint))
    )
    return minimum + cold + hot


def skew_gauss(
    temperature: np.ndarray | float,
    scale: float,
    location_temperature: float,
    sigma: float,
    skew: float,
) -> np.ndarray:
    temperature = np.asarray(temperature, dtype=float)
    z = (temperature - location_temperature) / sigma
    return scale * np.exp(-(z**2)) * (1 + erf(skew * z / np.sqrt(2)))


def skew_gauss_peak(
    temperature: np.ndarray | float,
    maximum: float,
    optimum_temperature: float,
    sigma: float,
    skew: float,
) -> np.ndarray:
    """Skew-Gaussian response parameterized by its maximum and optimum."""

    temperature = np.asarray(temperature, dtype=float)
    mode_offset = _skew_gaussian_mode_offset(skew)
    z = (temperature - optimum_temperature) / sigma + mode_offset
    raw_response = np.exp(-(z**2)) * (1 + erf(skew * z / np.sqrt(2)))
    mode_response = np.exp(-(mode_offset**2)) * (
        1 + erf(skew * mode_offset / np.sqrt(2))
    )
    return maximum * raw_response / mode_response


def skew_gaussinv(
    temperature: np.ndarray | float,
    scale: float,
    location_temperature: float,
    sigma: float,
    skew: float,
) -> np.ndarray:
    temperature = np.asarray(temperature, dtype=float)
    z = (temperature - location_temperature) / sigma
    return scale * np.exp(z**2) * (1 + erf(skew * z / np.sqrt(2)))


def _skew_gaussian_mode_offset(skew: float) -> float:
    """Return the standardized mode of the unscaled skew-Gaussian kernel."""

    if abs(skew) < 1e-12:
        return 0.0
    magnitude = abs(float(skew))

    def log_derivative(z: float) -> float:
        normal_density = np.exp(-0.5 * (magnitude * z) ** 2) / np.sqrt(2 * np.pi)
        normal_cdf = 0.5 * (1 + erf(magnitude * z / np.sqrt(2)))
        return -2 * z + magnitude * normal_density / normal_cdf

    positive_mode = brentq(log_derivative, 0.0, 1.0)
    return float(np.copysign(positive_mode, skew))


def fit_stage_delay(data: pd.DataFrame, name: str) -> FitResult:
    return _fit(
        data,
        name=name,
        function_name="gaussinv",
        function=gaussinv,
        p0=_stage_initial_values(data),
        bounds=([0.0, -100.0, 1e-6], [np.inf, 100.0, np.inf]),
    )


def fit_stage_development_rate(data: pd.DataFrame, name: str) -> FitResult:
    fit_data = data.copy()
    fit_data["value"] = 1.0 / fit_data["value"]
    rate_fit = _fit(
        fit_data,
        name=name,
        function_name="gauss",
        function=gauss,
        p0=(
            float(fit_data["value"].max()),
            float(fit_data["temperature"].median()),
            10.0,
        ),
        bounds=([0.0, -100.0, 1e-6], [np.inf, 100.0, np.inf]),
    )
    return FitResult(
        name=name,
        function="gaussinv",
        parameters=(
            1.0 / rate_fit.parameters[0],
            rate_fit.parameters[1],
            rate_fit.parameters[2],
        ),
        rss=rate_fit.rss,
        r2=rate_fit.r2,
        n=rate_fit.n,
    )


def fit_stage_delay_q10_deactivation(data: pd.DataFrame, name: str) -> FitResult:
    """Fit development rates but parameterize the response by delay at 20 C."""

    fit_data = data.copy()
    fit_data["value"] = 1.0 / fit_data["value"]
    rate_fit = fit_q10_deactivation_response(
        fit_data,
        name=name,
    )
    return FitResult(
        name=name,
        function="q10_deactivation_delay",
        parameters=(
            1.0 / rate_fit.parameters[0],
            *rate_fit.parameters[1:],
        ),
        rss=rate_fit.rss,
        r2=rate_fit.r2,
        n=rate_fit.n,
    )


def fit_shared_stage_delays(data: pd.DataFrame) -> dict[str, FitResult]:
    """Fit E/L/P delays with one shared inverse-Gaussian temperature shape.

    The fitted observation is observed delay. Each stage has its own scale, and
    all stages share the same temperature-shape parameters.
    """

    fit_data = data.loc[data["stage"].isin(STAGES), ["stage", "temperature", "value"]]
    fit_data = fit_data.loc[fit_data["value"] > 0].copy()
    if fit_data.empty:
        raise ValueError("Cannot fit shared stage delays without positive durations.")

    stage_index = {stage: index for index, stage in enumerate(STAGES)}
    temperatures = fit_data["temperature"].to_numpy(dtype=float)
    stages = fit_data["stage"].map(stage_index).to_numpy(dtype=int)
    observed_delay = fit_data["value"].to_numpy(dtype=float)

    stage_medians = fit_data.groupby("stage")["value"].median()
    p0 = [float(stage_medians[stage]) for stage in STAGES]
    p0.extend([float(fit_data["temperature"].median()), 10.0])

    def delay_model(x: tuple[np.ndarray, np.ndarray], *params: float) -> np.ndarray:
        temp, stage_ids = x
        scales = np.asarray(params[: len(STAGES)], dtype=float)
        minimum_temperature = float(params[-2])
        sigma = float(params[-1])
        return scales[stage_ids.astype(int)] * np.exp(
            ((minimum_temperature - temp) / sigma) ** 2
        )

    popt, _ = curve_fit(
        delay_model,
        (temperatures, stages),
        observed_delay,
        p0=p0,
        bounds=(
            [0.0] * len(STAGES) + [-100.0, 1e-6],
            [np.inf] * len(STAGES) + [100.0, np.inf],
        ),
        maxfev=100_000,
    )
    predicted_delay = delay_model((temperatures, stages), *popt)
    rss_by_stage: dict[str, float] = {}
    r2_by_stage: dict[str, float] = {}
    n_by_stage: dict[str, int] = {}
    for stage in STAGES:
        mask = fit_data["stage"].to_numpy() == stage
        obs = observed_delay[mask]
        pred = predicted_delay[mask]
        rss = float(np.sum((obs - pred) ** 2))
        tss = float(np.sum((obs - obs.mean()) ** 2))
        rss_by_stage[stage] = rss
        r2_by_stage[stage] = 1.0 - rss / tss if tss > 0 else np.nan
        n_by_stage[stage] = int(mask.sum())

    minimum_temperature = float(popt[-2])
    sigma = float(popt[-1])
    return {
        stage: FitResult(
            name=stage,
            function="gaussinv",
            parameters=(
                float(popt[index]),
                minimum_temperature,
                sigma,
            ),
            rss=rss_by_stage[stage],
            r2=r2_by_stage[stage],
            n=n_by_stage[stage],
        )
        for index, stage in enumerate(STAGES)
    }


def fit_shared_stage_delays_log(data: pd.DataFrame) -> dict[str, FitResult]:
    """Fit E/L/P delays on the log scale with a shared temperature shape."""

    fit_data = data.loc[data["stage"].isin(STAGES), ["stage", "temperature", "value"]]
    fit_data = fit_data.loc[fit_data["value"] > 0].copy()
    if fit_data.empty:
        raise ValueError("Cannot fit shared stage delays without positive durations.")

    stage_index = {stage: index for index, stage in enumerate(STAGES)}
    temperatures = fit_data["temperature"].to_numpy(dtype=float)
    stages = fit_data["stage"].map(stage_index).to_numpy(dtype=int)
    observed_delay = fit_data["value"].to_numpy(dtype=float)
    observed_log_delay = np.log(observed_delay)

    stage_medians = fit_data.groupby("stage")["value"].median()
    p0 = [float(np.log(stage_medians[stage])) for stage in STAGES]
    p0.extend([float(fit_data["temperature"].median()), 10.0])

    def log_delay_model(x: tuple[np.ndarray, np.ndarray], *params: float) -> np.ndarray:
        temp, stage_ids = x
        log_scales = np.asarray(params[: len(STAGES)], dtype=float)
        minimum_temperature = float(params[-2])
        sigma = float(params[-1])
        return log_scales[stage_ids.astype(int)] + (
            (minimum_temperature - temp) / sigma
        ) ** 2

    popt, _ = curve_fit(
        log_delay_model,
        (temperatures, stages),
        observed_log_delay,
        p0=p0,
        bounds=(
            [-np.inf] * len(STAGES) + [-100.0, 1e-6],
            [np.inf] * len(STAGES) + [100.0, np.inf],
        ),
        maxfev=100_000,
    )
    predicted_delay = np.exp(log_delay_model((temperatures, stages), *popt))
    rss_by_stage: dict[str, float] = {}
    r2_by_stage: dict[str, float] = {}
    n_by_stage: dict[str, int] = {}
    for stage in STAGES:
        mask = fit_data["stage"].to_numpy() == stage
        obs = observed_delay[mask]
        pred = predicted_delay[mask]
        rss = float(np.sum((np.log(obs) - np.log(pred)) ** 2))
        tss = float(np.sum((np.log(obs) - np.log(obs).mean()) ** 2))
        rss_by_stage[stage] = rss
        r2_by_stage[stage] = 1.0 - rss / tss if tss > 0 else np.nan
        n_by_stage[stage] = int(mask.sum())

    minimum_temperature = float(popt[-2])
    sigma = float(popt[-1])
    return {
        stage: FitResult(
            name=stage,
            function="gaussinv",
            parameters=(
                float(np.exp(popt[index])),
                minimum_temperature,
                sigma,
            ),
            rss=rss_by_stage[stage],
            r2=r2_by_stage[stage],
            n=n_by_stage[stage],
        )
        for index, stage in enumerate(STAGES)
    }


def fit_shared_stage_development_rates(data: pd.DataFrame) -> dict[str, FitResult]:
    """Fit E/L/P development rates, returning equivalent delay parameters."""

    fit_data = data.loc[data["stage"].isin(STAGES), ["stage", "temperature", "value"]]
    fit_data = fit_data.loc[fit_data["value"] > 0].copy()
    if fit_data.empty:
        raise ValueError("Cannot fit shared stage rates without positive durations.")

    stage_index = {stage: index for index, stage in enumerate(STAGES)}
    temperatures = fit_data["temperature"].to_numpy(dtype=float)
    stages = fit_data["stage"].map(stage_index).to_numpy(dtype=int)
    observed_delay = fit_data["value"].to_numpy(dtype=float)
    observed_rate = 1.0 / observed_delay

    stage_medians = fit_data.groupby("stage")["value"].median()
    p0 = [float(1.0 / stage_medians[stage]) for stage in STAGES]
    p0.extend([float(fit_data["temperature"].median()), 10.0])

    def rate_model(x: tuple[np.ndarray, np.ndarray], *params: float) -> np.ndarray:
        temp, stage_ids = x
        rate_scales = np.asarray(params[: len(STAGES)], dtype=float)
        optimum_temperature = float(params[-2])
        sigma = float(params[-1])
        return rate_scales[stage_ids.astype(int)] * np.exp(
            -(((optimum_temperature - temp) / sigma) ** 2)
        )

    popt, _ = curve_fit(
        rate_model,
        (temperatures, stages),
        observed_rate,
        p0=p0,
        bounds=(
            [0.0] * len(STAGES) + [-100.0, 1e-6],
            [np.inf] * len(STAGES) + [100.0, np.inf],
        ),
        maxfev=100_000,
    )
    predicted_rate = rate_model((temperatures, stages), *popt)
    predicted_delay = 1.0 / predicted_rate
    rss_by_stage: dict[str, float] = {}
    r2_by_stage: dict[str, float] = {}
    n_by_stage: dict[str, int] = {}
    for stage in STAGES:
        mask = fit_data["stage"].to_numpy() == stage
        obs = observed_rate[mask]
        pred = predicted_rate[mask]
        rss = float(np.sum((obs - pred) ** 2))
        tss = float(np.sum((obs - obs.mean()) ** 2))
        rss_by_stage[stage] = rss
        r2_by_stage[stage] = 1.0 - rss / tss if tss > 0 else np.nan
        n_by_stage[stage] = int(mask.sum())

    optimum_temperature = float(popt[-2])
    sigma = float(popt[-1])
    return {
        stage: FitResult(
            name=stage,
            function="gaussinv",
            parameters=(float(1.0 / popt[index]), optimum_temperature, sigma),
            rss=rss_by_stage[stage],
            r2=r2_by_stage[stage],
            n=n_by_stage[stage],
        )
        for index, stage in enumerate(STAGES)
    }


def fit_juvenile_mortality(
    data: pd.DataFrame,
    *,
    stage_fits: dict[str, FitResult] | None = None,
    stage_counts: dict[str, int] | None = None,
) -> FitResult:
    if stage_fits is not None and stage_counts is not None:
        return _fit_juvenile_mortality_to_survival(
            data,
            function_name="gaussinv",
            function=gaussinv,
            p0=(float(data["value"].median()), 15.0, 15.0),
            bounds=([0.0, 10.0, 10.0], [np.inf, 20.0, np.inf]),
            stage_fits=stage_fits,
            stage_counts=stage_counts,
        )
    return _fit(
        data,
        name="Juvenile mortality rate",
        function_name="gaussinv",
        function=gaussinv,
        p0=(float(data["value"].median()), 15.0, 15.0),
        bounds=([0.0, 10.0, 10.0], [np.inf, 20.0, np.inf]),
    )


def fit_adult_delay(data: pd.DataFrame) -> FitResult:
    peak = data.loc[data["value"].idxmax()]
    return _fit(
        data,
        name="Adult duration",
        function_name="gauss",
        function=gauss,
        p0=(float(peak["value"]), float(peak["temperature"]), 10.0),
        bounds=([0.0, -100.0, 1e-6], [np.inf, 100.0, np.inf]),
    )


def fit_adult_mortality_rate(data: pd.DataFrame) -> FitResult:
    fit_data = data.copy()
    fit_data["value"] = 1.0 / fit_data["value"]
    rate_fit = _fit(
        fit_data,
        name="Adult mortality rate",
        function_name="gaussinv",
        function=gaussinv,
        p0=_stage_initial_values(fit_data),
        bounds=([0.0, -100.0, 1e-6], [np.inf, 100.0, np.inf]),
    )
    return rate_fit


def fit_lifetime_fecundity(data: pd.DataFrame) -> FitResult:
    peak = data.loc[data["value"].idxmax()]
    return _fit(
        data,
        name="Lifetime fecundity",
        function_name="gauss",
        function=gauss,
        p0=(float(peak["value"]), float(peak["temperature"]), 10.0),
        bounds=([0.0, -100.0, 1e-6], [np.inf, 100.0, np.inf]),
    )


def fit_skew_adult_delay(data: pd.DataFrame) -> FitResult:
    return _fit_skew_peak(data, name="Adult duration")


def fit_skew_lifetime_fecundity(data: pd.DataFrame) -> FitResult:
    return _fit_skew_peak(data, name="Lifetime fecundity")


def fit_skew_juvenile_mortality(
    data: pd.DataFrame,
    *,
    stage_fits: dict[str, FitResult] | None = None,
    stage_counts: dict[str, int] | None = None,
) -> FitResult:
    if stage_fits is not None and stage_counts is not None:
        return _fit_juvenile_mortality_to_survival(
            data,
            function_name="skew_gaussinv",
            function=skew_gaussinv,
            p0=(float(data["value"].median()), 15.0, 15.0, 0.0),
            bounds=([0.0, 10.0, 10.0, -20.0], [np.inf, 20.0, np.inf, 20.0]),
            stage_fits=stage_fits,
            stage_counts=stage_counts,
        )
    return _fit(
        data,
        name="Juvenile mortality rate",
        function_name="skew_gaussinv",
        function=skew_gaussinv,
        p0=(float(data["value"].median()), 15.0, 15.0, 0.0),
        bounds=([0.0, 10.0, 10.0, -20.0], [np.inf, 20.0, np.inf, 20.0]),
    )


def maturation_delay_summary(
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
            ].assign(value=lambda data: data[["L1", "L2", "L3"]].sum(axis=1)),
            "value",
        ),
        ("Pupa", development.loc[development["P"] > 0, ["temperature", "P"]], "P"),
    ]

    for stage, data, column in stage_specs:
        stage_data = data[["temperature", column]].rename(columns={column: "value"})
        stage_data = stage_data.loc[stage_data["value"] > 0].copy()
        summary = (
            stage_data.groupby("temperature", as_index=False)
            .agg(value=("value", "mean"), n=("value", "size"))
            .sort_values("temperature")
        )
        summary["stage"] = stage
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def adult_delay_summary(adult_survival: pd.DataFrame) -> pd.DataFrame:
    female = adult_survival.loc[adult_survival["AF"] > 0, ["temperature", "AF"]].rename(
        columns={"AF": "value"}
    )
    return (
        female.groupby("temperature", as_index=False)
        .agg(value=("value", "mean"), n=("value", "size"))
        .sort_values("temperature")
    )


def lifetime_fecundity_summary(fertility: pd.DataFrame) -> pd.DataFrame:
    totals = fertility.groupby(["temperature", "female"], as_index=False).agg(
        total_eggs=("eggs", "sum")
    )
    return (
        totals.groupby("temperature", as_index=False)
        .agg(value=("total_eggs", "mean"), n=("female", "size"))
        .sort_values("temperature")
    )


def daily_fecundity_summary(fertility: pd.DataFrame) -> pd.DataFrame:
    return (
        fertility.dropna(subset=["eggs"])
        .groupby("temperature", as_index=False)
        .agg(value=("eggs", "mean"), n=("eggs", "size"))
        .sort_values("temperature")
    )


def stage_duration_observations(
    development: pd.DataFrame,
    adult_survival: pd.DataFrame,
    fertility: pd.DataFrame,
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
            ].assign(value=lambda data: data[["L1", "L2", "L3"]].sum(axis=1)),
            "value",
        ),
        ("Pupa", development.loc[development["P"] > 0, ["temperature", "P"]], "P"),
    ]
    for stage, data, column in stage_specs:
        stage_data = data[["temperature", column]].rename(columns={column: "duration"})
        stage_data = stage_data.loc[stage_data["duration"] > 0].copy()
        stage_data["stage"] = stage
        rows.append(
            stage_data[["stage", "temperature", "duration"]]
        )

    adult_lifetimes = pd.concat(
        [
            adult_survival.loc[adult_survival["AM"] > 0, ["temperature", "AM"]].rename(
                columns={"AM": "duration"}
            ),
            adult_survival.loc[adult_survival["AF"] > 0, ["temperature", "AF"]].rename(
                columns={"AF": "duration"}
            ),
        ],
        ignore_index=True,
    )
    adult_lifetimes = adult_lifetimes.loc[adult_lifetimes["duration"] > 0].copy()
    adult_lifetimes["stage"] = "Adult"
    rows.append(
        adult_lifetimes[["stage", "temperature", "duration"]]
    )
    return pd.concat(rows, ignore_index=True)


def erlang_stage_counts(
    duration_observations: pd.DataFrame,
    *,
    minimum: int = 1,
    maximum: int = 40,
    search_maximum: int = 400,
) -> tuple[dict[str, int], pd.DataFrame]:
    """Choose capped Erlang-chain lengths from observed durations.

    The Erlang shape is common across temperatures within a stage, while its
    mean is profiled separately for every temperature.

    The likelihood-selected integer shape is subsequently capped at `maximum`
    for numerical tractability. If the profile optimum reaches
    `search_maximum`, `raw_substage_count_is_lower_bound` is true.
    """

    required = {
        "stage",
        "temperature",
        "duration",
    }
    missing = required - set(duration_observations.columns)
    if missing:
        raise ValueError(
            "Stage counts require columns: "
            + ", ".join(sorted(missing))
        )
    if minimum < 1 or maximum < minimum or search_maximum < maximum:
        raise ValueError(
            "Stage-count bounds must satisfy 1 <= minimum <= maximum <= search_maximum."
        )

    rows = []
    for stage, stage_data in duration_observations.groupby("stage", sort=False):
        candidate_scores = []
        for candidate in range(minimum, search_maximum + 1):
            negative_log_likelihood = 0.0
            fitted_temperature_count = 0
            for _, temperature_data in stage_data.groupby("temperature", sort=True):
                observed = temperature_data["duration"].to_numpy(dtype=float)
                if len(observed) == 0:
                    continue

                observed_mean = float(np.mean(observed))
                lower_mean = max(observed_mean * 0.1, 1e-8)
                upper_mean = max(observed_mean * 2.0, lower_mean * 1.01)

                def profile_objective(log_mean: float) -> float:
                    mean = float(np.exp(log_mean))
                    scale = mean / candidate
                    return -float(
                        gamma_distribution.logpdf(
                            observed, a=candidate, scale=scale
                        ).sum()
                    )

                result = minimize_scalar(
                    profile_objective,
                    bounds=(np.log(lower_mean), np.log(upper_mean)),
                    method="bounded",
                )
                if not result.success or not np.isfinite(result.fun):
                    negative_log_likelihood = np.inf
                    break
                negative_log_likelihood += float(result.fun)
                fitted_temperature_count += 1
            candidate_scores.append(
                (negative_log_likelihood, candidate, fitted_temperature_count)
            )

        best_score, raw_count, temperature_count = min(
            candidate_scores, key=lambda item: item[0]
        )
        count = min(maximum, max(minimum, int(raw_count)))
        rows.append(
            {
                "stage": stage,
                "stage_key": STAGE_COUNT_KEYS[stage],
                "selection_method": "Erlang profile likelihood",
                "raw_substage_count": int(raw_count),
                "raw_substage_count_is_lower_bound": bool(raw_count == search_maximum),
                "substage_count": count,
                "substage_cap": maximum,
                "negative_log_likelihood": best_score,
                "temperature_count": int(temperature_count),
                "observation_count": int(len(stage_data)),
            }
        )
    table = pd.DataFrame(rows)
    counts = {
        str(row.stage_key): int(row.substage_count)
        for row in table.itertuples(index=False)
    }
    return counts, table


def variance_matched_stage_counts(
    duration_observations: pd.DataFrame,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> tuple[dict[str, int], pd.DataFrame]:
    """Choose stage-chain lengths from observed duration variance.

    Counts are fixed across temperatures. For each stage, the rule averages
    temperature-specific CV^2 values and uses n = round(1 / mean(CV^2)).
    """

    rows = []
    for stage, stage_data in duration_observations.groupby("stage", sort=False):
        temp_stats = (
            stage_data.groupby("temperature", as_index=False)
            .agg(
                mean_duration=("duration", "mean"),
                variance=("duration", "var"),
                n=("duration", "size"),
            )
            .dropna(subset=["variance"])
        )
        temp_stats = temp_stats.loc[
            (temp_stats["mean_duration"] > 0) & (temp_stats["variance"] > 0)
        ].copy()
        if temp_stats.empty:
            mean_cv2 = np.nan
            raw_count = float(minimum)
        else:
            temp_stats["cv2"] = (
                temp_stats["variance"] / temp_stats["mean_duration"] ** 2
            )
            mean_cv2 = float(temp_stats["cv2"].mean())
            raw_count = 1.0 / mean_cv2 if mean_cv2 > 0 else float(minimum)
        count = max(minimum, int(np.round(raw_count)))
        if maximum is not None:
            count = min(maximum, count)
        rows.append(
            {
                "stage": stage,
                "stage_key": STAGE_COUNT_KEYS[stage],
                "mean_cv2": mean_cv2,
                "raw_substage_count": raw_count,
                "substage_count": count,
                "temperature_count": int(temp_stats["temperature"].nunique()),
                "observation_count": int(len(stage_data)),
            }
        )
    table = pd.DataFrame(rows)
    counts = {
        str(row.stage_key): int(row.substage_count)
        for row in table.itertuples(index=False)
    }
    return counts, table


def fit_adult_substage_fecundity_profile(
    fertility: pd.DataFrame,
    fecundity_fit: FitResult,
    adult_fit: FitResult,
    *,
    adult_stage_count: int = 4,
) -> tuple[AdultFecundityProfileFit, pd.DataFrame]:
    """Fit shared adult-substage fecundity multipliers to daily egg timing.

    The fitted weights are constrained to be nonnegative and to sum to the
    number of adult substages. This preserves the fitted lifetime fecundity
    while allowing egg laying to shift earlier or later within the adult chain.
    """

    daily = (
        fertility.dropna(subset=["eggs"])
        .groupby(["temperature", "adult_day"], as_index=False)
        .agg(mean_eggs=("eggs", "mean"), live_females=("eggs", "size"))
        .sort_values(["temperature", "adult_day"])
    )
    if daily.empty:
        raise ValueError("Cannot fit adult fecundity profile without egg observations.")

    temperatures = daily["temperature"].to_numpy(dtype=float)
    adult_days = daily["adult_day"].to_numpy(dtype=float)
    observed = daily["mean_eggs"].to_numpy(dtype=float)
    observation_weights = daily["live_females"].to_numpy(dtype=float)
    adult_delay = predict(adult_fit, temperatures)
    base_daily = predict(fecundity_fit, temperatures) / adult_delay
    occupancy = adult_substage_occupancy(
        adult_days,
        adult_delay,
        adult_stage_count=adult_stage_count,
    )

    def weights_from_logits(logits: np.ndarray) -> np.ndarray:
        centered = logits - np.max(logits)
        raw = np.exp(centered)
        return adult_stage_count * raw / raw.sum()

    def objective_for_weights(weights: np.ndarray) -> float:
        predicted = base_daily * occupancy.dot(weights)
        residuals = observed - predicted
        return float(np.sum(observation_weights * residuals**2))

    def objective(logits: np.ndarray) -> float:
        return objective_for_weights(weights_from_logits(logits))

    result = minimize(
        objective,
        x0=np.zeros(adult_stage_count, dtype=float),
        method="L-BFGS-B",
        options={"maxiter": 10_000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Adult fecundity profile fit failed: {result.message}")

    weights = weights_from_logits(result.x)
    fitted = base_daily * occupancy.dot(weights)
    output = daily.copy()
    output["fitted_eggs"] = fitted
    output["baseline_eggs"] = base_daily
    output["adult_delay"] = adult_delay

    return (
        AdultFecundityProfileFit(
            stage_count=adult_stage_count,
            weights=np.asarray(weights, dtype=float),
            rss=objective_for_weights(weights),
            n=len(daily),
        ),
        output,
    )


def fit_adult_substage_reproduction_mortality_profile(
    fertility: pd.DataFrame,
    adult_survival: pd.DataFrame,
    fecundity_fit: FitResult,
    adult_fit: FitResult,
    *,
    adult_stage_count: int = 4,
    survival_weight: float = 1.0,
    smoothness_weight: float = 0.02,
) -> tuple[AdultTimingProfileFit, pd.DataFrame]:
    """Fit shared adult reproduction and mortality weights to daily egg timing.

    Reproduction and mortality weights are nonnegative and constrained to average
    1 across adult substages. Temperature still controls the total fecundity and
    adult-time scale; these weights only alter adult-age shape.
    """

    daily = adult_daily_reproduction_summary(fertility)
    survival_daily = adult_daily_survival_summary(adult_survival)
    if daily.empty or survival_daily.empty:
        raise ValueError("Cannot fit adult timing profile without egg observations.")

    temperatures = daily["temperature"].to_numpy(dtype=float)
    adult_days = daily["adult_day"].to_numpy(dtype=float)
    observed_eggs = daily["mean_eggs"].to_numpy(dtype=float)
    live_females = daily["live_females"].to_numpy(dtype=float)
    adult_delay = predict(adult_fit, temperatures)
    base_daily = predict(fecundity_fit, temperatures) / adult_delay
    survival_temperatures = survival_daily["temperature"].to_numpy(dtype=float)
    survival_adult_days = survival_daily["adult_day"].to_numpy(dtype=float)
    observed_survival = survival_daily["survival_fraction"].to_numpy(dtype=float)
    initial_females = survival_daily["initial_females"].to_numpy(dtype=float)
    survival_adult_delay = predict(adult_fit, survival_temperatures)

    def weights_from_logits(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        reproduction_logits = logits[:adult_stage_count]
        mortality_logits = logits[adult_stage_count:]
        return (
            _normalized_positive_weights(reproduction_logits, adult_stage_count),
            _normalized_positive_weights(mortality_logits, adult_stage_count),
        )

    def predictions(
        reproduction_weights: np.ndarray, mortality_weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        occupancy, survival = adult_substage_occupancy_with_mortality(
            adult_days,
            adult_delay,
            mortality_weights=mortality_weights,
        )
        _, survival_predictions = adult_substage_occupancy_with_mortality(
            survival_adult_days,
            survival_adult_delay,
            mortality_weights=mortality_weights,
        )
        egg_predictions = base_daily * occupancy.dot(reproduction_weights)
        return egg_predictions, survival_predictions

    def normalized_sse(
        reproduction_weights: np.ndarray, mortality_weights: np.ndarray
    ) -> tuple[float, float, float, float]:
        predicted_eggs, predicted_survival = predictions(
            reproduction_weights,
            mortality_weights,
        )
        egg_residual = observed_eggs - predicted_eggs
        survival_residual = observed_survival - predicted_survival
        egg_sse = float(np.sum(live_females * egg_residual**2))
        egg_scale = float(np.sum(live_females * observed_eggs**2))
        survival_sse = float(np.sum(initial_females * survival_residual**2))
        survival_scale = float(np.sum(initial_females * observed_survival**2))
        egg_normalized = egg_sse / egg_scale if egg_scale > 0 else np.nan
        survival_normalized = (
            survival_sse / survival_scale if survival_scale > 0 else np.nan
        )
        smoothness = smoothness_weight * (
            _log_weight_roughness(reproduction_weights)
            + _log_weight_roughness(mortality_weights)
        )
        total = egg_normalized + survival_weight * survival_normalized + smoothness
        return egg_normalized, survival_normalized, smoothness, total

    def objective(logits: np.ndarray) -> float:
        reproduction_weights, mortality_weights = weights_from_logits(logits)
        _, _, _, total = normalized_sse(reproduction_weights, mortality_weights)
        return total

    result = minimize(
        objective,
        x0=np.zeros(2 * adult_stage_count, dtype=float),
        method="L-BFGS-B",
        options={"maxiter": 20_000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Adult timing profile fit failed: {result.message}")

    reproduction_weights, mortality_weights = weights_from_logits(result.x)
    fitted_eggs, fitted_survival = predictions(reproduction_weights, mortality_weights)
    egg_normalized, survival_normalized, smoothness, total = normalized_sse(
        reproduction_weights,
        mortality_weights,
    )
    output = daily.copy()
    output["fitted_eggs"] = fitted_eggs
    output["adult_delay"] = adult_delay
    output["base_daily_eggs"] = base_daily
    survival_output = survival_daily.copy()
    survival_output["fitted_survival_fraction"] = fitted_survival
    survival_output["adult_delay"] = survival_adult_delay
    output = output.merge(
        survival_output[
            [
                "temperature",
                "adult_day",
                "survival_fraction",
                "initial_females",
                "fitted_survival_fraction",
            ]
        ],
        on=["temperature", "adult_day"],
        how="outer",
    ).sort_values(["temperature", "adult_day"])

    return (
        AdultTimingProfileFit(
            stage_count=adult_stage_count,
            reproduction_weights=np.asarray(reproduction_weights, dtype=float),
            mortality_weights=np.asarray(mortality_weights, dtype=float),
            egg_normalized_sse=egg_normalized,
            survival_normalized_sse=survival_normalized,
            smoothness_penalty=smoothness,
            total_normalized_sse=total,
            n=len(daily),
        ),
        output,
    )


def fit_parametric_adult_reproduction_mortality_profile(
    fertility: pd.DataFrame,
    adult_survival: pd.DataFrame,
    fecundity_fit: FitResult,
    adult_fit: FitResult,
    *,
    adult_stage_count: int = 12,
    survival_weight: float = 1.0,
) -> tuple[AdultTimingProfileFit, pd.DataFrame]:
    """Fit smooth parametric adult reproduction and mortality timing weights.

    Reproduction is represented as a single smooth hump over adult substages.
    Mortality is represented as a monotone increasing exponential hazard over
    adult substages. Both profiles are normalized to average 1.
    """

    daily = adult_daily_reproduction_summary(fertility)
    survival_daily = adult_daily_survival_summary(adult_survival)
    if daily.empty or survival_daily.empty:
        raise ValueError("Cannot fit adult timing profile without observations.")

    temperatures = daily["temperature"].to_numpy(dtype=float)
    adult_days = daily["adult_day"].to_numpy(dtype=float)
    observed_eggs = daily["mean_eggs"].to_numpy(dtype=float)
    live_females = daily["live_females"].to_numpy(dtype=float)
    adult_delay = predict(adult_fit, temperatures)
    base_daily = predict(fecundity_fit, temperatures) / adult_delay
    survival_temperatures = survival_daily["temperature"].to_numpy(dtype=float)
    survival_adult_days = survival_daily["adult_day"].to_numpy(dtype=float)
    observed_survival = survival_daily["survival_fraction"].to_numpy(dtype=float)
    initial_females = survival_daily["initial_females"].to_numpy(dtype=float)
    survival_adult_delay = predict(adult_fit, survival_temperatures)

    def predictions(
        vector: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        reproduction_weights, mortality_weights = parametric_adult_timing_weights(
            vector,
            adult_stage_count=adult_stage_count,
        )
        occupancy, _ = adult_substage_occupancy_with_mortality(
            adult_days,
            adult_delay,
            mortality_weights=mortality_weights,
        )
        _, survival_predictions = adult_substage_occupancy_with_mortality(
            survival_adult_days,
            survival_adult_delay,
            mortality_weights=mortality_weights,
        )
        egg_predictions = base_daily * occupancy.dot(reproduction_weights)
        return (
            egg_predictions,
            survival_predictions,
            reproduction_weights,
            mortality_weights,
        )

    def objective(vector: np.ndarray) -> float:
        predicted_eggs, predicted_survival, _, _ = predictions(vector)
        egg_residual = observed_eggs - predicted_eggs
        survival_residual = observed_survival - predicted_survival
        egg_sse = float(np.sum(live_females * egg_residual**2))
        egg_scale = float(np.sum(live_females * observed_eggs**2))
        survival_sse = float(np.sum(initial_females * survival_residual**2))
        survival_scale = float(np.sum(initial_females * observed_survival**2))
        return egg_sse / egg_scale + survival_weight * survival_sse / survival_scale

    result = minimize(
        objective,
        x0=np.asarray([0.45, np.log(0.18), np.log(2.0)], dtype=float),
        method="L-BFGS-B",
        bounds=[(0.0, 1.0), (np.log(0.04), np.log(0.8)), (np.log(1e-4), np.log(20.0))],
        options={"maxiter": 20_000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Parametric adult timing fit failed: {result.message}")

    fitted_eggs, fitted_survival, reproduction_weights, mortality_weights = predictions(
        result.x
    )
    egg_residual = observed_eggs - fitted_eggs
    survival_residual = observed_survival - fitted_survival
    egg_normalized = float(np.sum(live_females * egg_residual**2)) / float(
        np.sum(live_females * observed_eggs**2)
    )
    survival_normalized = float(
        np.sum(initial_females * survival_residual**2)
    ) / float(np.sum(initial_females * observed_survival**2))
    output = daily.copy()
    output["fitted_eggs"] = fitted_eggs
    output["adult_delay"] = adult_delay
    output["base_daily_eggs"] = base_daily
    survival_output = survival_daily.copy()
    survival_output["fitted_survival_fraction"] = fitted_survival
    survival_output["adult_delay"] = survival_adult_delay
    output = output.merge(
        survival_output[
            [
                "temperature",
                "adult_day",
                "survival_fraction",
                "initial_females",
                "fitted_survival_fraction",
            ]
        ],
        on=["temperature", "adult_day"],
        how="outer",
    ).sort_values(["temperature", "adult_day"])

    return (
        AdultTimingProfileFit(
            stage_count=adult_stage_count,
            reproduction_weights=np.asarray(reproduction_weights, dtype=float),
            mortality_weights=np.asarray(mortality_weights, dtype=float),
            egg_normalized_sse=egg_normalized,
            survival_normalized_sse=survival_normalized,
            smoothness_penalty=0.0,
            total_normalized_sse=egg_normalized + survival_weight * survival_normalized,
            n=len(daily) + len(survival_daily),
        ),
        output,
    )


def parametric_adult_timing_weights(
    vector: np.ndarray,
    *,
    adult_stage_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    center = float(vector[0])
    width = float(np.exp(vector[1]))
    mortality_slope = float(np.exp(vector[2]))
    x = (np.arange(adult_stage_count, dtype=float) + 0.5) / adult_stage_count
    reproduction_raw = np.exp(-0.5 * ((x - center) / width) ** 2)
    mortality_raw = np.exp(mortality_slope * (x - 0.5))
    return (
        adult_stage_count * reproduction_raw / reproduction_raw.sum(),
        adult_stage_count * mortality_raw / mortality_raw.sum(),
    )


def adult_daily_reproduction_summary(fertility: pd.DataFrame) -> pd.DataFrame:
    return (
        fertility.dropna(subset=["eggs"])
        .groupby(["temperature", "adult_day"], as_index=False)
        .agg(mean_eggs=("eggs", "mean"), live_females=("eggs", "size"))
        .sort_values(["temperature", "adult_day"])
    )


def adult_daily_survival_summary(adult_survival: pd.DataFrame) -> pd.DataFrame:
    rows = []
    female_lifetimes = adult_survival.loc[
        adult_survival["AF"] > 0, ["temperature", "AF"]
    ].copy()
    for temperature, data in female_lifetimes.groupby("temperature", sort=True):
        lifetimes = data["AF"].to_numpy(dtype=float)
        max_day = int(np.ceil(lifetimes.max()))
        initial_females = len(lifetimes)
        for adult_day in range(1, max_day + 1):
            rows.append(
                {
                    "temperature": temperature,
                    "adult_day": adult_day,
                    "initial_females": initial_females,
                    "live_females_survival": int(np.sum(lifetimes >= adult_day)),
                    "survival_fraction": float(np.mean(lifetimes >= adult_day)),
                }
            )
    return pd.DataFrame(rows)


def adult_substage_occupancy(
    adult_day: np.ndarray | float,
    adult_delay: np.ndarray | float,
    *,
    adult_stage_count: int,
) -> np.ndarray:
    """Conditional adult-chain substage occupancy for a live adult at adult day."""

    adult_day = np.asarray(adult_day, dtype=float)
    adult_delay = np.asarray(adult_delay, dtype=float)
    adult_day, adult_delay = np.broadcast_arrays(adult_day, adult_delay)
    adult_age = np.maximum(adult_day - 0.5, 0.0)
    transition_progress = adult_stage_count * adult_age / adult_delay
    substages = np.arange(adult_stage_count, dtype=float)
    terms = np.exp(
        -transition_progress[..., None]
        + substages * np.log(np.maximum(transition_progress[..., None], 1e-300))
        - np.array(
            [np.sum(np.log(np.arange(1, int(stage) + 1))) for stage in substages]
        )
    )
    terms = np.where(
        (transition_progress[..., None] == 0.0) & (substages == 0.0),
        1.0,
        terms,
    )
    return terms / terms.sum(axis=-1, keepdims=True)


def adult_substage_occupancy_with_mortality(
    adult_day: np.ndarray | float,
    adult_delay: np.ndarray | float,
    *,
    mortality_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    adult_day = np.asarray(adult_day, dtype=float)
    adult_delay = np.asarray(adult_delay, dtype=float)
    adult_day, adult_delay = np.broadcast_arrays(adult_day, adult_delay)
    mortality_weights = np.asarray(mortality_weights, dtype=float)
    stage_count = len(mortality_weights)
    adult_age = np.maximum(adult_day - 0.5, 0.0)
    occupancy = np.zeros((*adult_age.shape, stage_count), dtype=float)
    survival = np.zeros(adult_age.shape, dtype=float)

    for index, (age, delay) in enumerate(zip(adult_age.flat, adult_delay.flat)):
        transition_rate = stage_count / delay
        base_mortality = 1.0 / delay
        generator = np.zeros((stage_count, stage_count), dtype=float)
        for stage in range(stage_count):
            mortality = base_mortality * mortality_weights[stage]
            if stage < stage_count - 1:
                generator[stage, stage] = -(transition_rate + mortality)
                generator[stage, stage + 1] = transition_rate
            else:
                generator[stage, stage] = -mortality
        state = np.zeros(stage_count, dtype=float)
        state[0] = 1.0
        live_probabilities = state @ expm(generator * age)
        live_probability = float(live_probabilities.sum())
        survival.flat[index] = live_probability
        if live_probability > 0:
            occupancy.reshape(-1, stage_count)[index] = (
                live_probabilities / live_probability
            )
        else:
            occupancy.reshape(-1, stage_count)[index, -1] = 1.0
    return occupancy, survival


def juvenile_mortality_summary(
    development: pd.DataFrame, adult_survival: pd.DataFrame, cohort_size: int = 50
) -> pd.DataFrame:
    merged = adult_survival.merge(
        development, on=["temperature", "specimen"], how="inner"
    )
    merged["adult"] = (merged["AF"] > 0) | (merged["AM"] > 0)
    merged["preadult_days"] = merged[["E", "L1", "L2", "L3", "P"]].sum(axis=1)

    rows = []
    for temperature, data in merged.groupby("temperature", sort=True):
        adult_data = data.loc[data["adult"]]
        survival_probability = len(adult_data) / cohort_size
        if survival_probability <= 0 or adult_data.empty:
            continue
        mean_preadult_days = float(adult_data["preadult_days"].mean())
        rows.append(
            {
                "temperature": temperature,
                "value": -np.log(survival_probability) / mean_preadult_days,
                "survival_probability": survival_probability,
                "mean_preadult_days": mean_preadult_days,
                "n": cohort_size,
                "adult_count": len(adult_data),
            }
        )
    return pd.DataFrame(rows).sort_values("temperature")


def juvenile_mortality_summary_for_stage_chain(
    development: pd.DataFrame,
    adult_survival: pd.DataFrame,
    stage_counts: dict[str, int],
    cohort_size: int = 50,
) -> pd.DataFrame:
    """Summarize juvenile mortality from total adult emergence."""

    merged = adult_survival.merge(
        development, on=["temperature", "specimen"], how="inner"
    )
    merged["adult"] = (merged["AF"] > 0) | (merged["AM"] > 0)
    merged["larva_days"] = merged[["L1", "L2", "L3"]].sum(axis=1)
    merged["egg_days"] = merged["E"]
    merged["pupa_days"] = merged["P"]

    rows = []
    for temperature, data in merged.groupby("temperature", sort=True):
        adult_data = data.loc[data["adult"]].copy()
        survival_probability = len(adult_data) / cohort_size
        delay_source = adult_data if not adult_data.empty else data
        mean_delays = {
            "egg": _positive_mean(delay_source["egg_days"]),
            "larva": _positive_mean(delay_source["larva_days"]),
            "pupa": _positive_mean(delay_source["pupa_days"]),
        }
        mortality_rate = (
            juvenile_mortality_rate_for_stage_chain(
                survival_probability,
                mean_delays,
                stage_counts,
            )
            if survival_probability > 0
            and all(np.isfinite(value) for value in mean_delays.values())
            else np.nan
        )
        rows.append(
            {
                "temperature": temperature,
                "value": mortality_rate,
                "survival_probability": survival_probability,
                "mean_preadult_days": float(sum(mean_delays.values())),
                "n": cohort_size,
                "adult_count": len(adult_data),
            }
        )
    return pd.DataFrame(rows).sort_values("temperature")


def _positive_mean(values: pd.Series) -> float:
    positive = values.loc[values > 0]
    return float(positive.mean()) if len(positive) else np.nan


def juvenile_mortality_rate_for_stage_chain(
    survival_probability: float,
    mean_delays: dict[str, float],
    stage_counts: dict[str, int],
) -> float:
    """Find the constant juvenile hazard giving observed survival in a chain."""

    if survival_probability <= 0:
        return np.inf
    if survival_probability >= 1:
        return 0.0

    def predicted_survival(mortality_rate: float) -> float:
        survival = 1.0
        for stage_key in ("egg", "larva", "pupa"):
            count = int(stage_counts[stage_key])
            delay = float(mean_delays[stage_key])
            transition_rate = competing_risk_transition_rate(
                count, delay, mortality_rate
            )
            survival *= (transition_rate / (transition_rate + mortality_rate)) ** count
        return survival

    def objective(mortality_rate: float) -> float:
        return predicted_survival(mortality_rate) - survival_probability

    upper = min(
        int(stage_counts[stage_key]) / float(mean_delays[stage_key])
        for stage_key in ("egg", "larva", "pupa")
    )
    return float(brentq(objective, 0.0, upper * (1.0 - 1e-12)))


def competing_risk_transition_rate(
    stage_count: int,
    conditional_mean_delay: float | np.ndarray,
    mortality_rate: float | np.ndarray,
) -> np.ndarray:
    """Return progression rate consistent with duration among completers.

    In a chain with progression rate ``g`` and mortality rate ``m``, the time
    spent in each substage conditional on progression is exponential with rate
    ``g + m``. Therefore a measured mean duration ``tau`` among successful
    completers implies ``g + m = n / tau`` and ``g = n / tau - m``.
    Non-positive progression rates are clipped to zero, corresponding to no
    successful maturation under an incompatible extreme-rate combination.
    """

    total_event_rate = stage_count / np.maximum(
        np.asarray(conditional_mean_delay, dtype=float), 1e-12
    )
    return np.maximum(
        total_event_rate - np.asarray(mortality_rate, dtype=float),
        0.0,
    )


def predict(result: FitResult, temperature: np.ndarray | float) -> np.ndarray:
    functions = {
        "gauss": gauss,
        "gaussinv": gaussinv,
        "q10_deactivation_response": q10_deactivation_response,
        "q10_deactivation_inv": q10_deactivation_inv,
        "q10_deactivation_delay": q10_deactivation_delay,
        "double_logistic_mortality": double_logistic_mortality,
        "skew_gauss": skew_gauss,
        "skew_gauss_peak": skew_gauss_peak,
        "skew_gaussinv": skew_gaussinv,
    }
    return functions[result.function](temperature, *result.parameters)


def _fit_shared_stage_development_rates(
    data: pd.DataFrame,
    *,
    function_name: str,
    rate_function,
    shared_p0: tuple[float, ...],
    shared_bounds: tuple[list[float], list[float]],
) -> dict[str, FitResult]:
    fit_data = data.loc[data["stage"].isin(STAGES), ["stage", "temperature", "value"]]
    fit_data = fit_data.loc[fit_data["value"] > 0].copy()
    if fit_data.empty:
        raise ValueError("Cannot fit shared stage rates without positive durations.")

    stage_index = {stage: index for index, stage in enumerate(STAGES)}
    temperatures = fit_data["temperature"].to_numpy(dtype=float)
    stages = fit_data["stage"].map(stage_index).to_numpy(dtype=int)
    observed_delay = fit_data["value"].to_numpy(dtype=float)
    observed_rate = 1.0 / observed_delay

    stage_medians = fit_data.groupby("stage")["value"].median()
    p0 = [float(1.0 / stage_medians[stage]) for stage in STAGES]
    p0.extend(shared_p0)

    def rate_model(x: tuple[np.ndarray, np.ndarray], *params: float) -> np.ndarray:
        temp, stage_ids = x
        rate_scales = np.asarray(params[: len(STAGES)], dtype=float)
        shared = tuple(float(value) for value in params[len(STAGES) :])
        return rate_function(temp, rate_scales[stage_ids.astype(int)], *shared)

    popt, _ = curve_fit(
        rate_model,
        (temperatures, stages),
        observed_rate,
        p0=p0,
        bounds=(
            [0.0] * len(STAGES) + shared_bounds[0],
            [np.inf] * len(STAGES) + shared_bounds[1],
        ),
        maxfev=100_000,
    )
    predicted_rate = rate_model((temperatures, stages), *popt)
    rss_by_stage: dict[str, float] = {}
    r2_by_stage: dict[str, float] = {}
    n_by_stage: dict[str, int] = {}
    for stage in STAGES:
        mask = fit_data["stage"].to_numpy() == stage
        obs = observed_rate[mask]
        pred = predicted_rate[mask]
        rss = float(np.sum((obs - pred) ** 2))
        tss = float(np.sum((obs - obs.mean()) ** 2))
        rss_by_stage[stage] = rss
        r2_by_stage[stage] = 1.0 - rss / tss if tss > 0 else np.nan
        n_by_stage[stage] = int(mask.sum())

    shared = tuple(float(value) for value in popt[len(STAGES) :])
    return {
        stage: FitResult(
            name=stage,
            function=function_name,
            parameters=(float(popt[index]), *shared),
            rss=rss_by_stage[stage],
            r2=r2_by_stage[stage],
            n=n_by_stage[stage],
        )
        for index, stage in enumerate(STAGES)
    }


def _fit(
    data: pd.DataFrame,
    *,
    name: str,
    function_name: str,
    function,
    p0: tuple[float, ...],
    bounds: tuple[list[float], list[float]],
) -> FitResult:
    fit_data = data.dropna(subset=["temperature", "value"]).copy()
    x = fit_data["temperature"].to_numpy(dtype=float)
    y = fit_data["value"].to_numpy(dtype=float)
    params, _ = curve_fit(function, x, y, p0=p0, bounds=bounds, maxfev=100_000)
    fitted = function(x, *params)
    rss = float(np.sum((y - fitted) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - rss / tss) if tss > 0 else np.nan
    return FitResult(
        name=name,
        function=function_name,
        parameters=tuple(float(value) for value in params),
        rss=rss,
        r2=r2,
        n=len(fit_data),
    )


def fit_q10_deactivation_response(
    data: pd.DataFrame,
    *,
    name: str,
) -> FitResult:
    """Fit a Q10 response with normalized logistic high-temperature decline."""

    fit_data = data.dropna(subset=["temperature", "value"]).copy()
    fit_data = fit_data.loc[fit_data["value"] > 0].sort_values("temperature")
    if fit_data.empty:
        raise ValueError(f"Cannot fit {name} without positive observations.")

    temperature = fit_data["temperature"].to_numpy(dtype=float)
    observed = fit_data["value"].to_numpy(dtype=float)
    response_at_reference = max(
        float(np.interp(20.0, temperature, observed)),
        1e-8,
    )
    peak_temperature = float(temperature[np.argmax(observed)])
    lower = np.asarray([0.0, 0.05, 10.0, 0.01], dtype=float)
    upper = np.asarray([np.inf, 20.0, 45.0, 2.0], dtype=float)

    best_parameters: np.ndarray | None = None
    best_prediction: np.ndarray | None = None
    best_rss = np.inf
    for q10 in (0.5, 1.0, 2.0, 4.0):
        for threshold in (
            18.0,
            float(np.clip(peak_temperature, 12.0, 40.0)),
            25.0,
            30.0,
            35.0,
        ):
            for slope in (0.08, 0.25, 0.7):
                start = np.asarray(
                    [response_at_reference, q10, threshold, slope],
                    dtype=float,
                )
                start = np.maximum(start, lower + 1e-10)
                start = np.minimum(start, upper - 1e-10)
                try:
                    parameters, _ = curve_fit(
                        q10_deactivation_response,
                        temperature,
                        observed,
                        p0=start,
                        bounds=(lower, upper),
                        maxfev=200_000,
                    )
                except (RuntimeError, ValueError, FloatingPointError):
                    continue
                prediction = q10_deactivation_response(temperature, *parameters)
                rss = float(np.sum((observed - prediction) ** 2))
                if np.isfinite(rss) and rss < best_rss:
                    best_rss = rss
                    best_parameters = np.asarray(parameters, dtype=float)
                    best_prediction = np.asarray(prediction, dtype=float)

    if best_parameters is None or best_prediction is None:
        raise RuntimeError(f"Q10-deactivation fit failed for {name}.")

    tss = float(np.sum((observed - observed.mean()) ** 2))
    return FitResult(
        name=name,
        function="q10_deactivation_response",
        parameters=tuple(float(value) for value in best_parameters),
        rss=best_rss,
        r2=1.0 - best_rss / tss if tss > 0 else np.nan,
        n=len(fit_data),
    )


def _fit_juvenile_mortality_to_survival(
    data: pd.DataFrame,
    *,
    function_name: str,
    function,
    p0: tuple[float, ...],
    bounds: tuple[list[float], list[float]],
    stage_fits: dict[str, FitResult],
    stage_counts: dict[str, int],
) -> FitResult:
    if "survival_probability" not in data.columns:
        return _fit(
            data,
            name="Juvenile mortality rate",
            function_name=function_name,
            function=function,
            p0=p0,
            bounds=bounds,
        )

    fit_data = data.loc[
        data["survival_probability"].notna() & (data["survival_probability"] >= 0)
    ].copy()
    temperatures = fit_data["temperature"].to_numpy(dtype=float)
    observed = fit_data["survival_probability"].to_numpy(dtype=float)

    def survival_model(temperature: np.ndarray | float, *params: float) -> np.ndarray:
        return juvenile_survival_from_mortality_response(
            temperature,
            function,
            params,
            stage_fits=stage_fits,
            stage_counts=stage_counts,
        )

    if {"n", "adult_count"}.issubset(fit_data.columns):
        trials = np.maximum(fit_data["n"].to_numpy(dtype=float), 1.0)
        successes = np.clip(
            fit_data["adult_count"].to_numpy(dtype=float),
            0.0,
            trials,
        )

        def objective(params: np.ndarray) -> float:
            predicted = np.clip(
                survival_model(temperatures, *params), 1e-12, 1.0 - 1e-12
            )
            return float(
                -np.sum(
                    successes * np.log(predicted)
                    + (trials - successes) * np.log1p(-predicted)
                )
            )

        result = minimize(
            objective,
            np.asarray(p0, dtype=float),
            bounds=list(zip(bounds[0], bounds[1])),
            method="L-BFGS-B",
            options={"maxiter": 100_000},
        )
        params = result.x
    else:
        params, _ = curve_fit(
            survival_model,
            temperatures,
            observed,
            p0=p0,
            bounds=bounds,
            maxfev=100_000,
        )
    fitted = survival_model(temperatures, *params)
    rss = float(np.sum((observed - fitted) ** 2))
    tss = float(np.sum((observed - observed.mean()) ** 2))
    r2 = float(1 - rss / tss) if tss > 0 else np.nan
    return FitResult(
        name="Juvenile mortality rate",
        function=function_name,
        parameters=tuple(float(value) for value in params),
        rss=rss,
        r2=r2,
        n=len(fit_data),
    )


def juvenile_survival_from_mortality_response(
    temperature: np.ndarray | float,
    mortality_function,
    mortality_params: tuple[float, ...] | np.ndarray,
    *,
    stage_fits: dict[str, FitResult],
    stage_counts: dict[str, int],
) -> np.ndarray:
    temperatures = np.asarray(temperature, dtype=float)
    survival = np.ones_like(temperatures, dtype=float)
    mortality = np.asarray(
        mortality_function(temperatures, *mortality_params), dtype=float
    )
    for stage in STAGES:
        stage_key = STAGE_COUNT_KEYS[stage]
        count = int(stage_counts[stage_key])
        stage_delay = np.asarray(predict(stage_fits[stage], temperatures), dtype=float)
        stage_rate = competing_risk_transition_rate(count, stage_delay, mortality)
        survival *= (stage_rate / (stage_rate + mortality)) ** count
    return survival


def _fit_skew_peak(data: pd.DataFrame, *, name: str) -> FitResult:
    peak = data.loc[data["value"].idxmax()]
    return _fit(
        data,
        name=name,
        function_name="skew_gauss_peak",
        function=skew_gauss_peak,
        p0=(float(peak["value"]), float(peak["temperature"]), 10.0, 0.0),
        bounds=([0.0, -100.0, 1e-6, -20.0], [np.inf, 100.0, np.inf, 20.0]),
    )


def _normalized_positive_weights(logits: np.ndarray, total: int) -> np.ndarray:
    centered = logits - np.max(logits)
    raw = np.exp(centered)
    return total * raw / raw.sum()


def _log_weight_roughness(weights: np.ndarray) -> float:
    log_weights = np.log(np.maximum(weights, 1e-12))
    return float(np.sum(np.diff(log_weights) ** 2))


def _stage_initial_values(data: pd.DataFrame) -> tuple[float, float, float]:
    trough = data.loc[data["value"].idxmin()]
    return float(trough["value"]), float(trough["temperature"]), 10.0
