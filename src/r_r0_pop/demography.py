from __future__ import annotations

import math

import numpy as np
import pandas as pd

from r_r0_pop.data import BASER_POOLED_ADULT_FEMALE_FRACTION


def net_reproductive_rate(schedule: pd.DataFrame) -> float:
    """Calculate standard R0 = sum(lx * mx)."""

    return float((schedule["lx"] * schedule["mx"]).sum())


def euler_lotka_r(
    schedule: pd.DataFrame,
    *,
    lower: float = -2.0,
    upper: float = 2.0,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> float:
    """Solve sum(exp(-r*x) * lx * mx) = 1 by bisection."""

    ages = schedule["age_days"].to_numpy(dtype=float)
    reproduction = (schedule["lx"] * schedule["mx"]).to_numpy(dtype=float)
    if reproduction.sum() <= 0:
        return math.nan

    def f(rate: float) -> float:
        return float(np.sum(np.exp(-rate * ages) * reproduction) - 1.0)

    f_lower = f(lower)
    f_upper = f(upper)
    while f_lower < 0:
        upper = lower
        f_upper = f_lower
        lower *= 2
        f_lower = f(lower)
    while f_upper > 0:
        lower = upper
        f_lower = f_upper
        upper *= 2
        f_upper = f(upper)

    for _ in range(max_iterations):
        midpoint = (lower + upper) / 2
        f_midpoint = f(midpoint)
        if abs(f_midpoint) < tolerance or (upper - lower) / 2 < tolerance:
            return float(midpoint)
        if f_midpoint > 0:
            lower = midpoint
        else:
            upper = midpoint

    return float((lower + upper) / 2)


def generation_time_euler(schedule: pd.DataFrame, r: float | None = None) -> float:
    """Calculate Euler-Lotka mean generation time from a reproduction schedule."""

    ages = schedule["age_days"].to_numpy(dtype=float)
    reproduction = (schedule["lx"] * schedule["mx"]).to_numpy(dtype=float)
    if reproduction.sum() <= 0:
        return math.nan
    if r is None:
        r = euler_lotka_r(schedule)
    if not math.isfinite(r):
        return math.nan
    discounted = np.exp(-r * ages) * reproduction
    total = float(discounted.sum())
    if total <= 0:
        return math.nan
    return float(np.sum(ages * discounted) / total)


def build_reproduction_schedule(
    fertility: pd.DataFrame,
    female_preadult: pd.DataFrame,
    *,
    offspring_sex_ratio: float = BASER_POOLED_ADULT_FEMALE_FRACTION,
) -> pd.DataFrame:
    """Build age-specific lx and mx values from Baser individual fertility data.

    `mx` is expressed as female offspring per live female-day by multiplying egg
    counts by `offspring_sex_ratio`. `lx` is conditional on an initial female
    egg: it combines juvenile survival estimated from emergence of either sex
    with the observed adult female survival fraction in the fertility trial.
    Adult age begins at emergence. Each female's recorded preoviposition days
    are inserted as live adult days with zero fecundity, and her reproductive
    day `d` is placed at adult age `preoviposition_days + d - 1`.

    When `female_preadult` contains individual `preadult_days`, each adult-age
    contribution is convolved with that empirical duration distribution. The
    fertility and life-table cohorts contain no individual identifiers in
    common, so this independent empirical convolution uses the observed timing
    distribution without inventing pairings. A one-row temperature summary
    containing `mean_female_preadult_days` remains supported for sensitivity
    analyses and backwards compatibility.
    """

    schedules: list[pd.DataFrame] = []
    for temp, fert_temp in fertility.groupby("temperature", sort=True):
        preadult_row = female_preadult.loc[female_preadult["temperature"] == temp]
        if preadult_row.empty:
            continue
        juvenile_survival = float(preadult_row["p_survive_to_adult"].iloc[0])
        if "preadult_days" in preadult_row:
            preadult_distribution = preadult_row.loc[
                preadult_row["preadult_days"] > 0,
                ["preadult_days"]
                + (
                    ["preadult_weight"]
                    if "preadult_weight" in preadult_row
                    else []
                ),
            ].copy()
            if "preadult_weight" not in preadult_distribution:
                preadult_distribution["preadult_weight"] = (
                    1.0 / len(preadult_distribution)
                )
            weight_total = float(preadult_distribution["preadult_weight"].sum())
            if weight_total <= 0:
                continue
            preadult_distribution["preadult_weight"] /= weight_total
        else:
            preadult_distribution = pd.DataFrame(
                {
                    "preadult_days": [
                        float(
                            preadult_row["mean_female_preadult_days"].iloc[0]
                        )
                    ],
                    "preadult_weight": [1.0],
                }
            )

        female_summary = (
            fert_temp.groupby("female", as_index=False)
            .agg(
                preoviposition_days=("preoviposition_days", "first"),
                observed_egg_days=("eggs", "count"),
            )
        )
        available_females = female_summary.loc[
            female_summary["observed_egg_days"] > 0, "female"
        ]
        fert_temp = fert_temp.loc[
            fert_temp["female"].isin(available_females)
        ].copy()
        female_count = int(available_females.nunique())
        if female_count == 0:
            continue

        individual_records = []
        for female, female_data in fert_temp.groupby("female", sort=False):
            preoviposition_values = (
                female_data["preoviposition_days"].dropna().unique()
            )
            if len(preoviposition_values) != 1:
                raise ValueError(
                    f"Expected one preoviposition duration for female {female!r} "
                    f"at {temp:g} C; found {len(preoviposition_values)}."
                )
            preoviposition_days = float(preoviposition_values[0])
            if preoviposition_days < 0 or not preoviposition_days.is_integer():
                raise ValueError(
                    "Preoviposition duration must be a nonnegative whole number "
                    f"of days; found {preoviposition_days} for female {female!r} "
                    f"at {temp:g} C."
                )
            preoviposition_days = int(preoviposition_days)

            for adult_age in range(preoviposition_days):
                individual_records.append(
                    {
                        "adult_age_days": adult_age,
                        "eggs": 0.0,
                        "individual_phase": "adult_zero_fecundity",
                    }
                )

            for row in female_data.sort_values("adult_day").itertuples(index=False):
                if pd.isna(row.eggs):
                    continue
                individual_records.append(
                    {
                        "adult_age_days": preoviposition_days
                        + int(row.adult_day)
                        - 1,
                        "eggs": float(row.eggs),
                        "individual_phase": "reproductive",
                    }
                )

        aligned = pd.DataFrame(individual_records)
        records = []
        for adult_age, age_data in aligned.groupby("adult_age_days", sort=True):
            live_count = len(age_data)
            zero_fecundity_count = int(
                (age_data["individual_phase"] == "adult_zero_fecundity").sum()
            )
            reproductive_count = live_count - zero_fecundity_count
            eggs_per_live_female = float(age_data["eggs"].mean())
            adult_survival = live_count / female_count
            if zero_fecundity_count == live_count:
                schedule_phase = "adult_zero_fecundity"
            elif zero_fecundity_count == 0:
                schedule_phase = "reproductive"
            else:
                schedule_phase = "mixed"

            for preadult in preadult_distribution.itertuples(index=False):
                records.append(
                    {
                        "temperature": temp,
                        "age_days": float(preadult.preadult_days) + adult_age,
                        "adult_age_days": adult_age,
                        "preadult_days": float(preadult.preadult_days),
                        "preadult_weight": float(preadult.preadult_weight),
                        "lx": (
                            juvenile_survival
                            * adult_survival
                            * float(preadult.preadult_weight)
                        ),
                        "mx": eggs_per_live_female * offspring_sex_ratio,
                        "live_females": live_count,
                        "zero_fecundity_females": zero_fecundity_count,
                        "reproductive_females": reproductive_count,
                        "eggs_per_live_female": eggs_per_live_female,
                        "schedule_phase": schedule_phase,
                    }
                )

        if records:
            schedules.append(pd.DataFrame(records))

    if not schedules:
        return pd.DataFrame(
            columns=[
                "temperature",
                "age_days",
                "adult_age_days",
                "preadult_days",
                "preadult_weight",
                "lx",
                "mx",
                "live_females",
                "zero_fecundity_females",
                "reproductive_females",
                "eggs_per_live_female",
                "schedule_phase",
            ]
        )
    return (
        pd.concat(schedules, ignore_index=True)
        .sort_values(["temperature", "age_days", "adult_age_days"])
        .reset_index(drop=True)
    )


def summarize_rates(schedule: pd.DataFrame) -> pd.DataFrame:
    """Calculate R0 and Euler-Lotka r for each temperature."""

    rows = []
    for temp, temp_schedule in schedule.groupby("temperature", sort=True):
        reproductive = temp_schedule.loc[
            (temp_schedule["lx"] * temp_schedule["mx"]) > 0
        ]
        rows.append(
            {
                "temperature": temp,
                "R0": net_reproductive_rate(temp_schedule),
                "r_euler": euler_lotka_r(temp_schedule),
                "generation_time_euler": generation_time_euler(temp_schedule),
                "first_reproductive_age_days": (
                    reproductive["age_days"].min()
                    if not reproductive.empty
                    else math.nan
                ),
                "last_reproductive_age_days": (
                    reproductive["age_days"].max()
                    if not reproductive.empty
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)
