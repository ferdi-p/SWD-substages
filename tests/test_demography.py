import math

import pandas as pd

from r_r0_pop.demography import (
    build_adult_age_schedule,
    build_reproduction_schedule,
    euler_lotka_r,
    generation_time_euler,
    net_reproductive_rate,
)
from r_r0_pop.life_history_fits import daily_fecundity_summary


def test_reproduction_schedule_uses_juvenile_survival_without_initial_sex_ratio():
    fertility = pd.DataFrame(
        {
            "temperature": [20.0, 20.0],
            "female": [1, 2],
            "adult_day": [1, 1],
            "eggs": [10.0, 10.0],
            "preoviposition_days": [0.0, 0.0],
        }
    )
    preadult = pd.DataFrame(
        {
            "temperature": [20.0],
            "mean_female_preadult_days": [4.0],
            "p_survive_to_adult": [0.8],
        }
    )

    schedule = build_reproduction_schedule(
        fertility,
        preadult,
        offspring_female_fraction=0.5,
    )

    assert schedule.loc[0, "lx"] == 0.8
    assert schedule.loc[0, "mx"] == 5.0
    assert net_reproductive_rate(schedule) == 4.0


def test_reproduction_schedule_aligns_each_female_from_emergence():
    fertility = pd.DataFrame(
        {
            "temperature": [20.0, 20.0, 20.0],
            "female": [1, 2, 3],
            "adult_day": [1, 1, 1],
            "eggs": [2.0, 4.0, float("nan")],
            "preoviposition_days": [1.0, 3.0, float("nan")],
        }
    )
    preadult = pd.DataFrame(
        {
            "temperature": [20.0],
            "mean_female_preadult_days": [4.0],
            "p_survive_to_adult": [0.8],
        }
    )

    schedule = build_reproduction_schedule(
        fertility,
        preadult,
        offspring_female_fraction=0.5,
    )

    assert schedule["adult_age_days"].tolist() == [0, 1, 2, 3]
    assert schedule["age_days"].tolist() == [4.0, 5.0, 6.0, 7.0]
    assert schedule["live_females"].tolist() == [2, 2, 1, 1]
    assert schedule["zero_fecundity_females"].tolist() == [2, 1, 1, 0]
    assert schedule["reproductive_females"].tolist() == [0, 1, 0, 1]
    assert schedule["schedule_phase"].tolist() == [
        "adult_zero_fecundity",
        "mixed",
        "adult_zero_fecundity",
        "reproductive",
    ]
    assert schedule["eggs_per_live_female"].tolist() == [0.0, 1.0, 0.0, 4.0]
    assert schedule["lx"].tolist() == [0.8, 0.8, 0.4, 0.4]
    assert schedule["mx"].tolist() == [0.0, 0.5, 0.0, 2.0]
    assert math.isclose(net_reproductive_rate(schedule), 1.2)


def test_adult_age_schedule_and_daily_rate_include_preoviposition_zeros():
    fertility = pd.DataFrame(
        {
            "temperature": [20.0, 20.0, 20.0],
            "female": [1, 1, 2],
            "adult_day": [1, 2, 1],
            "eggs": [2.0, 4.0, float("nan")],
            "preoviposition_days": [2.0, 2.0, float("nan")],
        }
    )

    schedule = build_adult_age_schedule(fertility)
    summary = daily_fecundity_summary(fertility).iloc[0]

    assert schedule["adult_day"].tolist() == [1, 2, 3, 4]
    assert schedule["mean_eggs"].tolist() == [0.0, 0.0, 2.0, 4.0]
    assert schedule["live_females"].tolist() == [1, 1, 1, 1]
    assert summary["n"] == 4
    assert summary["value"] == 1.5


def test_reproduction_schedule_ignores_trailing_missing_egg_records():
    fertility = pd.DataFrame(
        {
            "temperature": [20.0, 20.0, 20.0],
            "female": [1, 1, 1],
            "adult_day": [1, 2, 3],
            "eggs": [2.0, float("nan"), float("nan")],
            "preoviposition_days": [1.0, 1.0, 1.0],
        }
    )
    preadult = pd.DataFrame(
        {
            "temperature": [20.0],
            "mean_female_preadult_days": [4.0],
            "p_survive_to_adult": [0.8],
        }
    )

    schedule = build_reproduction_schedule(
        fertility,
        preadult,
        offspring_female_fraction=0.5,
    )

    assert schedule["adult_age_days"].tolist() == [0, 1]
    assert schedule["live_females"].tolist() == [1, 1]
    assert schedule["eggs_per_live_female"].tolist() == [0.0, 2.0]
    assert math.isclose(net_reproductive_rate(schedule), 0.8)


def test_reproduction_schedule_convolves_empirical_preadult_durations():
    fertility = pd.DataFrame(
        {
            "temperature": [20.0],
            "female": [1],
            "adult_day": [1],
            "eggs": [4.0],
            "preoviposition_days": [0.0],
        }
    )
    preadult = pd.DataFrame(
        {
            "temperature": [20.0, 20.0],
            "preadult_days": [4.0, 8.0],
            "preadult_weight": [0.25, 0.75],
            "p_survive_to_adult": [0.8, 0.8],
        }
    )

    schedule = build_reproduction_schedule(
        fertility,
        preadult,
        offspring_female_fraction=0.5,
    )

    assert schedule["age_days"].tolist() == [4.0, 8.0]
    assert all(
        math.isclose(observed, expected)
        for observed, expected in zip(schedule["lx"], [0.2, 0.6])
    )
    assert math.isclose(net_reproductive_rate(schedule), 1.6)


def test_net_reproductive_rate_sums_lx_mx():
    schedule = pd.DataFrame(
        {
            "age_days": [1.0, 2.0, 3.0],
            "lx": [1.0, 0.5, 0.25],
            "mx": [0.0, 2.0, 4.0],
        }
    )

    assert net_reproductive_rate(schedule) == 2.0


def test_euler_lotka_r_for_single_reproductive_age():
    schedule = pd.DataFrame(
        {
            "age_days": [2.0],
            "lx": [1.0],
            "mx": [4.0],
        }
    )

    assert math.isclose(euler_lotka_r(schedule), math.log(4.0) / 2.0, rel_tol=1e-8)


def test_generation_time_euler_for_single_reproductive_age():
    schedule = pd.DataFrame(
        {
            "age_days": [5.0],
            "lx": [1.0],
            "mx": [3.0],
        }
    )

    assert math.isclose(generation_time_euler(schedule), 5.0)
