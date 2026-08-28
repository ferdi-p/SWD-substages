import math

import pandas as pd
import pytest

from r_r0_pop.data import (
    BASER_POOLED_ADULT_FEMALE_FRACTION,
    BASER_POOLED_ADULT_FEMALE_FRACTION_UNROUNDED,
    female_preadult_distribution,
    female_preadult_summary,
    pooled_adult_female_fraction,
)


def test_pooled_adult_female_fraction_pools_counts_over_temperature():
    adults = pd.DataFrame(
        {
            "temperature": [13.0, 13.0, 18.0, 18.0, 18.0, 31.0],
            "AF": [12.0, 0.0, 10.0, 9.0, 0.0, 0.0],
            "AM": [0.0, 8.0, 0.0, 0.0, 7.0, 0.0],
        }
    )

    assert pooled_adult_female_fraction(adults) == 3 / 5


def test_baser_pooled_adult_female_fraction_matches_observed_counts():
    assert math.isclose(BASER_POOLED_ADULT_FEMALE_FRACTION_UNROUNDED, 152 / 257)
    assert BASER_POOLED_ADULT_FEMALE_FRACTION == 152 / 257


def test_pooled_adult_female_fraction_rejects_ambiguous_sex_record():
    adults = pd.DataFrame({"AF": [10.0], "AM": [8.0]})

    with pytest.raises(ValueError, match="both female and male"):
        pooled_adult_female_fraction(adults)


def test_female_preadult_summary_conditions_survival_on_initial_female():
    development = pd.DataFrame(
        {
            "temperature": [20.0, 20.0, 20.0],
            "specimen": [1, 2, 3],
            "E": [1.0, 1.0, 1.0],
            "L1": [1.0, 1.0, 1.0],
            "L2": [1.0, 1.0, 1.0],
            "L3": [1.0, 1.0, 1.0],
            "P": [1.0, 1.0, 1.0],
        }
    )
    adults = pd.DataFrame(
        {
            "temperature": [20.0, 20.0, 20.0],
            "specimen": [1, 2, 3],
            "AF": [5.0, 0.0, 0.0],
            "AM": [0.0, 5.0, 5.0],
        }
    )

    summary = female_preadult_summary(development, adults, cohort_size=10).iloc[0]

    assert summary["female_adults"] == 1
    assert summary["adult_count"] == 3
    assert summary["p_survive_to_adult"] == 0.3
    assert summary["mean_female_preadult_days"] == pytest.approx(5.0)


def test_female_preadult_distribution_retains_individual_timings():
    development = pd.DataFrame(
        {
            "temperature": [20.0, 20.0, 20.0],
            "specimen": [1, 2, 3],
            "E": [1.0, 2.0, 1.0],
            "L1": [1.0, 1.0, 1.0],
            "L2": [1.0, 1.0, 1.0],
            "L3": [1.0, 1.0, 1.0],
            "P": [1.0, 2.0, 1.0],
        }
    )
    adults = pd.DataFrame(
        {
            "temperature": [20.0, 20.0, 20.0],
            "specimen": [1, 2, 3],
            "AF": [5.0, 5.0, 0.0],
            "AM": [0.0, 0.0, 5.0],
        }
    )

    distribution = female_preadult_distribution(development, adults, cohort_size=10)

    assert distribution["preadult_days"].tolist() == pytest.approx(
        [5.0, 7.0]
    )
    assert distribution["preadult_weight"].tolist() == [0.5, 0.5]
    assert distribution["p_survive_to_adult"].tolist() == [0.3, 0.3]
