import math

import pandas as pd
import pytest

from r_r0_pop.precision import round_parameter, round_parameter_table


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (27.260026, 27.3),
        (0.050353617, 0.0504),
        (125.567875, 126.0),
        (-4.5329367, -4.53),
        (0.0, 0.0),
    ],
)
def test_round_parameter_uses_three_significant_figures(value, expected):
    assert round_parameter(value) == expected


def test_round_parameter_preserves_nonfinite_values():
    assert math.isnan(round_parameter(float("nan")))
    assert round_parameter(float("inf")) == float("inf")


def test_round_parameter_table_only_changes_coefficient_columns():
    table = pd.DataFrame(
        {
            "name": ["example"],
            "c1": [27.260026],
            "c2": [0.050353617],
            "rss": [0.123456789],
        }
    )

    rounded = round_parameter_table(table)

    assert rounded.loc[0, "c1"] == 27.3
    assert rounded.loc[0, "c2"] == 0.0504
    assert rounded.loc[0, "rss"] == table.loc[0, "rss"]
