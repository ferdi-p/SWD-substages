from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from r_r0_pop.paths import FIGURE_DIR, OUTPUT_DIR
from r_r0_pop.plotting import clean_axis, save_figure


M3_COLOR = "#c44536"


def main() -> None:
    profile_path = (
        OUTPUT_DIR / "model_complexity" / "adult_exit_chain_fecundity_profile.csv"
    )
    legacy_weights_path = (
        OUTPUT_DIR / "model_complexity" / "adult_exit_chain_fecundity_weights.csv"
    )
    output = FIGURE_DIR / "m3_adult_fecundity_profile.pdf"
    profile = pd.read_csv(
        profile_path if profile_path.exists() else legacy_weights_path
    )
    profile_column = (
        "fecundity_profile"
        if "fecundity_profile" in profile.columns
        else "reproduction_weight"
    )

    fig, ax = plt.subplots(figsize=(3.4, 2.65))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.26, top=0.96)
    ax.plot(
        profile["adult_substage"],
        profile[profile_column],
        color=M3_COLOR,
        linewidth=2.4,
        marker="o",
        markersize=4.5,
        markerfacecolor=M3_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.7,
    )
    ax.set_xlabel("Adult substage, $i$")
    ax.set_ylabel("Relative fecundity weight")
    ax.set_xticks(profile["adult_substage"])
    ax.set_xlim(0.5, float(profile["adult_substage"].max()) + 0.5)
    ax.set_ylim(bottom=0.0)
    clean_axis(ax)
    save_figure(fig, output)


if __name__ == "__main__":
    main()
