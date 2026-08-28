from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

from r_r0_pop.paths import MANUSCRIPT_FIGURE_DIR, OUTPUT_DIR
from r_r0_pop.plotting import clean_axis, save_figure


M3_COLOR = "#c44536"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the fitted M3 adult-substage fecundity profile."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MANUSCRIPT_FIGURE_DIR / "m3_adult_fecundity_profile.pdf",
        help=(
            "Output figure path. The model-only workflow redirects this to "
            "outputs/."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile_path = (
        OUTPUT_DIR / "model_complexity" / "adult_exit_chain_fecundity_profile.csv"
    )
    profile = pd.read_csv(profile_path)

    fig, ax = plt.subplots(figsize=(3.4, 2.65))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.26, top=0.96)
    ax.plot(
        profile["adult_substage"],
        profile["fecundity_profile"],
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
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.set_xlim(0.5, float(profile["adult_substage"].max()) + 0.5)
    ax.set_ylim(bottom=0.0)
    clean_axis(ax)
    save_figure(fig, args.output)


if __name__ == "__main__":
    main()
