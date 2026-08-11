from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from r_r0_pop.data import BaserPaths, write_baser_processed_data
from r_r0_pop.paths import PROCESSED_DATA_DIR, SOURCE_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Baser et al. Excel workbooks into tidy CSV files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SOURCE_DATA_DIR,
        help="Directory containing Baser workbook files.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory for processed tidy CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings(
        "ignore",
        message="Conditional Formatting extension is not supported and will be removed",
        category=UserWarning,
        module="openpyxl",
    )

    paths = write_baser_processed_data(
        BaserPaths.from_data_dir(args.data_dir), args.processed_dir
    )
    print(f"Wrote {paths.development}")
    print(f"Wrote {paths.adult_survival}")
    print(f"Wrote {paths.fertility}")


if __name__ == "__main__":
    main()
