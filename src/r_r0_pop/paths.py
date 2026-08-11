"""Repository paths shared by the analysis entry points.

Keeping these locations in one module makes the scripts independent of the
caller's working directory while leaving input locations overridable on the
command line.
"""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DATA_DIR = REPOSITORY_ROOT / "data" / "raw" / "baser"
PROCESSED_DATA_DIR = REPOSITORY_ROOT / "data" / "processed" / "baser"

OUTPUT_DIR = REPOSITORY_ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"
SUPPLEMENTARY_FIGURE_DIR = FIGURE_DIR / "supplementary"
