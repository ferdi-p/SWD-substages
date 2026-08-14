"""Repository paths shared by the paper analysis entry points.

Keeping these locations in one module makes the scripts independent of the
caller's working directory while leaving every path overridable on the command
line.
"""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DATA_DIR = (
    REPOSITORY_ROOT / "Baser et al 2025" / "DSuzukiiLifeTables-main"
)
PROCESSED_DATA_DIR = REPOSITORY_ROOT / "data" / "processed" / "baser"

OUTPUT_DIR = REPOSITORY_ROOT / "outputs"
REPORT_DIR = REPOSITORY_ROOT / "reports"

MANUSCRIPT_DIR = REPOSITORY_ROOT / "manuscript"
MANUSCRIPT_FIGURE_DIR = MANUSCRIPT_DIR / "figures"
SUPPLEMENTARY_FIGURE_DIR = MANUSCRIPT_DIR / "supplementary_figures"
