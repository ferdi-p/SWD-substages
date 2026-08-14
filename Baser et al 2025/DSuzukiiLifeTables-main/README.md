# Optional Baser et al. source data

This directory is the expected location for the original workbooks:

- `LifeTablesDataset.xlsx`
- `Fertility.xlsx`

They are not redistributed in this repository. The version-controlled analysis
inputs are the tidy CSV files under `data/processed/baser/`; a fresh clone uses
those files directly.

If the original workbooks are available, place them here and run
`make preprocess` from the repository root to reconstruct the processed CSVs.

Data source:

> Baser N., Rossini L., Anfora G., Temel K., Gualano S., Garone E., Santoro F.
> (2025). Thermal Development, Mortality, and Fertility of an Apulian Strain of
> *Drosophila suzukii* at Different Temperatures. *Insects* 16:60.
> DOI: 10.3390/insects16010060.

The `LICENSE` file is the license distributed with the upstream dataset.
