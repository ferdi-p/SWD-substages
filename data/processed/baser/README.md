# Processed Baser et al. data

The three CSV files in this directory are the version-controlled inputs used by
the analysis. They are direct transformations of the life-table and fertility
workbooks published with:

> Baser, N., Rossini, L., Anfora, G., Temel, K. M., Gualano, S., Garone, E.,
> and Santoro, F. (2025). Thermal Development, Mortality, and Fertility of an
> Apulian Strain of *Drosophila suzukii* at Different Temperatures. *Insects*,
> 16, 60. https://doi.org/10.3390/insects16010060

The article identifies the raw dataset and scripts as publicly available at
https://github.com/lucaros1190/DSuzukiiLifeTables. The `UPSTREAM_LICENSE` file
is the GPL-3.0 license distributed with that source repository.

The original Excel workbooks are not duplicated here. They are unnecessary for
a normal analysis run. To recreate the CSV files, place
`LifeTablesDataset.xlsx` and `Fertility.xlsx` under `data/raw/baser/` and run:

```bash
make preprocess
```
