# Processed Baser et al. data

The three CSV files in this directory are the version-controlled inputs used by
the analysis. They are direct transformations of the life-table and fertility
workbooks published with:

> Baser, N., Rossini, L., Anfora, G., Temel, K. M., Gualano, S., Garone, E.,
> and Santoro, F. (2025). Thermal Development, Mortality, and Fertility of an
> Apulian Strain of *Drosophila suzukii* at Different Temperatures. *Insects*,
> 16, 60. https://doi.org/10.3390/insects16010060

The article identifies the raw dataset and scripts as publicly available at
https://github.com/lucaros1190/DSuzukiiLifeTables. The upstream README and
GPL-3.0 license are retained under
`Baser et al 2025/DSuzukiiLifeTables-main/`.

The original Excel workbooks are not redistributed here and are unnecessary
for a normal analysis run. To recreate the CSV files, place
`LifeTablesDataset.xlsx` and `Fertility.xlsx` under
`Baser et al 2025/DSuzukiiLifeTables-main/` and run:

```bash
make preprocess
```
