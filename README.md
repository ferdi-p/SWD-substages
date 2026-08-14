# Stage-chain population models for *Drosophila suzukii*

This repository contains the processed data and Python analysis supporting the
study “How much stage structure is needed to model population dynamics? A case
study of an invasive fruit fly.” It compares three nested,
temperature-dependent population models for *Drosophila suzukii*:

- **M1:** one state for each juvenile stage and one adult state;
- **M2:** juvenile stage chains and one adult state;
- **M3:** the M2 juvenile chains plus an adult exit chain with
  age-dependent fecundity.

The manuscript itself is intentionally not included in this code-and-data
repository.

## Reproduce the analysis

Python 3.12 or newer is required. The exact tested environment is recorded in
`requirements-lock.txt`.

```bash
make setup
make verify
```

`make verify` runs the tests, rebuilds the direct demographic calculations,
model comparisons, seasonal simulations, intervention analysis, tables, and
figures, and checks the generated products for internal consistency. Rebuilt
products are written under `outputs/`.

For subsequent runs:

```bash
make model       # rebuild all analyses, tables, and figures
make test        # run the automated tests only
make verify      # run tests and verify the complete workflow
make help        # list the available commands
```

## Repository contents

- `src/r_r0_pop/`: scientific implementation;
- `scripts/`: reproducible analysis entry points;
- `data/processed/baser/`: version-controlled analysis inputs;
- `tests/`: automated tests.

The analysis reconstructs direct estimates of net reproductive rate (`R0`),
intrinsic growth rate (`r`), and generation time; fits M1–M3; simulates seasonal
population dynamics; and evaluates juvenile- and adult-targeted mortality
scenarios. All analysis entry points accept `--help`.

## Data

The processed inputs are direct transformations of the life-table and
fertility workbooks from:

> Baser et al. (2025), “Thermal Development, Mortality, and Fertility of an
> Apulian Strain of *Drosophila suzukii* at Different Temperatures,” *Insects*
> 16:60, DOI: 10.3390/insects16010060.

The original Excel workbooks are not redistributed and are not required to run
the analysis. Full provenance and reconstruction instructions are provided in
`data/processed/baser/README.md`.

## License

Original software in this repository is released under the MIT License. The
retained upstream Baser et al. metadata and license remain under their original
terms.
