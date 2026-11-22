# Complete-Constraint Projection over Galois Rings

Code and data for **Complete-Constraint Projection over Galois Rings:
Lee Decoding over Z4**, by Zhihao Li, Qunxiong Zheng, and Zhongxiao Wang.

Complete-Constraint Projection (CCP) constructs complete residue-field
constraints for decoding over Galois rings. This repository provides the
Z4 Lee-decoding experiments: component comparisons, CCP--Prange versus
projection baselines, and CCP--MMT versus native LMMT.

## Installation

Requires Python 3.10 or later. Run from the repository root:

```sh
python -m pip install -r requirements.txt -r requirements-plots.txt
```

## Reproduce the Results

| Experiment | Command |
| --- | --- |
| Component comparison | `python reproduce.py table` |
| Prange comparison | `python src/analyze_lee_rejection.py` |
| Prange comparison figure | `python plot_lee_comparison.py` |
| CCP--MMT versus LMMT | `python src/analyze_lee_mmt_stability.py` |
| MMT parameters and search checks | `python src/analyze_lee_mmt_search_modes.py` |

These commands use the included experimental records. They regenerate
tables, figures, and verification reports without collecting new timings.
The figure is written to `figures/`; analysis tables are written to each
experiment's `analysis/` directory. The component table is written to
`outputs/`.

## Validation

```sh
python reproduce.py tests
python reproduce.py manifest
python reproduce.py audit --all
python reproduce.py equivalence
```

These checks cover the implementation, package checksums, recorded component
decodes, and the equivalence of the constraint constructions.

## Repository Layout

```text
src/         Decoders, experiment runners, and analysis scripts
tests/       Correctness and regression tests
reference/   Component-comparison data and package checksums
results/     Prange and MMT data, including training and provenance records
figures/     Manuscript figure
docs/        Measurement and MMT protocols
```

Large Prange records are split into numbered files and read automatically.
Keep all data directories intact: the earlier matrix and tuning records
are required by the MMT verification.

For new measurements, use separate output directories:

```sh
python reproduce.py benchmark --output outputs/component
python src/run_lee_scale_confirmation.py --output outputs/prange
python src/run_lee_mmt_stability.py --output outputs/mmt
```

New wall times depend on the machine and system load. See
[measurement details](docs/REPRODUCIBILITY.md) and
[MMT protocol](docs/MMT_PROTOCOL.md) for the experimental design.

## Citation

Please cite the accompanying manuscript when using this code or data.
