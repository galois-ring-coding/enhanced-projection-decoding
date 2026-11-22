# Complete-Constraint Projection over Galois Rings

Reproduction package for **Complete-Constraint Projection over Galois Rings:
Lee Decoding over Z4**, by Zhihao Li, Qunxiong Zheng, and Zhongxiao Wang.

This release contains the current manuscript's component comparison,
standard-rejection Prange scale comparison, and CCP--MMT versus native LMMT
comparison. 
## Install

Use Python 3.10 or later. From this directory:

```sh
python -m pip install -r requirements.txt
python -m pip install -r requirements-plots.txt
```

## Reproduce the Manuscript Results

| Result | Command |
| --- | --- |
| Component comparison | `python reproduce.py table` |
| Prange scale comparison | `python src/analyze_lee_rejection.py` |
| Scale figure | `python plot_lee_comparison.py` |
| CCP--MMT versus LMMT | `python src/analyze_lee_mmt_stability.py` |
| MMT parameters and successful-selection checks | `python src/analyze_lee_mmt_search_modes.py` |

These commands analyze recorded measurements; they do not collect new timings.
Some also replay recorded decodes to verify correctness. The plotting command
writes `figures/lee_scale_ccp.pdf` and `.png`.

Additional checks:

```sh
python reproduce.py manifest
python reproduce.py tests
python reproduce.py audit --all
python reproduce.py equivalence
python reproduce.py models
```

The table command uses the original method identifiers: `Enhanced Proj`
means CCP, `PI unforced` means Tower unforced, and `PI supported` means
Tower supported. The model command also retains a legacy rank calculation;
only its Lee estimates are used in the current manuscript.

## What Is Included

| Path | Contents |
| --- | --- |
| `reference/lee/` | Component-comparison matrices, inputs, counters, and timings |
| `results/lee_scale_rejection_20261119/` | Current Prange scale records and analyses |
| `results/lee_mmt_stability_20261107/` | Current MMT training, test records, and analyses |
| Three other `results/` directories | Only the five provenance files listed below |
| `src/` | Experiment entry points and their shared dependencies |
| `tests/` | Tests of the retained implementation and analyses |
| `figures/` | Regenerated manuscript scale figure |
| `docs/` | Measurement protocol and publication instructions |

The MMT audit needs these earlier records:

- `results/lee_sweep_20261103/subtypes_main/codes.json`
- `results/lee_list_confirmation_20261105/mothers.json`
- `results/lee_list_retuned_20261105/mothers.json`
- `results/lee_list_retuned_20261105/training.jsonl`
- `results/lee_list_retuned_20261105/tuning.json`

The matrix files verify that the final training and test panels are new.
The training and tuning files verify the frozen parameter shortlist and
its provenance. Earlier test measurements are not required and are omitted.

Some dependency modules retain historical names and unused command-line
entry points. They remain because the current scripts import their helpers
and verify their frozen source hashes. Only the commands documented here
are supported by this trimmed data release.

See [measurement details](docs/REPRODUCIBILITY.md),
[MMT protocol](docs/MMT_PROTOCOL.md), and
[publication instructions](docs/PUBLICATION.md).
`reference/sha256.json` checks the source/reference package;
`RELEASE_MANIFEST.json` additionally hashes all included data and figures.
Generated `outputs/` and Python caches are not distributed.

## Fresh Measurements

Use a new output directory to preserve the archived observations:

```sh
python reproduce.py benchmark --output outputs/fresh_component
python src/run_lee_scale_confirmation.py --output outputs/fresh_scale
python src/run_lee_mmt_stability.py --output outputs/fresh_mmt
```

Wall times depend on the machine and system load. These commands can take
substantially longer than analyzing the recorded results.
