# Enhanced Projection Decoding over Galois Rings

Experimental artifact for **Enhanced Projection Decoding over Galois Rings:
Complete Constraints and Prefix Information**, by Zhihao Li, Qunxiong Zheng,
and Zhongxiao Wang.

This repository contains the code and reference data for the manuscript's
finite-length Lee decoding comparison over Z4, the independent search-equivalence
check, and supplementary checks of complete constraints and rank lifting.
The supported entry point is `reproduce.py`; no external datasets are needed.

## Installation

The reference benchmark used Python 3.13.5, NumPy 2.1.3, and Windows 11.
From the repository root, create a virtual environment and install the dependency:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on Linux/macOS. Then run:

```sh
python -m pip install -r requirements.txt
```

## Quick Reproduction

```sh
python reproduce.py manifest
python reproduce.py tests
python reproduce.py table
python reproduce.py models
python reproduce.py statistics
```

These commands verify the archived source/data hashes, run 16 tests, regenerate
the paper's five-method table from recorded measurements, and evaluate the
current Lee and rank exponent examples. The statistics command regenerates
the cluster bootstrap intervals and per-code adverse comparisons reported in
the paper. Generated files go to `outputs/`.
The table command reads archived timings; it does not measure new timings.

| Method | First trials | Recovery trials | Weight tests | Proxy ratio | Time (ms) |
|---|---:|---:|---:|---:|---:|
| Proj ring | 8.440 | 3.214 | 3.529 | 1.000 | 8.427 |
| Proj binary | 8.440 | 0.310 | 2.589 | 0.525 | 6.641 |
| PI unforced | 4.705 | 4.708 | 3.038 | 0.582 | 7.349 |
| Enhanced Proj | 4.705 | 0.310 | 1.668 | 0.338 | 4.210 |
| PI supported | 4.705 | 0.310 | 1.668 | 0.342 | 4.419 |

The table is exported as CSV, Markdown, and a LaTeX `tabular` fragment
(requiring `booktabs`). Proxy ratios are ratios of mean arithmetic costs to
Proj ring. Weight tests correspond to the historical `candidates` counter.

## Recompute and Validate

```sh
python reproduce.py audit --all
python reproduce.py equivalence
python reproduce.py benchmark --smoke
python reproduce.py benchmark
python reproduce.py rank
python reproduce.py theory
```

| Command | Result |
|---|---|
| `audit --all` | Rechecks all 96 distance certificates and recorded inputs; replays all 5,760 paper-method calls and compares counters. |
| `audit` | Same recorded-data checks, replaying the first error for each code and method (480 calls). |
| `equivalence` | Rebuilds independent EP and supercode systems; checks 672 paired cases and 4,511 search events. |
| `benchmark --smoke` | Generates a small fresh benchmark and timings to check the execution pipeline. |
| `benchmark` | Regenerates the full benchmark with eight mother matrices per parameter triple, 12 errors per code, and three timing repetitions. |
| `rank` | Checks 40 small complete rank decoding/lifting cases. |
| `theory` | Runs small exhaustive affine-fiber, Lee-completion, and rank-prefix checks. |

Use `--output PATH` to choose a separate output directory. For example:

```sh
python reproduce.py benchmark --output outputs/fresh
python reproduce.py table --input outputs/fresh/lee_full --output outputs/fresh/table
```

Random seeds are fixed by the scripts and stored in the raw records. New wall
times depend on the machine and system load; the archived timings came from a
shared machine. All experiments run on the CPU. See
[Reproducibility details](docs/REPRODUCIBILITY.md) for measurement conventions,
data provenance, and the scope of supplementary code.

## Repository Layout

```text
reproduce.py             Supported reproduction commands
requirements.txt         Pinned NumPy dependency
src/                     Original experiment implementations and helpers
tests/                   Regression and exhaustive small-instance tests
reference/lee/           Original code descriptions, per-instance records, summary
reference/checks/        Original supplementary validation records
reference/sha256.json    Source, test, and reference-data checksums
docs/                    Reproducibility and publication notes
outputs/                 Generated results (ignored by Git)
```

## Citation

If you use this code or the experimental data in your research, please cite:

Zhihao Li, Qunxiong Zheng, and Zhongxiao Wang.
*Enhanced Projection Decoding over Galois Rings: Complete Constraints and Prefix Information.*
