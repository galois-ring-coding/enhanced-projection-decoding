# Packaging Validation

The following checks passed on 2026-10-16 using Python 3.13.5 and NumPy 2.1.3
on Windows. These are validations of the packaged artifact, not replacement
measurements for the paper's historical timings.

| Check | Outcome |
|---|---|
| Regression suite | 16 tests passed |
| Full recorded-data audit | 96 exact distance certificates checked; 5,760 calls replayed across 1,152 paired instances; errors and counters matched |
| Independent construction equivalence | 672 paired cases and 4,511 search events matched |
| Fresh smoke benchmark | Three codes, six paired instances, 42 counted calls; all decoders correct |
| Rank validation | 40 complete small decoding/lifting cases passed |
| Exhaustive theory checks | 320 affine-fiber, 56 Lee-completion, and 24 rank-prefix checks passed |
| Numerical examples | Lee exponents 0.01217379218, 0.04146469452, 0.00090808107; rank distance margin 0.34 |
| Original benchmark source hash | Matches the hash stored in the historical summary |

The regenerated five-method table agrees with the paper at the displayed
precision. The full timed benchmark was not rerun during packaging; the full
audit replayed the counted calls and checked historical timing aggregation.
