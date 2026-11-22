# Current CCP--MMT versus Native LMMT Protocol

The current comparison uses results/lee_mmt_stability_20261107.
It tests six fixed targets: (n,K,t,k0)=(64,12,4,3), (64,12,4,6),
(64,12,8,9), (96,18,6,5), (96,18,6,9), and (96,18,12,14).

## Training and Testing

Each length has eight fresh training matrices and 32 additional test matrices.
Matrices satisfy d_H(D1)>16 at n=64 and d_H(D1)>24 at n=96.
Their hashes are checked against earlier panels and against one another.

For each target, the shared parameter shortlist is the union of the best
two previous training candidates per method and policy, plus the all-zero
candidate. Earlier test timings do not determine this shortlist.
The retained earlier training.jsonl and tuning.json verify its provenance.

Replicated training uses two errors and two search seeds per composition.
Both methods independently choose parameters by minimizing failed calls,
then the exact-sphere-weighted penalized mean online time. Parameters are
frozen before testing. The full-family policy permits zero base weights;
the positive-base policy restricts training selection to positive weights.

Test allocations per composition are four errors and two seeds for a=0,
two errors and two seeds for a=1, and one error and two seeds for a>=2.
All compositions are included and weighted by their exact Lee-sphere mass.
The final archive contains 12,096 training records, 12,800 counted test
calls, and 38,400 separate timed calls. All test calls complete correctly.

## Search and Measurement

The common resource limits are 15 seconds, 16,384 attempted selections per
search stage, 200,000 list entries, and two million candidate checks.
The two binary CCP stages use separate random streams.
Public-input parameter adjustments and every-fourth-attempt zero-base
trials apply under both training-selection policies. Rejected pivot
selections count toward that schedule and the resource limits.

Timing starts with an empty coordinate-word cache. Online time includes
list construction, failed trials, lookups, substitution, and verification.
Code-dependent preprocessing is recorded separately. Each input has one
instrumented decode and three uninstrumented timings in alternating order.

The analysis resamples ambient-matrix clusters within each length.
Simultaneous approximate 95% intervals cover all six targets and both
selection policies. Unresolved orderings are retained.
Prior-reversal diagnostics describe previous training measurements;
they do not replace the held-out comparison.

## Commands

Run from the package root:

```sh
python src/analyze_lee_mmt_stability.py
python src/analyze_lee_mmt_search_modes.py
```

The first command checks source/data hashes, distance certificates,
train/test separation, parameter selection, and representative decoder
replays, then regenerates timing tables and intervals.

The second reconstructs successful coordinate selections from the stored
seeds, rank tests, and public schedule. It checks archived phase counters
without collecting new timings. Its outputs distinguish positive accepted
free weight, cancellation to zero despite nonzero base weights, explicit
zero-base trials, and direct recovery. These frequencies describe the
scheduled implementations; they do not isolate MMT's incremental benefit
over Prange.
