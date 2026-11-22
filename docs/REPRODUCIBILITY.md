# Measurement and Data Details

## Common Conventions

Online time includes syndrome-dependent construction, failed trials,
candidate processing, recovery, and final verification. Code generation,
distance certification, and code-dependent preprocessing are excluded.
Each timing observation is the median of three uninstrumented runs.
Counter collection is separate from timing. Methods use paired inputs and
search seeds. Bootstrap resampling keeps related codes, errors, seeds, and
methods together within each independent ambient-matrix cluster.

The retained raw measurements and scientific source files are unchanged.
Analysis commands regenerate summaries from those records; new measurements
are not expected to reproduce historical wall times exactly.

## Component Comparison

The four parameter triples are (16,4,2), (24,6,3), (40,8,4), and (64,12,5).
Eight ambient matrices per triple, with three subtypes per matrix, give
96 codes in 32 independent clusters. Each code has 12 uniform exact-Lee-sphere
errors: 1,152 paired instances and 5,760 counted paper-method calls.
All codes satisfy d_H(D1)>2t.

The five reported method IDs are dcc_ring, dcc_binary, sc_plain, enhanced,
and sc_support. The latter three are printed by the original wrapper as
PI unforced, Enhanced Proj, and PI supported. Raw records include two
additional historical methods; they are excluded from the paper table.

Prange selections reject singular matrices. Zero-syndrome and zero-budget
shortcuts use no randomized trials. Weight-test counts exclude final
verification and direct zero-budget recovery.
For arithmetic modulo 2^lambda, additions, multiplications, and inversions
have proxy weights lambda, lambda^2, and lambda^3.
Proxy ratios and time ratios compare method means.

`python reproduce.py audit --all` verifies 96 distance certificates and
replays the 5,760 paper-method calls, checking errors and counters.
`reference/lee/summary.json` records the environment and source hash.
`python reproduce.py equivalence` independently checks 672 paired
construction/search cases and 4,511 search events.

## Standard-Rejection Prange Scale Comparison

The triples are (48,9,3), (64,12,4), and (96,18,6).
Each length has 16 independent ambient matrices and all subtypes k0=0,...,K:
672 codes in total. Sampling uses d_H(D1)>4t; the decoding theory requires
only d_H(D1)>2t.

For every matrix and composition, two errors and four search seeds are shared
across subtypes and methods. Additional independent uniform-sphere errors
validate sampling and are excluded from the composition-weighted estimates.
There are 37,248 counted decodes and 111,744 timed calls.

The methods ep_reject and proj_binary_t share conditional binary recovery,
rank rejection, budgets, and stage-specific random streams. Only their
initial equations and selection sizes differ. The audit verifies the
recorded inputs, timing medians, rejection counts, matched recovery traces,
free-code boundary, and representative replayed calls.

Composition means use exact-Lee-sphere probabilities. The a>0 group
renormalizes these probabilities over that group. Bootstrap intervals
resample ambient matrices. Figure error bars are pointwise 95% intervals;
paired time-ratio intervals have simultaneous coverage over subtypes within
each length and error group.

Dashed trial-count curves are support-coverage lower bounds, not exact
success predictions under rank rejection. The cost exponents use the
manuscript's random-matrix heuristic, not a proved code-average cost bound.

## MMT Comparison and Integrity Checks

See [MMT protocol](MMT_PROTOCOL.md) for the final training and test design.
The five earlier files listed in README.md are retained solely for its
matrix-independence and training-provenance checks.

`reference/sha256.json` covers the retained source, tests, documentation,
and component data. `RELEASE_MANIFEST.json` covers the full released file
set, including the later experiments. Experiment configurations additionally
retain the source hashes recorded when those measurements were collected.
The source/reference manifest is regenerated when the package contents change.
