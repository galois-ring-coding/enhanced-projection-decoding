# Reproducibility Details

## Main Lee Experiment

The parameter triples `(n,K,t)` are `(16,4,2)`, `(24,6,3)`, `(40,8,4)`,
and `(64,12,5)`. Eight mother matrices per triple and
`k0 in {1, floor(K/2), K-1}` give 96 codes in 32 mother-matrix clusters.
Codes are conditioned on `d_H(D1) > 2t`, checked by exact binary enumeration.
Twelve errors per code are sampled uniformly on the exact Lee sphere, giving
1,152 paired instances.

The paper-method subset contains 5,760 counted calls and 17,280 timed calls.
Methods use matched inputs, trial seeds, and arithmetic backends.

| Paper label | Raw method ID | Role |
|---|---|---|
| Proj ring | `dcc_ring` | Ordinary first projection and ring recovery |
| Proj binary | `dcc_binary` | Ordinary first projection and support-aware binary recovery |
| PI unforced | `sc_plain` | Prefix-informed supercode search without forcing known support into recovery sets |
| Enhanced Proj | `enhanced` | Complete initial constraints and support-aware binary recovery |
| PI supported | `sc_support` | Matched prefix-informed supercode construction with forced support |

Original raw runs also include `sc_all` and `enhanced_all`. The raw files and
original scientific source are preserved unchanged; `reproduce.py` selects the
five methods in the current manuscript when reporting the table. The full runner
still executes seven methods, so its raw totals exceed the paper's totals.

Trials count sampled sets, including singular rejected sets. Weight tests count
vectors reaching a search or recovery weight check, not all sampled sets or
distinct errors. Zero-syndrome and zero-budget shortcuts contribute no random
trials, but their arithmetic is counted. Enhanced Proj and PI supported have
matched search counters; constructing targets and residuals through different
computations can give different arithmetic proxies.

At precision `2^lambda`, additions, multiplications, and inversions have proxy
weights `lambda`, `lambda^2`, and `lambda^3`. The reported proxy ratio is the
ratio of method means, not the mean of per-instance ratios. Online time is the
mean of per-instance medians over three runs. Code generation, distance
certification, and code-dependent preprocessing are excluded. Method-specific
preparation is recorded separately; theoretical total work includes
preprocessing once. New runs should be interpreted using the same boundaries.

The implementation caps randomized searches at 10,000 trials and raises an error
on exhaustion. The theoretical decoder uses unbounded independent trials.

`python reproduce.py statistics` resamples the 32 mother-matrix clusters with
replacement 2,000 times using seed 711944. For each metric it computes the ratio
of mean Proj ring cost to mean PI supported cost; sorted bootstrap values at
zero-based indices 50 and 1950 give the reported percentile endpoints. All
clusters have the same number of instances. The arithmetic-proxy interval is
[2.518, 3.314] and the online-time interval is [1.693, 2.127]. The command also
reports codes on which PI supported has larger mean cost than Proj ring.

## Source and Data Provenance

`src/lee_revision.py` is the main benchmark. Its SHA-256 recorded in the original
summary is `56f0acb8a66e45f407c6abff2d1c8de6d6bed90c4eebf252a1c4f352ca8c5a20`.
`reference/sha256.json` additionally records the packaged files' hashes; these
are packaging-time integrity checks, not a claim that every dependency was
hashed during the historical benchmark. Git attributes preserve source and raw
data bytes to avoid newline-induced hash changes.

`reference/lee/codes.json` stores generated matrices and distance certificates;
`instances.jsonl` stores inputs, seeds, counters, and timings; `summary.json`
stores environment information and aggregates. Use `audit --all` to recompute
distances and replay every paper-method call. Historical times are checked for
aggregation consistency rather than reproduced bit for bit.

## Other Checks and Models

`z4_separation.py` constructs EP constraints directly from torsion equations and
compares them with the supercode construction. The supported `equivalence`
command checks 672 paired cases over slack values 0, 1, and 2. Slack is the
number of selected coordinates beyond binary check rank; it is used in this
equivalence test, not in the main standard-Prange benchmark.

The `models` command evaluates the current Lee mean-composition example at
`(kappa0,kappa,W)=(0.1,0.3,0.08)` and the specified rank support-trapping example.
The Lee calculation evaluates fixed-composition exponents at limiting mean
proportions; it does not average decoding work over the whole sphere.
Legacy source and historical check files include additional model/certificate
fields from earlier analyses. Those fields are not used to regenerate the
current manuscript's exponents.

`ring_constraints.py` checks integer residue-ring systems (including Z8 and Z9),
not arbitrary Galois extensions. The older `Code` helper uses systematic mother
generators and row valuations. The main Lee runner separately supports the
random ambient transformations used in the paper. `rank_schur_lift.py` supplies
small checks over GR(2^r,m); these checks validate constructions and are not a
large-scale experimental validation of the asymptotic rank theorem. Hamming
and rank complexity results in the current manuscript are theoretical.

The supported wrapper avoids running unrelated historical script entry points.
