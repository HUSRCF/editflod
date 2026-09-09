# Frozen mechanism benchmark v0.1

The first unified table is assembled from method reports only after checking
that every method in a split has the same split-level manifest fingerprint and
ordered pair IDs. The machine-readable artifacts are:

- `/tmp/ospedit-real-sample/frozen_benchmark_v1.json`
- `/tmp/ospedit-real-sample/frozen_benchmark_v1.csv`

The set contains one multi-repeat-identifiable pair per split: H63T, T49V, and
D31A. It is a mechanism table, not a benchmark-scale estimate.

Subsequent experimental-context auditing found that this S/B-only set is not a
valid primary benchmark. H63T changes the target-chain hetero context
(`GOL/HEM/OH/SO4` versus `AZI/HEM/SO4`), and D31A is an antibody-antigen
complex with three protein chains. Only T49V satisfies the strict first-version
single-chain and matching-hetero gates. The table below is retained as
provenance and mechanism diagnostics; its three-row aggregate must not be used
as biological performance evidence. The machine-readable context audit is
`/tmp/ospedit-real-sample/pairs2022_identifiable_multirepeat_context_audit.json`.

| split | method | local error | site error | distance error | distance cosine | predicted / true norm | remote frame drift |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | copy | 0.1159 | 0.0933 | 0.3431 | undefined | 0 / 52.3 | 0 |
| train | FoldFlow C2 | 0.1168 | 0.0980 | 0.3436 | -0.047 | 1.35 / 52.3 | 0.0051 |
| train | PreMut | 0.1167 | 0.0948 | 0.3430 | 0.028 | 2.53 / 52.3 | 0.0295 |
| train | ESMFold default | 0.2736 | 0.2469 | 0.5801 | 0.213 | 83.4 / 52.3 | 0.6009 |
| train | ESMFold zero extra | 0.3155 | 0.3021 | 0.6254 | 0.173 | 89.3 / 52.3 | 0.6328 |
| dev | copy | 0.3572 | 0.1749 | 0.3589 | undefined | 0 / 44.7 | 0 |
| dev | FoldFlow C2 | 0.3556 | 0.1471 | 0.3584 | 0.065 | 1.01 / 44.7 | 0.0029 |
| dev | PreMut | 0.3600 | 0.1752 | 0.3605 | -0.066 | 2.27 / 44.7 | 0.0273 |
| dev | ESMFold default | 0.3999 | 0.3192 | 0.7129 | 0.152 | 83.8 / 44.7 | 0.6353 |
| dev | ESMFold zero extra | 0.9279 | 0.5012 | 0.8603 | 0.049 | 99.6 / 44.7 | 0.6291 |
| test | copy | 0.7179 | 0.7920 | 0.3617 | undefined | 0 / 41.0 | 0 |
| test | FoldFlow C2 | 0.7188 | 0.7965 | 0.3623 | -0.085 | 0.79 / 41.0 | 0.0041 |
| test | PreMut | 0.7176 | 0.7823 | 0.3655 | -0.098 | 3.21 / 41.0 | 0.0285 |
| test | ESMFold default | 1.0769 | 1.0200 | 0.8753 | 0.082 | 93.9 / 41.0 | 0.4391 |
| test | ESMFold zero extra | 1.0246 | 1.0546 | 0.8469 | 0.165 | 93.9 / 41.0 | 0.5415 |

Errors and drift are in Angstrom. FoldFlow C2 improves the single dev pair but
worsens train and replacement test. Raw PreMut slightly improves test local
and site error but worsens its distance response. ESMFold has a nonzero
positive distance direction, but its response magnitude and scaffold drift are
too large. No current method satisfies accurate response, low unrelated drift,
and low cost together.

## Cost status

A resident-model accounting table is now available at:

- `/tmp/ospedit-real-sample/resident_runtime_v1.json`
- `/tmp/ospedit-real-sample/resident_runtime_v1.csv`

It separates one-time model setup from the measured incremental cost of the
three H63T/T49V/D31A candidates. Cache retrieval is excluded; PreMut and
ESMFold generation costs are replayed from their persistent-process reports.

| method | device | setup (s) | warm candidate mean (s) | observed resident total, N=3 (s) | linear resident estimate, N=32 (s) | calls / candidate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FoldFlow C2 | CPU, FoldFlow environment | 7.215 | 1.532 | 11.809 | 56.223 | 2 network, 2 sequence |
| PreMut raw | CPU, dock environment | 2.204 | 1.676 | 7.232 | 55.834 | 1 network, 0 sequence |
| ESMFold default | AMD Radeon AI PRO R9700 | 25.599 | 2.169 | 32.105 | 95.001 | 1 network, 1 sequence |
| ESMFold zero extra | AMD Radeon AI PRO R9700 | 25.599 | 0.352 | 26.654 | 36.861 | 1 network, 1 sequence |

These numbers establish a reproducible accounting contract, not a speed
ranking. FoldFlow and PreMut ran on CPU while ESMFold ran on GPU. The N=32
column is a linear extrapolation from three different parent/mutation pairs;
it is not a measured same-parent cache workload. That workload, on matched
hardware, remains required before a cross-method acceleration claim.

A second table measures 15 real single substitutions of the same 164-residue
parent `2LZM_A`:

- `/tmp/ospedit-real-sample/runtime_2LZM_resident_v1.json`
- `/tmp/ospedit-real-sample/runtime_2LZM_resident_v1.csv`

| method | device | setup (s) | warm candidate mean (s) | observed resident total, N=15 (s) |
| --- | --- | ---: | ---: | ---: |
| FoldFlow C2 | CPU, FoldFlow environment | 7.140 | 2.439 | 43.731 |
| PreMut raw, parent parse cached | CPU, dock environment | 2.183 | 1.892 | 30.556 |
| ESMFold default | AMD Radeon AI PRO R9700 | 25.650 | 1.825 | 53.027 |
| ESMFold zero extra | AMD Radeon AI PRO R9700 | 25.650 | 0.448 | 32.376 |

This resolves the absence of a measured same-parent workload up to N=15, but
not the preregistered N=32/128 workloads or matched-hardware comparison. It
also shows why the three-short-protein linear projections should not be used
as throughput estimates: per-candidate cost changes with sequence length.

The PreMut adapter optimization is numerically conservative: all 15 optimized
predictions and reconstructed input backbones were elementwise identical to
the uncached runner. Caching the filtered parent ATOM table reduced its N=15
total from `32.775 s` to `30.556 s`; the published path's second randomized
coordinate construction for KNN edges was deliberately retained.

PreMut's measured warm cost is dominated by preprocessing: `4.939 s` total
for three candidates, compared with `0.087 s` of network inference. Reusing or
incrementally updating its parent graph is therefore the relevant optimization
target, rather than only compressing its GNN forward.

## Reassembly

Use explicit entries so duplicate copy-parent methods in source reports cannot
be selected implicitly:

```bash
python scripts/assemble_benchmark_table.py \
  --entry copy_parent=results/mechanism_test.json::C0_copy_parent \
  --entry foldflow_c2=results/mechanism_test.json::C2_single_noise_local_frame_difference \
  --entry premut_raw=results/premut_test.json::B1_premut_raw \
  --entry esmfold_default=results/esmfold_test.json::B2_esmfold_default \
  --output results/benchmark.json \
  --csv-output results/benchmark.csv
```

Repeat entries for train and dev. The assembler rejects duplicate output
methods within a split and any fingerprint or pair-order mismatch.

Resident-runtime reports use a separate explicit assembler so shared model
loads cannot be inferred ambiguously from suite totals:

```bash
python scripts/assemble_runtime_table.py \
  --entry foldflow_c2=results/foldflow_all.json::C2_single_noise_local_frame_difference::metadata.endpoint_setup_seconds::cpu \
  --entry 'premut_raw=results/premut_all.json::B1_premut_raw::results/premut_batch.json#model_load_seconds::cpu' \
  --entry 'esmfold_default=results/esmfold_all.json::B2_esmfold_default::results/esmfold_batch.json#model_load_seconds::gpu' \
  --candidate-counts 1 3 32 128 \
  --output results/resident_runtime.json \
  --csv-output results/resident_runtime.csv
```

The runtime assembler also rejects manifest or ordered-pair mismatches and
marks cross-device ranking as disallowed in its machine-readable output. It
stores the observed candidate count and explicitly labels all N-count fields
as linear projections; only `observed_resident_total_seconds` is measured.
