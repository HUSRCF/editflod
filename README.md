# ospedit

Reference-conditioned protein structure editing experiments.

Install the local package in editable mode:

```bash
python -m pip install -e .
ospedit-eval examples/toy_pair.json
```

Audit a JSON or JSONL pair manifest before running experiments:

```bash
ospedit-eval --audit-manifest data/manifest/pairs.jsonl
```

To also verify that the referenced source files have not changed:

```bash
ospedit-eval --audit-manifest data/manifest/pairs.jsonl --verify-checksums
```

Generate a JSON pair from two residue-mapped single-chain structures:

```bash
ospedit-eval --parent-structure parent.pdb --mutant-structure mutant.pdb \
  --parent-chain A --mutant-chain A --pair-id example --output pair.json
```

The repository includes a minimal CA-only smoke fixture:

```bash
ospedit-eval --parent-structure examples/toy_parent.pdb \
  --mutant-structure examples/toy_mutant.pdb --parent-chain A \
  --mutant-chain A --pair-id toy_pdb --manifest-output /tmp/toy_manifest.jsonl
ospedit-eval --audit-manifest /tmp/toy_manifest.jsonl
```

The same chain is available as a temporary-file smoke test:

```bash
bash examples/run_p0_smoke.sh
```

Run the full local gate with:

```bash
make check
```

This runs tests, lint, core-module type checking, bytecode compilation through
the test path, the PDB/manifest smoke test, and the synthetic mechanism
harness, plus a wheel-content check.

For an orchestration-only C0-C5 check without a FoldFlow installation, run the
synthetic harness:

```bash
python examples/run_mechanism_smoke.py
```

Its output is synthetic engineering data and must not be used as a biological
benchmark.

Once the FoldFlow environment and checkpoint loader are available, run the
real C0-C5 grid with the injected endpoint factory:

```bash
python scripts/run_mechanism_grid.py \
  --manifest data/manifest/pairs.jsonl \
  --endpoint-factory my_foldflow_loader:build_endpoint \
  --split dev --output results/mechanism_dev.json
```

The factory is deliberately external: it should construct and return an
object implementing `EndpointModel.endpoint`. This keeps checkpoint and
upstream configuration details out of the core package while making the
experiment command reproducible and explicit.

Calibrate noise and update scales on the frozen development split with the
reproducible sweep tool:

```bash
python scripts/run_mechanism_sweep.py \
  --manifest data/manifest/pairs.jsonl \
  --endpoint-factory my_foldflow_loader:build_endpoint \
  --split dev \
  --noise-levels 0.25,0.5,0.75 \
  --two-noise-levels 0.25,0.75 \
  --two-noise-weights 1.0,1.0 \
  --difference-step-sizes 0.05,0.1,0.2 \
  --methods C0_copy_parent,C3_two_noise_shared_difference \
  --output results/mechanism_sweep_dev.json
```

The sweep records every configuration, per-method metrics, manifest
fingerprint, and runtime counters. Select and freeze a configuration from the
development report before evaluating the held-out test split; the command does
not perform test-set selection. Reports are written atomically after each
configuration and carry `status=running` until the full sweep completes;
the report also records expected versus completed configuration counts. The
selector accepts only `status=completed` reports with matching counts.
If a run is interrupted, repeat the command with `--resume` and the same
manifest, split, and output path to skip configurations already present in the
partial report.
Use `--methods` to limit an exploratory sweep; the selected method list is
recorded in each run's metadata. For quick CPU diagnostics, `--max-records N`
uses the first `N` records in manifest order and records that limit; remove it
for the complete development split. Use `--record-selection family_round_robin`
with a limit when a quick diagnostic should cover families more evenly; the
strategy is recorded and must match when resuming.

Freeze a configuration with an explicit development-set constraint before
opening the held-out test split:

```bash
python scripts/select_mechanism_config.py \
  --input results/mechanism_sweep_dev.json \
  --method C3_two_noise_shared_difference \
  --max-remote-drift 0.05 \
  --max-local-regression 0.05 \
  --max-network-calls 12 \
  --output results/mechanism_config_frozen.json
```

The selector records the source manifest fingerprint, selected metrics, and
sorting policy. A missing or non-finite candidate is never silently selected.
All runs in one sweep must also share a manifest fingerprint.
Selection requires a `dev` sweep by default; `--allow-non-dev` is available
only for explicit diagnostic runs and should not be used to freeze parameters.
Apply the frozen parameters to a held-out run by passing
`--frozen-config results/mechanism_config_frozen.json` to
`run_mechanism_grid` and explicitly setting `--split test`; the report retains
the frozen configuration provenance. When the selector recorded a source
manifest path, the grid command rejects a different manifest instead of
silently mixing calibration and evaluation data. If the frozen config records
a selected `method`, any explicit `--methods` filter must include that method;
this prevents a held-out run from silently evaluating a different editor than
the one selected on `dev`.

Write the same pair as an auditable manifest record:

```bash
ospedit-eval --parent-structure parent.pdb --mutant-structure mutant.pdb \
  --pair-id example --parent-id parent-1 --family-id family-1 --split dev \
  --manifest-output data/manifest/pairs.jsonl
```

Use `--manifest-append` when importing another pair into an existing JSONL
manifest; without it, `--manifest-output` intentionally replaces the file:

```bash
ospedit-eval --parent-structure parent-2.pdb --mutant-structure mutant-2.pdb \
  --pair-id example-2 --parent-id parent-2 --family-id family-2 --split dev \
  --manifest-output data/manifest/pairs.jsonl --manifest-append
```

The generated manifest embeds the parsed backbone coordinates for the current
prototype and also records source/target files, chains, residue mapping,
family, split, and label source. `validate_manifest` rejects family leakage
across train/dev/test. During pair construction, the mutant backbone is
Kabsch-aligned onto the parent using all shared finite backbone atoms. This is
required because experimental coordinate files have arbitrary global poses;
the embedded mutant coordinates and all parent-local targets therefore live in
the parent coordinate frame.

For multiple experimental pairs, use the CSV importer so path resolution,
checksums, and manifest validation are handled consistently:

```bash
ospedit-build-manifest --pairs-csv data/pairs.csv \
  --output data/manifest/pairs.jsonl
```

Required CSV columns are `pair_id`, `parent_structure`, `mutant_structure`,
`parent_id`, `family_id`, and `split`; optional columns are
`parent_chain`, `mutant_chain`, and `label_source`. Paths are resolved relative
to the CSV file. Add `--append` to extend an existing manifest.

To create a deterministic family/parent-grouped split, use
`ospedit-split-manifest`. Add `--require-nonempty` to reject empty splits and
`--report results/split.report.json` to persist the seed, fractions, grouping
mode, split counts, and resulting manifest fingerprint.

For a local PreMut release, use the dedicated importer:

```bash
python scripts/build_premut_manifest.py \
  --csv /path/to/MutData2022.csv \
  --pdb-root /path/to/MutData2022_PDB \
  --dataset-name premut22 \
  --output data/manifest/premut22.jsonl
```

It treats PreMut mutation positions as zero-based, validates the reported
wild-type and mutant residue, preserves structure
checksums, performs family-grouped splitting, and prints accepted, missing,
and invalid row counts. Add `--strict-missing` to fail on any missing PDB or
malformed row.

Add `--report results/premut_import.json` to write machine-readable counters,
split counts, source paths, and the resulting manifest fingerprint.

For the Platinum protein--ligand mutation database, build a structure-pair
manifest from the public flat file and locally downloaded RCSB PDB files:

```bash
python scripts/build_platinum_manifest.py \
  --csv data/platinum_flat_file.csv \
  --pdb-root data/platinum_pdb \
  --output data/manifest/platinum.jsonl \
  --report results/platinum_import.json
```

The importer keeps single-point substitutions with both experimental
structures, requires the mutation identity and author residue number to match,
and checks that a ligand declared by Platinum is present in both coordinate
files. By default it permits only a continuous terminal-overlap crop with at
least 95% coordinate coverage; internal gaps and reorderings are rejected.
Use `--require-exact-residue-ids` for the stricter exact-ID diagnostic, or
change `--min-mapping-coverage` only as an explicitly reported sensitivity
analysis. Mapping mode, coverage, and terminal trim counts are stored in every
record.

For the much larger MicroMiner monomer release, first select a deterministic
candidate pool without using mutant RMSD as a ranking target:

```bash
python scripts/select_microminer_candidates.py \
  --input /path/to/filtered_single_mutations_pdb_monomer.tsv \
  --output data/microminer_candidates.csv \
  --report results/microminer_selection.json --limit 2048
```

Then use RCSB entry metadata to reject different UniProt chains, multichain
protein entries, mismatched hetero contexts, and mismatched experimental
methods before downloading coordinates:

```bash
python scripts/audit_microminer_metadata.py \
  --candidates data/microminer_candidates.csv \
  --metadata-cache data/cache/microminer_rcsb_metadata.json \
  --output data/microminer_candidates_context.csv \
  --report results/microminer_metadata_audit.json
```

Finally, build and locally audit the structure manifest:

```bash
python scripts/build_microminer_manifest.py \
  --candidates data/microminer_candidates_context.csv \
  --structure-root data/microminer_pdb \
  --metadata-report results/microminer_metadata_audit.json \
  --output data/manifest/microminer.jsonl \
  --report results/microminer_import.json
python scripts/audit_structure_context.py \
  --manifest data/manifest/microminer.jsonl \
  --output results/microminer_context.json \
  --filtered-output data/manifest/microminer_context.jsonl
```

The importer deliberately requires exactly one observed full-chain sequence
difference and matching author residue numbers. MicroMiner's source table
describes locally matched mutation environments, so many nominal single-
mutation rows contain additional full-chain differences and are correctly
rejected. When a metadata report is supplied, shared UniProt accessions are
embedded and used as provisional family groups; all imported records remain
`train` until sequence-family clustering freezes a leakage-safe split.

When available, pass PreMut's cluster file to avoid grouping unrelated
proteins solely by mutation label:

```bash
python scripts/build_premut_manifest.py ... \
  --cluster-dict /path/to/MutData2022_cluster_dict
```

Run the currently available copy-parent baseline over a manifest:

```bash
ospedit-eval --manifest data/manifest/pairs.jsonl --eval-split dev \
  --results-output results/copy_parent_dev.json
```

For a saved student, set `--editor student --student-checkpoint ...` and use
`--batch-size` to measure padded batch inference rather than one forward per
pair.

The report contains per-pair metrics, family-macro metrics, and explicit cost
counts for regular and batched evaluation. Batched reports include
`conditional_batches` (batches that contain an actual edit); both paths report
condition branches, network calls, sequence encoder calls, structure updates,
and wall time. External baselines and learned adapters are not silently
substituted.

The raw PreMut baseline has an isolated subprocess adapter with strict
source/checkpoint/cache identity and parent-frame validation. Setup, command,
and the current three-pair mechanism result are documented in
`docs/premut_setup.md`.

ESMFold default and zero-extra-recycling baselines share a persistent model
runner and receive the same global rigid-alignment allowance before frame
metrics. Environment, checkpoint identities, cost accounting, and current
results are documented in `docs/esmfold_setup.md`.

`scripts/assemble_benchmark_table.py` combines explicitly selected methods
only when split fingerprints and pair IDs match. The current unified mechanism
table and its cost-accounting limitation are in `docs/benchmark_v0_1.md`.

`scripts/assemble_runtime_table.py` separately combines resident-model setup
and warm-candidate costs. Each entry names the suite method, an explicit setup
source, and the device; the assembler rejects manifest/pair-order mismatches
and marks cross-device ranking as disallowed. Its N-candidate totals are linear
accounting projections until a true same-parent workload is measured.

For mechanism comparisons in Python, use `evaluate_editor_suite(records,
editors, split=..., batch_size=...)`. It materializes the selected records once and evaluates
every named editor on the identical set, returning separate per-pair,
family-level, and runtime reports per method.

`build_mechanism_editors(...)` provides a standard C0-C5 editor mapping with
fixed names and explicit noise/step-size arguments. Add external baselines to
that mapping before passing it to `evaluate_editor_suite`.
When an endpoint model is supplied, the mapping also includes
`C4_repeated_single_noise_matched_budget`, which spends the same four endpoint
queries as C3 at one noise level, and
`C5_independent_noise_difference`, the matched two-branch control for testing
whether shared noise itself improves the response.

Use `write_suite_report(suite, path)` to persist the complete mechanism result
as strict JSON; the report includes a deterministic `manifest_fingerprint`,
and non-finite metrics are encoded as `null` for downstream tools.
`write_suite_csv(suite, path)` or the mechanism-grid CLI's `--csv-output`
produces a one-row-per-method table with family-macro metrics and runtime
counts for quick comparison.
`SuiteEvaluation.run_metadata` is persisted as `run_metadata`; the mechanism
grid CLI fills it with the endpoint factory, resolved manifest path, split,
noise/step/scale settings, and Python/platform information.

The mechanism-grid CLI accepts `--neighborhood-radius <angstroms>` to include
the C6 mutation-neighborhood preservation ablation in the same reproducible
C0-C5 run.

Structural reports also include backbone bond-length violation fraction and
the number of checked bonds, plus independent backbone angle and non-adjacent
backbone clash fractions. Missing atoms are excluded from each denominator;
the metric is reported as `NaN` when no required bonds are available.
Distance-change reports additionally expose `local_distance_change_error` and
`remote_distance_change_error`, so mutation response and scaffold behavior can
be inspected separately. They also report signed `distance_change_cosine`,
predicted/true signal norms, and local/remote response cosines; a copy-parent
prediction has zero predicted signal and therefore an undefined cosine. All
distance-change errors exclude diagonal self-distances and non-finite residue
pairs.
Frame-audit metrics `parent_to_prediction_frame_rmsd` and
`remote_scaffold_frame_drift` are also reported without an additional Kabsch
fit; they measure actual movement in the parent coordinate frame. Aligned
RMSDs remain the shape-comparison metrics.
CLI reports encode such non-finite metrics as JSON `null` for strict downstream
parsers.

The first prototype evaluates whether a model can predict mutation-induced
structural changes from a parent structure while reducing unrelated scaffold
drift. The model-independent core uses a shared `(L, A, 3)` atom representation
and a small `FieldModel` adapter:

```python
from ospedit import ConditionalDifferenceEditor, StructurePair, evaluate_pair

editor = ConditionalDifferenceEditor(model=my_model, step_size=0.2)
prediction = editor.predict(pair)
metrics = evaluate_pair(pair, prediction)
```

The parent-frame mechanism is available as `LocalFrameDifferenceEditor`. Its
adapter implements `endpoint(coords, sequence, noise_level) -> (rotations,
origins)`, returning one global 3x3 residue frame and origin per residue. The
editor computes source/target differences with SO(3) log maps in the parent
frame, then applies SO(3) exp-map updates to the original backbone.

`MultiNoiseLocalFrameDifferenceEditor` combines several such shared-input
queries with explicit weights. Its runtime cost is reported as two condition
branches per noise level (source and target), so a two-level experiment is
not counted as one forward pass.

Adapters that need stochastic input can declare an optional `noise_state`
keyword. The editor then passes the same deterministic `SharedNoise` object to
the source and target calls at each noise level. Legacy three-argument
adapters remain supported, but their internal randomness cannot be audited as
shared noise.

This applies to both endpoint adapters and the simpler coordinate-field
`ConditionalDifferenceEditor`. In the latter, the target-only editor receives
the same explicit state as well; this keeps matched-budget comparisons honest
when the underlying field is stochastic.

Before admitting a real teacher, use `condition_response_diagnostic` to check
that source/target endpoint responses are finite, reproducible, and nonzero
for real edits while remaining zero for identical sequences. The configurable
`assess_teacher_admission` gate rejects reports with no finite mutation signal
or signal below the supplied development threshold; it is an engineering
filter, not a scientific accuracy claim.

The first one-pass student skeleton is `ParentEditStudent` (optional Torch
dependency). It consumes cached parent-local node features plus source/target
one-hot edit features and emits one 6D local-frame delta per residue. The
sample-level no-edit gate is exact, while edited samples can still propagate
updates beyond the mutated residue. It is an untrained architecture boundary,
not a reported model result. `PairDataset`, `collate_pair_records`,
`masked_delta_loss`, and `train_student` now provide the minimal supervised
training path over experimental pair records. Optional admitted teacher-delta
distillation is enabled with `--teacher-cache`, `--teacher-noise-level`, and
`--distill-weight`; it is additive to the experimental target loss and remains
disabled by default.

The student target has explicit unit scales: `--translation-scale` is measured
in Angstroms and `--rotation-scale` in radians. Use the same values when
constructing `StudentEditor`; the CLI wires these scales through so the loss
does not silently treat the two physical units as interchangeable.
Use `scripts/audit_response_scale.py` to report the per-pair experimental
parent-to-mutant delta range before selecting a bounded student dataset. The
tool can optionally write a thresholded manifest with `--filtered-output`; it
always validates the filtered records before writing them. The report includes
the content fingerprint of the input manifest for later provenance checks.
The final output layer is zero-initialized, so a newly created student starts
as an exact copy-parent editor. `--delta-norm-weight` enables an optional
normalized-update penalty for stability diagnostics; it defaults to `0.0`.
`--mutation-loss-weight` optionally upweights the explicitly mutated residues
without freezing non-mutated positions; it also defaults to `0.0`.
`--neighborhood-loss-weight` optionally upweights all valid residues within
10 Angstrom of a mutation, matching the local evaluation region; it also
defaults to `0.0`. Set `--neighborhood-radius` to reproduce a different local
radius; the selected radius is stored in the checkpoint configuration.
`--geometry-features` enables optional invariant chain-context and
mutation-distance channels; this setting is stored in the checkpoint and must
match any parent-context cache used at inference.
Positional encoding is enabled by default; pass `--no-positional-encoding` to
reproduce the pre-position baseline in student or sweep runs.
Legacy checkpoints that predate this option are evaluated with positional
encoding disabled to preserve their original behavior.
The training CLI also exposes `--delta-loss smooth_l1` and
`--delta-loss-beta` for robust-loss ablations; MSE remains the default.
`--max-normalized-delta` applies an optional `tanh` bound to each predicted
normalized channel, providing a directly measurable update-stability ablation.
To repeat a bound sweep without manually duplicating commands:

```bash
python scripts/run_student_bound_sweep.py \
  --manifest data/manifest/pairs.jsonl --output-dir results/bounds \
  --bounds 0.1 0.25 0.5 1.0 --split train --eval-split dev --epochs 25
```
The script writes one checkpoint per bound plus a strict `summary.json`.
Pass `--seeds 0 1 2` to repeat every bound across seeds; checkpoints receive a
seed suffix and `summary.json` adds per-bound mean/std aggregates. Without
`--seeds`, the legacy single-`--seed` output names are preserved.
The summary also records the full manifest fingerprint, so multi-seed results
can be checked for data-version consistency.
Each run includes family-macro local error, parent-frame RMSD,
distance-change error, and mean inference time for direct comparison.
It also records copy-parent metrics and student-minus-copy error deltas, so a
bound cannot appear favorable merely because it preserves the parent scaffold.
The metric set includes both locally re-aligned mutation-site error and
`mutation_site_global_rmsd`, which measures the mutation site after one
whole-chain parent-to-mutant alignment.
The sweep also accepts `--teacher-cache`, `--teacher-noise-level`, and
`--distill-weight` to compare bounded direct supervision with admitted teacher
distillation under identical architecture and bound settings.
When a cache is supplied, the sweep validates it before launching child runs
and records its manifest fingerprint in `summary.json`.

Before generating teacher labels, run the endpoint admission gate:

```bash
python scripts/run_teacher_admission.py \
  --manifest data/manifest/pairs.jsonl \
  --endpoint-factory my_foldflow_loader:build_endpoint \
  --split dev --output results/teacher_admission.json
```
The report records per-pair mutation responses and sets
`ready_for_distillation` only when every selected pair has a finite, nonzero
mutation response and passes the repeatability threshold. The default
`--repeat-tolerance 1e-6` compares two identical shared-noise diagnostic runs;
raise it only when the endpoint's documented numerical nondeterminism requires
that accommodation.

After the report is ready, materialize endpoint labels with the audited cache
builder:

```bash
python scripts/build_teacher_cache.py \
  --manifest data/manifest/pairs.jsonl \
  --admission-report results/teacher_admission.json \
  --endpoint-factory my_foldflow_loader:build_endpoint \
  --split train --output-dir results/teacher_cache
```

The builder requires a matching manifest fingerprint and noise grid, refuses
rejected pairs, and writes one compressed `.npz` per pair plus `index.json`.
Each file contains source/target endpoint frames and the parent-local 6D
condition difference for every requested noise level; the index records the
endpoint, seed, split, and diagnostic provenance.
For a research-use cache, pass a direction-audit report from
`evaluate_teacher_cache.py` with `--direction-report`; the builder then refuses
to proceed unless that report's direction gate is accepted. Without this
argument the output is explicitly a plumbing/debug cache. Direction reports
must use the `ospedit.teacher_evaluation.v1` schema emitted by the evaluator;
other JSON reports are rejected.

Audit a generated cache against experimental parent-to-mutant deltas before
using it for research distillation:

```bash
python scripts/evaluate_teacher_cache.py \
  --manifest data/manifest/pairs.jsonl \
  --teacher-cache results/teacher_cache/index.json \
  --split train --noise-level 0.25 \
  --min-mutation-cosine 0.2 --output results/teacher_delta_eval.json
```

Add `--verify-checksums` when the manifest carries source/target SHA-256
checksums; evaluation then fails before loading labels if an experimental
structure file is missing or has changed.

The optional direction gate reports whether the mean mutation-response cosine
and RMSE satisfy the supplied thresholds; it is separate from the nonzero,
reproducibility-only teacher admission gate. Translation and rotation RMSE
are also reported separately (Angstroms and radians), and can be gated with
`--max-mutation-translation-rmse` and `--max-mutation-rotation-rmse`. A
direction-gated cache also requires the report's manifest fingerprint, split,
and noise level to match. When any direction threshold is supplied and the
gate rejects the report, the evaluator still writes the full report and exits
with status `2`, so shell/CI pipelines can stop before cache construction.

Direction reports include record, family-macro, and parent-macro summaries.
The CLI applies scientific gates to the parent-macro summary by default so
multiple mutations of one parent cannot silently receive extra weight. Use
`--gate-aggregation record` only for an explicitly record-weighted analysis.

After caching several noise levels, evaluate two-level transport combinations
without additional endpoint calls:

```bash
python scripts/evaluate_teacher_combinations.py \
  --manifest data/manifest/pairs.jsonl \
  --teacher-cache results/teacher_cache/index.json --split train \
  --noise-levels 0.1 0.25 0.5 0.75 \
  --output results/teacher_combinations.json
```

The calibration excludes combinations with nonpositive total weight by
default, since those reverse or cancel a time-invariant target-minus-source
response. Retain them only as an explicit diagnostic with
`--allow-nonpositive-net-weight`.

Estimate whether mutation changes exceed experimental conformational
background using same-sequence repeat structures:

First reject pairs whose experimental context makes mutation attribution
ambiguous:

```bash
python scripts/audit_structure_context.py \
  --manifest data/manifest/pairs.jsonl \
  --verify-checksums \
  --filtered-output data/manifest/context_clean.jsonl \
  --output results/structure_context_audit.json
```

The strict default requires one protein chain in both coordinate files, the
same target-chain hetero-residue set, and the same experimental method. This
implements the first-version single-chain scope and rejects ligand-state or
binding-partner changes before they can be attributed to a mutation. The
report retains explicit reasons; relax a gate only as a separately named
analysis, not by silently changing the primary set.

`--ignore-hetero` can define a named sensitivity analysis for crystallization
additives. The default remains `HOH` only. A shared ligand recorded by a source
database does not by itself make a pair context-clean, since cofactors or other
biologically relevant hetero compounds may still differ.

Then estimate whether the remaining mutation changes exceed experimental
conformational background:

```bash
python scripts/audit_repeat_structures.py \
  --pairs-csv data/repeat_pairs.csv \
  --mutation-manifest data/manifest/context_clean.jsonl \
  --background-aggregation max \
  --min-repeat-structures 2 \
  --min-neighborhood-signal-to-background 2 \
  --min-distance-signal-to-background 2 \
  --filtered-output data/manifest/identifiable.jsonl \
  --output results/repeat_structure_audit.json
```

Repeat structures remain separate from mutation labels. The audit reports
site, 10-Angstrom neighborhood, and distance-change signal-to-background
ratios after backbone alignment. With multiple repeats, the default `max`
aggregation uses the largest observed background for each metric, so a pair
cannot pass by selecting its most favorable repeat.

Discover candidate repeats from the official RCSB exact-sequence service and
write the CSV consumed by that audit with:

```bash
python scripts/discover_rcsb_repeats.py \
  --manifest data/manifest/context_clean.jsonl \
  --download-dir data/rcsb_repeat_cache \
  --pairs-output data/repeat_pairs.csv \
  --report results/rcsb_repeat_discovery.json
```

The tool caches downloaded PDB entries but rechecks the observed coordinate
sequence, excludes the source entry by PDB ID and chain even when it appears
under a different path, and applies the same single-chain, target-hetero, and
experimental-method context rules. Any `--ignore-hetero` list is recorded in
the report and therefore defines a separate sensitivity analysis.

Select independent PreMut groups with one parent and at least two additional
same-sequence wild-type structures before downloading and auditing them:

```bash
python scripts/select_premut_repeat_candidates.py \
  --csv data/MutData2022.csv \
  --dataset-name premut22n \
  --pdb-root data/pdb \
  --max-candidates 40 \
  --repeat-structures 2 \
  --max-mutation-index 255 \
  --mutations-output data/repeat_mutations.csv \
  --repeats-output data/repeat_pairs.csv \
  --report results/repeat_candidates.json
```

Pass earlier candidate reports with `--exclude-report` to continue a scan
without reusing mutants or primary parents.

For a parent-balanced search of alternative mutations after each parent's
first candidate has been audited, add `--allow-excluded-parents`. Previously
audited mutants remain excluded, while another mutation of the same parent may
be selected. Combine it with `--allow-repeated-parent` only for an explicitly
mutation-dense scan; otherwise each round still contributes at most one new
candidate per parent.

Merge independently audited subsets through the structured manifest API:

```bash
python scripts/merge_manifests.py \
  --manifest data/manifest/subset_a.jsonl data/manifest/subset_b.jsonl \
  --output data/manifest/combined.jsonl \
  --report results/manifest_merge.json --verify-checksums
```

The merger rejects duplicate pair IDs, split leakage, invalid input manifests,
and output paths that overwrite an input.

For the upstream FoldFlow-2 release, the repository includes an environment-
driven factory. Run the command from the isolated FoldFlow environment after
setting `OSPEDIT_FOLDFLOW_ROOT` to the upstream checkout and
`OSPEDIT_FOLDFLOW_CHECKPOINT` to `ff2_base.pth` (optionally set
`OSPEDIT_FOLDFLOW_DEVICE`), then use
`examples.foldflow_endpoint_factory:build_endpoint` as the factory value.

Variable-length batches pass a residue padding mask through the Transformer and
training loss, so padded positions cannot contribute attention or updates.
The supervised loss is normalized per pair before averaging a batch, preventing
longer proteins from silently dominating mixed-length training.

`train_records` is the high-level entry point when the data is already loaded
as audited `PairRecord` objects. It builds local targets and padded batches,
with deterministic shuffling controlled by `seed`. It also excludes
nonexperimental labels by default; pass `allow_nonexperimental=True` only for
an explicitly separate synthetic/teacher run.

Use `save_student_checkpoint` and `load_student_checkpoint` to resume local
experiments. Checkpoints use an explicit format version and retain optimizer
state, epoch, loss history, and the supplied configuration dictionary.
Resume and evaluation validate the checkpoint's recorded `parent_dim`,
`hidden_dim`, `blocks`, and `heads` against the requested/input architecture
before loading weights.
Resume also validates the recorded target scales, neighborhood radius, loss
kind/beta, and loss weights before continuing.
New training checkpoints also store a deterministic `manifest_fingerprint`;
resume refuses to continue when the supplied manifest has changed.

The same path is available as a CLI for a frozen manifest split:

```bash
ospedit-train \
  --manifest data/manifest/pairs.jsonl \
  --split train \
  --output checkpoints/student_epoch_1.pt \
  --epochs 1 --batch-size 1 --grad-accumulation-steps 16
```

Use `--no-gradient-clip` to disable clipping, or set an explicit
`--gradient-clip-norm`; the selected value is stored in the checkpoint config.

Add a same-manifest development evaluation after training:

```bash
ospedit-train --manifest data/manifest/pairs.jsonl \
  --split train --eval-split dev \
  --output checkpoints/student_epoch_1.pt --eval-batch-size 8
```

The command audits the manifest first and records its model/training arguments
in the checkpoint config. It excludes synthetic and teacher-labelled records by
default; use `--allow-nonexperimental` only for an explicitly separate run.

Evaluate a saved student through the same evaluator:

```bash
ospedit-eval pair.json \
  --editor student \
  --student-checkpoint checkpoints/student_epoch_1.pt
```

Both manifest evaluation and training accept `--verify-checksums` to enforce
the recorded source/target structure hashes before running.

Resume an existing run with:

```bash
ospedit-train --manifest data/manifest/pairs.jsonl \
  --split train --resume checkpoints/student_epoch_1.pt \
  --output checkpoints/student_epoch_2.pt --epochs 1
```

When `--resume` is provided, omitted architecture flags are restored from the
checkpoint; explicitly supplied conflicting values are rejected.

`train_student` materializes its batch iterable once so multi-epoch training
also works with generators, and applies optional gradient clipping (default
norm `1.0`). Gradient accumulation scales each window by its actual size,
including a shorter final window.

Wrap a trained student for the common evaluation API with `StudentEditor`:

```python
from ospedit import StudentEditor

editor = StudentEditor(model, device="cuda")
prediction = editor.predict(pair)
```

This performs one student forward and applies its local-frame delta to the
parent backbone. Runtime reports count the student as one network call and no
per-mutant sequence encoder call.

Pass a `ParentContextCache` to `StudentEditor` to reuse parent-local features
across many mutations of the same parent. The cache key includes the parent
sequence, atom layout, and coordinate bytes; call `.clear()` between datasets
or experiments. Runtime reports expose `parent_cache_hits`,
`parent_cache_misses`, and `parent_cache_entries` for auditing the actual reuse.
Hit/miss values are deltas for that evaluation call, so repeated reports with
the same editor remain comparable. Pass `ParentContextCache(max_entries=N)`
for a bounded LRU when processing many different parents.

Use `group_records_by_parent(records, split=...)` to construct the same-parent
candidate workloads required for 1/32/128 throughput measurements.
`parent_workloads(records, candidate_counts=(1, 32, 128), seed=...)` adds
deterministic sampling and can exclude parents without a complete workload.
Pass its result to `evaluate_parent_workloads(workloads, editor,
batch_size=...)` to run and retain one report per parent/candidate count.
Persist those nested reports with `write_parent_workload_report(reports, path)`;
candidate counts are stored as JSON object keys for stable downstream parsing.
Use `flatten_parent_workload_reports(reports)` for one row per parent and
candidate count when preparing CSVs or plots.
`write_parent_workload_csv(reports, path)` writes the same rows with stable
columns and blank cells for non-finite metrics.

For throughput measurements, `predict_student_batch(model, pairs, ...)` pads
mixed-length pairs once, performs one student forward, and returns one edited
coordinate tensor per pair.

`StructurePair.from_json()` accepts a development fixture with fields
`pair_id`, `parent_sequence`, `mutant_sequence`, `parent_coords`,
`mutant_coords`, and `mutation_indices` (zero-based residue indices).
The optional `atom_names` field is inferred as `[N, CA, C, O]` for four-atom
backbones and `[CA]` for C-alpha-only fixtures; other layouts must declare it.

The current scaffold intentionally leaves the folding/flow-model adapter
external. This keeps the initial experiment focused on the copy, target-update,
and source-target conditional-difference comparisons.

An optional `FoldFlow2EndpointAdapter` is included for the next integration
step. It accepts an already-loaded FoldFlow-2 model plus a repository-specific
batch builder, enables `conditional_generation()` once, and converts the
model's `[B, L, 7]` `(quaternion, translation)` `rigids` output into the
endpoint contract. FoldFlow's environment, checkpoint loading, and data
transforms remain caller-supplied and optional.

The adapter also re-enables conditional generation immediately before a
forward if an external `.eval()` call reset FoldFlow's internal flag.

`FoldFlow2BatchBuilder` now provides the minimal batch-field construction for
that adapter. The caller supplies only the installed FoldFlow/OpenFold
coordinate-to-rigids conversion, including its noise schedule and scaling;
this prevents the scaffold from silently using an incompatible Cartesian
noise convention. For clean N/CA/C frames, `backbone_to_rigids` provides a
scalar-first tensor-7 conversion; a separate flow-matcher callback should add
the model's actual noise and scaling before inference.

`FoldFlow2MarginalConverter` packages that final bridge: inject an OpenFold
`Rigid` constructor and a marginal-sampling callback, and it returns validated
`[L, 7]` noisy rigids while forwarding the shared noise object and time level.
`make_forward_marginal_sampler` supplies the matching wrapper for FoldFlow's
`SE3FlowMatcher.forward_marginal`; its `target_rigid_sampler` remains explicit
so the SO(3)/R(3) stochastic path cannot be confused with Cartesian Gaussian
noise.
`openfold_rigid_from_tensor7` is provided as a lazy constructor for the
OpenFold `Rigid` object in a properly provisioned FoldFlow environment.
Run `ospedit-foldflow-check` to inspect that environment, or
`ospedit-foldflow-check --strict` in CI/launch scripts to fail fast when the
official runtime is unavailable.

See [docs/foldflow_setup.md](docs/foldflow_setup.md) for the isolated upstream
environment and adapter wiring sequence.

Before a real mechanism run, use the combined gate:

```bash
ospedit-p1-preflight --manifest data/manifest/pairs.jsonl --verify-checksums
```

Add `--output results/p1_preflight.json` to persist the same report while it is
also printed to stdout.
For the first single-point-only experiment, add `--max-mutations 1` (the
default is no mutation-count restriction).
Use `--require-split-capacity` when the run requires a nonempty train/dev/test
split; this checks both connected-group capacity and the manifest's assigned
split counts, turning either warning into a hard failure.
For a local FoldFlow checkout, pass `--foldflow-root /path/to/FoldFlow` to
make the source path part of the same command-level audit.

It fails unless both the manifest audit and the FoldFlow environment check pass.

Implemented protocol pieces are deliberately narrower than
`docs/protocol_v0_1.yaml`: external baselines, FoldFlow-2 checkpoint loading,
and side-chain packing are not yet wired. PDB/mmCIF parsing and manifest
generation are available in the current prototype.
