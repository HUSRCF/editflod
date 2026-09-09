# Real-Data Smoke Run

## MicroMiner capacity audit (2026-09-09)

The official MicroMiner monomer archive was added as a high-capacity source
of candidate experimental structure pairs. The downloaded archive MD5 was
`1992f83a3356c0dbf846d242688ee4cb`, matching the repository record. Its TSV
contains 4,868,764 data rows. Every row in this release has one duplicated
terminal field beyond the 14-column header; the selector validates that the
two terminal values agree before removing the duplicate rather than silently
shifting columns.

`scripts/select_microminer_candidates.py` streams the archive and applies
`fullSeqId >= 0.98`, `alignmentLDDT >= 0.9`, at least 8 site residues,
distinct PDB entries, standard non-identity substitutions, unique query and
hit chains, and deterministic SHA-256 ranking. It found 457,590 eligible rows.
A 2,048-row candidate pool was then prefiltered with
`scripts/audit_microminer_metadata.py` using RCSB GraphQL metadata. Requiring
the named chains to share a UniProt accession, one protein-chain instance per
entry, matching non-water hetero sets, and the same experimental method kept
100 pairs across 197 unique PDB entries.

All 197 structures were downloaded and re-parsed. The strict importer kept 16
pairs with equal observed chain length, exactly one full-chain sequence
difference, matching mutation identities and author residue numbers, exact
residue-ID mapping, and length 64--256. The principal rejection counts were
55 rows with two or more full-chain substitutions, 22 unequal observed chain
lengths, and 7 out-of-scope chains. This confirms that MicroMiner's
single-mutation label is local-environment based and cannot replace a
full-chain sequence audit.

All 16 imported pairs passed the coordinate-level context gate and span 11
shared-UniProt family groups. Their maximum normalized per-residue response
has median `1.313` and range `0.501--12.220`; three pairs exceed `10`, so the
set includes both near-copy and large-response cases. They remain `train`
pending sequence-family clustering and are not yet a locked mechanism split.
The reproducible scratch artifacts are:

```text
/tmp/ospedit-real-sample/microminer_candidates_2048_v1.csv
/tmp/ospedit-real-sample/microminer_candidates_2048_metadata_strict_v2.csv
/tmp/ospedit-real-sample/microminer_manifest_2048_strict_v2.jsonl
/tmp/ospedit-real-sample/microminer_context_manifest_2048_strict_v2.jsonl
/tmp/ospedit-real-sample/microminer_response_2048_strict_v2.json
```

The source dataset is the [MicroMiner data record](https://www.fdr.uni-hamburg.de/record/13411),
and the method is described in the [MicroMiner paper](https://academic.oup.com/bib/article/24/6/bbad357/7311328).

Exact-sequence RCSB discovery queried 14 unique observed parent sequences,
downloaded 320 unique entries, and found 103 context-matched repeat instances
for 6 of the 16 pairs. Five pairs had at least two repeats. Under the
predeclared conservative maximum-background aggregation and the dual
`neighborhood >= 2` and `distance change >= 2` signal/background gate, no pair
passed. `1KF7_A -> 1IZR_A (F46A)` was closest: its neighborhood ratio was
`2.292`, but its distance-change ratio was only `0.981`. The other pair-level
distance-change ratios were at most `1.038`.

The repeat discovery command now downloads with a bounded worker pool and
atomically promotes complete files into its cache. This reduced the RCSB
coordinate-fetch bottleneck while retaining deterministic result ordering and
per-entry failure reporting. The authoritative reports are:

```text
/tmp/ospedit-real-sample/microminer_repeat_discovery_2048_strict_v2.json
/tmp/ospedit-real-sample/microminer_repeat_audit_2048_strict_v2.json
```

MicroMiner therefore materially expands strict, context-clean pair capacity,
but it does not add a new repeat-identifiable mechanism pair under the current
primary evidence rule. These 16 records are suitable for sensitivity analyses
and data-pipeline development, not for relaxing the frozen identifiability
criterion after observing results.

### Response-stratified discovery audit

A second, separately labeled discovery-only sample tested whether the first
pool was dominated by very small structural responses. The selector allocated
4,096 candidates equally across MicroMiner site-backbone RMSD strata
`<0.15`, `0.15--0.30`, `0.30--0.60`, and `>=0.60 Angstrom`, using seed 29.
Because this selection reads an experimental structural-difference field, its
report sets `eligible_for_unbiased_test=false`; none of these records may be
used to estimate population-level test performance.

RCSB metadata screening retained 192 pairs (`99/54/25/14` by the four source
strata) across 374 unique PDB entries. All coordinates were downloaded. Strict
full-chain validation retained 35 single substitutions, of which 32 passed the
coordinate-level context gate. They cover 18 shared-UniProt groups before the
three context rejections. The response-scale median rose from `1.313` in the
unbiased pool to `2.394`, showing that the stratification enriched structural
change without being treated as unbiased evidence.

Repeat discovery completed for 25 of 27 unique parent sequences. Two 252-aa
sequences from the same `P84131` family consistently received HTTP 400 from
the RCSB sequence-search API and remain an explicit coverage gap. For the
successful queries, PDB-to-mmCIF fallback resolved all 456 coordinate entries.
The audit found 197 context-matched repeats for 12 mutation pairs; 10 pairs
had at least two repeats. None passed the same conservative dual
signal/background gate. The closest result was
`2ZIL_A -> 1GB8_A (V74M)`, with neighborhood ratio `2.893` and distance-change
ratio `1.533`.

Thus response stratification improves raw response capacity but does not solve
label identifiability. The result strengthens the decision to keep teacher
distillation disabled rather than weakening the frozen distance-change gate.
Artifacts are:

```text
/tmp/ospedit-real-sample/microminer_discovery_candidates_4096_v1.json
/tmp/ospedit-real-sample/microminer_discovery_manifest_4096_v1.jsonl
/tmp/ospedit-real-sample/microminer_discovery_context_manifest_4096_v1.jsonl
/tmp/ospedit-real-sample/microminer_discovery_response_4096_v1.json
/tmp/ospedit-real-sample/microminer_discovery_repeat_discovery_4096_v3.json
/tmp/ospedit-real-sample/microminer_discovery_repeat_audit_4096_v1.json
```

## Platinum capacity and identifiability audit (2026-09-09)

The public Platinum flat file was added as a second experimental-pair source.
`scripts/build_platinum_manifest.py` aggregates duplicate affinity rows and
requires a single substitution, both WT/mutant structures, mutation identity
and author-number agreement, and at least one Platinum-declared ligand in both
coordinate files. A continuous terminal-overlap mapping is allowed only at
95% or greater coverage; internal coordinate gaps remain rejected and mapping
provenance is embedded in each record.

Among 54 author-declared monomer candidates, the 64--256 aa import accepted 9
pairs: 7 exact residue-ID mappings and 2 terminal-overlap mappings. Thirteen
failed residue mapping, 31 were outside the length scope, and one lacked the
declared ligand in both structures. The manifest and report are:

```text
/tmp/ospedit-real-sample/platinum_monomer_manifest_v2.jsonl
/tmp/ospedit-real-sample/platinum_monomer_report_v2.json
```

The default context gate (`HOH` ignored, all other target-chain hetero names
matched exactly) retained 4/9. A separately named crystallization-additive
sensitivity analysis retained 7/9 after ignoring `CL`, `SO4`, `PEG`, `NA`, and
`CD`; it is not silently substituted for the primary strict context policy.
An exploratory 320-aa capacity run retained 20/30 under the same sensitivity
policy, representing 19 undirected structure pairs, 13 parents, and 7 family
groups. This shows that the 256-aa ceiling is a material capacity constraint,
but does not authorize changing the frozen first-version scope.

Exact-sequence RCSB search found two same-sequence, single-chain, ATP/Zn
background structures for `3DGL_A -> 3DGO_A (Y43F)`: `3LTC_A` and `2P09_A`.
`CL` and the pentaethylene-glycol component `1PE` were treated as explicitly
reported crystallization additives for this exploratory repeat audit. With
only `3LTC_A`, the pair appeared identifiable (neighborhood ratio 5.80,
distance-change ratio 2.19). Under the protocol-required two repeats and
conservative maximum background, those ratios fell to 1.285 and 0.923,
respectively, so the pair was rejected. The authoritative two-repeat report is:

```text
/tmp/ospedit-real-sample/platinum_rcsb_repeat_audit_tool_v2.json
```

The search and filtering were reproduced end to end with
`scripts/discover_rcsb_repeats.py`. It queried 6 unique observed parent
sequences, cached 144 RCSB entries, and emitted exactly the two `Y43F` repeat
rows after source-entry deduplication. Its report and downstream CSV are:

```text
/tmp/ospedit-real-sample/platinum_rcsb_repeat_discovery_tool_v2.json
/tmp/ospedit-real-sample/platinum_rcsb_repeat_pairs_tool_v2.csv
```

This leaves the count of context-clean, repeat-identifiable connected groups
unchanged at one. Platinum currently improves pair-source coverage and exposes
a reproducible mapping policy; it has not yet solved the experimental
identifiability bottleneck.

## Coordinate-frame correction (2026-09-09)

The original experimental manifests embedded parent and mutant coordinates in
their raw PDB poses. Metrics based on a fresh Kabsch fit or internal distance
matrices remained rigid-transform invariant, but `target_local_delta()` used
those raw coordinates directly. As a result, student supervision and teacher
direction audits mixed the biological response with an arbitrary file-level
rotation and translation.

Pair construction now Kabsch-aligns every mutant backbone onto its parent using
all shared finite backbone atoms. A regression test verifies that a mutant
containing only a rigid pose change produces zero structural response after
import. The PreMut importer also records the alignment protocol and row-level
rejection reasons in its audit report.

This correction invalidates the quantitative interpretation of all student
training, response-threshold filtering, and teacher-delta direction results
below that used the old `pairs2022_split.jsonl` coordinates. Their artifacts
remain pipeline smoke tests only. The C0/C3/C5 mechanism comparisons based on
Kabsch-aligned errors, distance changes, and parent-to-prediction drift do not
depend on the mutant file pose and remain valid as mechanism diagnostics.

The 13 accepted MutData2022 pairs were rebuilt at
`/tmp/ospedit-real-sample/pairs2022_aligned_frozen.jsonl`. A split seed of `4`
was chosen before model evaluation solely to balance connected-group record
counts, yielding `7 train / 2 dev / 4 test`, 9 connected groups, and no parent
or family leakage. Strict checksum and FoldFlow-environment preflight passed.
The new response audit is
`/tmp/ospedit-real-sample/pairs2022_aligned_frozen_scale.json`: maximum
normalized per-residue deltas now range from `1.085` to `10.904`, rather than
the previous `0.98--86.03`. The audit additionally reports physical components;
the high-response records reach `4.1--6.7 Angstrom` maximum translation and
approximately `2.1--2.7 radian` maximum frame rotation.

An independent five-pair expansion accepted four records and rejected
`5O41_A -> 2G0R_A (L29F)` because its residue identifiers require explicit
alignment. After rigid-pose correction, the accepted records have maximum
normalized responses of `1.50--4.17`. They are stored at
`/tmp/ospedit-real-sample/pairs2022_expand_aligned.jsonl`; they have not yet
been used for model selection.

A clean direct-supervision baseline was then rerun on the aligned frozen split
with `bound=0.1`, five epochs, and seeds `0/1/2`. On the two-record dev split,
mean local error improved by only `0.00052 Angstrom` versus copy-parent and
mutation-site error by `0.00213 Angstrom`, with `0.00408 Angstrom` parent-frame
drift. On the four-record high-response test split, the corresponding changes
were `-0.00019 Angstrom` and `-0.00156 Angstrom`, with `0.00300 Angstrom`
drift. Reports are stored under `aligned_student_baseline/` and
`aligned_student_baseline_test/` in the same scratch directory. These values
are free of the coordinate-pose bug, but the student still recovers only a
negligible fraction of the required response and should be treated as a
near-copy baseline rather than a successful editor.

The FoldFlow endpoint admission and direction audit were also repeated on the
two aligned dev records at noise level `0.25`. Both records passed the nonzero
and exact-repeatability admission gate. Their mutation-site direction cosines
were `0.638` and `0.739` (mean `0.689`), with mean translation RMSE `0.246
Angstrom` and rotation RMSE `0.095 radian`. The corrected report is
`/tmp/ospedit-real-sample/aligned_teacher_eval_dev.json`, backed by the debug
cache at `aligned_teacher_cache_dev/`. This is materially stronger than the
invalid old-frame audit, but both dev records share parent `1REX_A`; it is a
single-parent mechanism signal, not sufficient teacher admission for research
distillation. The next required check is a family/parent-balanced train-side
direction audit under the corrected coordinate protocol.

That train-side audit has now been completed for 7 records from 6 independent
parents. At noise level `0.25`, all endpoint responses were nonzero and exactly
repeatable, but parent-macro mutation cosine was `-0.220`; four of seven record
cosines were negative. Admission repeatability therefore does not imply useful
teacher direction, and the train cache is not admitted for research
distillation.

Noise levels `0.10`, `0.25`, `0.50`, and `0.75` were cached and compared. Their
parent-macro mutation cosines were respectively `-0.131`, `-0.220`, `-0.085`,
and `-0.0004`. All four formal gates with minimum cosine `0.2` rejected and
exited with status `2`. A 216-candidate two-level diagnostic initially showed
that apparent positive results were dominated by combinations with negative
net weight, which reverse the target-minus-source semantics. After requiring
positive net weight, 90 candidates remained and the best cosine was only
`0.141` (`-0.5 * delta(0.5) + delta(0.75)`). No combination was promoted to
dev evaluation. The calibrated cache and report are stored at
`aligned_teacher_cache_train_multit/` and
`aligned_teacher_combinations_train.json` under the scratch directory.

To estimate label identifiability, six same-sequence repeat structures were
audited for the train parents. Across the seven mutation records, the median
mutant-to-background ratio was `1.17` in the 10-Angstrom neighborhood and
`1.09` for all-pairs distance-change RMS. D17A had a neighborhood ratio of
only `0.32`; only H63T exceeded `2.0` in that metric (`2.22`). The report is
`/tmp/ospedit-real-sample/repeat_structure_audit_train.json`. Most current
experimental differences are therefore not clearly separated from
same-sequence structural variation. The 13-pair set remains useful for
mechanism diagnostics, but the next data milestone must prioritize replicated
structures and a predeclared signal-to-background criterion before further
student or teacher optimization.

Twelve additional PreMut2022 mutation groups with at least two candidate wild
structures were downloaded as mutation/parent/repeat triples. Ten mutation
pairs passed strict residue mapping and seven were within the `64--256`
residue scope. This first pass used one repeat per mutation pair. T49V passed
the predeclared neighborhood and distance-change `signal/background >= 2`
gate; H63T and D122N from the earlier scan also appeared to pass.

That conclusion was re-audited with at least two same-sequence repeats per
pair and conservative maximum-background aggregation. H63T retained ratios
`2.08` (neighborhood) and `3.39` (distance change), and T49V retained `7.33`
and `3.29`. D122N fell to `1.19` and `0.37` and is therefore no longer treated
as identifiable. The earlier three-record manifest and its D122N mechanism
result are retained only as provenance for the superseded single-repeat
analysis below.

Before held-out queries, noise level `0.5` was frozen using the single train
record H63T (`mutation cosine=0.214`) and written to
`identifiable_teacher_config_frozen.json`. The dev T49V teacher direction
cosine was `0.437` and passed the `0.2` direction gate, while test D122N was
only `0.038` and failed. With a fixed structure-update scale of `0.05`, C2
changed predicted distance matrices by norms `1.35`, `1.01`, and `1.04` on
train/dev/test, versus true norms `52.3`, `44.7`, and `105.4`. Dev local and
mutation-site errors improved by `0.00162` and `0.02778 Angstrom`; test local
error improved by `0.00143 Angstrom` but mutation-site error worsened by
`0.00107 Angstrom`; train worsened on both. Even on high signal-to-background
records, the current endpoint edit remains near-copy and does not generalize
as a correct mutation-response predictor.

The candidate scan was then automated with
`scripts/select_premut_repeat_candidates.py`. A first batch contributed seven
additional in-scope audited mutation pairs and no strict dual-gate pass. A
second independent-parent batch contributed 20 in-scope pairs and one pass:
`1J1X_H -> 1IC4_H (D31A)`, with neighborhood ratio `8.63` and distance-change
ratio `2.85`. Thus only one of 27 newly audited in-scope pairs passed the
multi-repeat gate. The current minimum three-parent mechanism set is H63T
(train), T49V (dev), and D31A (test), stored at
`/tmp/ospedit-real-sample/pairs2022_identifiable_multirepeat_frozen.jsonl`
with fingerprint
`a330edd87085249426a23e3cbafa0e55b62218e90e47e1689bad34b2874ec711`.

Noise level `0.5` and update scale `0.05` were frozen before querying D31A.
The FoldFlow endpoint passed finite-response and repeatability admission, but
failed the held-out direction gate: mutation-site cosine was `-0.608`, local
cosine `-0.077`, and all-residue cosine `-0.049`. C2 also worsened local error
from `0.71790` to `0.71882 Angstrom` and mutation-site error from `0.79203` to
`0.79647 Angstrom`. Its predicted distance-change norm was `0.79`, versus the
experimental `41.05`, with distance-change cosine `-0.085`. This replacement
test confirms the current FoldFlow difference endpoint is not admitted for
distillation; student training remains deferred while identifiable data
capacity and alternative teachers are investigated.

## Raw PreMut baseline

The published raw PreMut checkpoint was integrated through an isolated,
deterministic subprocess runner. The runner restores the parent coordinate
translation omitted by the upstream prediction script and returns its
reconstructed input backbone for validation. On the official 8b0s example,
the reconstructed input differed from the source by only `6.65e-7 Angstrom`
RMSD and two seed-0 predictions were exactly equal.

PreMut was then evaluated on the frozen H63T/T49V/D31A mechanism set with
seeds 0, 1, and 2. At seed 0, local errors for copy versus PreMut were
`0.11590/0.11666` (train), `0.35718/0.36003` (dev), and
`0.71790/0.71757 Angstrom` (test). Mutation-site errors were
`0.09329/0.09480`, `0.17492/0.17524`, and `0.79203/0.78225 Angstrom`.
The test distance-change error worsened from `0.36165` to `0.36551`, with
cosine `-0.098` and predicted/true change norms `3.21/41.05`. D31A was exactly
seed-invariant; the two mutations requiring random missing-side-chain
initialization showed small but nonzero variation. PreMut therefore remains a
required strong baseline, but raw PreMut does not consistently beat copy on
this minimum set. Full commands, provenance, limitations, and seed ranges are
in `docs/premut_setup.md`.

## ESMFold refolding baselines

Default and zero-extra-recycling ESMFold were run through one persistent ROCm
model per split, then globally aligned to the parent before coordinate-frame
metrics. The explicit zero setting still executes one folding trunk pass.
Across H63T/T49V/D31A, copy-parent local errors were
`0.116/0.357/0.718 Angstrom`; default ESMFold produced
`0.274/0.400/1.077`, and zero-extra recycling produced
`0.315/0.928/1.025`. Predicted distance-change norms were `83--100`, versus
experimental norms `41--52`, and remote frame drift was `0.44--0.63
Angstrom`. Thus sequence-only refolding moves in a weakly positive aggregate
direction on some pairs but introduces too much unrelated change.

On the R9700, pure inference was `3.75--4.15 s` for default and `0.31--0.41 s`
for zero-extra recycling, with about `8.6 GB` peak allocated VRAM. Model load
was about 25 seconds and is separately reported and amortized across jobs.
Exact environment, checkpoint checksums, command, and per-pair table are in
`docs/esmfold_setup.md`.

## Resident runtime accounting

FoldFlow C2, raw PreMut, and both ESMFold modes were rerun on the same ordered
H63T/T49V/D31A manifest and assembled only after exact fingerprint and pair-ID
validation. The machine-readable outputs are
`/tmp/ospedit-real-sample/resident_runtime_v1.json` and `.csv`.

The observed model-setup / mean warm-candidate / resident N=3 times were
`7.215 / 1.532 / 11.809 s` for FoldFlow C2, `2.204 / 1.676 / 7.232 s` for
PreMut, `25.599 / 2.169 / 32.105 s` for default ESMFold, and
`25.599 / 0.352 / 26.654 s` for zero-extra ESMFold. FoldFlow and PreMut ran on
CPU; ESMFold ran on an AMD Radeon AI PRO R9700. These are runtime-instrumentation
results, not a fair cross-method speed ranking.

For PreMut, preprocessing consumed `4.939 s` across the three candidates while
network inference consumed only `0.087 s`. For a same-parent mutation scan,
parent graph reuse is therefore a more material systems hypothesis than GNN
compression alone. The table's N=32 and N=128 totals are linear projections
from the three heterogeneous records, not measured same-parent throughput.

A follow-up used 15 audited experimental single substitutions of the same
164-residue parent `2LZM_A`. The manifest passed structural validation and
source/target checksum verification. Observed setup / warm candidate /
resident N=15 times were `7.140 / 2.439 / 43.731 s` for FoldFlow C2,
`2.183 / 1.892 / 30.556 s` for PreMut with parent parsing cached,
`25.650 / 1.825 / 53.027 s` for
default ESMFold, and `25.650 / 0.448 / 32.376 s` for zero-extra ESMFold. The
artifact is `/tmp/ospedit-real-sample/runtime_2LZM_resident_v1.json` with a CSV
companion. This is the first measured same-parent curve point, but the methods
still ran on different hardware and N=32/128 remain unmeasured.

The PreMut cache only reuses the filtered parent ATOM table. Its 15 predictions
were elementwise identical to the original resident runner, and the published
second coordinate construction used for KNN edges remains intact. This reduced
the observed N=15 total by about `6.8%`, while leaving graph construction as
the dominant cost.

## Experimental-context gate

The repeat-structure S/B gate was followed by an explicit audit of protein
chain and hetero-residue context. `scripts/audit_structure_context.py` now
requires, by default, a single protein chain in both parent and mutant files,
matching target-chain hetero-residue sets, and matching experimental methods.
It writes both detailed rejection reasons and an optional filtered manifest.

This audit materially changes the interpretation of the minimum H63T/T49V/D31A
set. H63T compares different heme-ligand/crystallization contexts, and D31A is
an antibody-antigen complex. Only T49V passes both the strict context gate and
the earlier dual S/B gate. The old three-pair metrics remain useful engineering
diagnostics but are no longer primary biological evidence.

Two further parent-balanced scans were completed. A 40-candidate batch yielded
H92G, and a near-exhaustive 72-candidate independent-parent batch yielded P176A
and A128R under the S/B-only rule. Context inspection rejected all three:
H92G changes heme ligand state, P176A compares inhibitor-bound closed parent
with ligand-free mutant, and A128R is a heterodimeric nitrile hydratase chain.
Their combined frozen-parameter FoldFlow C2 evaluation had mean distance-change
cosine `-0.036` and worsened mean local error from `0.6717` to `0.6739
Angstrom`; they are retained as a confounding audit, not a clean test set.

A subsequent parent-balanced alternative-mutation round covered 102 parents.
Of 77 successfully mapped mutation pairs, 15 passed the context gate and 10
also met the 64--256 length plus two-repeat requirements. None passed both S/B
thresholds. The closest was `2PHY_A -> 1F9I_A (Y41F)`, with neighborhood ratio
`1.999` and distance ratio `2.472`; it remains rejected under the predeclared
threshold of `2.0`. This result shows that searching alternative mutations of
previously audited parents does not yet recover a development-scale clean set.

Relevant artifacts are:

- `/tmp/ospedit-real-sample/pairs2022_identifiable_multirepeat_context_audit.json`
- `/tmp/ospedit-real-sample/pairs2022_identifiable_external3_context_audit.json`
- `/tmp/ospedit-real-sample/repeat_expand_next80_audit.json`
- `/tmp/ospedit-real-sample/repeat_expand_alternatives_round1_context_audit.json`
- `/tmp/ospedit-real-sample/repeat_expand_alternatives_round1_audit.json`

The data conclusion is now stronger than “more pairs are needed”: the current
PreMut2022 pairing source, under the first-version single-chain and strict
experimental-attribution rules, does not provide enough clean independent
groups. Further model tuning on the old three-pair set would be misleading;
the next data source or curation strategy must explicitly match ligand and
binding-partner state before teacher selection resumes.

This note records a reproducible pipeline check on one public PreMut
MutData2023 pairing. It is an integration smoke test, not a benchmark or a
claim about model quality.

## Pair

- parent: `8G65_A`
- mutant: `1G1G_A`
- mutation: `C214A` (zero-based index `214`)
- length: `298`
- source: PreMut MutData2023 pairing table; structures downloaded from RCSB

The two parsed chains have identical length and exactly one sequence mismatch.

## Commands

```bash
curl -L --fail https://files.rcsb.org/download/8G65.pdb -o /tmp/8G65.pdb
curl -L --fail https://files.rcsb.org/download/1G1G.pdb -o /tmp/1G1G.pdb

ospedit-eval \
  --parent-structure /tmp/8G65.pdb \
  --mutant-structure /tmp/1G1G.pdb \
  --parent-chain A --mutant-chain A \
  --pair-id premut_1G1G_A_8G65_A \
  --parent-id 8G65_A --family-id premut-2023-1 --split dev \
  --manifest-output /tmp/pairs.jsonl

ospedit-eval --audit-manifest /tmp/pairs.jsonl --verify-checksums
ospedit-eval --manifest /tmp/pairs.jsonl --eval-split dev \
  --results-output /tmp/copy.json
```

## Observed smoke values

The copy-parent report contained zero backbone geometry violations and zero
remote scaffold drift by construction. Its local backbone error was about
`0.554` Angstrom and its distance-change error was about `2.33` Angstrom. These
values establish that the sample has a measurable structural response; they do
not establish that copy-parent is a competitive method.

The same parent also has a second public candidate, `1G1H_A`, with the same
`C214A` substitution and length. Adding it with `--manifest-append` produced a
two-record `8G65_A` workload. The copy-parent workload report measured zero
structure updates for both candidates and about `7.1e-6` seconds total for the
two evaluations on this CPU run. This is a cost-accounting baseline only; it is
not an inference-speed claim for a production implementation.

## FoldFlow endpoint smoke

An isolated `ospedit_foldflow_py310` environment was created with Python
`3.10.21`, Torch `1.13.1`, NumPy `1.24.3`, and the upstream FoldFlow source at
commit `9d2c260`. The official `ff2_base.pth` checkpoint (SHA-256
`2d19aa3624cb9f34081ae028745cfcfc4142dd634635c226d4104d9298637666`) loaded
successfully, and one CPU endpoint call on the 298-residue parent returned
`(298, 3, 3)` rotations and `(298, 3)` translations in about `4.6` seconds.

The first real C2/C3 mechanism query used the default step scale of `1.0` and
was too aggressive: local backbone error increased to `0.671`/`0.853` Angstrom
with remote scaffold drift `0.142`/`0.193` Angstrom. After wiring
`difference_step_size` into endpoint editors, a one-pair C2 sweep gave:

| difference step | local error (Angstrom) | remote drift (Angstrom) | distance-change error |
| ---: | ---: | ---: | ---: |
| 0.05 | 0.5533 | 0.0070 | 2.3299 |
| 0.10 | 0.5534 | 0.0141 | 2.3299 |
| 0.25 | 0.5582 | 0.0353 | 2.3302 |
| 1.00 | 0.6710 | 0.1423 | 2.3409 |

This is a single-pair development observation, not a benchmark claim. It
supports calibrating the update scale before expanding the C0-C5 matrix; the
copy-parent baseline remains the required zero-response control.

With `difference_step_size=0.05`, the full C0/C2/C3/C4/C5 matrix was then run
on both available `8G65_A` candidates (two records, same `C214A` edit). Family
means were:

| method | local error (Angstrom) | remote drift (Angstrom) | distance-change error | network calls |
| --- | ---: | ---: | ---: | ---: |
| C0 copy parent | 0.5805 | 0.0000 | 2.3147 | 0 |
| C2 shared single noise | 0.5802 | 0.0070 | 2.3146 | 4 |
| C3 shared two noise levels | 0.5778 | 0.0156 | 2.3147 | 8 |
| C4 repeated single noise | 0.5803 | 0.0049 | 2.3147 | 8 |
| C5 independent noise control | 3.2031 | 3.5429 | 5.5993 | 4 |

The run took about 140 seconds on CPU. C3 gives a small local improvement in
this two-record development set, while C5's large degradation provides a
direct sanity check that shared noise is doing real variance cancellation.
This remains a mechanism result, not a generalization or benchmark claim.

## 16-pair C2 development run

The first expanded manifest contains 16 single-point pairs from MutData2023,
covering four mutation families (`C214A`, `T156A`, `T156C`, and `T156D`) and
multiple parent structures. Strict preflight passed for all 16 records. A C0
versus calibrated C2 (`difference_step_size=0.05`) run took about 117 seconds
on CPU:

| family | C0 local error | C2 local error | C2 remote drift | C2 distance-change error |
| --- | ---: | ---: | ---: | ---: |
| C214A | 0.6066 | 0.6059 | 0.0071 | 2.3200 |
| T156A | 0.1068 | 0.1059 | 0.0030 | 0.0953 |
| T156C | 0.1127 | 0.1163 | 0.0031 | 0.0901 |
| T156D | 0.1088 | 0.1078 | 0.0039 | 0.1056 |

The family-level signal is small and not uniformly positive (`T156C` worsens),
so these results support further controlled C3/C5 analysis rather than a
general performance claim. The manifest and report remain under
`/tmp/ospedit-real-sample` and are intentionally outside the repository.

The NumPy-to-Torch conversion needed by the upstream `Rigid` constructor is
now handled by a scoped compatibility context in `ospedit.foldflow_batch`; the
upstream checkout itself is unmodified for this run.

## Cross-family C3/C5 check

To test whether the shared-noise effect was specific to one parent, one
representative pair from each of the four families was evaluated with C0, C3,
and C5 at the same `0.05` step scale (about 68 seconds on CPU):

| family | C0 local | C3 local | C3 drift | C5 local | C5 drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| C214A | 0.5540 | 0.5511 | 0.0156 | 3.2065 | 3.5429 |
| T156A | 0.0728 | 0.0715 | 0.0106 | 2.7431 | 3.1501 |
| T156C | 0.1206 | 0.1623 | 0.0044 | 2.7453 | 3.1498 |
| T156D | 0.0761 | 0.0779 | 0.0111 | 2.7510 | 3.1495 |

C5 degrades every family, supporting shared-noise cancellation as a real
mechanism effect. C3's local accuracy gain remains small and fails on T156C;
the next experiment should focus on scale/noise calibration and larger
family-balanced samples before student distillation.

## Mutation-neighborhood gate ablation

As a signal-enhancement test, C6 restricted the C2 update to residues within
10 Angstrom of the mutated C-alpha positions. On one representative from each
of the four MutData2023 families, C6 produced remote drift of only
`0.0002--0.0003` Angstrom, but did not improve local accuracy over C3:

| family | C6 local error | C6 remote drift | C6 distance-change error |
| --- | ---: | ---: | ---: |
| C214A | 0.5535 | 0.0003 | 2.3299 |
| T156A | 0.0724 | 0.0002 | 0.0651 |
| T156C | 0.1255 | 0.0002 | 0.0918 |
| T156D | 0.0769 | 0.0002 | 0.0772 |

The gate is therefore retained as a negative preservation ablation, not as a
replacement for C3: suppressing remote updates further does not recover the
missing local mutation signal.

## 13-pair independent-family C3/C5 run

The 13 MutData2022 pairs were evaluated with C0, C3, and C5 at step `0.05`
(about 309 seconds on CPU). The split-level means were:

| split | n | C0 local | C3 local | C3 drift | C5 local | C5 drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 6 | 0.5571 | 0.5638 | 0.0080 | 3.1112 | 3.3694 |
| dev | 3 | 0.1565 | 0.1563 | 0.0061 | 2.7629 | 3.4462 |
| test | 4 | 0.1127 | 0.1192 | 0.0046 | 3.3213 | 3.2526 |

C5 is worse than C0 on all 13 records, while C3 remains close to C0 and adds
only a few thousandths of an Angstrom of remote drift. C3 is slightly better
on the small dev subset but worse on test, so there is not yet evidence for a
general accuracy gain. This result strengthens the shared-noise mechanism
diagnostic while setting a conservative bar for any future signal-enhancement
or student-distillation change.

## Independent MutData2022 candidates

To reduce the connected-series bias, 16 MutData2022 metadata candidates were
downloaded separately. Three were rejected because parent and mutant residue
identifiers require an explicit alignment; the remaining 13 pairs passed all
checks and are stored at `/tmp/ospedit-real-sample/pairs2022.jsonl`. Their
family-parent connected-component split is deterministic and non-empty:
`train=6`, `dev=3`, `test=4` (seed `20260909`).

The strict P1 preflight was rerun in the isolated Python 3.10 environment
with `OSPEDIT_FOLDFLOW_ROOT=/tmp/foldflow-upstream`. It exited `0`, found no
missing packages, reported `9` connected groups and the same `6/3/4` split,
and returned `ready=true`. This separates a valid data/environment gate from
the intentionally rejected 32-record manifest below.

## Expanded candidate manifest

The metadata scan also produced a 32-record MutData2023 candidate manifest at
`/tmp/ospedit-real-sample/pairs32.jsonl`. All records pass checksum, chain,
length, and single-mutation checks in the Python 3.10 FoldFlow environment.
The set covers 15 mutation descriptions, but many structures come from the
same crystallographic series; it is therefore a development expansion, not a
claim of 32 independent protein families. The next locked split should group
these related structures before any tuning.

## One-pass student smoke

The supervised student path was exercised on the independent 13-pair split
with 6 train records and 3 dev records (5 epochs, hidden size 64, two blocks).
It produced `/tmp/ospedit-real-sample/student_pairs2022_smoke.pt` with the
manifest fingerprint embedded in its configuration and a batched dev report.
This is an integration smoke only: the dataset is too small, the final loss
is not calibrated for a biological claim, and no teacher distillation was
used. A research student run requires substantially more family-balanced
training data plus explicit coordinate/loss normalization.

## Scaled student diagnostic

The student was rerun on the same frozen split with explicit target scales
(`translation_scale=1.0 Angstrom`, `rotation_scale=0.25 radian`) and 25 epochs.
On the four-record test split, local errors were `0.716`, `0.419`, `2.886`, and
`2.858` Angstrom; predicted distance-change norms reached `109--525`, versus
experimental norms of `20--86`. Adding a small normalized-update penalty
(`delta_norm_weight=0.01`) did not stabilize the run (local errors
`0.904`, `0.441`, `3.119`, and `3.047`). These are diagnostic failures, not
benchmark results: the current six-record training set is insufficient for a
generalizing student, and the raw parent-local feature representation still
needs a better geometry-aware target or regularizer before distillation.

The output head is now zero-initialized so a fresh student is exactly
reference-preserving. A 25-epoch rerun with this initialization reduced the
four test local errors to `0.679`, `0.379`, `1.752`, and `1.934` Angstrom, but
still overpredicted distance-change norms (`92--342` versus `20--86`). This is
a useful stability improvement over random output initialization, not yet a
successful editor; training remains data- and representation-limited.

## Real FoldFlow teacher admission and cache

The repository factory was exercised in the isolated `ospedit_foldflow_py310`
environment with the upstream checkout at `/tmp/foldflow-upstream` and the
`ff2_base.pth` checkpoint. The first 2-record smoke manifest passed admission
and produced `/tmp/ospedit-real-sample/teacher_cache_real_2/index.json`.

The same path was then run on the audited 13-record MutData2022 manifest, using
its three-record `dev` split and one noise level (`0.25`). All three pairs were
admitted; maximum mutation responses were `0.14788`, `0.15983`, and `0.07223`,
and every repeated shared-noise query had translation and rotation error `0`.
The resulting checksum-enabled cache is at
`/tmp/ospedit-real-sample/teacher_cache_real_pairs2022_dev_checksum/index.json`,
with one compressed label file per pair and a SHA-256 recorded for every file.

The cache was evaluated with `scripts/evaluate_teacher_cache.py` using the
exact source manifest recorded in its index (`pairs2022_split.jsonl`). The
three-record `dev` direction report is
`/tmp/ospedit-real-sample/teacher_eval_pairs2022_dev.json`. Mean mutation-site
cosine was `0.2183` with mutation RMSE `2.3371`; mean remote cosine was
`-0.0519` and remote RMSE was `5.7032`. This is a diagnostic negative result:
the current FoldFlow endpoint response is not yet a reliable mutation-direction
teacher on this small split. Passing a different manifest is intentionally
rejected by the cache reader because the cache fingerprint covers the complete
source manifest, not only the cached split.

The direction gate was also exercised end to end on this cache. A threshold of
`--min-mutation-cosine 0.3` wrote
`/tmp/ospedit-real-sample/teacher_eval_gate_rejected.json`, recorded the reason
`mean mutation cosine is below threshold`, and exited with status `2`. The
lower diagnostic threshold `0.2` wrote
`/tmp/ospedit-real-sample/teacher_eval_gate_accepted.json` and exited `0`.
These runs verify gate plumbing only; they do not turn the small-split teacher
response into a validated biological signal.

## Geometry-feature student diagnostic

An additional five-epoch direct-supervision student run enabled
`--geometry-features` with the same six-record train and four-record test split,
hidden size 64, two blocks, and rotation scale `0.25`. The checkpoint is
`/tmp/ospedit-real-sample/student_geometry_smoke.pt`. Mean test local backbone
error was `0.1453 Angstrom` for the student versus `0.1127` for copy-parent;
mean distance-change error was `0.2725` versus `0.2125`. The geometry feature
path is therefore operational but does not beat the copy baseline on this
small split and remains a diagnostic, not a biological result.

A matched five-epoch rerun with `--mutation-loss-weight 1.0` changed the mean
test local error only from `0.1453` to `0.1447 Angstrom` and the mean
distance-change error from `0.2725` to `0.2721`; copy-parent remained at
`0.1127` and `0.2125`, respectively. This small shift does not justify treating
mutation weighting as a solved optimization.

A matched run with `--neighborhood-loss-weight 1.0` produced mean test local
error `0.1458 Angstrom` and distance-change error `0.2739`, slightly worse than
the mutation-weighted run and still worse than C0. The local-region weighting
switch is therefore implemented for controlled experiments, but this tiny
split provides no evidence that it improves the student.

The CLI-to-checkpoint wiring for this control was also exercised with a
one-epoch real smoke using radius `5.0 Angstrom` and neighborhood weight `1.0`.
The checkpoint `/tmp/ospedit-real-sample/student_radius5_smoke.pt` records both
values and contains a three-record `dev` evaluation, confirming that the
parameterized neighborhood target is reproducible end to end.
These are endpoint-response labels for
mechanism and distillation development, not evidence of mutant-structure
accuracy; the split remains too small for a biological claim.

The cache was consumed by a one-bound, one-epoch student sweep smoke with
`distill_weight=0.25` and `max_normalized_delta=0.1`. Its summary is at
`/tmp/ospedit-real-sample/distill_sweep_smoke/summary.json`; the run reported
local backbone error `0.1564`, distance-change error `0.2314`, and mean
inference time `0.0750 s/sample`. This confirms subprocess argument
propagation and summary provenance, not a quality comparison.

## Train-cache direct versus distillation diagnostic

The six-record `train` split was admitted with the same FoldFlow endpoint and
noise level, producing the checksum-enabled cache at
`/tmp/ospedit-real-sample/teacher_cache_real_pairs2022_train_checksum/index.json`.
Using the locked four-record `test` split, both students used bound `0.1`,
hidden size `64`, two Transformer blocks, four heads, seed `0`, and 25 epochs:

| training target | local backbone error | distance-change error | parent-frame RMSD |
| --- | ---: | ---: | ---: |
| direct experimental supervision | 0.1650 | 0.2518 | 0.1077 |
| + admitted teacher delta (`0.25`) | 0.1617 | 0.2513 | 0.1028 |

Reports are stored at
`/tmp/ospedit-real-sample/student_direct_test_smoke/summary.json` and
`/tmp/ospedit-real-sample/student_distill_test_smoke/summary.json`. The
improvement is small on four test records and is a pipeline/mechanism signal,
not evidence that distillation improves biological accuracy; family-balanced
expansion is required before selecting a method.

A fixed-bound weight probe (`0`, `0.1`, `0.25`, `1.0`) on the same split gave
local errors `0.1650`, `0.1642`, `0.1617`, and `0.1563`, respectively; the
corresponding parent-frame RMSDs were `0.1077`, `0.1054`, `0.1028`, and
`0.0946`. Distance-change errors stayed near `0.251` throughout. The apparent
local trend is based on four test records and one seed, so it is recorded only
as a weight-selection diagnostic; no default weight is frozen from it.

As a small stability check, the same three weights were rerun with seeds `1`
and `2` (all other settings unchanged). Mean +/- population standard
deviation across the two seeds was:

| `distill_weight` | local backbone error | distance-change error | parent-frame RMSD |
|---:|---:|---:|---:|
| 0 | 0.1580 +/- 0.0016 | 0.2538 +/- 0.0012 | 0.0983 +/- 0.0036 |
| 0.25 | 0.1579 +/- 0.0046 | 0.2501 +/- 0.0004 | 0.0969 +/- 0.0051 |
| 1.0 | 0.1529 +/- 0.0046 | 0.2519 +/- 0.0016 | 0.0901 +/- 0.0052 |

The local-error ordering is consistent across these two seeds, but the sample
and family count remain too small to select a production default. Per-run
reports are under `/tmp/ospedit-real-sample/student_multiseed/`.

## Teacher delta direction audit

The endpoint-response cache was compared directly with the experimental
parent-to-mutant local-frame delta using
`scripts/evaluate_teacher_cache.py`. On the six train pairs, mean mutation
cosine was only `0.1476` and mutation delta RMSE was `10.3542`; on the three dev
pairs the corresponding values were `0.2183` and `2.3371`. Remote cosine was
near zero (`0.0029` train, `-0.0519` dev). Reports are stored at
`/tmp/ospedit-real-sample/teacher_eval_train.json` and
`/tmp/ospedit-real-sample/teacher_eval_dev.json`.

The audit now reports translation and rotation channels separately because
their units differ. On mutation residues, translation RMSE was `14.6158`
Angstrom and rotation RMSE `0.8797` radians for train, versus `3.2420`
Angstrom and `0.6418` radians for dev. These channel values should be used for
future physical-unit gates instead of interpreting the combined 6D RMSE as a
single geometric quantity. Channel-aware reports are written to
`teacher_eval_train_channels.json` and `teacher_eval_dev_channels.json` in the
same scratch directory.

This is a negative but useful result: the current FoldFlow endpoint exposes a
stable, nonzero sequence-conditioned response, yet that response is not well
aligned with the observed experimental endpoint change. The cache remains
usable for controlled distillation plumbing only; any scientific teacher gate
must include this direction audit, and no claim that the current teacher is a
good pseudo-label source is justified.

During this run the integration boundary exposed and fixed two shape/runtime
issues: FoldFlow's marginal sampler requires an unbatched `[L]` Rigid while the
network consumes `[1,L]`, and its upstream ESM wrapper half-casts weights even
on CPU, where PyTorch 1.13 lacks the required float16 LayerNorm kernel. The
factory now handles both cases explicitly. The production GPU path remains
unchanged and should still be benchmarked separately.

An attempted CUDA C3/C4 sweep on 2026-09-09 exposed an environment mismatch:
the host Torch reports a CUDA device, but the isolated FoldFlow Torch `1.13.1`
build is CPU-only (`Torch not compiled with CUDA enabled`). The sweep therefore
failed during endpoint construction before any data evaluation. The endpoint
factory now reports this condition explicitly; GPU calibration requires a
CUDA-enabled Torch build inside the FoldFlow environment.

The optional invariant geometry-context path was also exercised for two epochs
on the same six-record training split. It produced a valid checkpoint with a
batched three-record dev evaluation and frame-audit metrics. This confirms the
feature/cache/checkpoint plumbing; the short run is not an accuracy result.

### Real P1 mechanism grid (dev, 3 pairs)

The frozen FoldFlow-2 endpoint was run through the five endpoint-based
mechanism variants (`C0`, `C2`, `C3`, `C4`, `C5`) on CPU. The report is
`/tmp/ospedit-real-sample/mechanism_dev_real.json`. Mean pair metrics were:

| method | local backbone error (Angstrom) | remote scaffold drift (Angstrom) | distance-change error | total seconds | model calls |
|---|---:|---:|---:|---:|---:|
| C0 copy parent | 0.1565 | 0.0000 | 0.2316 | 0.00 | 0 |
| C2 shared single-noise | 0.1661 | 0.0228 | 0.2380 | 21.62 | 6 |
| C3 shared two-noise | 0.2486 | 0.0616 | 0.2529 | 43.43 | 12 |
| C4 repeated single-noise | 0.1632 | 0.0153 | 0.2360 | 43.28 | 12 |
| C5 independent-noise | 18.7879 | 28.4908 | 50.8951 | 21.89 | 6 |

This small run does not support a quality improvement claim: copying the
parent remains the strongest local-error baseline, while independent noise is
catastrophic. It does establish the expected shared-noise stability pattern
and provides a concrete P1 negative result. Runtime accounting now reports
actual conditional model evaluations rather than merely counting batches.

The same dev grid was rerun after switching the environment audit to the
command-level `--foldflow-root` workflow, with `difference_step_size=0.05`.
The refreshed report is `/tmp/ospedit-real-sample/mechanism_dev_real_latest.json`.
The mean local errors for C0/C2/C3/C4/C5 were `0.1565/0.1564/0.1563/0.1565/2.7629`
Angstrom, while remote scaffold drift was
`0.0000/0.0027/0.0061/0.0019/3.4615` Angstrom. C5 again failed sharply, and
the report accounted for `6/12/12/12/6` conditional network calls respectively.
This confirms the command-level environment path reproduces the earlier
shared-noise stability pattern; it does not change the negative accuracy
conclusion on this three-pair development subset.

### Direct student scale probe

Using the same six train and three dev pairs, a 50-epoch direct-supervision
student without an output bound diverged to a mean dev local error of about
`3.31 Angstrom` and predicted distance-change norms around `741--777`, far
above the experimental changes. Adding `--max-normalized-delta 0.1` reduced the
mean local error to about `0.201 Angstrom` and kept remote scaffold drift near
`0.123 Angstrom`, although copy-parent remained better at `0.157 Angstrom`.
This confirms output-scale control is necessary for student training, but the
small split still provides no improvement claim.

A matched bound run with `--mutation-loss-weight 4` gave mean dev local error
about `0.208 Angstrom`, versus `0.201 Angstrom` without the extra weighting,
and slightly increased remote drift. On this split, emphasizing the mutated
residue alone is therefore not a useful default; the response target needs
better local geometry or side-chain supervision rather than a larger scalar
weight.

Enabling the invariant geometry features under the same bound (`--geometry-features
--max-normalized-delta 0.1`) reduced mean local error to roughly `0.184
Angstrom` and remote drift to `0.103 Angstrom`. This is substantially better
than the unbounded student, but still worse than copy-parent (`0.157 Angstrom`)
and does not yet recover the observed distance changes. Geometry context is
therefore retained as a candidate ablation, not a selected default.

A matched 25-epoch geometry-feature run on the frozen split was worse than the
zero-initialized baseline: test local errors were `1.766`, `1.541`, `1.413`, and
`2.093` Angstrom, with predicted distance-change norms of `238--556`. The
additional channels are therefore retained as an explicit negative ablation,
not enabled by default. The next student iteration should change the target
parameterization or add geometry-aware endpoint supervision rather than add
more parent feature channels.

Smooth-L1 target supervision (`beta=0.25`) was also tested for 25 epochs with
the same architecture and split. It did not improve stability: test local
errors were `1.274`, `0.664`, `2.028`, and `2.105` Angstrom, while predicted
distance-change norms remained `176--501`. Robust regression alone is therefore
not the current bottleneck; future work should constrain the geometric update
parameterization or gate its spatial support.

An explicit normalized-delta bound was then tested (`max_normalized_delta=0.5`,
25 epochs, otherwise identical). This was the most stable student variant so
far: test local errors were `0.480`, `0.421`, `0.421`, and `0.478` Angstrom;
predicted distance-change norms were `76--186`, and parent-frame RMSDs were
`0.429--0.415` Angstrom. It still does not beat copy-parent on this tiny test
set, but the bounded parameterization substantially reduces the prior
unbounded-response failure and is now the primary student ablation.

### Bounded-delta sweep

Keeping the split, seed, architecture, and optimizer fixed, the normalized
bound sweep was:

| max normalized delta | test local error range (Angstrom) | predicted distance-change norm range | parent-frame RMSD range (Angstrom) |
| ---: | ---: | ---: | ---: |
| 0.10 | 0.130--0.197 | 16--38 | 0.086--0.087 |
| 0.25 | 0.233--0.276 | 40--95 | 0.210--0.213 |
| 0.50 | 0.421--0.480 | 76--186 | 0.415--0.429 |
| 1.00 | 0.624--0.794 | 141--301 | 0.653--0.789 |

On this four-record test subset, `0.10` is the best preservation-response
trade-off and is the current development candidate. The sweep is too small to
freeze a biological default; the value must be confirmed on a larger
family-held-out development set.

The automated sweep was then run over all 13 audited MutData2022 pairs, using
the frozen `train=6/dev=3/test=4` split and the same 25-epoch configuration:

| max normalized delta | test family-macro local error (Angstrom) | frame RMSD (Angstrom) | distance-change error (Angstrom) | mean seconds/sample |
| ---: | ---: | ---: | ---: | ---: |
| 0.10 | 0.1506 | 0.0865 | 0.2584 | 0.0493 |
| 0.25 | 0.2489 | 0.2120 | 0.3919 | 0.0498 |
| 0.50 | 0.4503 | 0.4255 | 0.6433 | 0.0489 |
| 1.00 | 0.7198 | 0.7337 | 1.0330 | 0.0500 |

The larger family-macro check preserves the same monotonic trend as the small
diagnostic subset. It supports using `0.10` as the development candidate while
leaving the final claim pending a larger, independently sourced evaluation.

For a direct same-test-set control, copy-parent on these four test pairs has
family-macro local error `0.1127 Angstrom`, distance-change error `0.2125
Angstrom`, and parent-frame RMSD `0.0`. The `0.10` student is currently
`0.1506`, `0.2584`, and `0.0865`, respectively. Thus the bounded student has
not yet beaten the copy baseline; its present value is that it produces a
non-zero, bounded response while remaining substantially more stable than the
unbounded student. Accuracy improvement remains an open research result.

The refreshed 25-epoch bound sweep also reports strict mutation-site metrics.
On the three-record dev split, copy-parent mutation-site global RMSD was
`0.2247 Angstrom`; the bounded student values were `0.3048`, `0.3033`,
`0.3583`, and `0.8799 Angstrom` for bounds `0.10`, `0.25`, `0.50`, and `1.0`.
This agrees with the neighborhood metric: every tested student bound is worse
than copy-parent on this split.

An independent train-to-test sweep on the same frozen 13-pair manifest gives
the same ordering. Copy-parent test local error and mutation-site global RMSD
were `0.1127` and `0.1115 Angstrom`; at bound `0.10`, the student values were
`0.1744` and `0.1576 Angstrom`, with student-minus-copy gaps `+0.0617` and
`+0.0460`. Bounds `0.25/0.50/1.0` increased local error to `0.264/0.440/0.793`
Angstrom. This test report is stored under
`/tmp/ospedit-real-sample/bound_sweep_site_test/` and was not used for tuning.

The student was also rerun with the default sinusoidal residue positional
encoding. On the same four test pairs it gave local errors `0.129`, `0.150`,
`0.125`, and `0.206 Angstrom`, with parent-frame RMSDs `0.092--0.095
Angstrom`; this was neutral to slightly worse than the pre-position bounded
baseline. Positional encoding remains optional and is exposed in the bound
sweep script for exact ablations.

With the `0.10` bound fixed, a mutation-focused loss weight of `4.0` was also
tested. It kept parent-frame drift at `0.087--0.088 Angstrom` and produced local
errors of `0.129--0.193 Angstrom`, with distance-change norms `16--38`. The
effect is small on four test pairs and does not establish a winner over weight
`0`; it remains an optional ablation for larger family-held-out training.

For the separate 50-epoch direct-supervision run on the six-record train and
three-record dev split, relaxing the bound was also harmful: bounds `0.25`
and `0.50` produced mean dev local errors `0.298` and `0.486 Angstrom`,
respectively, compared with `0.201 Angstrom` at `0.10`. This independent
probe is consistent with the earlier 25-epoch sweep, while remaining too small
to freeze the bound globally.

A matched 50-epoch Smooth-L1 run (`beta=0.25`) at bound `0.1` produced mean
dev local error about `0.216 Angstrom`, versus `0.201 Angstrom` for MSE, with
similar remote drift. Robust loss alone is not an improvement on this split
and remains an explicit negative ablation.

The local MutData2023 sample contains 30 of 32 pairs under only two parent
structures, so the connected-component split correctly yields `30 train / 0
dev / 2 test`; splitting those parent groups would leak scaffold information.
A 25-epoch bounded geometry student trained on the 30 records reached `0.593
Angstrom` test local error versus `0.579 Angstrom` for copy-parent. Its
distance-change error was `2.326` versus `2.327` for copy-parent, a negligible
change on this two-record test. This is useful as a data-audit result, not as
evidence of generalization.

`ospedit-p1-preflight` now reports this limitation before a run: the manifest
has 4 unique parents and 16 families but only 2 connected parent-family groups,
with a largest parent group of 15 records. It emits a warning that a nonempty
train/dev/test split is impossible under the no-leakage grouping rule.
For experiments that require all three splits, pass
`--require-split-capacity` so this condition fails the preflight instead of
remaining a warning.
The report also records the current `train/dev/test` record counts, so a
manifest with enough groups but an already-empty assigned split is rejected in
the same strict mode.

The PreMut importer was validated against three existing MutData2022 pairs.
PreMut's `Mutation INFO` index is zero-based (for example `D_59_N` maps to
manifest residue index 59); the importer now enforces that convention. The
probe converted all three rows with `accepted=3`, `missing=0`, and `invalid=0`.

The importer was then run on all 32 locally available MutData2023 pairs with
the upstream cluster dictionary. It accepted all `32/32` rows with no missing
or invalid records and produced the expected `30 train / 0 dev / 2 test`
grouped split. The cluster file does not contain every parent in this small
release, so unmatched parents correctly use the documented mutation-label
fallback.

## Frozen-config test replay

The new sweep/selector path was exercised end to end on the same 13-pair
manifest. A single real `dev` configuration was selected with explicit limits
of `0.1` Angstrom remote drift and `0.1` Angstrom local regression versus C0;
the stricter `0.05` drift limit correctly produced no candidate. The selected
configuration was then replayed on the locked `test` split using
`mechanism_config_real_dev.json`.

The four-record test family means were:

| method | local backbone error | distance-change error | remote drift |
| --- | ---: | ---: | ---: |
| C0 copy parent | 0.1639 | 0.2959 | 0.0000 |
| C2 single-noise difference | 0.2078 | 0.3219 | 0.0922 |
| C3 two-noise shared difference | 0.2836 | 0.3551 | 0.1438 |
| C4 repeated single-noise control | 0.2225 | 0.3143 | 0.0808 |
| C5 independent-noise difference | 57.0350 | 143.1648 | 67.5213 |

The frozen test report is `/tmp/ospedit-real-sample/mechanism_test_real_frozen.json`
with CSV at `mechanism_test_real_frozen.csv`. This is a negative result for
the current endpoint and scale: C0 remains strongest, while the provenance
and held-out replay path now work end to end.

A three-step real C3/C4 scale sweep was also attempted with the new
`--methods` filter (C0, C3, and C4 only). It exceeded the 300-second CPU
budget before the atomic report write, so no partial result was retained. This
is an execution-throughput limitation rather than a quality result; future
calibration should use the GPU environment or a smaller development subset.

The new bounded diagnostic path was then validated on one deterministic `dev`
record (`--max-records 1`). Three difference steps completed in about one
minute on CPU. C3 local error changed from `0.1643` at step `0.05` to `0.1623`
and `0.1598` at steps `0.1` and `0.2`, while remote drift increased from
`0.0037` to `0.0073` and `0.0146` Angstrom. C4 showed the same direction but
remained closer to C0. This is an exploratory single-record signal only; it is
not used to freeze a production configuration.

The two-noise weight path was also checked on the same record at step `0.2`.
Equal weights `(1,1)` gave C3 local error `0.1598 Angstrom`, distance-change
error `0.1964`, and remote drift `0.0146`; low-noise emphasis `(2,0)` gave
`0.1637`, `0.1994`, and `0.0111`; high-noise emphasis `(0,2)` gave `0.1634`,
`0.1965`, and `0.0276`. The response changes as expected, but equal weights
are best on this single sample and no weight default is selected from it.

Using the grid `--methods` filter, the fixed `test` split was evaluated again
at difference step `0.2` with only C0 and C3. C3 obtained mean local error
`0.1618 Angstrom`, distance-change error `0.2150`, and remote drift `0.0182`;
C0 obtained `0.1127`, `0.2125`, and `0.0000`. Compared with the earlier
step-`1.0` C3 test result (`0.2836` local error and `0.1438` drift), the smaller
step is substantially more stable, but still does not beat C0. The report is
`/tmp/ospedit-real-sample/mechanism_test_step02_c3.json`.

The matched-budget C4 control was evaluated on the same test split at step
`0.2`. C4 reached mean local error `0.1158 Angstrom`, distance-change error
`0.2134`, and remote drift `0.0100`, versus C0's `0.1127`, `0.2125`, and
`0.0000`. This is much more stable than the earlier step-`1.0` C4 result and
is close to C0, but it requires 16 network calls for four records. The report
is `/tmp/ospedit-real-sample/mechanism_test_step02_c4.json`.

The current incremental sweep/selector implementation was then exercised on
one deterministic `dev` record with C0/C3, step `0.2`, equal two-noise weights,
and a four-call budget. It produced a completed `1/1` sweep and a frozen
configuration containing the full noise grid, weights, method list, and
manifest fingerprint:
`/tmp/ospedit-real-sample/mechanism_sweep_real_current_v2.json` and
`mechanism_config_real_current_v2.json`. This confirms the latest provenance
and cost constraints work on a real endpoint; it remains a one-record
diagnostic, not a model-selection claim.

## Current C0-C5 grid rerun

After the runtime-summary schema update, the three-record `dev` split was
rerun with the default grid settings and checksum verification. The report is
stored at `/tmp/ospedit-real-sample/mechanism_dev_real_current.json` (CSV:
`mechanism_dev_real_current.csv`). The report exposes both
`condition_branches` and `network_calls`; the edited methods reported C2=6,
C3=12, C4=12, and C5=6 condition branches for the three records.

At this uncalibrated step scale, C4 had lower local error and remote drift than
C3, while C5 again produced catastrophic drift. C0 remained the strongest
local-error baseline on this tiny development split. These results are a
runtime/accounting and mechanism sanity check, not a benchmark claim; scale
selection and family-balanced evaluation remain open.

The same near-response student configuration (`bound=0.1`, 10 epochs, fixed
seed) was then evaluated on its four-record held-out `test` split. Local
backbone error was `0.11258 Angstrom` versus `0.11271` for copy-parent, and
mutation-site error was `0.10925` versus `0.11155`; distance-change error was
slightly worse (`0.21275` versus `0.21254`) and parent-frame RMSD was
`0.00650 Angstrom`. This is a small, near-copy test signal with no claim of
generalization; it supports repeating the low-bound experiment on more
independent families.

Target-scale audit explains why the bounded student is currently conservative:
with `translation_scale=1.0 Angstrom` and `rotation_scale=0.25 radian`, the
maximum per-residue normalized target delta across the six train pairs ranged
from `11.12` to `86.03`, and the three dev pairs ranged from `0.98` to `76.91`.
The student output bound is `0.1`, so large-response pairs cannot be matched by
that configuration. These pairs also mix substantial parent/mutant state
changes with near-local edits. The next student dataset should stratify or
filter response magnitude before selecting output bounds; changing the bound
alone would confound scale and state coverage.

As a concrete diagnostic, `scripts/audit_response_scale.py` with
`--max-normalized-norm 12` produced
`/tmp/ospedit-real-sample/pairs2022_near12.jsonl` (9 records: `3 train / 2 dev /
4 test`). The subset has 6 connected parent-family groups, 6 unique parents,
and no parent contributes more than 3 records, so its split-capacity check
passes. The corresponding response report is
`/tmp/ospedit-real-sample/response_scale_pairs2022_max12.json`. P1 preflight
found no manifest errors; its overall `ready=false` in the default environment
only because FoldFlow dependencies are unavailable, so this subset still
requires the isolated FoldFlow environment gate before a real teacher run.

A five-epoch direct-supervision bound sweep on this near-response subset tested
`0.1`, `0.5`, `1.0`, and `2.0` with fixed seed and scales. The `0.1` bound
gave mean dev local error `0.1464` versus copy-parent `0.1482`, and mutation-site
error `0.2071` versus `0.2253`; bounds `0.5`, `1.0`, and `2.0` increased local
error to `0.1656`, `0.2247`, and `0.3568` respectively. Parent-frame RMSD also
rose from `0.0231` to `0.3711` across the range. The full sweep is at
`/tmp/ospedit-real-sample/near12_bound_sweep/summary.json`. This is only a
two-record dev diagnostic and is not a frozen hyperparameter result; it
motivates repeating the low-bound hypothesis after adding independent families.

The low-bound hypothesis was repeated over seeds `0`, `1`, and `2` with bounds
`0.1`, `0.5`, and `1.0` (the three-record train/two-record dev subset). For
`0.1`, the mean overall local-error difference from copy-parent was `+0.00016
Angstrom` (population standard deviation `0.00077`), while mutation-site error
improved by `-0.01504 Angstrom` (standard deviation `0.00089`). Bounds `0.5`
and `1.0` regressed overall local error by `+0.02407` and `+0.08242 Angstrom`
and increased parent-frame RMSD. Per-seed reports are under
`/tmp/ospedit-real-sample/near12_bound_seed{0,1,2}/summary.json`. This supports
a mutation-site-only signal for the low bound, not a general student win.

The same comparison was rerun through the multi-seed sweep CLI in one command
(`bounds=0.1,0.5,1.0`, `seeds=0,1,2`, five epochs). The consolidated report is
`/tmp/ospedit-real-sample/near12_multiseed_final/summary.json` and contains 9
run records plus per-bound mean/std aggregates. Its values match the manual
three-directory aggregation above, confirming the reproducible multi-seed
reporting path.

The same 9-run multi-seed sweep was evaluated on the held-out four-record
`test` split at `/tmp/ospedit-real-sample/near12_multiseed_test/summary.json`.
For `bound=0.1`, overall local-error difference from copy-parent averaged
`+0.00071 Angstrom` (std `0.00066`), while mutation-site error improved by
`-0.00272 Angstrom` (std `0.00114`). Bounds `0.5` and `1.0` regressed overall
local error by `+0.03014` and `+0.09263 Angstrom`. The low-bound test effect is
therefore directionally consistent but small, and remains a development signal
rather than a generalization claim.

## Current real distillation smoke

The train-side admitted cache was consumed by a direct-plus-distillation
student run on the six-record `train` split (one epoch, hidden size 64, two
blocks, `distill_weight=0.1`, normalized output bound `0.1`, checksum
verification enabled). The checkpoint is
`/tmp/ospedit-real-sample/student_distill_train_current.pt`; its configuration
stores the teacher-cache fingerprint, teacher noise level `0.25`, and the full
manifest fingerprint. A three-record `dev` evaluation completed in the same
run. The final training loss was `234.686`; this is a pipeline smoke only, not
evidence that distillation improves mutant-structure accuracy.

A fixed 10-epoch rerun used the same train cache, architecture, seed, and
distillation weight. Its checkpoint is
`/tmp/ospedit-real-sample/student_distill_train_10e_current.pt` and final loss
was `234.666`. On the three-record `dev` split, mean local backbone error was
`0.15710` for the student versus `0.15652` for copy-parent; mean
distance-change error was `0.23141` versus `0.23165`, while remote scaffold
frame drift increased to `0.01048 Angstrom` and predicted distance-change
norm averaged `3.85` (copy-parent: `0`). This is a negative accuracy result:
the student learned a nonzero update but did not improve local accuracy on the
current tiny split.
