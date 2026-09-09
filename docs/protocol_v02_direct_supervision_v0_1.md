# Protocol v0.2 direct-supervision diagnostic

The frozen structure-prediction layer contains 101 experimental pairs in 27
connected sequence/family/parent groups. The deterministic split has 71 train,
15 dev, and 15 test records. The largest 44-record connected group is wholly in
train. Parent and family identifiers have zero cross-split overlap.

Three students were trained without FoldFlow labels for 20 epochs (1420
optimizer steps), using hidden dimension 32, two primary blocks, batch size 1,
family-balanced loss, learning rate 0.001, mutation weight 4, neighborhood
weight 1, and no output bound. The spatial models use 16 neighbors. The hybrid
adds one global attention block after two spatial message blocks.

All values are parent-family macro means under
`ospedit.structure_metrics.v2`.

| Split / method | Local error (A) | Site error (A) | Distance-change error (A) | Distance-change cosine | Remote target error (A) | Remote frame drift (A) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train / copy | 0.2941 | 0.3678 | 0.2362 | undefined | 0.3425 | 0.0000 |
| Train / Transformer | 0.2887 | 0.3117 | 0.2390 | 0.155 | 0.3441 | 0.1202 |
| Train / spatial graph | 0.2843 | 0.2647 | 0.2348 | 0.115 | 0.3414 | 0.0386 |
| Train / graph + global | 0.2785 | 0.2444 | 0.2365 | 0.205 | 0.3451 | 0.0972 |
| Dev / copy | 0.3374 | 0.2795 | 0.2406 | undefined | 0.3287 | 0.0000 |
| Dev / Transformer | 0.3600 | 0.3212 | 0.2816 | -0.006 | 0.3680 | 0.1203 |
| Dev / spatial graph | 0.3419 | 0.2913 | 0.2420 | -0.004 | 0.3301 | 0.0370 |
| Dev / graph + global | 0.3458 | 0.2964 | 0.2504 | 0.038 | 0.3412 | 0.0975 |

The expanded dataset removes the seven-pair capacity ambiguity but does not yet
produce a useful held-out editor. Spatial relations make the model markedly
less destructive than the Transformer. Adding global attention improves train
fit and response cosine, but worsens dev preservation and target error. This is
evidence of stronger memorization, not improved generalization.

The hybrid did not pass the dev gate, so it was not evaluated on test. The
already evaluated Transformer and spatial graph were also below copy-parent on
test; those reports are retained as diagnostics, not model-selection evidence.

Next gate: characterize the v0.2 target-response distribution and representation
floor by family. Training should then distinguish low-response preservation
examples from resolvable-response examples instead of forcing one uncalibrated
update policy across both.

## Representation and regional-loss follow-up

The v0.2 oracle confirms that the output representation can express most of the
held-out target change:

| Split | Copy local (A) | Unbounded oracle local (A) | Bound-0.1 oracle local (A) | Unbounded distance-change error (A) |
| --- | ---: | ---: | ---: | ---: |
| Train | 0.2941 | 0.0829 | 0.2123 | 0.0000 |
| Dev | 0.3374 | 0.0879 | 0.2425 | 0.0000 |
| Test | 0.4806 | 0.1944 | 0.4057 | 0.0000 |

We then changed the training objective from one whole-chain weighted mean to
independently normalized whole-chain, mutation-site, and mutation-neighborhood
losses (`ospedit.student_loss.v2`). This made the model produce larger local
updates but did not improve dev direction: local error was 0.3625 A, site error
0.3256 A, and distance-change cosine 0.021.

An update-scale sweep on dev used scales 0, 0.1, 0.25, 0.5, 0.75, and 1.0.
Scale 0 (copy-parent) was best for local, site, and remote target error. Scale
0.1 changed distance-change error from 0.24057 to 0.24056 A, which is negligible.
Thus the failure is not explained by an overly large inference step; held-out
directions themselves are not reliable.

The split remains suitable for development but small for broad claims: train
contains 18 connected families, dev 5, and test 4. The largest dev and test
families contain 9/15 and 12/15 records respectively. Family-macro reporting is
mandatory, and further gains require more cross-family signal rather than more
per-mutation rows from one parent.

## Biochemical edit-feature ablation

One pre-registered follow-up added seven fixed target-minus-source residue-class
features to the spatial graph student: acidic, basic, aromatic, polar,
hydrophobic, glycine, and proline membership. All other settings matched the
region-normalized 20-epoch run. This tests whether a minimal biochemical prior
improves held-out direction without introducing a sequence language model.

| Dev method | Local error (A) | Site error (A) | Distance-change error (A) | Distance-change cosine | Remote target error (A) | Remote frame drift (A) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Copy | 0.3374 | 0.2795 | 0.2406 | undefined | 0.3287 | 0.0000 |
| Region loss, no biochemical features | 0.3625 | 0.3256 | 0.2536 | 0.021 | 0.3492 | 0.1175 |
| Region loss + biochemical features | 0.3585 | 0.3633 | 0.2581 | -0.042 | 0.3468 | 0.1022 |

The extra descriptors slightly reduced local error and drift relative to the
otherwise matched student, but worsened mutation-site error, distance-change
error, and response cosine. The model therefore failed the dev gate and was not
evaluated on test. Fixed residue classes alone do not resolve the missing
cross-family response signal.

## Response scope and representation recoverability

`ospedit.response_learnability_audit.v1` compares copy-parent with the exact
unbounded local-frame oracle and reports translation/rotation response scope.
The following values are family-macro means; the test column is frozen-dataset
characterization only and was not used for model selection.

| Split | Families | Copy local (A) | Oracle local (A) | Recoverable local fraction | Local translation energy | Remote translation energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 18 | 0.2941 | 0.0828 | 0.697 | 0.179 | 0.676 |
| Dev | 5 | 0.3374 | 0.0879 | 0.734 | 0.175 | 0.648 |
| Test | 4 | 0.4806 | 0.1944 | 0.628 | 0.381 | 0.516 |

The representation can remove roughly 70% of train/dev local error and almost
all pairwise distance-change error, so the local-frame output is not the main
bottleneck. In contrast, most translation energy lies more than 15 A from the
mutation in train and dev. The paired endpoint therefore asks the model to
predict widespread structural differences that may include experimental-state
variation; this audit alone cannot attribute those differences to mutation.

## Conditioning and mutation-distance diagnostics

A seed-0 target-identity ablation retained parent residue one-hot channels and
the mutation marker but zeroed all target residue one-hot channels. It worsened
dev family-macro local error from 0.3625 to 0.3699 A and mutation-site error from
0.3256 to 0.4625 A. The full model therefore uses target identity, although
neither variant beats copy-parent.

We also repeated the region-loss graph student with and without the optional
mutation-distance geometry channels over seeds 0, 1, and 2:

| Dev configuration | Local error (A) | Site error (A) | Distance-change error (A) | Distance-change cosine | Remote frame drift (A) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base spatial graph | 0.3586 +/- 0.0087 | 0.3507 +/- 0.0472 | 0.2552 +/- 0.0038 | -0.023 +/- 0.032 | 0.1010 +/- 0.0154 |
| + mutation-distance geometry | 0.3614 +/- 0.0061 | 0.3909 +/- 0.0449 | 0.2593 +/- 0.0084 | -0.051 +/- 0.016 | 0.1094 +/- 0.0442 |

The apparent seed-0 drift reduction did not replicate. Explicit mutation
distance does not improve held-out response under the current full-endpoint
target. The next intervention should separate local mutation-conditioned
supervision from remote endpoint variation or add matched-state controls;
adding another propagation layer is not supported by these results.

## Localized-target and localized-output ablation

The response-scope audit motivated one explicitly restricted local editor. It
uses the exact experimental local-frame delta through 10 A, a cosine taper from
10 to 15 A, and a zero target beyond 15 A. The same window is applied to the
predicted delta at inference. This tests a preservation-first local task; it
does not assume distal mutation responses are universally absent.

On dev, the record-macro unbounded localized oracle has local error 0.1047 A, local
distance-change cosine 1.000, and zero remote frame drift. Thus the restricted
target retains a substantial representable local signal.

| Seed-0 dev method | Local error (A) | Site error (A) | Distance-change error (A) | Distance-change cosine | Remote target error (A) | Remote frame drift (A) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Copy | 0.3374 | 0.2795 | 0.2406 | undefined | 0.3287 | 0.0000 |
| Full target + localized output | 0.3613 | 0.4557 | 0.2437 | 0.014 | 0.3287 | 0.0000 |
| Localized target + localized output | 0.3592 | 0.4035 | 0.2420 | 0.037 | 0.3284 | 0.0000 |

Localized supervision improves the matched post-hoc-window control on local,
site, distance-change, and direction metrics, so separating the target has a
measurable effect. It still fails the copy-parent gate, particularly at the
mutation site, and was therefore not repeated across seeds or evaluated on
test. Hard preservation solves unrelated remote drift but not held-out mutation
response prediction.

## Same-sequence background-control audit

All available repeat-structure batches were recomputed with the corrected
standard atom RMSD definition and consolidated by mutation pair, repeat PDB,
and repeat chain. This last key matters because one PDB file can contain
multiple same-sequence chains with different coordinates. The background-only
index contains 247 distinct control chains and covers 37/101 mutation pairs,
30/56 parents, and 18/27 families.

| Split | Covered pairs | Total pairs | Covered families | Total families |
| --- | ---: | ---: | ---: | ---: |
| Train | 19 | 71 | 11 | 18 |
| Dev | 11 | 15 | 3 | 5 |
| Test | 7 | 15 | 4 | 4 |

The endpoint signal is the copy-parent error, not an assumed mutation-only
effect. On dev, its median ratio to the maximum same-sequence background is
0.769 locally, 0.545 at the mutation site, and 0.606 for distance change. Using
the median rather than maximum background raises these ratios to 1.081, 1.299,
and 0.987, respectively, but does not establish a clean response target.

On train, maximum background and endpoint signal have Spearman correlations of
0.695 locally, 0.665 at the site, and 0.560 for distance change. This is
consistent with experimental-state variation contributing materially to both
the repeat controls and parent-mutant endpoints. It is not evidence that the
background causes the endpoint difference.

Among records with at least two repeat controls, no train, dev, or test pair
simultaneously passes the frozen maximum-background local and distance
signal/background threshold of 2. At a threshold of 1, only 5 train, 3 dev, and
1 test pairs pass. The localized seed-0 student also worsens local error versus
copy-parent on all 11 covered dev records, including the three with both ratios
above 1.5.

Consequently, repeat-derived target weighting or background subtraction is not
admitted for training: train coverage is only 19/71, the controls have not all
been matched for crystal environment and ligand/contact state, and weights
derived from parent-mutant endpoint signal would leak target information. The
tracked reports are:

- `reports/protocol_v02_background_control_coverage_v1.json`: parent-side
  same-sequence controls, per-control metrics, max/median aggregation, and
  coverage.
- `reports/protocol_v02_background_signal_diagnostic_v1.json`: explicitly
  diagnostic endpoint-to-background ratios and correlations.

The next data gate is therefore a matched-environment response set, not another
student hyperparameter sweep. Repeat controls must be checked for biological
assembly, target-chain contacts, ligand state, construct, and crystal form;
training labels must remain independent of any confidence feature available
only after observing the mutant endpoint.

### Coordinate-file context prescreen

A follow-up prescreen compares hetero residues within 6 A of the target chain,
the presence of other protein-chain contacts within 5 A, and experimental
method. It retains 200/247 controls and leaves at least one control for 27/37
pairs (16 train, 9 dev, and 2 test). The 47 rejections comprise 39 proximal
hetero mismatches, 14 protein-contact-presence mismatches, and 2 method
mismatches; reasons can overlap.

This filter does not rescue response identifiability. Among dev records with at
least two prescreened controls, median endpoint/max-background ratios are 0.593
locally, 0.520 at the mutation site, and 0.551 for distance change. No split has
a record passing the frozen maximum-background local and distance thresholds
of 2. The result strengthens the decision not to train with target-derived
confidence weights.

The prescreen explicitly reports `biological_assembly_verified: false`.
Asymmetric-unit contacts, ligand names, and experiment method are useful
automatic checks but cannot establish biological-assembly, construct, crystal
form, or occupancy equivalence. The detailed artifact is
`reports/protocol_v02_repeat_control_context_v1.json`.

### RCSB assembly and crystallographic metadata audit

The coordinate-file prescreen was followed by an official RCSB Data API audit.
Target author chains are mapped to label asymmetry identifiers before comparing
biological-assembly membership. Crystal-form compatibility additionally
requires matching space group and unit-cell lengths/angles within the frozen
10% and 5 degree tolerances.

For same-sequence controls, 198/247 pass the assembly tier and 108/247 pass the
crystal-form tier. The latter cover 24 mutation pairs and 10 families: 13/7 in
train, 9/2 in dev, and 2/1 in test. Intersecting those controls with compatible
parent-mutant endpoints leaves 12 train pairs, 6 dev pairs, and no test pairs.
With at least two controls and both local and distance endpoint/max-background
ratios at least 1, only four train records and one dev record remain.

The endpoint audit finds 93/101 assembly-compatible and 86/101 crystal-form
compatible pairs. A stronger same-primary-citation plus crystal-form tier has
17 pairs across 11 families; adding exact declared crystal-growth signatures
leaves four pairs across three families. These tiers are evidence layers, not
proof that mutation is the only changed experimental variable.

Five records entered a target-aware manual review queue. Abstracts, PDB titles,
citations, and deposited crystallization metadata identified state, redox,
cofactor, pH, construct-source, or multi-conformer concerns in every case. No
record was admitted as a strict mutation-attribution benchmark. Four remain
useful as explicitly state-aware challenge cases; the oxidized-parent versus
reduced-mutant FMN case is rejected for mutation-only attribution.

This result prevents a false positive: automated response ratios can prioritize
manual review, but cannot define training weights or a held-out benchmark. The
tracked RCSB endpoint/control reports, review queue, and static decisions make
that evidence chain reproducible without treating endpoint-derived information
as an inference feature.

### Endpoint provenance queue and mutation annotations

A second review queue deliberately excludes observed response magnitude. It
contains the 17 same-primary-citation, crystal-form-compatible endpoints (14
train, one dev, two frozen test) and records whether the deposited crystal-growth
signatures also match. Four satisfy that stronger declared-growth tier.

Legacy PDB `SEQADV` engineered-mutation records were compared with each manifest
edit. They support 91/101 edits directly or through two annotated variants at
the same site; six pairs have no usable engineered-mutation annotation; four
pairs contain annotations only at other sites. Absence is not a rejection
because `SEQADV` is optional. The four discrepancies are manual-review flags.

One flag is already confirmed as a provenance mismatch:
`microminer_5FNX_A_5FZU_A_K177E` compares two N19D inhibitor structures at
different pH values, while the manifest edit is a K177E sequence discrepancy at
author residue 177. It is not a controlled K177E mutation endpoint and must not
be used for mutation-response claims. This finding motivates a source-level
annotation gate for newly discovered structural-neighbor pairs.

An ingestion shadow run on the existing 192-row MicroMiner discovery artifact
produced 35 otherwise usable pairs in diagnostic mode and 26 under the strict
SEQADV option. The nine removed records are exactly the five no-annotation and
four other-site-annotation cases in that artifact. This validates the gate's
mechanics but does not make strict mode the default: the five absent annotations
still require manual evidence rather than automatic rejection.

### Response-independent context decisions and endpoint groups

The 15 non-test records in the citation/crystal queue were reviewed without
using observed displacement or prediction metrics. Eight are retained as
matched-context direct-supervision candidates (seven train and one dev), three
are explicit redox-state mismatch challenges, two depend on oligomeric assembly,
one is the confirmed provenance rejection above, and one is a reverse duplicate
alias. No record is promoted to strict mutation-attribution truth because the
literature evidence is abstract/metadata level rather than full-method review.

The candidate set is a mechanism-development probe, not a new generalization
benchmark: it has only one development record and no reviewed test record. The
two frozen test entries were intentionally not reviewed while defining these
rules. The static decisions are validated against the source queue so test
leakage, missing records, invalid dispositions, and summary drift fail loudly.

An independent manifest audit finds 98 unique physical endpoint groups among
101 records. Two groups are duplicated in the same direction across source
imports; one DHFR endpoint pair appears in both directions. These rows are not
independent experimental evidence. Future student runs should use endpoint-group
balancing, optionally nested within family balancing, and evaluation should
aggregate or bootstrap by physical endpoint group. This changes statistical
weighting, not the validity of learning a directional reverse edit.

### Matched-context representation limit

The static decisions materialize an eight-record probe (seven train, one dev;
six families) without consulting observed response magnitude. The
coordinate-bearing manifest remains outside Git, while the tracked selection
report stores both source and selected manifest fingerprints.

On the five train families, copy-parent local backbone error has a family-macro
mean of 0.1969 Angstrom and the unbounded exact local-frame oracle reaches
0.0858 Angstrom. The mean recoverable fraction is 0.575. On the single dev
record, the corresponding values are 0.2506, 0.0692, and 0.724. The rigid-frame
output therefore has material capacity to beat copy-parent, but it cannot
reconstruct all residue-internal experimental differences.

Across all eight records, the record-macro local error is 0.0754 Angstrom for
the unbounded oracle and 0.1198 Angstrom after applying the student's current
component-wise normalized bound of 0.1. The bound is therefore an active
capacity constraint, not merely a regularizer. Initial overfit experiments
must include an unbounded or substantially relaxed output and report the
bounded oracle beside learned results. These numbers characterize
representation capacity only; the probe has no reviewed test record and does
not establish generalization.

### Direct-supervision overfit probe

Two unbounded students were trained for 500 epochs (3,500 optimizer steps) on
the seven matched-context train records with family-balanced loss, seed 0, no
teacher signal, hidden dimension 32, and two blocks. This is a capacity check,
not a tuned generalization experiment.

The spatial-graph student reduced train family-macro local error from the
copy-parent value of 0.1969 to 0.1113 Angstrom and mutation-site error from
0.3571 to 0.0938 Angstrom; its distance-change cosine was 0.833. The local
feature Transformer reached 0.1067, 0.1031, and 0.871 respectively. Both models
therefore learned substantial non-zero training responses. The earlier
near-copy result did not demonstrate that a one-pass student was intrinsically
unable to learn the task.

Neither model beat copy-parent on the single held-out dev record. Copy-parent
local and site errors were 0.2506 and 0.1307 Angstrom. The spatial graph reached
0.2770 and 0.2320 with distance-change cosine 0.021; the Transformer reached
0.2842 and 0.1938 with cosine 0.041. Both introduced unsupported drift. These
post-training measurements were not used for additional tuning.

This probe passes the train-fit gate but provides no evidence that spatial
edges improve generalization: the Transformer is slightly better on some train
metrics, the graph is slightly better on train site error, and one dev record
cannot resolve the comparison. The next data/model experiment must expand
response-independent train/dev coverage and use multiple seeds before choosing
an architecture. FoldFlow distillation remains paused.

### Expanded structure-prediction development layer

Requiring a shared primary citation is useful for strict mutation attribution,
but it reduced the development set to eight records and one dev example. A
broader layer was therefore frozen using only response-independent endpoint
criteria. It requires compatible biological-assembly signatures, experimental
method, construct length, space group and unit cell; matching proximal ligand
names; matching counts of proximal protein chains and contacting residues; and
no conflicting other-site-only `SEQADV` annotation. Existing manual redox,
assembly, provenance, and reverse-alias decisions remain vetoes.

The resulting layer contains 55 train records from 10 families and eight dev
records from four families. It represents 61 unique physical endpoint groups;
the two same-direction duplicates remain available but must use endpoint-group
balanced loss. The selector processes no frozen-test records and does not use
observed response magnitude. Coordinate-bearing output remains outside Git;
the tracked report records all source-report hashes and the selected manifest
fingerprint.

On this layer, train family-macro copy-parent local error is 0.2354 Angstrom and
the unbounded rigid-frame oracle reaches 0.0820 Angstrom, a mean recoverable
fraction of 0.658. Dev family-macro values are 0.2815, 0.0702, and 0.735. Across
all 63 records, imposing `bound=0.1` raises record-macro oracle local error from
0.0728 to 0.1364 Angstrom. Subsequent multi-seed training must therefore remain
unbounded initially and compare both student architectures under the same
family- and endpoint-group weighting.

This is a structure-prediction supervision layer, not a high-confidence claim
that every observed endpoint difference was caused only by the annotated
mutation. Matching ligand names does not establish occupancy or redox state,
and matching contact counts does not establish identical interface geometry.

### Fixed-budget multi-seed architecture comparison

The expanded layer was used for a predeclared comparison of the local-feature
Transformer and spatial graph. Each architecture used seeds 0/1/2, hidden
dimension 32, two blocks, 64 epochs (3,520 optimizer steps), unbounded outputs,
family and endpoint-group balanced loss, and no teacher signal. Frozen-test
records were rejected by the runner.

Neither architecture passed dev copy-parent noninferiority. Copy-parent dev
family-macro local error is 0.2815 Angstrom. The Transformer obtained
0.3005 +/- 0.0049 and the spatial graph obtained 0.2965 +/- 0.0103 Angstrom.
Their mutation-site errors were 0.3070 and 0.3167 versus copy-parent 0.2559.
Mean distance-change cosine was -0.0068 and -0.0214 respectively. Both models
introduced remote scaffold drift, although the graph's mean drift was smaller.

The Transformer fit train more strongly, while the graph was slightly less bad
on dev local error; neither fact justifies architecture selection after both
failed the primary gate. The result also shows why a single favorable seed is
not adequate: spatial-graph dev local regression ranged from 0.0058 to 0.0293
Angstrom across seeds.

The experimental endpoint deltas are dominated by remote motion: remote
translation and rotation account for about 66% of train family energy, compared
with about 18% locally. That signal may contain true long-range response, but it
also contains endpoint-specific nuisance motion. The next fixed intervention is
therefore a soft localized-target ablation using the already defined 10
Angstrom local region plus a 5 Angstrom cosine taper. It will be compared with
the full-target runs and must still be judged against copy-parent; it is not a
claim that all legitimate remote response should be suppressed.

### Soft localized-target result

The fixed 10 Angstrom radius plus 5 Angstrom cosine taper retains most local
oracle capacity: unbounded record-macro local and mutation-site errors are
0.0770 and 0.0720 Angstrom, compared with 0.0728 and 0.0682 for the full oracle.
It reduces oracle remote drift from 0.1926 to 0.0145 Angstrom, while lowering
whole-structure distance-change cosine from 1.0 to 0.519 as expected.

The same localization was then applied to training targets and inference for
both architectures over the identical three seeds and step budget. It improved
dev local error relative to full-target training for all six paired runs. The
Transformer mean improved from 0.3005 to 0.2893 Angstrom and the spatial graph
from 0.2965 to 0.2873. Mean remote drift fell to 0.0022 and 0.0018 Angstrom.

Localization still did not beat copy-parent local error of 0.2815 Angstrom;
mean regressions were 0.0078 and 0.0058 Angstrom. Mutation-site error also
remained worse than copy, and distance-change cosine remained slightly
negative. The intervention therefore improves the preservation-response tradeoff
but does not yet constitute successful editing. The next fixed experiment will
retain this taper and increase the explicitly normalized mutation-site and
neighborhood loss terms, testing whether local responses are currently diluted
by the many zeroed remote targets.

### Localized regional-weighting result

The fixed follow-up assigned weight 1.0 to each separately normalized
mutation-site and 10 Angstrom neighborhood loss, while retaining the localized
target, architectures, three seeds, and 3,520-step budget. The extra regional
supervision improved training fit, but it did not transfer to held-out families.

Transformer dev family-macro local error increased from 0.2893 to 0.2955
Angstrom relative to unweighted localization, and its mutation-site error
increased from 0.2818 to 0.3167. Spatial-graph local error increased from 0.2873
to 0.2988 Angstrom, while mutation-site error increased from 0.2787 to 0.3645.
The mutation-site regression occurred in all six paired runs. Copy-parent remains
better at 0.2815 local and 0.2559 mutation-site error.

The spatial graph also worsened distance-change cosine in every seed, reaching a
mean of -0.0706, and increased remote drift to 0.0051 Angstrom. The Transformer
mean cosine remained uninformative at -0.0155. This result rejects the narrow
hypothesis that local response was merely diluted by zeroed remote targets.
Further regional-weight sweeps are not justified. The next diagnostic will
decompose held-out error by family and record to determine whether the failure is
associated with particular endpoint contexts, response magnitudes, or a general
cross-family transfer problem; no response-derived filtering will be introduced.

The record-level decomposition found no local-error improvement in any of 24
record-seed comparisons per architecture. All four dev families had positive
mean student-minus-copy local error for both architectures. Mutation-site error
improved in only 2/24 Transformer and 1/24 spatial-graph comparisons; graph
distance-change error improved in 0/24. The failed family-macro result is
therefore not caused by one outlying family.

The next diagnostic will use only the training partition to create
response-independent, physical-endpoint-group holdouts within seen families.
This is not a replacement for frozen dev. It separates two hypotheses: failure
to transfer even within a represented family points back to representation or
targets, while within-family success combined with unseen-family failure points
to coverage and the need for transferable pretrained context.

### Within-family endpoint holdout

A response-independent diagnostic was built from the original training split.
Only four families had at least two physical endpoint groups; six single-group
families (seven records) were excluded. The resulting probe contains 36 train
and 12 dev records, with every dev family represented in train and no endpoint
group crossing the boundary. Frozen dev and test records were not used. Runs
used 98 epochs, or 3,528 optimizer steps, to match the prior 3,520-step budget.

The Transformer improved mutation-site family-macro error in all three seeds,
from copy-parent 0.2656 to 0.2162 +/- 0.0167 Angstrom. It still worsened the
whole local neighborhood by 0.0167 Angstrom and produced only 0.0154 mean
distance-change cosine. The spatial graph worsened both local and site error.
Thus a site-level response is learnable inside represented families, but the
current models do not transfer a coherent neighborhood update.

Exact directed substitutions are also sparse: only 4/12 probe-dev records use
an edit observed in probe train. Across record-seed comparisons, Transformer
local error improves by 0.0105 Angstrom for seen edits and worsens by 0.0110 for
unseen edits. Site error improves in both strata, by 0.0312 and 0.0131 Angstrom.
The next fixed ablation will expose the existing physicochemical edit-difference
features. This tests transferable mutation chemistry without changing the split,
loss, architecture budget, or endpoint selection.

### Biochemical edit-feature ablation

Adding the seven fixed target-minus-source physicochemical channels did not pass
the within-family gate. Transformer local error increased to 0.3054 Angstrom,
or 0.0201 worse than copy, and its site improvement shrank from 0.0495 to 0.0160
Angstrom. The graph's local regression decreased from 0.0160 to 0.0085 Angstrom,
but its site regression increased from 0.0311 to 0.0543.

For unseen directed edits, Transformer local regression was essentially
unchanged (0.01095 to 0.01091 Angstrom). Its unseen-edit site improvement grew,
but the seen-edit site delta changed from -0.0312 to +0.0032 Angstrom. The fixed
descriptor set therefore does not provide a stable transferable edit encoding
and is not selected as the default.

The next intervention will augment training with the reverse direction of each
experimental endpoint pair while leaving dev directions untouched. Reverse
examples are not independent evidence and will remain in the same endpoint
group for weighting. They expand probe-train directed edit types from 31 to 58
without adding teacher labels or crossing endpoint splits.

### Reverse-endpoint augmentation result

All 36 probe-train records received a reverse direction, while the 12 dev
directions remained unchanged. Endpoint-group balancing kept the number and
total weighting of physical pairs fixed. The Transformer used 49 epochs over 72
records, retaining the 3,528-step budget.

Reverse training reduced dev family-macro local regression from 0.0167 to
0.00120 Angstrom and remote drift from 0.00665 to 0.00087 Angstrom. However, it
also reduced the prior mutation-site improvement from 0.0495 to 0.00311
Angstrom. Distance-change cosine was 0.0007 and distance-change error remained
slightly worse than copy. The model predicted a much smaller change norm and
behaved primarily as a more stable near-copy baseline.

Although directed training edit types increased from 31 to 58, none of the
eight previously unseen dev substitutions became exactly covered; they are not
simply reverse forms of the training edits. Reverse augmentation is therefore
retained as a preservation control, not selected as the response model. The
next model intervention must explicitly improve coherent neighborhood response
or bring transferable sequence/structure context; more shrinkage toward zero is
not sufficient.
