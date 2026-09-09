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
