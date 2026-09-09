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
