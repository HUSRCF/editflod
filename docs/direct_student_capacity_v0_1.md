# Direct student capacity check v0.1

This check follows protocol v0.2 and does not use FoldFlow supervision. It asks
whether the output representation and two directly supervised students can fit
the seven training pairs in `pairs2022_aligned_frozen.jsonl`. The dataset is too
small and context-confounded for a generalization claim.

All coordinate metrics use `ospedit.structure_metrics.v2` (standard atom RMSD).
Both students used hidden dimension 32, two blocks, batch size 1, learning rate
0.001, mutation loss weight 4, neighborhood loss weight 1, no output bound, and
seed 0. The graph used 16 parent spatial neighbors. Results below are
parent-family macro means after 700 optimizer steps.

| Method | Local error (A) | Mutation-site error (A) | Distance-change error (A) | Distance-change cosine | Remote target error (A) | Remote frame drift (A) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Copy parent | 0.2665 | 0.2992 | 0.3195 | undefined | 0.4117 | 0.0000 |
| Transformer student | 0.1836 | 0.0932 | 0.1958 | 0.603 | 0.2817 | 0.2655 |
| Spatial graph student | 0.2159 | 0.0912 | 0.2272 | 0.476 | 0.3133 | 0.2225 |

The direct students clearly learn non-zero training responses and outperform
copy-parent on target accuracy. The first graph implementation is more
reference-preserving than the matched Transformer but worse on local and
distance-change recovery. This is a mixed result, not evidence that explicit
spatial edges improve the overall tradeoff.

The unbounded local-frame oracle has zero distance-change error but 0.1362 A
local atom RMSD, exposing the rigid-residue representation floor. Component
bound 0.1 raises the local floor to 0.4582 A, so subsequent capacity tests should
not restore that bound without a specific preservation analysis.

Tracked machine-readable artifacts:

- `reports/oracle_pairs2022_aligned_v1.json`
- `reports/transformer_overfit7_700steps_v1.json`
- `reports/spatial_graph_overfit7_700steps_v1.json`

Next decision: improve spatial propagation (four to six blocks or a global
context path) and evaluate a held-out-family split only after the graph model
matches the Transformer on this training-capacity gate.

## Frozen extrapolation diagnostic

The same 700-step checkpoints were evaluated once on the existing dev and test
splits without further tuning. They do not establish generalization.

| Split / method | Local error (A) | Site error (A) | Distance-change error (A) | Distance-change cosine | Remote frame drift (A) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dev / copy | 0.2179 | 0.2462 | 0.2252 | undefined | 0.0000 |
| Dev / Transformer | 0.5299 | 0.6226 | 0.5257 | 0.140 | 0.5906 |
| Dev / spatial graph | 0.3325 | 0.4007 | 0.4108 | 0.106 | 0.4143 |
| Test / copy | 1.2705 | 0.9948 | 0.7667 | undefined | 0.0000 |
| Test / Transformer | 1.2708 | 1.0074 | 0.7651 | 0.063 | 0.1812 |
| Test / spatial graph | 1.2631 | 0.9833 | 0.7668 | 0.061 | 0.1193 |

Both models overfit: dev performance is substantially worse than copy-parent,
and test performance is essentially at the copy baseline. The spatial graph is
less destructive than the Transformer, but its tiny test local/site changes are
not accompanied by better distance-change recovery. The next scientific
constraint is therefore data-supported generalization, not more teacher scans.
