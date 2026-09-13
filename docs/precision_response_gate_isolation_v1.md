# Precision response-gate isolation (2026-09-13)

Runs used the `BIO` conda environment on Precision (two W7900 GPUs),
`protocol_v02_within_family_probe_v1.jsonl`, cached ESM2-T6 context, CUDA,
batch size 4, hidden size 32, two transformer blocks, and learning rate
`1e-4`. The remote checkout was `/media/990Pro/ospedit`.

## Results

### Response-supervised, unbounded delta

Eight epochs with `gate_response_loss_weight=0.05` completed with finite
weights and final loss `0.03228`. On the training split, the aggregate gate
means were approximately `0.245` on response residues and `0.243` on stable
residues. The corresponding per-family AUROC values ranged from `0.52` to
`0.81`; this is a diagnostic signal, not evidence of a generalizable gate.
The model predicted only a small fraction of the measured response energy
(roughly 0.5--2.1% by family), so stability was obtained mostly by shrinking
the update.

### Response-supervised, bounded delta

Two epochs with `max_normalized_delta=0.25` also completed with finite
weights and final loss `0.03238`. Gate response/stable means again differed by
less than `0.001` per family. The bound reduced predicted change magnitude
further; it did not provide evidence of selective localization.

## Interpretation

The response loss itself is not an immediate source of non-finite values in
the short unbounded run, and the current clamp-based bound is stable in the
short bounded run. A later 30-epoch isolation reproduced the failure only for
CUDA with response loss enabled: the same balanced configuration without the
response loss completed on CUDA, and the response-loss configuration completed
on CPU. With CUDA, the first non-finite gradient was reported at
`epoch=28, batch=1` in `context.layers.1.norm2.weight`. This identifies an
interaction between the response-loss backward path and the CUDA Transformer
kernel/precision path, rather than a bound-only failure or an unconditional
ROCm failure.

Neither run supports continuing a sparsity-weight sweep:
the gate remains close to a uniform multiplier. The next experiment should
use direct gate targets together with explicit delta-scale control, then
evaluate gate AUPRC/AUROC and response/stable gate separation on held-out
families. These runs are calibration diagnostics only; they are not a final
generalization result.
